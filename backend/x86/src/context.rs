use crate::compile;
use crate::cpu::Cpu;
use crate::cpu_argument::{ArgLock, Argument, CpuArray};
use crate::printer::X86printer;
///! Defines the CPU evaluation context.
use telamon::codegen::ParamVal;

use telamon::codegen;
use telamon::context::{self, AsyncCallback, EvalMode, KernelEvaluator, ScalarArgument};
use telamon::device::{self, Device};
use telamon::explorer;
use telamon::ir;

use crossbeam;
use fxhash::FxHashMap;
use itertools::Itertools;
use libc;
use log::debug;
use std::convert::TryFrom;
use std::f64;
use std::io::Write;
use std::sync::{mpsc, Arc, MutexGuard};
use std::{self, fmt};
use tempfile;
use utils::*;

/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;

/// A CPU evaluation context.
pub struct Context {
    cpu_model: Arc<Cpu>,
    parameters: FxHashMap<String, Arc<dyn Argument>>,
}

impl Context {
    /// Returns a parameter given its name.
    pub fn get_param(&self, name: &str) -> &dyn Argument {
        self.parameters[name].as_ref()
    }

    fn bind_param(&mut self, name: String, value: Arc<dyn Argument>) {
        //assert_eq!(param.t, value.t());
        self.parameters.insert(name, value);
    }

    fn allocate_array(&self, size: usize) -> CpuArray {
        CpuArray::new(size)
    }

    /// Generates a structure holding parameters for function call
    fn gen_args(&self, func: &codegen::Function) -> Vec<ThunkArg> {
        func.device_code_args()
            .map(|pval| match pval {
                ParamVal::External(par, _) => {
                    ThunkArg::ArgRef(Arc::clone(&self.parameters[&par.name]))
                }
                ParamVal::GlobalMem(_, size, _) => {
                    ThunkArg::TmpArray((self as &dyn context::Context).eval_size(size))
                }
                ParamVal::Size(size) => {
                    ThunkArg::Size((self as &dyn context::Context).eval_size(size) as i32)
                }
                ParamVal::DivMagic(s, t) => {
                    assert_eq!(*t, ir::Type::I(32));

                    let u32_size = (self as &dyn context::Context).eval_size(s);
                    let div_magic =
                        codegen::i32_div_magic(i32::try_from(u32_size).unwrap());
                    ThunkArg::Size(div_magic)
                }
                ParamVal::DivShift(s, t) => {
                    assert_eq!(*t, ir::Type::I(32));

                    let u32_size = (self as &dyn context::Context).eval_size(s);
                    let div_shift =
                        codegen::i32_div_shift(i32::try_from(u32_size).unwrap());
                    ThunkArg::Size(div_shift)
                }
            })
            .collect_vec()
    }
}

impl Default for Context {
    /// Create a new evaluation context.
    fn default() -> Context {
        let default_cpu = Cpu::dummy_cpu();
        Context {
            cpu_model: Arc::new(default_cpu),
            parameters: FxHashMap::default(),
        }
    }
}

impl context::ArgMap for Context {
    fn bind_erased_scalar(
        &mut self,
        param: &ir::Parameter,
        value: Box<dyn ScalarArgument>,
    ) {
        assert_eq!(param.t, value.get_type());

        self.bind_param(param.name.clone(), Arc::new(value));
    }

    fn bind_erased_array(
        &mut self,
        param: &ir::Parameter,
        t: ir::Type,
        len: usize,
    ) -> Arc<dyn context::ArrayArgument> {
        let size = len * unwrap!(t.len_byte()) as usize;
        let array = Arc::new(self.allocate_array(size));
        self.bind_param(param.name.clone(), Arc::clone(&array) as Arc<dyn Argument>);
        array
    }
}

impl device::ParamsHolder for Context {
    fn param_as_size(&self, name: &str) -> Option<u32> {
        self.get_param(name).size()
    }
}

impl context::Context for Context {
    fn device(&self) -> Arc<dyn Device> {
        Arc::<Cpu>::clone(&self.cpu_model)
    }

    fn params(&self) -> &dyn device::ParamsHolder {
        self
    }

    fn printer(&self) -> &dyn codegen::DevicePrinter {
        &*self.cpu_model
    }

    /// Evaluation in sequential mode
    fn evaluate(&self, func: &codegen::Function, _mode: EvalMode) -> Result<f64, ()> {
        let mut printer = X86printer::default();
        let fun_str = printer.wrapper_function(func);
        function_evaluate(&fun_str, &self.gen_args(func))
    }

    /// returns a vec containing num_sample runs of function_evaluate
    fn benchmark(&self, func: &codegen::Function, num_samples: usize) -> Vec<f64> {
        let mut printer = X86printer::default();
        let fun_str = printer.wrapper_function(func);
        let args = self.gen_args(func);
        let mut res = vec![];
        for _ in 0..num_samples {
            res.push(function_evaluate(&fun_str, &args).unwrap_or(std::f64::INFINITY));
        }
        res
    }

    fn async_eval<'c>(
        &self,
        num_workers: usize,
        _mode: EvalMode,
        inner: &(dyn Fn(&mut dyn context::AsyncEvaluator<'c>) + Sync),
    ) {
        let (send, recv) = mpsc::sync_channel(EVAL_BUFFER_SIZE);
        crossbeam::scope(move |scope| {
            // Start the explorer threads.
            for _ in 0..num_workers {
                let mut evaluator = AsyncEvaluator {
                    context: self,
                    sender: send.clone(),
                };
                scope
                    .builder()
                    .name("Telamon - Explorer Thread".to_string())
                    .spawn(move |_| inner(&mut evaluator))
                    .unwrap();
            }
            // Start the evaluation thread.
            let eval_thread_name = "Telamon - CPU Evaluation Thread".to_string();
            scope
                .builder()
                .name(eval_thread_name)
                .spawn(move |_| {
                    while let Ok((candidate, fun_str, code_args, callback)) = recv.recv()
                    {
                        callback.call(
                            candidate,
                            &mut Code {
                                source: &fun_str,
                                arguments: &code_args,
                            },
                        );
                    }
                })
                .unwrap();
        })
        .unwrap();
    }
}

enum HoldThunk<'a> {
    Scalar(*mut libc::c_void),
    ArrRef(MutexGuard<'a, Vec<i8>>),
    Size(i32),
    Arr(Vec<i8>),
}

enum ThunkArg {
    ArgRef(Arc<dyn Argument>),
    Size(i32),
    TmpArray(u32),
}

struct Code<'a> {
    source: &'a str,
    arguments: &'a [ThunkArg],
}

impl<'a> fmt::Display for Code<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.source)
    }
}

impl<'a> KernelEvaluator for Code<'a> {
    fn evaluate(&mut self) -> Option<f64> {
        function_evaluate(self.source, self.arguments).ok()
    }
}

enum RawArg {
    Scalar(*mut libc::c_void),
    Size(i32),
    Array(*mut libc::c_void),
}

/// Given a function string and its arguments as ThunkArg, compile to a binary, executes it and
/// returns the time elapsed. Converts ThunkArgs to HoldTHunk as we want to allocate memory for
/// temporary arrays at the last possible moment
fn function_evaluate(fun_str: &str, args: &[ThunkArg]) -> Result<f64, ()> {
    debug!("running code {}", fun_str);
    let temp_dir = unwrap!(tempfile::tempdir());
    let templib_name = temp_dir
        .path()
        .join("lib_compute.so")
        .to_string_lossy()
        .into_owned();
    let mut source_file = unwrap!(tempfile::tempfile());
    unwrap!(source_file.write_all(fun_str.as_bytes()));
    let compile_status = compile::compile(source_file, &templib_name);
    if !compile_status.success() {
        panic!("Could not compile file:\n{}", fun_str);
    }
    // Lock the arguments and allocate temporary arrays
    //
    // `thunks` owns the array values
    let mut thunks = args
        .iter()
        .map(|arg| match arg {
            ThunkArg::ArgRef(arg_ref) => match arg_ref.arg_lock() {
                ArgLock::Scalar(ptr) => HoldThunk::Scalar(ptr),
                ArgLock::Arr(guard) => HoldThunk::ArrRef(guard),
            },
            ThunkArg::Size(size) => HoldThunk::Size(*size),
            ThunkArg::TmpArray(size) => {
                let arr = vec![0; *size as usize];
                HoldThunk::Arr(arr)
            }
        })
        .collect_vec();
    // This contains the argument values, which might be references into `thunks`
    //
    // `raw_args` owns array pointers, *NOT* values
    let mut raw_args = thunks
        .iter_mut()
        .map(|arg| match arg {
            HoldThunk::ArrRef(arg) => {
                RawArg::Array(arg.as_mut_ptr() as *mut libc::c_void)
            }
            &mut HoldThunk::Scalar(ptr) => RawArg::Scalar(ptr),
            &mut HoldThunk::Size(size) => RawArg::Size(size),
            HoldThunk::Arr(array) => {
                RawArg::Array(array.as_mut_ptr() as *mut libc::c_void)
            }
        })
        .collect::<Vec<_>>();
    // This contains pointers to the arguments values held in `raw_args`
    //
    // `ptrs` is only references to the pointers stored in `raw_args
    let ptrs = raw_args
        .iter_mut()
        .map(|raw| match raw {
            &mut RawArg::Scalar(ptr) => ptr,
            RawArg::Array(array) => array as *mut *mut libc::c_void as *mut libc::c_void,
            RawArg::Size(size) => size as *mut i32 as *mut libc::c_void,
        })
        .collect::<Vec<_>>();
    let time = compile::link_and_exec(&templib_name, &String::from("entry_point"), ptrs);
    Ok(time)
}

type AsyncPayload<'b> = (
    explorer::Candidate,
    String,
    Vec<ThunkArg>,
    AsyncCallback<'b>,
);

pub struct AsyncEvaluator<'b> {
    context: &'b Context,
    sender: mpsc::SyncSender<AsyncPayload<'b>>,
}

impl<'b, 'c> context::AsyncEvaluator<'c> for AsyncEvaluator<'b>
where
    'c: 'b,
{
    fn add_dyn_kernel(
        &mut self,
        candidate: explorer::Candidate,
        callback: context::AsyncCallback<'c>,
    ) {
        let (fun_str, code_args);
        {
            let dev_fun = codegen::Function::build(&candidate.space);
            code_args = self.context.gen_args(&dev_fun);
            let mut printer = X86printer::default();
            fun_str = printer.wrapper_function(&dev_fun);
        }
        unwrap!(self.sender.send((candidate, fun_str, code_args, callback)));
    }
}
