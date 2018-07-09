///! Defines the CPU evaluation context.
use codegen::ParamVal;
use crossbeam;
use device::{self, Device, ScalarArgument, EvalMode};
use device::context::AsyncCallback;
use device::x86::compile;
use device::x86::cpu_argument::{CpuArray, Argument, CpuScalarArg, ArgLock};
use device::x86::cpu::Cpu;
use device::x86::printer::wrapper_function;
use explorer;
use ir;
use itertools::Itertools;
use libc;
use std;
use std::f64;
use std::io::Write;
use std::sync::{mpsc, Arc, MutexGuard};
use tempfile;
use utils::*;


/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;

/// A CPU evaluation context.
pub struct Context {
    cpu_model: Cpu,
    parameters: HashMap<String, Arc<Argument>>,
}

impl Context {
    /// Create a new evaluation context.
    pub fn new() -> Context {
        let default_cpu = Cpu::dummy_cpu();
        Context {
            cpu_model: default_cpu,
            parameters: HashMap::default(),
        }
    }

    /// Returns a parameter given its name.
    pub fn get_param(&self, name: &str) -> &Argument { self.parameters[name].as_ref() }

    fn bind_param(&mut self, name: String, value: Arc<Argument>) {
        //assert_eq!(param.t, value.t());
        self.parameters.insert(name, value);
    }

    fn allocate_array(&self, size: usize) -> CpuArray  {
        CpuArray::new(size)
    }

    /// Generates a structure holding parameters for function call
    fn gen_args(&self, func: &device::Function) -> Vec<ThunkArg> {
        func.device_code_args().map(|pval| match pval {
            ParamVal::External(par, _) => ThunkArg::ArgRef(Arc::clone(&self.parameters[&par.name])),
            ParamVal::GlobalMem(_, size, _) => ThunkArg::TmpArray((self as &device::Context).eval_size(size)),
            ParamVal::Size(size) => ThunkArg::Size((self as &device::Context).eval_size(size) as i32),
        }).collect_vec()
    }
}


impl device::ArgMap for Context {

    type Array = CpuArray;

    fn bind_scalar<S: ScalarArgument>(&mut self, param: &ir::Parameter, value: S) {
        assert_eq!(param.t, S::t());
        self.bind_param(param.name.clone(), Arc::new(Box::new(value) as Box<CpuScalarArg>));
    }

    fn bind_array<S: ScalarArgument>(&mut self, param: &ir::Parameter, len: usize)
        -> Arc<Self::Array>
    {
        let size = len * std::mem::size_of::<S>();
        let array = Arc::new(self.allocate_array(size));
        self.bind_param(param.name.clone(), Arc::clone(&array) as Arc<Argument>);
        array
    }
}

impl device::Context for Context {
    fn device(&self) -> &Device { &self.cpu_model }

    fn param_as_size(&self, name: &str) -> Option<u32> { self.get_param(name).size() }


    /// Evaluation in sequential mode
    fn evaluate(&self, func: &device::Function, _mode: EvalMode) -> Result<f64, ()> {
        let fun_str = wrapper_function(func);
        function_evaluate(&fun_str, &self.gen_args(func))
    }

    /// returns a vec containing num_sample runs of function_evaluate
    fn benchmark(&self, func: &device::Function, num_samples: usize) -> Vec<f64> {
        let fun_str = wrapper_function(func);
        let args =  self.gen_args(func);
        let mut res = vec![];
        for _ in 0..num_samples {
            res.push(function_evaluate(&fun_str, &args).unwrap_or(std::f64::INFINITY));
        }
        res
    }

    fn async_eval<'b, 'c>(&self, num_workers: usize, _mode: EvalMode,
                          inner: &(Fn(&mut device::AsyncEvaluator<'b, 'c>) + Sync)){
        let (send, recv) = mpsc::sync_channel(EVAL_BUFFER_SIZE);
        crossbeam::scope(move |scope| {
            // Start the explorer threads.
            for _ in 0..num_workers {
                let mut evaluator = AsyncEvaluator {
                    context: self,
                    sender: send.clone(),
                };
                unwrap!(scope.builder().name("Telamon - Explorer Thread".to_string())
                        .spawn(move || inner(&mut evaluator)));
            }
            // Start the evaluation thread.
            let eval_thread_name = "Telamon - CPU Evaluation Thread".to_string();
            unwrap!(scope.builder().name(eval_thread_name).spawn(move || {
                while let Ok((candidate, fun_str, code_args, callback)) = recv.recv() {
                    let eval = function_evaluate(&fun_str, &code_args).unwrap();
                    callback.call(candidate, eval);
                }
            }));
        });
    }
}


enum HoldThunk<'a> {
    Scalar(*mut libc::c_void),
    ArrRef(MutexGuard<'a, Vec<i8>>),
    Size(i32),
    Arr(Vec<i8>),
}

enum ThunkArg {
    ArgRef(Arc<Argument>),
    Size(i32),
    TmpArray(u32),
}

/// Given a function string and its arguments as ThunkArg, compile to a binary, executes it and
/// returns the time elapsed. Converts ThunkArgs to HoldTHunk as we want to allocate memory for
/// temporary arrays at the last possible moment
fn function_evaluate(fun_str: &String, args: &Vec<ThunkArg>) -> Result<f64, ()> {
    let temp_dir = tempfile::tempdir().unwrap();
    let templib_name = temp_dir.path().join("lib_compute.so").to_string_lossy().into_owned();
    let mut source_file = tempfile::tempfile().unwrap();
    source_file.write_all(fun_str.as_bytes()).unwrap();
    let compile_status = compile::compile(source_file, &templib_name);
    if !compile_status.success() {
        panic!("Could not compile file");
    }
    let mut thunks = args.iter().map(|arg| match arg {
        ThunkArg::ArgRef(arg_ref) => match arg_ref.arg_lock() {
            ArgLock::Scalar(ptr) => HoldThunk::Scalar(ptr),
            ArgLock::Arr(guard) => HoldThunk::ArrRef(guard),
        },
        ThunkArg::Size(size) => HoldThunk::Size(*size),
        ThunkArg::TmpArray( size) => {
            let arr = vec![0; *size as usize];
            HoldThunk::Arr(arr)
        },
    }).collect_vec();
    let ptrs = thunks.iter_mut().map(|arg| match arg {
        HoldThunk::ArrRef(arg) => arg.as_mut_ptr() as *mut libc::c_void,
        HoldThunk::Scalar(ptr) => *ptr,
        HoldThunk::Size(size) =>  size as *mut _ as *mut libc::c_void,
        HoldThunk::Arr(array) =>  array.as_mut_ptr() as *mut libc::c_void,
    }).collect_vec();
    let time = compile::link_and_exec(&templib_name, &String::from("entry_point"), ptrs);
    Ok(time)
}


type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>,  String, Vec<ThunkArg>, AsyncCallback<'a, 'b>);

pub struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
    where 'a: 'b, 'c: 'b
{
    fn add_kernel(&mut self, candidate: explorer::Candidate<'a>, callback: device::AsyncCallback<'a, 'c> ) {
        let (fun_str, code_args);
        {
            let dev_fun = device::Function::build(&candidate.space);
            code_args = self.context.gen_args(&dev_fun);
            fun_str = wrapper_function(&dev_fun);
        }
        unwrap!(self.sender.send((candidate, fun_str, code_args, callback)));
    }
}
