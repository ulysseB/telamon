///! Defines the CPU evaluation context.
use codegen::ParamVal;
use crossbeam;
use device::{self, Device, ScalarArgument, EvalMode};
use device::context::AsyncCallback;
use device::x86::compile;
use device::x86::cpu_argument::{CpuArray, Argument, CpuScalarArg};
use device::x86::cpu::Cpu;
use device::x86::printer::wrapper_function;
use explorer;
use ir;
use itertools::Itertools;
use libc;
use std;
use std::f64;
use std::io::Write;
use std::sync::{mpsc, Arc};
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
    /// Create a new evaluation context. The GPU model if infered.
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

    fn gen_args(&self, func: &device::Function) -> Vec<ThunkArg> {
        func.device_code_args().map(|pval| match pval {
            ParamVal::External(par, _) => ThunkArg::ArgRef(Arc::clone(&self.parameters[&par.name])),
            ParamVal::GlobalMem(_, size, _) => ThunkArg::TmpArray((self as &device::Context).eval_size(size)),
            //ParamVal::GlobalMem(_, size, _) => ThunkArg::TmpArray(
            //    CpuArray::new((self as &device::Context).eval_size(size)as usize)),
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


    fn evaluate(&self, func: &device::Function, _mode: EvalMode) -> Result<f64, ()> {
        let fun_str = wrapper_function(&func);
        function_evaluate(fun_str, self.gen_args(func))
    }

    fn benchmark(&self, _function: &device::Function, _num_samples: usize) -> Vec<f64> {
        //let gpu = &self.gpu_model;
        //let kernel = Kernel::compile(function, gpu, self.executor, 4);
        //kernel.evaluate_real(self, num_samples)
        vec![]
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
                let mut cpt_candidate = 0;
                while let Ok((candidate, fun_str, code_args, callback)) = recv.recv() {
                    cpt_candidate += 1;
                    let eval = function_evaluate(fun_str, code_args).unwrap();
                    callback.call(candidate, eval, cpt_candidate);
                }
            }));
        });
    }
}


enum HoldThunk {
    PlaceHolder,
    Arr(CpuArray),
}

enum ThunkArg {
    ArgRef(Arc<Argument>),
    Size(i32),
    TmpArray(u32),
}

fn function_evaluate(fun_str: String, mut args: Vec<ThunkArg>) -> Result<f64, ()> {
    println!("{}", fun_str);
    //panic!();
    let temp_dir = tempfile::tempdir().unwrap();
    let templib_name = temp_dir.path().join("lib_compute.so").to_string_lossy().into_owned();
    let mut source_file = tempfile::tempfile().unwrap();
    source_file.write_all(fun_str.as_bytes()).unwrap();
    let compile_status = compile::compile(source_file, &templib_name);
    if !compile_status.success() {
        panic!("Could not compile file");
    }
    let thunks = args.iter().map(|arg| match arg {
        &ThunkArg::ArgRef(_) =>  HoldThunk::PlaceHolder,
        &ThunkArg::Size(_) => HoldThunk::PlaceHolder,
        &ThunkArg::TmpArray( size) => {
            let arr = CpuArray::new(size as usize); 
            HoldThunk::Arr(arr)
        },
    }).collect_vec();
    let ptrs = args.iter_mut().enumerate() .map(|(ind, arg)| match arg {
        &mut ThunkArg::ArgRef(ref mut arg_arc) =>  arg_arc.raw_ptr(),
        &mut ThunkArg::Size(ref mut size) =>  size as *mut _ as *mut libc::c_void,
        &mut ThunkArg::TmpArray(_) => {
            if let &HoldThunk::Arr(ref arr) = &thunks[ind] {
                arr.raw_ptr()
            } else {panic!("There should be an Arr at this position !")}
        },
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
