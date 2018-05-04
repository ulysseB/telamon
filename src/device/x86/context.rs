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
            let eval_thread_name = "Telamon - GPU Evaluation Thread".to_string();
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


fn function_evaluate(fun_str: String, fun_name: String) -> Result<f64, ()> {
    //let templib_name = tempfile::tempdir().unwrap().path().join("lib_compute.so").to_string_lossy()
    //    .into_owned();
    let temp_dir = tempfile::tempdir().unwrap();
    let templib_name = temp_dir.path().join("lib_compute.so").to_string_lossy().into_owned();
    println!("{}", fun_str);
    println!("{}", fun_name);
    println!("{}", templib_name);
    panic!();
    let mut source_file = tempfile::tempfile().unwrap();
    source_file.write_all(fun_str.as_bytes()).unwrap();
    let compile_status = compile::compile(source_file, &templib_name);
    if let Some(code) = compile_status.code() {
        println!("gcc exited with code {}", code);
    }
    if !compile_status.success() {
        panic!("Could not compile file");
    }
    let time = compile::link_and_exec(&templib_name, &String::from("execute"), args);
    Ok(time)
}

fn dummy_evaluate() -> Result<f64, ()> {
    // Does not look like a very reliable way to do this...
    // Directory from which we launched the binary, assuming it's telamon for now
    let mut telamon_path = String::from(std::env::current_dir().unwrap().to_str().unwrap());
    telamon_path.push_str("/");
    let libname = String::from("hello");
    let libpath = String::from("/tmp/");
    let mut complete_lib_path = libpath.clone();
    complete_lib_path.push_str("lib");
    complete_lib_path.push_str(&libname);
    complete_lib_path.push_str(".so");
    let mut source_path = telamon_path.clone();
    source_path.push_str("src/device/x86/template/hello_world.c");
    //let source_path = String::from("template/hello_world.c");
    compile::compile(libname, source_path, libpath);
    let time = compile::link_and_exec(complete_lib_path, String::from("hello"));
    Ok(time)
}

type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>,  String, AsyncCallback<'a, 'b>);

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
