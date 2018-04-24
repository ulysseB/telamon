///! Defines the CPU evaluation context.
use crossbeam;
use device::{self, Device, ScalarArgument};
use device::context::AsyncCallback;
use device::x86::compile;
use device::x86::cpu_argument::{CpuArray, Argument, CpuScalarArg};
use device::x86::cpu::Cpu;
use device::x86::printer::function;
use explorer;
use tempfile;
use ir;
use std;
use std::f64;
use std::io::Write;
use std::sync::{mpsc, Mutex, Arc};
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

    fn allocate_array(&mut self, size: usize) -> CpuArray  {
        CpuArray::new(size)
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


    fn evaluate(&self, function: &device::Function) -> Result<f64, ()> {
        Ok(1.0)
    }

    fn benchmark(&self, function: &device::Function, num_samples: usize) -> Vec<f64> {
        //let gpu = &self.gpu_model;
        //let kernel = Kernel::compile(function, gpu, self.executor, 4);
        //kernel.evaluate_real(self, num_samples)
        vec![]
    }

    fn async_eval<'b, 'c>(&self, num_workers: usize,
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
            let res = scope.builder().name(eval_thread_name).spawn(move || {
                let mut cpt_candidate = 0;
                while let Ok((candidate, fun_str, fun_name, callback)) = recv.recv() {
                    cpt_candidate += 1;
                    let eval = function_evaluate(fun_str, fun_name).unwrap();
                    callback.call(candidate, eval, cpt_candidate);
                }
            });
        });
    }
}

fn function_evaluate(fun_str: String, fun_name: String) -> Result<f64, ()> {
    let templib_name = tempfile::tempdir().unwrap().path().join("lib_compute.so").to_string_lossy()
        .into_owned();
    let mut source_file = tempfile::tempfile().unwrap();
    source_file.write_all(fun_str.as_bytes()).unwrap();
    compile::compile(source_file, &templib_name);
    let time = compile::link_and_exec(&templib_name, &fun_name);
    Ok(time)
}




type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>,  String, String, AsyncCallback<'a, 'b>);

pub struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
    where 'a: 'b, 'c: 'b
{
    fn add_kernel(&mut self, candidate: explorer::Candidate<'a>, callback: device::AsyncCallback<'a, 'c> ) {
        let (fun_str, fun_name);
        {
            let dev_fun = device::Function::build(&candidate.space);
            fun_name = dev_fun.name.clone();
            fun_str = function(&dev_fun);
        }
        self.sender.send((candidate, fun_str, fun_name, callback));
    }
}
