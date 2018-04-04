///! Defines the CUDA evaluation context.
use crossbeam;
use device;
use device::{Device, Argument};
use explorer;
use ir;
use std;
use std::f64;
use std::sync::{atomic, mpsc};
use utils::*;
use device::context::AsyncCallback;
use device::x86::compile;
use device::x86::cpu::Cpu;


/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;

/// A CUDA evaluation context.
pub struct Context<'a> {
    cpu_model: Cpu,
    parameters: HashMap<String, Box<device::Argument + 'a>>,
}

impl<'a> Context<'a> {
    /// Create a new evaluation context. The GPU model if infered.
    pub fn new() -> Context<'a> {
        let default_cpu = Cpu::dummy_cpu();
        Context {
            cpu_model: default_cpu,
            parameters: HashMap::default(),
        }
        //let gpu_name = executor.device_name();
        //if let Some(gpu) = Cpu::from_name(&gpu_name) {
        //    Context {
        //        gpu_model: gpu,
        //        executor,
        //        parameters: HashMap::default(),
        //    }
        //} else {
        //    panic!("Unknown cpu model: {}, \
        //           please add it to devices/cuda_gpus.json.", gpu_name);
        //}
    }
}

impl<'a> device::Context<'a> for Context<'a> {
    fn device(&self) -> &Device { &self.cpu_model }

    fn bind_param(&mut self, param: &ir::Parameter, value: Box<Argument + 'a>) {
        unimplemented!()
    }

    fn allocate_array(&mut self, id: ir::mem::Id, size: usize) -> Box<Argument + 'a> {
        unimplemented!()
    }

    fn get_param(&self, name: &str) -> &Argument { self.parameters[name].as_ref() }

    fn evaluate(&self, function: &device::Function) -> Result<f64, ()> {
        //let kernel = Kernel::compile(function, &self.gpu_model, self.executor);
        //Ok(kernel.evaluate(self)? as f64 / self.gpu_model.smx_clock)
        let libname = String::from("hello");
        let libpath = String::from("/tmp/");
        let mut complete_lib_path = libpath.clone();
        complete_lib_path.push_str(&libname);
        let source_path = String::from("template/hello_world.c");
        compile::compile(libname, source_path, libpath);
        let time = compile::link_and_exec(complete_lib_path, String::from("hello"));
        Ok(time)
    }

    fn async_eval<'b, 'c>(&self, num_workers: usize,
                          inner: &(Fn(&mut device::AsyncEvaluator<'b, 'c>) + Sync)){
        unimplemented!()
    }
}

type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>, AsyncCallback<'a, 'b>);

pub struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context<'b>,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
    blocked_time: &'b atomic::AtomicUsize
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
    where 'a: 'b, 'c: 'b
{
    fn add_kernel(&mut self, candidate: explorer::Candidate<'a>, callback: device::AsyncCallback<'a, 'c> ) {
        unimplemented!()
    }
}
