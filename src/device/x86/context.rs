///! Defines the CUDA evaluation context.
use crossbeam;
use device;
use device::{Device, Argument};
use device::context::AsyncCallback;
use device::x86::compile;
use device::x86::cpu::Cpu;
use device::x86::printer::function;
use device::x86::api::ArrayArg;
use explorer;
use tempfile;
use ir;
use std;
use std::f64;
use std::fs::File;
use std::io::Write;
use std::sync::{atomic, mpsc};
use utils::*;


/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;

/// A CPU evaluation context.
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
        assert_eq!(param.t, value.t());
        self.parameters.insert(param.name.clone(), value);
    }

    fn allocate_array(&mut self, id: ir::mem::Id, size: usize) -> Box<Argument + 'a> {
        Box::new(ArrayArg::<i8>::new(id, vec![0; size]))
    }

    fn get_param(&self, name: &str) -> &Argument { self.parameters[name].as_ref() }

    fn evaluate(&self, function: &device::Function) -> Result<f64, ()> {
        Ok(1.0)
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

fn create_file(fun_str: String, fun_name: String, filepath: String) 
    -> Result<(), ()>
{
    let mut file = unwrap!(File::create(filepath));
    file.write_all(fun_str.as_bytes());
    Ok(())
}

fn function_evaluate(fun_str: String, fun_name: String) -> Result<f64, ()> {
    // Does not look like a very reliable way to do this...
    // Directory from which we launched the binary, assuming it's telamon for now
    let templib_name = tempfile::tempdir().unwrap().path().join("lib_compute.so").to_string_lossy()
        .into_owned();
    //let source_path = String::from("template/hello_world.c");
    let mut source_file = tempfile::tempfile().unwrap();
    source_file.write_all(fun_str.as_bytes()).unwrap();
    compile::compile(source_file, &templib_name);
    let time = compile::link_and_exec(&templib_name, &fun_name);
    Ok(time)
}

//fn dummy_evaluate() -> Result<f64, ()> {
//    // Does not look like a very reliable way to do this...
//    // Directory from which we launched the binary, assuming it's telamon for now
//    let mut telamon_path = String::from(std::env::current_dir().unwrap().to_str().unwrap());
//    telamon_path.push_str("/");
//    let libname = String::from("hello");
//    let libpath = String::from("/tmp/");
//    let mut complete_lib_path = libpath.clone();
//    complete_lib_path.push_str("lib");
//    complete_lib_path.push_str(&libname);
//    complete_lib_path.push_str(".so");
//    let mut source_path = telamon_path.clone();
//    source_path.push_str("src/device/x86/template/hello_world.c");
//    //let source_path = String::from("template/hello_world.c");
//    compile::compile(source_path, complete_lib_path.clone());
//    let time = compile::link_and_exec(complete_lib_path, String::from("hello"));
//    Ok(time)
//}

type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>,  String, String, AsyncCallback<'a, 'b>);

pub struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context<'b>,
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
