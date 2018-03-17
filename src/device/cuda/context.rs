///! Defines the CUDA evaluation context.
use crossbeam;
use device;
use device::{Device, Argument};
use device::cuda::{ArrayArg, Executor, Gpu, Kernel, JITDaemon};
use device::cuda::kernel::Thunk;
use explorer;
use ir;
use std;
use std::f64;
use std::sync::{atomic, mpsc};
use utils::*;
use device::context::AsyncCallback;


//use std::boxed::FnBox;
/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;

/// A CUDA evaluation context.
pub struct Context<'a> {
    gpu_model: Gpu,
    executor: &'a Executor,
    parameters: HashMap<String, Box<device::Argument + 'a>>,
}

impl<'a> Context<'a> {
    /// Create a new evaluation context. The GPU model if infered.
    pub fn new(executor: &'a Executor) -> Context {
        let gpu_name = executor.device_name();
        if let Some(gpu) = Gpu::from_name(&gpu_name) {
            Context {
                gpu_model: gpu,
                executor,
                parameters: HashMap::default(),
            }
        } else {
            panic!("Unknown gpu model: {}, \
                   please add it to devices/cuda_gpus.json.", gpu_name);
        }
    }

    /// Creates a context from the given GPU.
    pub fn from_gpu(gpu: Gpu, executor: &'a Executor) -> Self {
        Context { gpu_model: gpu, executor, parameters: HashMap::default() }
    }

    /// Returns the GPU description.
    pub fn gpu(&self) -> &Gpu { &self.gpu_model }

    /// Returns the execution queue.
    pub fn executor(&self) -> &'a Executor { self.executor }
}

impl<'a> device::Context<'a> for Context<'a> {
    fn device(&self) -> &Device { &self.gpu_model }

    fn bind_param(&mut self, param: &ir::Parameter, value: Box<Argument + 'a>) {
        assert_eq!(param.t, value.t());
        self.parameters.insert(param.name.clone(), value);
    }

    fn allocate_array(&mut self, id: ir::mem::Id, size: usize) -> Box<Argument + 'a> {
        Box::new(ArrayArg(self.executor.allocate_array::<u8>(size), id))
    }

    fn get_param(&self, name: &str) -> &Argument { self.parameters[name].as_ref() }

    fn evaluate(&self, function: &device::Function) -> Result<f64, ()> {
        let kernel = Kernel::compile(function, &self.gpu_model, self.executor);
        Ok(kernel.evaluate(self)? as f64 / self.gpu_model.smx_clock)
    }

    fn async_eval<'b, 'c>(&self, num_workers: usize,
                          inner: &(Fn(&mut device::AsyncEvaluator<'b, 'c>) + Sync)){
        // Setup the evaluator.
        let blocked_time = &atomic::AtomicUsize::new(0);
        let (send, recv) = mpsc::sync_channel(EVAL_BUFFER_SIZE);
        let clock_rate = self.gpu_model.smx_clock;
        // Correct because the thread handle is not escaped.
        crossbeam::scope(move |scope| {
            // Start the explorer threads.
            for _ in 0..num_workers {
                let mut evaluator = AsyncEvaluator {
                    context: self,
                    sender: send.clone(),
                    ptx_daemon: self.executor.spawn_jit(),
                    blocked_time,
                };
                unwrap!(scope.builder().name("Telamon - Explorer Thread".to_string())
                        .spawn(move || inner(&mut evaluator)));
            }
            // Start the evaluation thread.
            let eval_thread_name = "Telamon - GPU Evaluation Thread".to_string();
            let res = scope.builder().name(eval_thread_name).spawn(move || {
                let mut cpt_candidate = 0;
                let mut best_eval = std::f64::INFINITY;
                while let Ok((candidate, thunk, callback)) = recv.recv() {
                    cpt_candidate += 1;
                    debug!("IN EVALUATOR: evaluating candidate for actions {:?}", candidate.actions);
                    //let eval = thunk.execute() as f64 / clock_rate;
                    let eval = if candidate.bound.value() < best_eval {
                        thunk.execute().map(|t| t as f64 / clock_rate)
                    } else {
                        warn!("candidate not evaluated because of bounds");
                        Ok(std::f64::INFINITY)
                    };
                    let eval = eval.unwrap_or_else(|()| {
                        panic!("evaluation failed for actions {:?}, with kernel {:?}",
                               candidate.actions, thunk)
                    });
                    if eval < best_eval {
                        best_eval = eval;
                    }
                    debug!("IN EVALUATOR: finished evaluating candidate for actions {:?}",
                           candidate.actions);
                    (callback)(candidate, eval, cpt_candidate);
                }
            });
            unwrap!(res);
            std::mem::drop(send);
        });
        let blocked_time = blocked_time.load(atomic::Ordering::SeqCst);
        info!("Total time blocked on `add_kernel`: {:.4e}ns", blocked_time as f64);
    }
}

type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>, Thunk<'b>, AsyncCallback<'a, 'b>);

pub struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context<'b>,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
    ptx_daemon: JITDaemon,
    blocked_time: &'b atomic::AtomicUsize
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
    where 'a: 'b, 'c: 'b
{
    fn add_kernel(&mut self, candidate: explorer::Candidate<'a>, callback: device::AsyncCallback<'a, 'c> ) {
        let thunk = {
            let dev_fun = device::Function::build(&candidate.space);
            let gpu = &self.context.gpu();
            debug!("compiling kernel with bound {} and actions {:?}",
                   candidate.bound, candidate.actions);
            // TODO(cc_perf): cuModuleLoadData is waiting the end of any running kernel
            let kernel = Kernel::compile_remote(
                &dev_fun, gpu, self.context.executor(), &mut self.ptx_daemon);
            kernel.gen_thunk(self.context)
        };
        let t0 = std::time::Instant::now();
        unwrap!(self.sender.send((candidate, thunk, callback)));
        let t = std::time::Instant::now() - t0;
        let t_usize = t.as_secs() as usize * 1_000_000_000 + t.subsec_nanos() as usize;
        self.blocked_time.fetch_add(t_usize, atomic::Ordering::Relaxed);
    }
}
