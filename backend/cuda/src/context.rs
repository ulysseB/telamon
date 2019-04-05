use crate::api::Argument;
use crate::kernel::Thunk;
use crate::{Executor, Gpu, JITDaemon, Kernel};
///! Defines the CUDA evaluation context.
use crossbeam;
use log::{debug, info};
use std::f64;
use std::fmt;
use std::sync::{atomic, mpsc, Arc};
use telamon::device::{
    self, AsyncCallback, Device, EvalMode, KernelEvaluator, ScalarArgument,
};
use telamon::{codegen, explorer, ir};
use utils::*;

/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;
// TODO(perf): enable optimizations when possible
const JIT_OPT_LEVEL: usize = 2;

/// A CUDA evaluation context.
pub struct Context<'a> {
    gpu_model: Gpu,
    executor: &'a Executor,
    parameters: FnvHashMap<String, Arc<dyn Argument + 'a>>,
}

impl<'a> Context<'a> {
    /// Create a new evaluation context. The GPU model if infered.
    pub fn new(executor: &'a Executor) -> Context {
        Context {
            gpu_model: Gpu::from_executor(executor),
            executor,
            parameters: FnvHashMap::default(),
        }
    }

    /// Creates a context from the given GPU.
    pub fn from_gpu(gpu: Gpu, executor: &'a Executor) -> Self {
        Context {
            gpu_model: gpu,
            executor,
            parameters: FnvHashMap::default(),
        }
    }

    /// Returns the GPU description.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu_model
    }

    /// Returns the execution queue.
    pub fn executor(&self) -> &'a Executor {
        self.executor
    }

    /// Returns a parameter given its name.
    pub fn get_param(&self, name: &str) -> &dyn Argument {
        self.parameters[name].as_ref()
    }

    /// Binds a parameter to the gien name.
    pub fn bind_param(&mut self, name: String, arg: Arc<dyn Argument + 'a>) {
        self.parameters.insert(name, arg);
    }

    /// Returns the optimization level to use.
    fn opt_level(mode: EvalMode) -> usize {
        match mode {
            EvalMode::TestBound => 1,
            EvalMode::FindBest | EvalMode::TestEval => JIT_OPT_LEVEL,
        }
    }
}

impl<'a> device::ArgMap<'a> for Context<'a> {
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
    ) -> Arc<dyn device::ArrayArgument + 'a> {
        let size = len * unwrap!(t.len_byte()) as usize;
        let array = Arc::new(self.executor.allocate_array::<i8>(size));
        self.bind_param(param.name.clone(), array.clone());
        array
    }
}

impl<'a> device::Context for Context<'a> {
    fn device(&self) -> &dyn Device {
        &self.gpu_model
    }

    fn param_as_size(&self, name: &str) -> Option<u32> {
        self.get_param(name).as_size()
    }

    fn stabilizer(&self) -> device::Stabilizer {
        device::Stabilizer::default().num_evals(20).num_outliers(4)
    }

    fn evaluate(&self, function: &codegen::Function, mode: EvalMode) -> Result<f64, ()> {
        let gpu = &self.gpu_model;
        let kernel = Kernel::compile(function, gpu, self.executor, Self::opt_level(mode));
        kernel
            .evaluate(self)
            .map(|t| t as f64 / self.gpu_model.smx_clock)
    }

    fn benchmark(&self, function: &codegen::Function, num_samples: usize) -> Vec<f64> {
        let gpu = &self.gpu_model;
        let kernel = Kernel::compile(function, gpu, self.executor, 4);
        kernel.evaluate_real(self, num_samples)
    }

    fn async_eval<'b, 'c>(
        &self,
        num_workers: usize,
        mode: EvalMode,
        inner: &(dyn Fn(&mut dyn device::AsyncEvaluator<'b, 'c>) + Sync),
    ) {
        // Setup the evaluator.
        let blocked_time = &atomic::AtomicUsize::new(0);
        let (send, recv) = mpsc::sync_channel(EVAL_BUFFER_SIZE);
        // Correct because the thread handle is not escaped.
        crossbeam::scope(move |scope| {
            // Start the explorer threads.
            for _ in 0..num_workers {
                let mut evaluator = AsyncEvaluator {
                    context: self,
                    sender: send.clone(),
                    ptx_daemon: self.executor.spawn_jit(Self::opt_level(mode)),
                    blocked_time,
                };
                unwrap!(scope
                    .builder()
                    .name("Telamon - Explorer Thread".to_string())
                    .spawn(move |_| inner(&mut evaluator)));
            }
            // Start the evaluation thread.
            let eval_thread_name = "Telamon - GPU Evaluation Thread".to_string();
            let res = scope.builder().name(eval_thread_name).spawn(move |_| {
                while let Ok((candidate, thunk, callback)) = recv.recv() {
                    callback.call(
                        candidate,
                        &mut RealtimeThunk {
                            thunk,
                            smx_clock: self.gpu_model.smx_clock,
                        },
                    );
                }
            });
            unwrap!(res);
            std::mem::drop(send);
        })
        .unwrap();
        let blocked_time = blocked_time.load(atomic::Ordering::SeqCst);
        info!(
            "Total time blocked on `add_kernel`: {:.4e}ns",
            blocked_time as f64
        );
    }
}

type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>, Thunk<'b>, AsyncCallback<'a, 'b>);

pub struct AsyncEvaluator<'a, 'b>
where
    'a: 'b,
{
    context: &'b Context<'b>,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
    ptx_daemon: JITDaemon,
    blocked_time: &'b atomic::AtomicUsize,
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
where
    'a: 'b,
    'c: 'b,
{
    fn add_any_kernel(
        &mut self,
        candidate: explorer::Candidate<'a>,
        callback: device::AsyncCallback<'a, 'c>,
    ) {
        let thunk = {
            let dev_fun = codegen::Function::build(&candidate.space);
            let gpu = &self.context.gpu();
            debug!(
                "compiling kernel with bound {} and actions {:?}",
                candidate.bound, candidate.actions
            );
            // TODO(cc_perf): cuModuleLoadData is waiting the end of any running kernel
            let kernel = Kernel::compile_remote(
                &dev_fun,
                gpu,
                self.context.executor(),
                &mut self.ptx_daemon,
            );
            kernel.gen_thunk(self.context)
        };
        let t0 = std::time::Instant::now();
        unwrap!(self.sender.send((candidate, thunk, callback)));
        let t = std::time::Instant::now() - t0;
        let t_usize = t.as_secs() as usize * 1_000_000_000 + t.subsec_nanos() as usize;
        self.blocked_time
            .fetch_add(t_usize, atomic::Ordering::Relaxed);
    }
}

struct RealtimeThunk<'a> {
    thunk: Thunk<'a>,
    smx_clock: f64,
}

impl<'a> fmt::Display for RealtimeThunk<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", &self.thunk)
    }
}

impl<'a> KernelEvaluator for RealtimeThunk<'a> {
    fn evaluate(&mut self) -> Option<f64> {
        Some(self.thunk.execute().ok()? as f64 / self.smx_clock)
    }
}
