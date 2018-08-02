//! MPPA evaluation context.
use codegen;
use crossbeam::sync::MsQueue;
use crossbeam;
use device::{self, ScalarArgument, ArrayArgument, mppa, EvalMode};
use device::context::AsyncCallback;
use device::Context as ContextTrait;
use device::mppa::{MppaPrinter, telajax};
use device::mppa::telajax::Buffer;
use explorer;
use ir;
use libc;
use search_space::SearchSpace;
use std;
use std::sync::{mpsc, Arc};
use std::time::Instant;
use utils::*;

const EXECUTION_QUEUE_SIZE: isize = 512;
/// Max number of candidates waiting to be evaluated.
const EVAL_BUFFER_SIZE: usize = 100;

pub trait Argument: Sync + Send {
    /// Returns a pointer to the object.
    fn raw_ptr(&self) -> *const libc::c_void;
    /// Returns the argument value if it can represent a size.
    fn as_size(&self) -> Option<u32> { None }
}

impl<T> Argument for T where T: device::ScalarArgument {
    fn raw_ptr(&self) -> *const libc::c_void {
        device::ScalarArgument::raw_ptr(self)
    }

    fn as_size(&self) -> Option<u32> {
        device::ScalarArgument::as_size(self)
    }
}

/// MPPA evaluation context.
pub struct Context<'a>  {
    device: mppa::Mppa,
    //executor: std::sync::MutexGuard<'static, telajax::Device>,
    executor: &'static telajax::Device,
    parameters: HashMap<String, Arc<Argument + 'a>>,
    wrappers: Cache<ir::Signature, telajax::Wrapper>,
    writeback_slots: MsQueue<telajax::Mem>,
}
unsafe impl<'a> Sync for Context<'a> {}

impl<'a> Context<'a> {
    /// Creates a new `Context`. Blocks until the MPPA device is ready to be used.
    pub fn new() -> Self {
        let executor = telajax::Device::get();
        let writeback_slots = MsQueue::new();
        for _ in 0..EXECUTION_QUEUE_SIZE {
            writeback_slots.push(executor.alloc(8));
        }
        Context {
            device: mppa::Mppa::default(),
            executor,
            parameters: HashMap::default(),
            wrappers: Cache::new(25),
            writeback_slots,
        }
    }

    fn bind_param(&mut self, name: String, value: Arc<Argument>) {
        self.parameters.insert(name, value);
    }


    /// Compiles and sets the arguments of a kernel.
    fn setup_kernel(&self, fun: &device::Function<'a>)
            -> (telajax::Kernel, telajax::Mem) 
            {
        let mut code: Vec<u8> = Vec::new();
        let mut printer = MppaPrinter::default();
        let kernel_code = printer.wrapper_function(fun);
        let wrapper = self.get_wrapper(fun);
        //let code = std::ffi::CString::new(code).unwrap();
        //debug!("{}", code.clone().into_string().unwrap()); // DEBUG
        let cflags = std::ffi::CString::new("").unwrap();
        let lflags = std::ffi::CString::new("").unwrap();
        let kernel_code = unwrap!(std::ffi::CString::new(kernel_code));
        let mut kernel = self.executor.build_kernel(&kernel_code, &cflags, &lflags, &*wrapper);
        kernel.set_num_clusters(1);
        let (mut arg_sizes, mut args): (Vec<_>, Vec<_>) = fun.params.iter().map(|p| {
            let arg = self.get_param(&p.name);
            (unwrap!(arg.as_size()) as usize, arg.raw_ptr())
        }).unzip();
        let out_mem = self.writeback_slots.pop();
        arg_sizes.push(std::mem::size_of::<*mut libc::c_void>());
        args.push(out_mem.raw_ptr());
        kernel.set_args(&arg_sizes[..], &args[..]);
        (kernel, out_mem)
    }

    /// Returns the wrapper for the given signature.
    fn get_wrapper(&self, fun: &device::Function) -> Arc<telajax::Wrapper>  {
        // TODO: There was a memoization here that allowed to cache a result for an already
        // generated signature wrapper. Maybe reimplement it
        let mut printer = MppaPrinter::default();
        let mut namer = mppa::Namer::default();
        let mut name_map = codegen::NameMap::new(fun, &mut namer);
        let ocl_code = printer.print_ocl_wrapper(fun, &mut name_map);
        let name = std::ffi::CString::new("wrapper").unwrap();
        let ocl_code = std::ffi::CString::new(ocl_code).unwrap();
        debug!("{}", ocl_code.clone().into_string().unwrap());
        Arc::new(self.executor.build_wrapper(&name, &ocl_code))
    }

    /// Returns a parameter given its name.
    pub fn get_param(&self, name: &str) -> &Argument { self.parameters[name].as_ref() }
}

impl<'a> device::Context for Context<'a> {
    fn device(&self) -> &device::Device { &self.device }


    fn benchmark(&self, function: &device::Function, num_samples: usize) -> Vec<f64> {
        unimplemented!()
    }


    fn evaluate(&self, fun: &device::Function, _mode: EvalMode) -> Result<f64, ()> {
        let (mut kernel, out_mem) = self.setup_kernel(fun);
        self.executor.execute_kernel(&mut kernel);
        let mut t: [u64; 1] = [0];
        self.executor.read_buffer(&out_mem, &mut t, &[]);
        self.writeback_slots.push(out_mem);
        Ok(t[0] as f64)
    }

    fn async_eval<'c, 'd>(&self, num_workers: usize, _mode: EvalMode,
                          inner: &(Fn(&mut device::AsyncEvaluator<'c, 'd>) + Sync)) {
        // FIXME: execute in parallel
        let (send, recv) = mpsc::sync_channel(EVAL_BUFFER_SIZE);
        //let mut evaluator = AsyncEvaluator { 
        //    context: self,
        //    sender: send};
        //inner(&mut evaluator);
        //self.executor.wait_all();
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
                while let Ok((candidate, mut kernel, callback)) = recv.recv() {
                    // TODO: measure time directly on MPPA
                    let t0 = Instant::now();
                    self.executor.execute_kernel(&mut kernel);
                    let t = Instant::now() - t0;
                    callback.call(candidate, t.subsec_nanos() as f64);
                }
            }));
        });
    }

    fn param_as_size(&self, name: &str) -> Option<u32> {
        self.get_param(name).as_size()
    }
}

impl< 'a> device::ArgMap for Context<'a> {
    type Array = Buffer;

    fn bind_scalar<S: ScalarArgument>(&mut self, param: &ir::Parameter, value: S) {
        assert_eq!(param.t, S::t());
        self.bind_param(param.name.clone(), Arc::new(value));
    }

    fn bind_array<S: ScalarArgument>(&mut self, param: &ir::Parameter, len: usize)
        -> Arc<Self::Array>
    {
        let size = len * std::mem::size_of::<S>();
        //let buffer_arc: Arc<Buffer<'a>> = Arc::new(self.executor.allocate_array(size));
        let buffer_arc = Arc::new(Buffer::new(self.executor, size));
        self.bind_param(param.name.clone(), buffer_arc.clone());
        buffer_arc
    }
}

type AsyncPayload<'a, 'b> = (explorer::Candidate<'a>, telajax::Kernel, AsyncCallback<'a, 'b>);

/// Asynchronous evaluator.
struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context<'b>,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
    where 'a: 'b, 'c: 'b {
    fn add_kernel(&mut self, candidate: explorer::Candidate<'a>, callback: device::AsyncCallback<'a, 'b>) {
        let (kernel, out_mem) = {
            let dev_fun = device::Function::build(&candidate.space);
            self.context.setup_kernel(&dev_fun)
        };
        unwrap!(self.sender.send((candidate, kernel, callback)));
    }
}

impl Argument for Buffer {
    fn as_size(&self) -> Option<u32> {
        Some(self.mem.read().unwrap().len() as u32)
    }

    fn raw_ptr(&self) -> *const libc::c_void {
        self.mem.read().unwrap().raw_ptr()
    }
}


///// Buffer in MPPA RAM.
//pub struct Buffer<'a> {
//    mem: std::sync::RwLock<telajax::Mem>,
//    executor: &'a std::sync::MutexGuard<'static, telajax::Device>,
//}
//
//impl<'a> Buffer<'a> {
//    fn new(len : usize, executor: &'a std::sync::MutexGuard<'static, telajax::Device>) -> Self {
//        let mem_block = executor.alloc(len);
//        Buffer {
//            mem : std::sync::RwLock::new(mem_block),
//            executor: executor,
//        }
//    }
//}
//
//impl<'a> Argument for Buffer<'a> {
//    fn as_size(&self) -> Option<u32> {
//        Some(self.mem.read().unwrap().len() as u32)
//    }
//
//    fn raw_ptr(&self) -> *const libc::c_void {
//        self.mem.read().unwrap().raw_ptr()
//    }
//}
//
//impl<'a> device::ArrayArgument for Buffer<'a> {
//    fn read_i8(&self) -> Vec<i8> {
//        let mem_block = unwrap!(self.mem.read());
//        let mut read_buffer = vec![0; mem_block.len()];
//        self.executor.read_buffer::<i8>(&mem_block, &mut read_buffer, &[]);
//        read_buffer
//    }
//
//    fn write_i8(&self, slice: &[i8]) {
//        let mut mem_block = unwrap!(self.mem.write());
//        self.executor.write_buffer::<i8>(slice, &mut mem_block, &[]);
//    }
//}
