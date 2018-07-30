//! MPPA evaluation context.
use crossbeam::sync::MsQueue;
use device::{self, mppa};
use device::Context as ContextTrait;
use device::mppa::telajax;
use explorer;
use ir;
use libc;
use search_space::SearchSpace;
use std;
use std::sync::Arc;
use utils::*;

const EXECUTION_QUEUE_SIZE: isize = 512;

/// MPPA evaluation context.
pub struct Context<'a> {
    device: mppa::MPPA,
    executor: std::sync::MutexGuard<'static, telajax::Device>,
    parameters: HashMap<String, Box<device::Argument + 'a>>,
    wrappers: Cache<ir::Signature, telajax::Wrapper>,
    writeback_slots: MsQueue<telajax::Mem>,
}

impl<'a> Context<'a> {
    /// Creates a new `Context`. Blocks until the MPPA device is ready to be used.
    pub fn new() -> Self {
        let mut executor = telajax::Device::get();
        let writeback_slots = MsQueue::new();
        for _ in 0..EXECUTION_QUEUE_SIZE {
            writeback_slots.push(executor.alloc(8));
        }
        Context {
            device: mppa::MPPA::default(),
            executor,
            parameters: HashMap::default(),
            wrappers: Cache::new(25),
            writeback_slots,
        }
    }

    /// Compiles and sets the arguments of a kernel.
    fn setup_kernel<'b>(&self, fun: &device::Function<'a, 'b>)
            -> (telajax::Kernel, telajax::Mem) {
        let mut code: Vec<u8> = Vec::new();
        let wrapper = self.get_wrapper(fun.signature());
        mppa::printer::print(fun, true, &mut code).unwrap();
        let code = std::ffi::CString::new(code).unwrap();
        debug!("{}", code.clone().into_string().unwrap()); // DEBUG
        let cflags = std::ffi::CString::new("").unwrap();
        let lflags = std::ffi::CString::new("").unwrap();
        let mut kernel = self.executor.build_kernel(&code, &cflags, &lflags, &*wrapper);
        kernel.set_num_clusters(1);
        let (mut arg_sizes, mut args): (Vec<_>, Vec<_>) = fun.params.iter().map(|p| {
            let arg = self.get_param(&p.name);
            (arg.size_of(), arg.raw_ptr())
        }).unzip();
        let out_mem = self.writeback_slots.pop();
        arg_sizes.push(std::mem::size_of::<*mut libc::c_void>());
        args.push(out_mem.raw_ptr());
        kernel.set_args(&arg_sizes, &args);
        (kernel, out_mem)
    }

    /// Returns the wrapper for the given signature.
    fn get_wrapper<'b>(&self, sig: &ir::Signature) -> Arc<telajax::Wrapper> {
        self.wrappers.get(sig, || {
            let mut ocl_code: Vec<u8> = Vec::new();
            mppa::printer::print_ocl_wrapper(sig, &mut ocl_code).unwrap();
            let name = std::ffi::CString::new("wrapper").unwrap();
            let ocl_code = std::ffi::CString::new(ocl_code).unwrap();
            debug!("{}", ocl_code.clone().into_string().unwrap());
            self.executor.build_wrapper(&name, &ocl_code)
        })
    }
}

impl<'a> device::Context<'a> for Context<'a> {
    fn device(&self) -> &device::Device { &self.device }

    fn bind_param(&mut self, param: &ir::Parameter, value: Box<device::Argument + 'a>) {
        assert_eq!(param.t, value.t());
        self.parameters.insert(param.name.to_string(), value);
    }

    fn get_param(&self, name: &str) -> &device::Argument {
        self.parameters[name].as_ref()
    }

    fn allocate_array(&mut self, id: ir::MemId, size: usize) -> Box<device::Argument> {
        let mem = self.executor.alloc(size);
        Box::new(Buffer { mem: mem, id: id, context: Default::default() })
    }

    fn evaluate(&self, fun: &device::Function) -> f64 {
        let (kernel, out_mem) = self.setup_kernel(fun);
        self.executor.execute_kernel(&kernel);
        let mut t: [u64; 1] = [0];
        self.executor.read_buffer(&out_mem, &mut t, &[]);
        self.writeback_slots.push(out_mem);
        t[0] as f64
    }

    fn bound(&self, _: &SearchSpace) -> f64 {
        0.0 // FIXME: bound the execution time
    }

    fn async_eval<'b, 'c>(&'c self, callback: &'c device::AsyncCallback<'b, 'c>,
                          inner: &mut FnMut(&device::AsyncEvaluator<'b>)) {
        // FIXME: execute in parallel
        let evaluator = AsyncEvaluator { context: self, callback: callback };
        inner(&evaluator);
        self.executor.wait_all();
    }
}

/// Asynchronous evaluator.
struct AsyncEvaluator<'a, 'b> where 'a: 'b {
    context: &'b Context<'b>,
    callback: &'b device::AsyncCallback<'a, 'b>,
}

impl<'a, 'b> device::AsyncEvaluator<'a> for AsyncEvaluator<'a, 'b> where 'a:'b {
    fn add_kernel(&self, candidate: explorer::Candidate<'a>) {
        let (kernel, out_mem) = {
            let dev_fun = device::Function::build(&candidate.space);
            self.context.setup_kernel(&dev_fun)
        };
        let kernel_event = self.context.executor.enqueue_kernel(&kernel);
        let mut t = vec![0u64];
        let read_event = self.context.executor.async_read_buffer(
            &out_mem, &mut t, &[kernel_event]);
        let callback = move || {
            self.context.writeback_slots.push(out_mem);
            (self.callback)(candidate, t[0] as f64);
        };
        self.context.executor.set_event_callback(&read_event, callback);
    }
}


/// Buffer in MPPA RAM.
pub struct Buffer<'a> {
    mem: telajax::Mem,
    id: ir::MemId,
    context: std::marker::PhantomData<&'a ()>,
}

impl<'a> Buffer<'a> {
    /// Copies a host buffer to the device buffer.
    pub fn copy_from_host<T: Copy>(&mut self, data: &[T], ctx: &Context) {
        ctx.executor.write_buffer(data, &mut self.mem, &[]);
    }

    /// Copies the buffer on the host.
    pub fn copy_to_host<T: Copy>(&self, data: &mut [T], ctx: &Context) {
        ctx.executor.read_buffer(&self.mem, data, &[]);
    }
}

impl<'a> device::Argument for Buffer<'a> {
    fn t(&self) -> ir::Type { ir::Type::PtrTo(self.id) }

    fn raw_ptr(&self) -> *const libc::c_void { self.mem.raw_ptr() }

    fn size_of(&self) -> usize { std::mem::size_of::<*mut libc::c_void>() }
}
