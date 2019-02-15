//! MPPA evaluation context.
use codegen::{self, ParamVal};
use crossbeam;
use crossbeam::sync::MsQueue;
use device::context::AsyncCallback;
use device::mppa::telajax::Buffer;
use device::mppa::{telajax, MppaPrinter};
use device::Context as ContextTrait;
use device::{self, mppa, ArrayArgument, EvalMode, ScalarArgument};
use explorer;
use ir;
use itertools::Itertools;
use libc;
use search_space::DimKind;
use search_space::SearchSpace;
use std;
use std::sync::{mpsc, Arc};
use std::time::Instant;
use utils::*;

const EXECUTION_QUEUE_SIZE: usize = 32;

pub trait Argument: Sync + Send {
    /// Returns a pointer to the object.
    fn raw_ptr(&self) -> *const libc::c_void;
    /// Returns the argument value if it can represent a size.
    fn as_size(&self) -> Option<u32> { None }
}

impl<T> Argument for T
where T: device::ScalarArgument
{
    fn raw_ptr(&self) -> *const libc::c_void { device::ScalarArgument::raw_ptr(self) }

    fn as_size(&self) -> Option<u32> { device::ScalarArgument::as_size(self) }
}

/// MPPA evaluation context.
pub struct Context<'a> {
    device: mppa::Mppa,
    executor: &'static telajax::Device,
    parameters: HashMap<String, Arc<Argument + 'a>>,
    wrappers: Cache<ir::Signature, telajax::Wrapper>,
    writeback_slots: MsQueue<telajax::Buffer>,
}
unsafe impl<'a> Sync for Context<'a> {}

/// We need to keep the arguments allocated for the kernel somewhere
enum KernelArg {
    GlobalMem(telajax::Buffer),
    Size(u32),
    External(*const libc::c_void),
}

impl KernelArg {
    fn raw_ptr(&self) -> *const libc::c_void {
        match self {
            KernelArg::GlobalMem(mem) => mem.raw_ptr(),
            KernelArg::Size(size) => size as *const _ as *const libc::c_void,
            KernelArg::External(ptr) => *ptr,
        }
    }
}

impl<'a> Context<'a> {
    /// Creates a new `Context`. Blocks until the MPPA device is ready to be
    /// used.
    pub fn new() -> Self {
        let executor = telajax::Device::get();
        let writeback_slots = MsQueue::new();
        for _ in 0..EXECUTION_QUEUE_SIZE {
            writeback_slots.push(executor.allocate_array(8));
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
    fn setup_kernel(
        &self,
        fun: &codegen::Function<'a>,
    ) -> (telajax::Kernel, Vec<KernelArg>)
    {
        let mut printer = MppaPrinter::default();
        let kernel_code = printer.wrapper_function(fun);
        std::fs::write("dump_kernel.c", &kernel_code).expect("Could not read file");
        //println!("KERNEL CODE\n{}", kernel_code);
        println!("Setting up kernel {}", fun.id);
        let wrapper = self.get_wrapper(fun);
        let cflags = std::ffi::CString::new("-mhypervisor").unwrap();
        let lflags = std::ffi::CString::new("-mhypervisor -lutask -lvbsp").unwrap();
        let kernel_code = unwrap!(std::ffi::CString::new(kernel_code));
        let mut kernel =
            self.executor
                .build_kernel(&kernel_code, &cflags, &lflags, &*wrapper);
        kernel.set_num_clusters(1);
        let mut namer = mppa::Namer::default();
        let name_map = codegen::NameMap::new(fun, &mut namer);
        let (mut arg_sizes, mut kernel_args): (Vec<_>, Vec<_>) = fun
            .device_code_args()
            .map(|p| {
                let name = name_map.name_param(p.key());
                match p {
                    ParamVal::External(par, _) => {
                        let name = name_map.name_param(p.key());
                        let arg = self.get_param(&name);
                        (get_type_size(p.t()), KernelArg::External(arg.raw_ptr()))
                    }
                    ParamVal::GlobalMem(_, size, _) => {
                        let size = self.eval_size(size);
                        let mem = self.executor.allocate_array(size as usize);
                        (
                            self::telajax::Mem::get_mem_size(),
                            KernelArg::GlobalMem(mem),
                        )
                    }
                    ParamVal::Size(size) => {
                        let size = self.eval_size(size);
                        //println!("param size {} : {}", name, size);
                        (get_type_size(p.t()), KernelArg::Size(size))
                    }
                }
            })
            .unzip();
        let out_mem = self.writeback_slots.pop();
        kernel_args.push(KernelArg::GlobalMem(out_mem));
        arg_sizes.push(self::telajax::Mem::get_mem_size());
        let args_ptr = kernel_args
            .iter()
            .map(|k_arg| k_arg.raw_ptr())
            .collect_vec();
        kernel.set_args(&arg_sizes[..], &args_ptr[..]);
        (kernel, kernel_args)
    }

    /// Returns the wrapper for the given signature.
    fn get_wrapper(&self, fun: &device::Function) -> Arc<telajax::Wrapper> {
        // TODO: There was a memoization here that allowed to cache a result for an
        // already generated signature wrapper. Maybe reimplement it
        let mut printer = MppaPrinter::default();
        let mut namer = mppa::Namer::default();
        let mut name_map = codegen::NameMap::new(fun, &mut namer);
        let ocl_code = printer.print_ocl_wrapper(fun, &mut name_map);
        //println!("{}", ocl_code);
        let name = std::ffi::CString::new(format!("wrapper_{}", fun.id)).unwrap();
        let ocl_code = std::ffi::CString::new(ocl_code).unwrap();
        Arc::new(self.executor.build_wrapper(&name, &ocl_code))
    }

    /// Returns a parameter given its name.
    pub fn get_param(&self, name: &str) -> &Argument { self.parameters[name].as_ref() }
}

fn get_type_size(t: ir::Type) -> usize {
    match t {
        ir::Type::I(u) | ir::Type::F(u) => (u / 8) as usize,
        ir::Type::PtrTo(_) => self::telajax::Mem::get_mem_size(),
        _ => panic!(),
    }
}

impl<'a> device::Context for Context<'a> {
    fn device(&self) -> &device::Device { &self.device }

    fn benchmark(&self, function: &device::Function, num_samples: usize) -> Vec<f64> {
        unimplemented!()
    }

    fn evaluate(&self, fun: &device::Function, _mode: EvalMode) -> Result<f64, ()> {
        let (mut kernel, mut kernel_args) = self.setup_kernel(fun);
        self.executor.execute_kernel_id(&mut kernel, fun.id);
        let out_mem = if let KernelArg::GlobalMem(mem) = kernel_args.pop().unwrap() {
            mem
        } else {
            panic!()
        };
        let ptr_i8 = out_mem.read_i8().as_ptr();
        let res: f64 = unsafe { *std::mem::transmute::<*const i8, *const f64>(ptr_i8) };
        self.writeback_slots.push(out_mem);
        Ok(res)
    }

    fn async_eval<'c, 'd>(
        &self,
        num_workers: usize,
        _mode: EvalMode,
        inner: &(Fn(&mut device::AsyncEvaluator<'c, 'd>) + Sync),
    )
    {
        // FIXME: execute in parallel
        let (send, recv) = mpsc::sync_channel(EXECUTION_QUEUE_SIZE);
        crossbeam::scope(move |scope| {
            // Start the explorer threads.
            for _ in 0..num_workers {
                let mut evaluator = AsyncEvaluator {
                    context: self,
                    sender: send.clone(),
                };
                unwrap!(
                    scope
                        .builder()
                        .name("Telamon - Explorer Thread".to_string())
                        .spawn(move || inner(&mut evaluator))
                );
            }
            // Start the evaluation thread.
            let eval_thread_name = "Telamon - CPU Evaluation Thread".to_string();
            unwrap!(scope.builder().name(eval_thread_name).spawn(move || {
                while let Ok((candidate, mut kernel, callback)) = recv.recv() {
                    // TODO: measure time directly on MPPA
                    let t0 = Instant::now();
                    panic!();
                    self.executor.execute_kernel(&mut kernel);
                    let t = Instant::now() - t0;
                    callback.call(candidate, t.subsec_nanos() as f64);
                }
            }));
        });
    }

    fn param_as_size(&self, name: &str) -> Option<u32> { self.get_param(name).as_size() }
}

impl<'a> device::ArgMap for Context<'a> {
    type Array = Buffer;

    fn bind_scalar<S: ScalarArgument>(&mut self, param: &ir::Parameter, value: S) {
        assert_eq!(param.t, S::t());
        self.bind_param(param.name.clone(), Arc::new(value));
    }

    fn bind_array<S: ScalarArgument>(
        &mut self,
        param: &ir::Parameter,
        len: usize,
    ) -> Arc<Self::Array>
    {
        let size = len * std::mem::size_of::<S>();
        //println!("Allocated {} bytes for param {}", size, param.name);
        let buffer_arc = Arc::new(Buffer::new(self.executor, size));
        self.bind_param(param.name.clone(), buffer_arc.clone());
        buffer_arc
    }
}

type AsyncPayload<'a, 'b> = (
    explorer::Candidate<'a>,
    telajax::Kernel,
    AsyncCallback<'a, 'b>,
);

/// Asynchronous evaluator.
struct AsyncEvaluator<'a, 'b>
where 'a: 'b
{
    context: &'b Context<'b>,
    sender: mpsc::SyncSender<AsyncPayload<'a, 'b>>,
}

impl<'a, 'b, 'c> device::AsyncEvaluator<'a, 'c> for AsyncEvaluator<'a, 'b>
where
    'a: 'b,
    'c: 'b,
{
    fn add_kernel(
        &mut self,
        candidate: explorer::Candidate<'a>,
        callback: device::AsyncCallback<'a, 'b>,
    )
    {
        let (kernel, out_mem) = {
            let dev_fun = device::Function::build(&candidate.space);
            self.context.setup_kernel(&dev_fun)
        };
        unwrap!(self.sender.send((candidate, kernel, callback)));
    }
}

impl Argument for Buffer {
    fn as_size(&self) -> Option<u32> { Some(self.mem.read().unwrap().len() as u32) }

    fn raw_ptr(&self) -> *const libc::c_void { self.mem.read().unwrap().raw_ptr() }
}
