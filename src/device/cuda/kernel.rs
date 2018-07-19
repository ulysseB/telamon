//! IR instances compiled into CUDA kernels.
use codegen;
use device;
use device::Context as ContextTrait;
use device::cuda::{api, Context, Gpu, JITDaemon};
#[cfg(feature="cuda")]
use device::cuda::PerfCounterSet;
use codegen::ParamVal;
use itertools::Itertools;
use std;

/// An IR instance compiled into a CUDA kernel.
pub struct Kernel<'a, 'b> {
    executor: &'a api::Executor,
    module: api::Module<'a>,
    function: &'b device::Function<'b>,
    expected_blocks_per_smx: u32,
    ptx: String,
}

impl<'a, 'b> Kernel<'a, 'b> {
    /// Compiles a device function.
    pub fn compile(fun: &'b device::Function<'b>,
                   gpu: &Gpu,
                   executor: &'a api::Executor,
                   opt_level: usize) -> Self {
        let ptx = gpu.print_ptx(fun);
        Kernel {
            module: executor.compile_ptx(&ptx, opt_level),
            executor, ptx,
            function: fun,
            expected_blocks_per_smx: gpu.blocks_per_smx(fun.space()),
        }
    }

    /// Compiles a device function, using a separate process.
    pub fn compile_remote(function: &'b device::Function<'b>, gpu: &Gpu,
                          executor: &'a api::Executor,
                          jit_daemon: &mut JITDaemon) -> Self {
        let ptx = gpu.print_ptx(function);
        let module =  executor.compile_remote(jit_daemon, &ptx);
        Kernel {
            executor, ptx, module, function,
            expected_blocks_per_smx: gpu.blocks_per_smx(function.space()),
        }
    }

    /// Runs a kernel and returns the number of cycles it takes to execute in cycles.
    pub fn evaluate(&self, args: &Context) -> Result<u64, ()> {
        let cuda_kernel = self.module.kernel(&self.function.name);
        self.gen_args(args).execute(&cuda_kernel, self.executor)
    }

    /// Runs a kernel and returns the number of cycles it takes to execute in nanoseconds,
    /// measured using cuda event rather than hardware counters.
    pub fn evaluate_real(&self, args: &Context, num_samples: usize) -> Vec<f64> {
        let cuda_kernel = self.module.kernel(&self.function.name);
        self.gen_args(args).time_in_real_conds(&cuda_kernel, num_samples, self.executor)
    }

    /// Instruments the kernel with the given performance counters.
    #[cfg(feature="cuda")]
    pub fn instrument(&self, args: &Context, counters: &PerfCounterSet) -> Vec<u64> {
        let cuda_kernel = self.module.kernel(&self.function.name);
        self.gen_args(args).instrument(&cuda_kernel, counters, self.executor)
    }

    /// Generates a Thunk than can then be run on the GPU.
    pub fn gen_thunk(self, args: &'a Context) -> Thunk<'a> {
        let args = self.gen_args(args);
        Thunk {
            name: self.function.name.clone(),
            module: self.module,
            executor: self.executor,
            ptx: self.ptx,
            args,
        }
    }

    fn gen_args(&self, args: &'a Context) -> ThunkArgs<'a> {
        let block_sizes = get_sizes(self.function.block_dims(), args);
        let thread_sizes = get_sizes(self.function.thread_dims().iter().rev(), args);
        let mut tmp_arrays = vec![];
        let params = self.function.device_code_args().map(|x| match *x {
            ParamVal::External(p, _) => ThunkArg::ArgRef(args.get_param(&p.name)),
            ParamVal::Size(ref s) => ThunkArg::Size(args.eval_size(s) as i32),
            ParamVal::GlobalMem(_, ref size, _) => {
                tmp_arrays.push(args.eval_size(size) as usize);
                ThunkArg::TmpArray(tmp_arrays.len() - 1)
            },
        }).collect_vec();
        ThunkArgs {
            blocks: block_sizes,
            threads: thread_sizes,
            tmp_arrays,
            args: params,
            expected_blocks_per_smx: self.expected_blocks_per_smx,
        }
    }
}

/// Generates an array to of dimension sizes.
fn get_sizes<'a, IT>(dims: IT, args: &Context) -> [u32; 3]
    where IT: IntoIterator<Item=&'a codegen::Dimension<'a>>
{
    let mut sizes = [1, 1, 1];
    for (i, s) in dims.into_iter().map(|x| args.eval_size(x.size())).enumerate() {
        assert!(i < 3);
        sizes[i] = s;
    }
    sizes
}

/// A kernel ready to execute.
pub struct Thunk<'a> {
    name: String,
    ptx: String,
    module: api::Module<'a>,
    executor: &'a api::Executor,
    args: ThunkArgs<'a>,
}

impl<'a> Thunk<'a> {
    /// Executes the kernel and returns the number of cycles it took to execute.
    pub fn execute(&self) -> Result<u64, ()> {
        let cuda_kernel = self.module.kernel(&self.name);
        self.args.execute(&cuda_kernel, self.executor)
    }
}

impl<'a> std::fmt::Debug for Thunk<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "kernel: {}", self.name)?;
        self.args.fmt(f)?;
        write!(f, "{}", self.ptx)
    }
}


/// The arguments of a kernel ready to execute.
struct ThunkArgs<'a> {
    blocks: [u32; 3],
    threads: [u32; 3],
    tmp_arrays: Vec<usize>,
    args: Vec<ThunkArg<'a>>,
    expected_blocks_per_smx: u32,
}

impl<'a> ThunkArgs<'a> {
    /// Executes the kernel.
    pub fn execute(&self, cuda_kernel: &api::Kernel, executor: &api::Executor)
        -> Result<u64, ()>
    {
        self.check_blocks_per_smx(cuda_kernel);
        let tmp_arrays = self.tmp_arrays.iter().map(|&size| {
            executor.allocate_array::<i8>(size)
        }).collect_vec();
        let params = self.args.iter().map(|x| match *x {
            ThunkArg::ArgRef(arg) => arg,
            ThunkArg::Size(ref arg) => arg,
            ThunkArg::TmpArray(id) => &tmp_arrays[id],
        }).collect_vec();
        cuda_kernel.execute(&self.blocks, &self.threads, &params)
    }

    /// Instruments the kernel.
    #[cfg(feature="cuda")]
    pub fn instrument(&self, cuda_kernel: &api::Kernel, counters: &PerfCounterSet,
                      executor: &api::Executor) -> Vec<u64> {
        self.check_blocks_per_smx(cuda_kernel);
        let tmp_arrays = self.tmp_arrays.iter().map(|&size| {
            executor.allocate_array::<i8>(size)
        }).collect_vec();
        let params = self.args.iter().map(|x| match *x {
            ThunkArg::ArgRef(arg) => arg,
            ThunkArg::Size(ref arg) => arg,
            ThunkArg::TmpArray(id) => &tmp_arrays[id],
        }).collect_vec();
        cuda_kernel.instrument(&self.blocks, &self.threads, &params, counters)
    }

    /// Evaluates the execution time of the kernel usig events rather than hardware
    /// counter.
    fn time_in_real_conds(&self, cuda_kernel: &api::Kernel, num_samples: usize,
                          executor: &api::Executor)
        -> Vec<f64>
    {
        let tmp_arrays = self.tmp_arrays.iter().map(|&size| {
            executor.allocate_array::<i8>(size)
        }).collect_vec();
        let params = self.args.iter().map(|x| match *x {
            ThunkArg::ArgRef(arg) => arg,
            ThunkArg::Size(ref arg) => arg,
            ThunkArg::TmpArray(id) => &tmp_arrays[id],
        }).collect_vec();
        // Heat-up caches.
        for _ in 0..100 {
            cuda_kernel.time_real_conds(&self.blocks, &self.threads, &params);
        }
        // Generate the samples.
        (0..num_samples).map(|_| {
            cuda_kernel.time_real_conds(&self.blocks, &self.threads, &params)
        }).collect()
    }

    fn check_blocks_per_smx(&self, cuda_kernel: &api::Kernel) {
        let blocks_per_smx = cuda_kernel.blocks_per_smx(&self.threads);
        if blocks_per_smx != self.expected_blocks_per_smx {
            warn!("mismatch in the number of blocks per SMX: expected {}, got {}",
                  self.expected_blocks_per_smx, blocks_per_smx);
        }
    }
}

impl<'a> std::fmt::Debug for ThunkArgs<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "blocks = {:?}, threads = {:?}", self.blocks, self.threads)?;
        for argument in &self.args { writeln!(f, "  {:?}", argument)?; }
        Ok(())
    }
}

/// An argument of a kernel ready to evaluate.
enum ThunkArg<'a> { ArgRef(&'a api::Argument), Size(i32), TmpArray(usize) }

impl<'a> std::fmt::Debug for ThunkArg<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ThunkArg::ArgRef(_) => write!(f, "context argument"),
            ThunkArg::Size(size) => write!(f, "size = {}", size),
            ThunkArg::TmpArray(size) => write!(f, "temporary array of size {}", size),
        }
    }
}
