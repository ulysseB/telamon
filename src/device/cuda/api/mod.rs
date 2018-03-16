//! Wrapper the CUDA API.
#![allow(dead_code)]
mod array;
mod counter;
mod executor;
mod module;
mod wrapper;
mod jit_daemon;

pub use self::array::{Array, ArrayArg};
pub use self::executor::*;
pub use self::counter::{PerfCounter, PerfCounterSet};
pub use self::module::{Module, Kernel};
pub use self::jit_daemon::JITDaemon;

use self::jit_daemon::DaemonSpawner;

#[cfg(test)]
mod tests {
    use super::*;
    use super::array;
    use ir;

    /// Tries to initialize a CUDA execution context.
    #[test]
    fn test_init() {
        let _ = Executor::init();
    }

    /// Tries to obtain the name of the GPU.
    #[test]
    fn test_getname() {
        let executor = Executor::init();
        executor.device_name();
    }

    /// Tries to compile an empty PTX module.
    #[test]
    fn test_empty_module() {
      let executor = Executor::init();
      let _ = executor.compile_ptx(".version 3.0\n.target sm_30\n.address_size 64\n");
    }

    /// Tries to compile an empty PTX kernel and execute it.
    #[test]
    fn test_empty_kernel() {
        let executor = Executor::init();
        let module = executor.compile_ptx(
            ".version 3.0\n.target sm_30\n.address_size 64\n
            .entry empty_fun() { ret; }");
        let kernel = module.kernel("empty_fun");
        let _ = kernel.execute(&[1,1,1], &[1,1,1], &mut []);
    }

    /// Tries to allocate an array.
    #[test]
    fn test_array_allocation() {
        let executor = Executor::init();
        let _: Array<f32> = executor.allocate_array(1024);
    }

    /// Allocates two identical arrays and ensures they are equal.
    #[test]
    fn test_array_copy() {
      let executor = Executor::init();
      let src = executor.allocate_array::<f32>(1024);
      let dst = src.clone();
      assert!(array::compare_f32(&src, &dst) < 1e-5);
    }

    /// Alocates a random array and copies using a PTX kernel.
    #[test]
    fn test_array_soft_copy() {
        let block_dim: u32 = 16;
        let executor = Executor::init();
        let mut src = executor.allocate_array::<f32>(block_dim as usize);
        let dst = executor.allocate_array::<f32>(block_dim as usize);
        array::randomize_f32(&mut src);
        let mut src = ArrayArg(src, ir::mem::Id::External(0));
        let mut dst = ArrayArg(dst, ir::mem::Id::External(1));
        let module = executor.compile_ptx(
            ".version 3.0\n.target sm_30\n.address_size 64\n
            .entry copy(
                .param.u64.ptr.global .align 16 src,
                .param.u64.ptr.global .align 16 dst
            ) {
                .reg.u64 %rd<4>;
                .reg.u32 %r<1>;
                .reg.f32 %f;
                ld.param.u64 %rd0, [src];
                ld.param.u64 %rd1, [dst];
                mov.u32 %r0, %ctaid.x;
                mad.wide.u32 %rd2, %r0, 4, %rd0;
                mad.wide.u32 %rd3, %r0, 4, %rd1;
                ld.global.f32 %f, [%rd2];
                st.global.f32 [%rd3], %f;
                ret;
            }");
        let kernel = module.kernel("copy");
        let _ = kernel.execute(&[block_dim, 1, 1], &[1, 1, 1], &mut [&mut src, &mut dst]);
        assert!(array::compare_f32(&src.0, &dst.0) < 1e-5);
    }
}
