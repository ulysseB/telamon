//! Declarations of foreign functions and structs.
use libc;

pub enum CuptiEventGroupSets {}
pub enum CudaContext {}
pub enum CudaModule {}
pub enum CudaFunction {}
pub enum CudaArray {}

#[repr(C)]
pub struct CubinObject {
    state: *const libc::c_void,
    pub data: *const u8,
    pub data_size: libc::size_t,
}

/// Imports the interface contained in cuda.c,
extern "C" {
    pub fn init_cuda(seed: u64) -> *mut CudaContext;
    pub fn free_cuda(context: *mut CudaContext);
    pub fn device_name(context: *const CudaContext) -> *mut libc::c_char;
    pub fn compile_ptx(
        context: *const CudaContext,
        ptx_code: *const libc::c_char,
        opt_level: libc::size_t,
    ) -> *mut CudaModule;
    pub fn load_cubin(
        context: *const CudaContext,
        image: *const libc::c_void,
    ) -> *mut CudaModule;
    pub fn free_module(module: *mut CudaModule);
    pub fn get_function(
        context: *const CudaContext,
        module: *const CudaModule,
        name: *const libc::c_char,
    ) -> *mut CudaFunction;
    pub fn launch_kernel(
        context: *const CudaContext,
        function: *mut CudaFunction,
        blocks: *const u32,
        threads: *const u32,
        params: *const *const libc::c_void,
        out: *mut u64,
    ) -> i32;
    pub fn time_with_events(
        context: *const CudaContext,
        function: *mut CudaFunction,
        blocks: *const u32,
        threads: *const u32,
        params: *const *const libc::c_void,
    ) -> f64;
    pub fn instrument_kernel(
        context: *const CudaContext,
        function: *const CudaFunction,
        blocks: *const u32,
        threads: *const u32,
        params: *const *const libc::c_void,
        events: *const CuptiEventGroupSets,
        event_ids: *mut u32,
        event_values: *mut u64,
    );
    pub fn allocate_array(context: *const CudaContext, size: u64) -> *mut CudaArray;
    pub fn free_array(context: *const CudaContext, array: *mut CudaArray);
    pub fn copy_DtoH(
        context: *const CudaContext,
        src: *const CudaArray,
        dst: *mut libc::c_void,
        size: u64,
    );
    pub fn copy_HtoD(
        context: *const CudaContext,
        src: *const libc::c_void,
        dst: *const CudaArray,
        size: u64,
    );
    pub fn copy_DtoD(
        context: *const CudaContext,
        src: *const CudaArray,
        dst: *mut CudaArray,
        size: u64,
    );
    pub fn randomize_float_array(
        ctx: *const CudaContext,
        dst: *mut CudaArray,
        size: u64,
        mean: f32,
        stddev: f32,
    );
    pub fn device_attribute(context: *const CudaContext, attr: u32) -> i32;
    pub fn create_cuptiEventGroupSets(
        context: *const CudaContext,
        num_event: u32,
        event_names: *const *const libc::c_char,
        event_ids: *mut u32,
    ) -> *mut CuptiEventGroupSets;
    pub fn free_cuptiEventGroupSets(
        context: *const CudaContext,
        sets: *mut CuptiEventGroupSets,
    );
    pub fn max_active_blocks_per_smx(
        function: *const CudaFunction,
        block_size: u32,
        dynamic_smem_size: libc::size_t,
    ) -> u32;
    pub fn compile_ptx_to_cubin(
        ctx: *const CudaContext,
        code: *const libc::c_char,
        code_size: libc::size_t,
        opt_level: libc::size_t,
    ) -> CubinObject;
    pub fn free_cubin_object(object: CubinObject);
}
