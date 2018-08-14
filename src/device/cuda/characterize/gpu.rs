//! Builds the description of a GPU.
use device::cuda::{self, Executor, InstDesc};
use device::cuda::DeviceAttribute::*;
use device::cuda::characterize::instruction;
use std;

const EMPTY_INST_DESC: InstDesc = InstDesc {
    latency: 0.0,
    issue: 0.0,
    alu: 0.0,
    mem: 0.0,
    l1_lines_from_l2: 0.0,
    l2_lines_read: 0.0,
    l2_lines_stored: 0.0,
    sync: 0.0,
    ram_bw: 0.0,
};

/// Returns the description of the GPU. Performance-related fields are not filled.
pub fn functional_desc(executor: &cuda::Executor) -> cuda::Gpu {
    let sm_major = executor.device_attribute(ComputeCapabilityMajor);
    let sm_minor = executor.device_attribute(ComputeCapabilityMinor);
    cuda::Gpu {
        name: executor.device_name(),
        sm_major: sm_major as u8,
        sm_minor: sm_minor as u8,
        addr_size: 64,
        shared_mem_per_smx: executor.device_attribute(MaxSharedMemoryPerSmx) as u32,
        shared_mem_per_block: executor.device_attribute(MaxSharedMemoryPerBlock) as u32,
        allow_nc_load: allow_nc_load(sm_major, sm_minor),
        allow_l1_for_global_mem: allow_l1_for_global_mem(sm_major, sm_minor),
        wrap_size: executor.device_attribute(WrapSize) as u32,
        thread_per_smx: thread_per_smx(sm_major, sm_minor),
        l1_cache_size: l1_cache_size(sm_major, sm_minor) as u32,
        l1_cache_line: 128,
        l2_cache_size: executor.device_attribute(L2CacheSize) as u32,
        l2_cache_line: 32,
        shared_bank_stride: shared_bank_stride(sm_major, sm_minor),
        num_smx: executor.device_attribute(SmxCount) as u32,
        max_block_per_smx: block_per_smx(sm_major, sm_minor),
        smx_clock: f64::from(executor.device_attribute(ClockRate)) /1.0E+6,

        thread_rates: EMPTY_INST_DESC,
        smx_rates: EMPTY_INST_DESC,
        gpu_rates: EMPTY_INST_DESC,

        add_f32_inst: EMPTY_INST_DESC,
        add_f64_inst: EMPTY_INST_DESC,
        add_i32_inst: EMPTY_INST_DESC,
        add_i64_inst: EMPTY_INST_DESC,
        mul_f32_inst: EMPTY_INST_DESC,
        mul_f64_inst: EMPTY_INST_DESC,
        mul_i32_inst: EMPTY_INST_DESC,
        mul_i64_inst: EMPTY_INST_DESC,
        mul_wide_inst: EMPTY_INST_DESC,
        mad_f32_inst: EMPTY_INST_DESC,
        mad_f64_inst: EMPTY_INST_DESC,
        mad_i32_inst: EMPTY_INST_DESC,
        mad_i64_inst: EMPTY_INST_DESC,
        mad_wide_inst: EMPTY_INST_DESC,
        div_f32_inst: EMPTY_INST_DESC,
        div_f64_inst: EMPTY_INST_DESC,
        div_i32_inst: EMPTY_INST_DESC,
        div_i64_inst: EMPTY_INST_DESC,
        syncthread_inst: EMPTY_INST_DESC,
        loop_init_overhead: EMPTY_INST_DESC,
        loop_iter_overhead: EMPTY_INST_DESC,
        loop_end_latency: 0f64,
        load_l2_latency: 0.0,
        load_ram_latency: 0.0,
        load_shared_latency: 0.0,
    }
}

/// Returns true if the GPU allows non-coherent loads.
fn allow_nc_load(sm_major: i32, sm_minor: i32) -> bool {
    match (sm_major, sm_minor) {
        (2, _) | (3, 0) | (3, 2) => false,
        (3, 5) | (3, 7) | (5, _) => true,
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returns true if the GPU allows L1 caching of global memory accesses.
fn  allow_l1_for_global_mem(sm_major: i32, sm_minor: i32) -> bool {
    match (sm_major, sm_minor) {
        (2, _) | (3, 7) | (5, 2) => true,
        (3, 0) | (3, 2) | (3, 5) | (5, 0) => false,
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returns the maximum number of resident blocks on an SMX.
fn block_per_smx(sm_major: i32, sm_minor: i32) -> u32 {
    match sm_major {
        2 => 8,
        3 => 16,
        5 => 32,
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returns the size of the L1 cache.
fn l1_cache_size(sm_major: i32, sm_minor: i32) -> u32 {
    match sm_major {
        2 | 3 => 16 * 1024,
        5 => 24 * 1024,
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returns the stride between shared memory banks.
fn shared_bank_stride(sm_major: i32, sm_minor: i32) -> u32 {
    match sm_major {
        2 => 4,
        3 | 5 => 8,
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returns the maximum number of resident thread per SMX.
fn thread_per_smx(sm_major: i32, sm_minor: i32) -> u32 {
    match sm_major {
        2 => 1536,
        3 | 5 => 2048,
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returns the amount of processing power available in a single SMX, in unit per cycle.
fn smx_rates(gpu: &cuda::Gpu, executor: &Executor) -> InstDesc {
    let (wrap_scheds, issues_per_wrap) = wrap_scheds_per_smx(gpu.sm_major, gpu.sm_minor);
    let issue = wrap_scheds * issues_per_wrap * gpu.wrap_size;
    let (alu, mem, sync) = match (gpu.sm_major, gpu.sm_minor) {
        (2, 0) => (32, 16, 32), // Sync unknown for 2.0
        (2, 1) => (48, 32, 32), // Sync unknown for 2.0
        (3, _) => (192, 32, 128),
        (5, _) => (128, 32, 64),
        (major, minor) => panic!("Unkown compute capability: {}.{}", major, minor)
    };
    let l1_lines_bw = instruction::smx_bandwidth_l1_lines(gpu, executor);
    let l2_lines_read_bw = instruction::smx_read_bandwidth_l2_lines(gpu, executor);
    let l2_lines_store_bw = instruction::smx_write_bandwidth_l2_lines(gpu, executor);
    InstDesc {
        latency: gpu.smx_clock,
        issue: f64::from(issue) * gpu.smx_clock,
        alu: f64::from(alu) * gpu.smx_clock,
        mem: f64::from(mem) * gpu.smx_clock,
        sync: f64::from(sync) * gpu.smx_clock,
        l1_lines_from_l2: l1_lines_bw * gpu.smx_clock,
        l2_lines_read: l2_lines_read_bw * gpu.smx_clock,
        l2_lines_stored: l2_lines_store_bw * gpu.smx_clock,
        ram_bw: ram_bandwidth(executor),
    }
}

/// Computes the processing power of a single thread.
fn thread_rates(gpu: &cuda::Gpu, smx_rates: &InstDesc) -> InstDesc {
    let (_, issues_per_wrap) = wrap_scheds_per_smx(gpu.sm_major, gpu.sm_minor);
    let wrap_size = f64::from(gpu.wrap_size);
    InstDesc {
        latency: smx_rates.latency,
        issue: f64::from(issues_per_wrap) * gpu.smx_clock,
        alu: smx_rates.alu / wrap_size,
        mem: smx_rates.mem / wrap_size,
        sync: smx_rates.sync / wrap_size,
        l1_lines_from_l2: smx_rates.l1_lines_from_l2, // FIXME: actually smaller
        l2_lines_read: smx_rates.l2_lines_read,
        l2_lines_stored: smx_rates.l2_lines_stored,
        ram_bw: smx_rates.ram_bw,
    }
}

/// Computes the total processing power from the processing power of a single SMX.
fn gpu_rates(gpu: &cuda::Gpu, smx_rates: &InstDesc) -> InstDesc {
    let num_smx = f64::from(gpu.num_smx);
    InstDesc {
        latency: smx_rates.latency,
        issue: smx_rates.issue * num_smx,
        alu: smx_rates.alu * num_smx,
        mem: smx_rates.mem * num_smx,
        sync: smx_rates.sync * num_smx,
        l1_lines_from_l2: smx_rates.l1_lines_from_l2 * num_smx,
        l2_lines_read: smx_rates.l2_lines_read * num_smx,
        l2_lines_stored: smx_rates.l2_lines_stored * num_smx,
        ram_bw: smx_rates.ram_bw,
    }
}

/// Returns the number of wrap scheduler in a single SMX and the number of independent
/// instruction issued per wrap scheduler.
fn wrap_scheds_per_smx(sm_major: u8, sm_minor: u8) -> (u32, u32) {
    match (sm_major, sm_minor) {
        (2, 0) => (2, 1),
        (2, 1) => (2, 2),
        (3, _) => (4, 2),
        (5, _) => (4, 1),
        _ => panic!("Unkown compute capability: {}.{}", sm_major, sm_minor)
    }
}

/// Returms the RAM bandwidth in bytes per SMX clock cycle.
fn ram_bandwidth(executor: &Executor) -> f64 {
    // TODO(model): take ECC into account.
    let mem_clock = f64::from(executor.device_attribute(MemoryClockRate)) / 1.0E+6;
    let mem_bus_width = executor.device_attribute(GlobalMemoryBusWidth) / 8;
    // Multiply by 2 because is a DDR, so it uses bith the up and down signals of the
    // clock.
    2.0 * mem_clock * f64::from(mem_bus_width)
}

/// Updates the gpu description with performance numbers.
pub fn performance_desc(executor: &Executor, gpu: &mut cuda::Gpu) {
    // TODO(model): l1 and l2 lines rates may not be correct on non-kepler architectures
    // Compute the processing.
    gpu.smx_rates = smx_rates(gpu, executor);
    gpu.thread_rates = thread_rates(gpu, &gpu.smx_rates);
    gpu.gpu_rates = gpu_rates(gpu, &gpu.smx_rates);
    // Compute instruction overhead.
    gpu.add_f32_inst = instruction::add_f32(gpu, executor);
    gpu.add_f64_inst = instruction::add_f64(gpu, executor);
    gpu.add_i32_inst = instruction::add_i32(gpu, executor);
    gpu.add_i64_inst = instruction::add_i64(gpu, executor);
    gpu.mul_f32_inst = instruction::mul_f32(gpu, executor);
    gpu.mul_f64_inst = instruction::mul_f64(gpu, executor);
    gpu.mul_i32_inst = instruction::mul_i32(gpu, executor);
    gpu.mul_i64_inst = instruction::mul_i64(gpu, executor);
    gpu.mad_f32_inst = instruction::mad_f32(gpu, executor);
    gpu.mad_f64_inst = instruction::mad_f64(gpu, executor);
    gpu.mad_i32_inst = instruction::mad_i32(gpu, executor);
    gpu.mad_i64_inst = instruction::mad_i64(gpu, executor);
    gpu.mad_wide_inst = instruction::mad_wide(gpu, executor);
    gpu.div_f32_inst = instruction::div_f32(gpu, executor);
    gpu.div_f64_inst = instruction::div_f64(gpu, executor);
    gpu.div_i32_inst = instruction::div_i32(gpu, executor);
    gpu.div_i64_inst = instruction::div_i64(gpu, executor);
    gpu.mul_wide_inst = gpu.mul_i32_inst; // TODO(model): benchmark mul wide.
    // Compute memory accesses overhead.
    gpu.load_l2_latency = instruction::load_l2(gpu, executor);
    gpu.load_ram_latency = instruction::load_ram(gpu, executor);
    gpu.load_shared_latency = instruction::load_shared(gpu, executor);
    // Compute loops overhead.
    gpu.syncthread_inst = instruction::syncthread(gpu, executor);
    let addf32_lat = gpu.add_f32_inst.latency;
    let syncthread_end_latency = instruction::syncthread_end_latency(
        gpu, executor, addf32_lat);
    assert!(syncthread_end_latency < std::f64::EPSILON);
    gpu.loop_end_latency = instruction::loop_iter_end_latency(gpu, executor, addf32_lat);
    gpu.loop_iter_overhead = instruction::loop_iter_overhead(gpu, executor);
    gpu.loop_init_overhead = cuda::InstDesc { issue: 1f64, alu: 1f64, .. EMPTY_INST_DESC };
}
