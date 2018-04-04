//! Describes CUDA-enabled GPUs.
use device::{self, Device};
use codegen::Function;
use ir::{self, Type};
use model::{self, HwPressure};
use search_space::{DimKind, Domain, InstFlag, MemSpace, SearchSpace};
use std;
use std::fs::File;
use std::io::{Read, Write};
use utils::*;

/// Specifies the performance parameters of an instruction.
#[derive(Default, Clone, Copy, Debug)]
pub struct InstDesc {
    /// The latency of the instruction.
    pub latency: f64,
    /// The number of instruction to issue.
    pub issue: f64,
    /// The number of instruction on the ALUs.
    pub alu: f64,
    /// The number of syncthread units used.
    pub sync: f64,
    /// The number of instruction on Load/Store units.
    pub mem: f64,
    /// The number of L1 cache lines that are fetched from the L2.
    pub l1_lines_from_l2: f64,
    /// The number of L2 cache lines that are fetched from the L2.
    pub l2_lines_from_l2: f64,
    /// The ram bandwidth used.
    pub ram_bw: f64,
}

impl InstDesc {
    fn dummy_inst_desc() -> Self {
        InstDesc {
            latency: 1.,
            issue: 1.,
            alu: 2.,
            sync: 1.,
            mem: 1.,
            l1_lines_from_l2: 0.,
            l2_lines_from_l2: 0.,
            ram_bw: 0.5,
        }
    }

    /// Multiplies concerned bottlenecks by the wrap use ratio.
    fn apply_use_ratio(self, ratio: f64) -> Self {
        unimplemented!()
    }
}

impl Into<HwPressure> for InstDesc {
    fn into(self) -> HwPressure {
        unimplemented!()
    }
}

/// Represents CUDA GPUs.
#[derive(Clone)]
pub struct Cpu {
    /// The name of the CPU.
    pub name: String,
    /// The compute capability major number.
    pub sm_major: u8,
    /// The compute capability minor number.
    pub sm_minor: u8,
    // TODO(perf): pointer size should be a parameter of the function and not of the GPU.
    /// The size of pointers.
    pub addr_size: u64,
    /// The size in bytes of the L1 cache.
    pub l1_cache_size: u32,
    /// The size in bytes of a L1 cache line.
    pub l1_cache_line: u32,
    /// The size in bytes of the L2 cache.
    pub l2_cache_size: u32,
    /// The size in bytes of a L2 cache line.
    pub l2_cache_line: u32,
    /// Latency of an L2 access.
    pub load_l2_latency: f64,
    /// Latency of a RAM access.
    pub load_ram_latency: f64,

    // Instructions performance description.
    pub add_f32_inst: InstDesc,
    pub add_f64_inst: InstDesc,
    pub add_i32_inst: InstDesc,
    pub add_i64_inst: InstDesc,
    pub mul_f32_inst: InstDesc,
    pub mul_f64_inst: InstDesc,
    pub mul_i32_inst: InstDesc,
    pub mul_i64_inst: InstDesc,
    pub mul_wide_inst: InstDesc,
    pub mad_f32_inst: InstDesc,
    pub mad_f64_inst: InstDesc,
    pub mad_i32_inst: InstDesc,
    pub mad_i64_inst: InstDesc,
    pub mad_wide_inst: InstDesc,
    pub div_f32_inst: InstDesc,
    pub div_f64_inst: InstDesc,
    pub div_i32_inst: InstDesc,
    pub div_i64_inst: InstDesc,
    pub syncthread_inst: InstDesc,

    /// Overhead for entring the loop.
    pub loop_init_overhead: InstDesc,
    /// Overhead for a single iteration of the loop.
    pub loop_iter_overhead: InstDesc,
    /// Latency for exiting the loop.
    pub loop_end_latency: f64,
}

impl Cpu {
    pub fn dummy_cpu() -> Self {
        Cpu {
            name: String::from("x86"),
            sm_major: 1,
            sm_minor: 1,
            addr_size: 8,
            l1_cache_size: u32::pow(2, 12),
            l1_cache_line: 64,
            l2_cache_size: u32::pow(2, 12),
            l2_cache_line: 64,
            load_l2_latency: 10.,
            load_ram_latency: 50.,
            add_f32_inst: InstDesc::dummy_inst_desc(),
            add_f64_inst: InstDesc::dummy_inst_desc(),
            add_i32_inst: InstDesc::dummy_inst_desc(),
            add_i64_inst: InstDesc::dummy_inst_desc(),
            mul_f32_inst: InstDesc::dummy_inst_desc(),
            mul_f64_inst: InstDesc::dummy_inst_desc(),
            mul_i32_inst: InstDesc::dummy_inst_desc(),
            mul_i64_inst: InstDesc::dummy_inst_desc(),
            mul_wide_inst: InstDesc::dummy_inst_desc(),
            mad_f32_inst: InstDesc::dummy_inst_desc(),
            mad_f64_inst: InstDesc::dummy_inst_desc(),
            mad_i32_inst: InstDesc::dummy_inst_desc(),
            mad_i64_inst: InstDesc::dummy_inst_desc(),
            mad_wide_inst: InstDesc::dummy_inst_desc(),
            div_f32_inst: InstDesc::dummy_inst_desc(),
            div_f64_inst: InstDesc::dummy_inst_desc(),
            div_i32_inst: InstDesc::dummy_inst_desc(),
            div_i64_inst: InstDesc::dummy_inst_desc(),
            syncthread_inst: InstDesc::dummy_inst_desc(),
            loop_init_overhead: InstDesc::dummy_inst_desc(),
            loop_iter_overhead: InstDesc::dummy_inst_desc(),
            loop_end_latency: 1.,
        }
    }
    /// Returns the GPU model corresponding to `name.
    //pub fn from_name(name: &str) -> Option<Cpu> {
    //    let mut file = unwrap!(File::open("data/cuda_gpus.json"));
    //    let mut string = String::new();
    //    unwrap!(file.read_to_string(&mut string));
    //    let gpus: Vec<Gpu> = unwrap!(json::decode(&string));
    //    gpus.into_iter().find(|x| x.name == name)
    //}

    /// Returns the PTX code for a Function.
    //pub fn print_ptx(&self, fun: &Function) -> String {
    //    p::function(fun, self)
    //}

    /// Returns the ratio of threads actually used per wrap.
    fn wrap_use_ratio(&self, max_num_threads: u64) -> f64 {
        unimplemented!()
    }

    /// Returns the description of a load instruction.
    //fn load_desc(&self, mem_info: &MemInfo, flags: InstFlag) -> InstDesc {
    //    // TODO(search_space,model): support CA and NC flags.
    //    assert!(InstFlag::MEM_COHERENT.contains(flags));
    //    // Compute possible latencies.
    //    let gbl_latency = if flags.intersects(InstFlag::MEM_GLOBAL) {
    //        let miss = mem_info.l2_miss_ratio/mem_info.l2_coalescing;
    //        miss*self.load_ram_latency + (1.0-miss)*self.load_l2_latency
    //    } else { std::f64::INFINITY };
    //    let shared_latency = if flags.intersects(InstFlag::MEM_SHARED) {
    //        self.load_shared_latency as f64
    //    } else { std::f64::INFINITY };
    //    // Compute the smx bandwidth used.
    //    let l1_lines_from_l2 = if flags.intersects(InstFlag::MEM_SHARED) {
    //        0.0
    //    } else { mem_info.l1_coalescing };
    //    let l2_lines_from_l2 = if flags.intersects(InstFlag::MEM_SHARED) {
    //        0.0
    //    } else { mem_info.l2_coalescing };
    //    InstDesc {
    //        latency: f64::min(gbl_latency, shared_latency),
    //        issue: mem_info.replay_factor,
    //        mem: mem_info.replay_factor,
    //        l1_lines_from_l2, l2_lines_from_l2,
    //        ram_bw: mem_info.l2_miss_ratio * f64::from(self.l2_cache_line),
    //        .. InstDesc::default()
    //    }
    //}

    /// Returns the description of a store instruction.
    //fn store_desc(&self, mem_info: &MemInfo, flags: InstFlag) -> InstDesc {
    //    // TODO(search_space,model): support CA flags.
    //    // TODO(model): understand how writes use the BW.
    //    assert!(InstFlag::MEM_COHERENT.contains(flags));
    //    let l2_lines_from_l2 = if flags.intersects(InstFlag::MEM_SHARED) {
    //        0.0
    //    } else { mem_info.l2_coalescing };
    //    // L1 lines per L2 is not limiting.
    //    InstDesc {
    //        issue: mem_info.replay_factor,
    //        mem: mem_info.replay_factor,
    //        l2_lines_from_l2,
    //        ram_bw: 2.0 * mem_info.l2_miss_ratio * f64::from(self.l2_cache_line),
    //        .. InstDesc::default()
    //    }
    //}

    /// Returns the overhead induced by all the iterations of a loop.
    fn dim_pressure(&self, kind: DimKind, size: u32) -> HwPressure {
        unimplemented!()
    }

    /// Retruns the overhead for a single instance of the instruction.
    fn inst_pressure(&self, space: &SearchSpace,
                         dim_sizes: &HashMap<ir::dim::Id, u32>,
                         inst: &ir::Instruction) -> HwPressure {
        unimplemented!()
    }

    /// Computes the number of blocks that can fit in an smx.
    pub fn blocks_per_smx(&self, space: &SearchSpace) -> u32 {
        unimplemented!()
    }
}

fn hello_world() -> String {
    let mut file = File::open("template/hello_world.c").expect("Could not find file");
    let mut contents = String::new();
    file.read_to_string(&mut contents);
    contents
}

impl device::Device for Cpu {
    fn print(&self, fun: &Function, out: &mut Write) { write!(out, "{}", hello_world()); }

    fn is_valid_type(&self, t: &Type) -> bool {
        match *t {
            Type::I(i) | Type::F(i) => i == 32 || i == 64,
            Type::Void | Type::PtrTo(_) => true,
        }
    }

    fn max_block_dims(&self) -> u32 { 1 }

    fn max_threads(&self) -> u32 { 24 }

    fn max_unrolling(&self) -> u32 { 512 }

    fn shared_mem(&self) -> u32 { 2u32.pow(25) }

    fn supports_nc_access(&self) -> bool {false}

    fn supports_l1_access(&self) -> bool {true}

    fn supports_l2_access(&self) -> bool {true}

    fn name(&self) -> &str { &self.name }

    fn lower_type(&self, t: ir::Type, space: &SearchSpace) -> Option<ir::Type> {
        unimplemented!()
    }

    fn hw_pressure(&self, space: &SearchSpace,
                   dim_sizes: &HashMap<ir::dim::Id, u32>,
                   _nesting: &HashMap<ir::BBId, model::Nesting>,
                   bb: &ir::BasicBlock) -> model::HwPressure {
        unimplemented!()
    }

    fn loop_iter_pressure(&self, kind: DimKind) -> (HwPressure, HwPressure) {
        unimplemented!()
    }

    fn thread_rates(&self) -> HwPressure {unimplemented!()}

    fn block_rates(&self, max_num_threads: u64) -> HwPressure {
        unimplemented!()
    }

    fn total_rates(&self, max_num_threads: u64) -> HwPressure {
        unimplemented!()
    }

    fn bottlenecks(&self) -> &[&'static str] {
        unimplemented!()
    }

    fn block_parallelism(&self, space: &SearchSpace) -> u32 {
        unimplemented!()
    }

    fn additive_indvar_pressure(&self, t: &ir::Type) -> HwPressure {
        unimplemented!()
    }

    fn multiplicative_indvar_pressure(&self, t: &ir::Type) -> HwPressure {
        unimplemented!()
    }
}
