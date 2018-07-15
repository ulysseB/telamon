//! Describes CUDA-enabled GPUs.
use codegen::Function;
use device::{self, cuda, Device};
use device::cuda::CudaPrinter;
use device::cuda::mem_model::{self, MemInfo};
use ir::{self, Type, Operator};
use model::{self, HwPressure};
use search_space::{DimKind, Domain, InstFlag, MemSpace, SearchSpace};
use std;
use std::io::Write;
use utils::*;

// FIXME: fix performance model
// - l1_lines constraint for stores ?
// - test if global pressure is needed
// - l1_lines per threads ?

/// Specifies the performance parameters of an instruction.
#[derive(Default, Serialize, Deserialize, Clone, Copy, Debug)]
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
    /// The number of L2 cache lines read.
    pub l2_lines_read: f64,
    /// Number of l2 cache lines stored.
    pub l2_lines_stored: f64,
    /// The ram bandwidth used.
    pub ram_bw: f64,
}

impl InstDesc {
    fn wasted_ratio(ratio: f64) -> Self {
        InstDesc {
            latency: 1.0,
            issue: ratio,
            alu: ratio,
            sync: ratio,
            mem: ratio,
            l1_lines_from_l2: 1.0,
            l2_lines_read: 1.0,
            l2_lines_stored: 1.0,
            ram_bw: 1.0,
        }
    }
}

impl Into<HwPressure> for InstDesc {
    fn into(self) -> HwPressure {
        let vec = vec![
            self.issue,
            self.alu,
            self.sync,
            self.mem,
            self.l1_lines_from_l2,
            self.l2_lines_read,
            self.l2_lines_stored,
            self.ram_bw,
        ];
        HwPressure::new(self.latency, vec)
    }
}

/// Represents CUDA GPUs.
#[derive(Clone, Serialize, Deserialize)]
pub struct Gpu {
    /// The name of the GPU.
    pub name: String,
    /// The compute capability major number.
    pub sm_major: u8,
    /// The compute capability minor number.
    pub sm_minor: u8,
    // TODO(perf): pointer size should be a parameter of the function and not of the GPU.
    /// The size of pointers.
    pub addr_size: u16,
    /// The amount of shared memory per SMX.
    pub shared_mem_per_smx: u32,
    /// The amount of shared memory available per block.
    pub shared_mem_per_block: u32,
    /// `true` when non-coherent loads are enabled on the GPU.
    pub allow_nc_load: bool,
    /// `ture` when L1 caching is enabled for global memory accesses.
    pub allow_l1_for_global_mem: bool,
    /// The size of a wrap.
    pub wrap_size: u32,
    /// The maximal number of resident thread per SMX.
    pub thread_per_smx: u32,
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
    /// The stride at wich replays occur in shared memory.
    pub shared_bank_stride: u32,
    /// Latency of a shared memory access.
    pub load_shared_latency: f64,
    /// The number of SMX in the GPU.
    pub num_smx: u32,
    /// Maximum number of block per SMX.
    pub max_block_per_smx: u32,
    /// The clock of an SMX, in GHz.
    pub smx_clock: f64,

    /// Amount of processing power available on a single thread.
    pub thread_rates: InstDesc,
    /// Amount of processing power available on a single SMX.
    pub smx_rates: InstDesc,
    /// Amount of processing power available on the whole GPU.
    pub gpu_rates: InstDesc,

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

impl Gpu {
    /// Returns the GPU model corresponding to `name.
    pub fn from_executor(executor: &cuda::Executor) -> Gpu {
        cuda::characterize::get_gpu_desc(executor)
    }

    /// Returns the PTX code for a Function.
    pub fn print_ptx(&self, fun: &Function) -> String {
        let mut printer = CudaPrinter::new();
        printer.function(fun, self)
    }

    /// Returns the description of a load instruction.
    fn load_desc(&self, mem_info: &MemInfo, flags: InstFlag) -> InstDesc {
        // TODO(search_space,model): support CA and NC flags.
        assert!(InstFlag::MEM_COHERENT.contains(flags));
        // Compute possible latencies.
        let gbl_latency = if flags.intersects(InstFlag::MEM_GLOBAL) {
            let miss = mem_info.l2_miss_ratio/mem_info.l2_coalescing;
            miss*self.load_ram_latency + (1.0-miss)*self.load_l2_latency
        } else { std::f64::INFINITY };
        let shared_latency = if flags.intersects(InstFlag::MEM_SHARED) {
            self.load_shared_latency as f64
        } else { std::f64::INFINITY };
        // Compute the smx bandwidth used.
        let l1_lines_from_l2 = if flags.intersects(InstFlag::MEM_SHARED) {
            0.0
        } else { mem_info.l1_coalescing };
        let l2_lines_read = if flags.intersects(InstFlag::MEM_SHARED) {
            0.0
        } else { mem_info.l2_coalescing };
        InstDesc {
            latency: f64::min(gbl_latency, shared_latency),
            issue: mem_info.replay_factor,
            mem: mem_info.replay_factor,
            l1_lines_from_l2, l2_lines_read,
            ram_bw: mem_info.l2_miss_ratio * f64::from(self.l2_cache_line),
            .. InstDesc::default()
        }
    }

    /// Returns the description of a store instruction.
    fn store_desc(&self, mem_info: &MemInfo, flags: InstFlag) -> InstDesc {
        // TODO(search_space,model): support CA flags.
        // TODO(model): understand how writes use the BW.
        assert!(InstFlag::MEM_COHERENT.contains(flags));
        let l2_lines_stored = if flags.intersects(InstFlag::MEM_SHARED) {
            0.0
        } else { mem_info.l2_coalescing };
        // L1 lines per L2 is not limiting.
        InstDesc {
            issue: mem_info.replay_factor,
            mem: mem_info.replay_factor,
            l2_lines_stored,
            ram_bw: 2.0 * mem_info.l2_miss_ratio * f64::from(self.l2_cache_line),
            .. InstDesc::default()
        }
    }

    /// Returns the overhead induced by all the iterations of a loop.
    fn dim_pressure(&self, kind: DimKind, size: u32) -> HwPressure {
        if kind == DimKind::LOOP {
            let mut pressure: HwPressure = self.loop_iter_overhead.into();
            pressure.repeat_sequential(f64::from(size));
            pressure.add_sequential(&self.loop_init_overhead.into());
            pressure
        } else if DimKind::THREAD.contains(kind) {
            // The repetition along the thread is taken into account by
            // `num_unmapped_thread` as the current thread is accounted as not mapped.
            self.syncthread_inst.into()
        } else { HwPressure::zero(self) }
    }

    /// Retruns the overhead for a single instance of the instruction.
    fn inst_pressure(&self, space: &SearchSpace,
                     dim_sizes: &HashMap<ir::dim::Id, u32>,
                     inst: &ir::Instruction,
                     ctx: &device::Context) -> HwPressure {
        use ir::Operator::*;
        let t = self.lower_type(inst.t(), space).unwrap_or_else(|| inst.t());
        match (inst.operator(), t) {
            (&BinOp(ir::BinOp::Add, ..), Type::F(32)) |
            (&BinOp(ir::BinOp::Sub, ..), Type::F(32)) => self.add_f32_inst.into(),
            (&BinOp(ir::BinOp::Add, ..), Type::F(64)) |
            (&BinOp(ir::BinOp::Sub, ..), Type::F(64)) => self.add_f64_inst.into(),
            (&BinOp(ir::BinOp::Add, ..), Type::I(32)) |
            (&BinOp(ir::BinOp::Sub, ..), Type::I(32)) => self.add_i32_inst.into(),
            (&BinOp(ir::BinOp::Add, ..), Type::I(64)) |
            (&BinOp(ir::BinOp::Sub, ..), Type::I(64)) => self.add_i64_inst.into(),
            (&Mul(..), Type::F(32)) => self.mul_f32_inst.into(),
            (&Mul(..), Type::F(64)) => self.mul_f64_inst.into(),
            (&Mul(..), Type::I(32)) |
            (&Mul(..), Type::PtrTo(_)) => self.mul_i32_inst.into(),
            (&Mul(ref op, _, _, _), Type::I(64)) => {
                let op_t = self.lower_type(op.t(), space).unwrap_or_else(|| op.t());
                if op_t == Type::I(64) {
                    self.mul_i64_inst.into()
                } else {
                    self.mul_wide_inst.into()
                }
            },
            (&Mad(..), Type::F(32)) => self.mad_f32_inst.into(),
            (&Mad(..), Type::F(64)) => self.mad_f64_inst.into(),
            (&Mad(..), Type::I(32)) |
            (&Mad(..), Type::PtrTo(_)) => self.mad_i32_inst.into(),
            (&Mad(ref op, _, _, _), Type::I(64)) => {
                let op_t = self.lower_type(op.t(), space).unwrap_or_else(|| op.t());
                if op_t == Type::I(64) {
                    self.mad_i64_inst.into()
                } else {
                    self.mad_wide_inst.into()
                }
            },
            (&BinOp(ir::BinOp::Div, ..), Type::F(32)) => self.div_f32_inst.into(),
            (&BinOp(ir::BinOp::Div, ..), Type::F(64)) => self.div_f64_inst.into(),
            (&BinOp(ir::BinOp::Div, ..), Type::I(32)) => self.div_i32_inst.into(),
            (&BinOp(ir::BinOp::Div, ..), Type::I(64)) => self.div_i64_inst.into(),
            (&Ld(..), _) | (&TmpLd(..), _) => {
                let flag = space.domain().get_inst_flag(inst.id());
                let mem_info = mem_model::analyse(space, self, inst, dim_sizes, ctx);
                self.load_desc(&mem_info, flag).into()
            },
            (&St(..), _) | (&TmpSt(..), _) => {
                let flag = space.domain().get_inst_flag(inst.id());
                let mem_info = mem_model::analyse(space, self, inst, dim_sizes, ctx);
                self.store_desc(&mem_info, flag).into()
            },
            // TODO(model): Instruction description for mov and cast.
            (&Mov(..), _) | (&Cast(..), _) =>  HwPressure::zero(self),
            _ => panic!(),
        }
    }

    /// Computes the number of blocks that can fit in an smx.
    pub fn blocks_per_smx(&self, space: &SearchSpace) -> u32 {
        let mut block_per_smx = self.max_block_per_smx;
        let num_thread = space.domain().get_num_threads().min;
        min_assign(&mut block_per_smx, self.thread_per_smx/num_thread);
        let shared_mem_used = space.domain().get_shared_mem_used().min;
        if shared_mem_used != 0 {
            min_assign(&mut block_per_smx, self.shared_mem_per_smx/shared_mem_used);
        }
        assert!(block_per_smx > 0,
                "not enough resources per block: shared mem used = {}, num threads = {}",
                shared_mem_used, num_thread);
        block_per_smx
    }

    /// Returns the pressure of an an instruction skipped by a predicate.
    fn skipped_pressure(&self) -> HwPressure {
        (InstDesc { issue: 1.0, .. InstDesc::default() }).into()
    }

    /// Computes the ratio `num_wraps*wrap_size/num_threads`. This ratio may be `>1`
    /// because the hardware creates additionnal threads to fill the wraps.
    fn waste_ratio(&self, lcm_threads_per_block: u64) -> f64 {
        let wrap_size = u64::from(self.wrap_size);
        let n_wraps = (lcm_threads_per_block + wrap_size - 1)/wrap_size;
        (n_wraps * wrap_size) as f64/lcm_threads_per_block as f64
    }
}

impl device::Device for Gpu {
    fn print(&self, fun: &Function, out: &mut Write) {
        let mut printer = CudaPrinter::new();
        printer.host_function(fun, self, out)
    }

    fn is_valid_type(&self, t: &Type) -> bool {
        match *t {
            Type::I(i) | Type::F(i) => i == 32 || i == 64,
            Type::Void | Type::PtrTo(_) => true,
        }
    }

    fn max_block_dims(&self) -> u32 { 3 }

    fn max_threads(&self) -> u32 { 1024 }

    fn max_unrolling(&self) -> u32 { 512 }

    fn can_vectorize(&self, dim: &ir::Dimension, op: &ir::Operator) -> bool {
        // TODO(search_space): compute vectorizable info for tmp Ld/St. On vectorization,
        // the layout must be constrained.
        match *op {
            Operator::St(_, ref operand, _, ref pattern) => {
                if let Some(type_len) = operand.t().len_byte() {
                    dim.size().as_int().into_iter().any(|x| x == 2 || x == 4) &&
                        pattern.stride(dim.id()) == ir::Stride::Int(type_len as i32)
                } else { false }
            },
            Operator::Ld(ref t, _, ref pattern) => {
                if let Some(type_len) = t.len_byte() {
                    dim.size().as_int().into_iter().any(|x| x == 2 || x == 4) &&
                        pattern.stride(dim.id()) == ir::Stride::Int(type_len as i32)
                } else { false }
            },
            Operator::BinOp(..) |
                Operator::Mul(..) |
                Operator::Mad(..) |
                Operator::Mov(..) |
                Operator::Cast(..) => false,
                Operator::TmpLd(..) |
                    Operator::TmpSt(..) => dim.size().as_int().into_iter().any(|x| x == 2 || x == 4),
        }
    }

    fn shared_mem(&self) -> u32 { self.shared_mem_per_block }

    fn supports_nc_access(&self) -> bool { self.allow_nc_load }

    fn supports_l1_access(&self) -> bool { self.allow_l1_for_global_mem }

    fn supports_l2_access(&self) -> bool { true }

    fn name(&self) -> &str { &self.name }

    fn lower_type(&self, t: ir::Type, space: &SearchSpace) -> Option<ir::Type> {
        match t {
            Type::PtrTo(mem_id) => {
                match space.domain().get_mem_space(mem_id) {
                    MemSpace::GLOBAL => Some(Type::I(self.addr_size)),
                    MemSpace::SHARED => Some(Type::I(32)),
                    _ => None,
                }
            },
            _ => Some(t),
        }
    }

    fn hw_pressure(&self, space: &SearchSpace,
                   dim_sizes: &HashMap<ir::dim::Id, u32>,
                   _nesting: &HashMap<ir::BBId, model::Nesting>,
                   bb: &ir::BasicBlock,
                   ctx: &device::Context) -> model::HwPressure {
        if let Some(inst) = bb.as_inst() {
            self.inst_pressure(space, dim_sizes, inst, ctx)
        } else if let Some(dim) = bb.as_dim() {
            let kind = space.domain().get_dim_kind(dim.id());
            self.dim_pressure(kind, dim_sizes[&dim.id()])
        } else { panic!() }
    }

    fn loop_iter_pressure(&self, kind: DimKind) -> (HwPressure, HwPressure) {
        if kind == DimKind::LOOP {
            let end_pressure = InstDesc {
                latency: self.loop_end_latency,
                .. InstDesc::default()
            };
            (self.loop_iter_overhead.into(), end_pressure.into())
        } else if DimKind::THREAD.contains(kind) {
            (self.syncthread_inst.into(), HwPressure::zero(self))
        } else { (HwPressure::zero(self), HwPressure::zero(self)) }
    }

    fn thread_rates(&self) -> HwPressure { self.thread_rates.into() }

    fn block_rates(&self) -> HwPressure { self.smx_rates.into() }

    fn total_rates(&self) -> HwPressure { self.gpu_rates.into() }

    fn bottlenecks(&self) -> &[&'static str] {
        &["issue",
          "alu",
          "syncthread",
          "mem_units",
          "l1_lines_from_l2",
          "l2_lines_read",
          "l2_lines_stored",
          "bandwidth"]
    }

    fn block_parallelism(&self, space: &SearchSpace) -> u32 {
        self.blocks_per_smx(space) * self.num_smx
    }

    fn additive_indvar_pressure(&self, t: &ir::Type) -> HwPressure {
        match *t {
            ir::Type::I(32) => self.add_i32_inst.into(),
            ir::Type::I(64) => self.add_i64_inst.into(),
            _ => panic!(),
        }
    }

    fn multiplicative_indvar_pressure(&self, t: &ir::Type) -> HwPressure {
        match *t {
            ir::Type::I(32) => self.mad_i32_inst.into(),
            ir::Type::I(64) => self.mad_i64_inst.into(),
            _ => panic!(),
        }
    }

    fn add_block_overhead(&self, predicated_dims_size: u64,
                          max_threads_per_blocks: u64,
                          pressure: &mut HwPressure) {
        assert!(predicated_dims_size > 0);
        assert!(max_threads_per_blocks > 0);
        let active_ratio = self.waste_ratio(max_threads_per_blocks/predicated_dims_size);
        pressure.multiply(&InstDesc::wasted_ratio(active_ratio).into());
        if predicated_dims_size > 1 {
            let full_ratio = self.waste_ratio(max_threads_per_blocks);
            let num_skipped = full_ratio * predicated_dims_size as f64 - active_ratio;
            assert!(num_skipped >= 0.);
            pressure.repeat_and_add_bottlenecks(num_skipped, &self.skipped_pressure());
        }
    }
}

/// Asigns min(lhs, rhs) to lhs.
fn min_assign<T: std::cmp::Ord>(lhs: &mut T, rhs: T) { if rhs < *lhs { *lhs = rhs; } }

// TODO(model): On the Quadro K4000:
// * The Mul wide latency is unknown.
// * The latency is not specialized per operand.
