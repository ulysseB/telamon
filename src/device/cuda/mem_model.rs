//! Memory accesses analysis.
use device::cuda;
use ir::{self, BasicBlock};
use itertools::Itertools;
use search_space::{DimKind, Domain, InstFlag, Order, SearchSpace};
use std;
use num::integer;
use utils::*;

/// Result of the memory analysis for one instruction. Vector instructions are considered
/// as one instruction.
#[derive(Debug)]
pub struct MemInfo {
    /// The proportion of instruction that produce a L2 miss.
    pub l2_miss_ratio: f64,
    /// The number of L1 cache line loaded for each instruction.
    pub l1_coalescing: f64,
    /// The number of L2 cache line loaded for each instruction.
    pub l2_coalescing: f64,
    /// The number of times the instruction must be issued to be completed.
    pub replay_factor: f64,
}

/// Runs the memory analysis.
pub fn analyse(space: &SearchSpace, gpu: &cuda::Gpu, inst: &ir::Instruction,
               sizes: &HashMap<ir::dim::Id, u32>) -> MemInfo {
    let flag = space.domain().get_inst_flag(inst.id());
    match *inst.operator() {
        ir::Operator::Ld(_, _, ref pattern) |
        ir::Operator::St(_, _, _, ref pattern) => {
            let is_shared = flag.is(InstFlag::MEM_SHARED);
            if flag.intersects(InstFlag::MEM_NC) {
                unknown_info(is_shared, gpu)
            } else {
                info(space, inst, pattern, is_shared, gpu, sizes)
            }
        },
        ir::Operator::TmpLd(..) | ir::Operator::TmpSt(..) => {
            let is_shared = flag.is(InstFlag::MEM_SHARED);
            unknown_info(is_shared, gpu)
        },
        _ => panic!()
    }
}

const NO_ACCESS_INFO: MemInfo = MemInfo {
    l2_miss_ratio: std::f64::INFINITY,
    l1_coalescing: std::f64::INFINITY,
    l2_coalescing: std::f64::INFINITY,
    replay_factor: std::f64::INFINITY,
};

/// Computes the memory access info for a given memory access.
fn info(space: &SearchSpace,
        inst: &ir::Instruction,
        pattern: &ir::AccessPattern,
        is_shared_access: Trivalent,
        gpu: &cuda::Gpu,
        sizes: &HashMap<ir::dim::Id, u32>) -> MemInfo {
    // TODO(model): The model can decrease if the maximal number decreases: the replay
    // assume a full wrap if possible. This is correct as if the wrap is not full, the
    // waste ratio will repeat the replay factor to achieve the same number. However,
    // it makes debugging the performance model harder.
    let info = match *pattern {
        // Without pattern accesses, we cannot analyse correctly the memory accesses
        ir::AccessPattern::Unknown { .. } => {
            unknown_info(is_shared_access, gpu)
        },
        // If the pattern is a tensor, we can analyse memory accesses
        ir::AccessPattern::Tensor { stride, ref dims, .. } => {
            let mut info = NO_ACCESS_INFO;
            let thread_dims = tensor_thread_dims(
                space, inst, stride, dims, sizes, gpu);
            if is_shared_access.maybe_true() {
                info.replay_factor = shared_replay_factor(
                    stride, &thread_dims, dims, sizes, space, gpu);
                info.l2_miss_ratio = 0.0;
            }
            if is_shared_access.maybe_false() {
                let (l1_coalescing, l2_coalescing, replay) =
                    global_coalescing(&thread_dims, space, gpu);
                info.l1_coalescing = l1_coalescing;
                info.l2_coalescing = l2_coalescing;
                info.replay_factor = f64::min(info.replay_factor, replay);
                // TODO(model): compute the miss ratio
                info.l2_miss_ratio = 0.0;
            }
            info
        },
    };
    trace!("mem_info for {:?}: {:?}", inst.id(), info);
    info
}

/// Computes the `MemInfo` when the access pattern is unknown.
fn unknown_info(is_shared_access: Trivalent, gpu: &cuda::Gpu) -> MemInfo {
    let mut info = NO_ACCESS_INFO;
    if is_shared_access.maybe_true() {
        info.l2_miss_ratio = 0.0;
        info.replay_factor = 1.0;
    }
    if is_shared_access.maybe_false() {
        info.l2_miss_ratio = 0.0;
        info.l1_coalescing = 1.0/(gpu.wrap_size as f64);
        info.l2_coalescing = 1.0/(gpu.wrap_size as f64);
        info.replay_factor = 1.0;
    }
    info
}

/// Computes the replay factor for a shared memory access.
fn shared_replay_factor(stride: i32,
                        thread_dims: &[ThreadDimInfo], 
                        tensor_dims: &[ir::dim::Id],
                        dim_sizes: &HashMap<ir::dim::Id, u32>,
                        space: &SearchSpace, gpu: &cuda::Gpu) -> f64 {
    let mut total_size = 1;
    let mut replay_factor = 1.0;
    for dim_info in sort_thread_dims(thread_dims, space, |d| d.shared_replay_freq) {
        let mut size = dim_info.size;
        total_size *= size;
        if total_size > gpu.wrap_size as u64 {
            let div = total_size / gpu.wrap_size as u64;
            size /= div;
            total_size = gpu.wrap_size as u64;
        }
        let replays = ((size-1) as f64 * dim_info.shared_replay_freq).floor();
        replay_factor *= 1.0 + replays;
        if total_size == gpu.wrap_size as u64 { break }
    }
    // Handle the case where a single thread must access two banks.
    let max_vector_factor = tensor_dims.iter()
        .filter(|&&d| space.domain().get_dim_kind(d).intersects(DimKind::VECTOR))
        .map(|d| dim_sizes[d]).max().unwrap_or(1);
    let vector_replay = div_ceil(max_vector_factor*stride as u32, gpu.shared_bank_stride);
    replay_factor = f64::max(replay_factor, vector_replay as f64);
    trace!("shared_replay: {}", replay_factor);
    replay_factor
}

/// Computes the L1, L2 coalescing and replay factor for a global memory access.
fn global_coalescing(thread_dims: &[ThreadDimInfo], space: &SearchSpace, gpu: &cuda::Gpu)
        -> (f64, f64, f64) {
    let mut total_size = 1;
    let mut l1_line_accessed = 1;
    let mut l2_line_accessed = 1;
    for dim_info in sort_thread_dims(thread_dims, space, |d| d.stride as f64) {
        let mut size = dim_info.size;
        total_size *= size;
        if total_size > gpu.wrap_size as u64 {
            let div = total_size / gpu.wrap_size as u64;
            size /= div;
            total_size = gpu.wrap_size as u64;
        }
        let l1_line_len = gpu.l1_cache_line as u64;
        let l2_line_len = gpu.l2_cache_line as u64;
        let l1_stride = std::cmp::min(dim_info.stride, l1_line_len);
        let l2_stride = std::cmp::min(dim_info.stride, l2_line_len);
        l1_line_accessed *= 1 + ((size-1)*l1_stride)/l1_line_len;
        l2_line_accessed *= 1 + ((size-1)*l2_stride)/l2_line_len;
        if total_size == gpu.wrap_size as u64 { break }
    }
    trace!("global_replay: {} (size: {})", l1_line_accessed, total_size);
    let l1_coalescing = l1_line_accessed as f64 / total_size as f64; 
    let l2_coalescing = l2_line_accessed as f64 / total_size as f64;
    (l1_coalescing, l2_coalescing, l1_line_accessed as f64)
}

#[derive(Debug)]
struct ThreadDimInfo {
    id: ir::dim::Id,
    is_active_thread: bool,
    size: u64,
    stride: u64,
    shared_replay_freq: f64,
}

/// Returns the size and stride of thread dimensions for a tensor access pattern.
fn tensor_thread_dims(space: &SearchSpace,
                      inst: &ir::Instruction,
                      base_stride: i32,
                      tensor_dims: &[ir::dim::Id],
                      sizes: &HashMap<ir::dim::Id, u32>,
                      gpu: &cuda::Gpu) -> Vec<ThreadDimInfo> {
    let base_stride = base_stride as u64;
    let non_zero_strides = tensor_dims.iter().rev().scan(base_stride, |stride, dim| {
        let current_stride = *stride;
        let size = sizes[dim];
        *stride *= size as u64;
        Some((dim, (size, current_stride)))
    }).collect::<HashMap<_, _>>();
    inst.iteration_dims().iter().flat_map(|&dim| {
        match space.domain().get_dim_kind(dim).is(DimKind::THREAD) {
            Trivalent::False => None,
            Trivalent::Maybe => Some((dim, false)),
            Trivalent::True => Some((dim, true)),
        }
    }).map(|(id, is_active_thread)| {
        let (size, stride) = non_zero_strides.get(&id).cloned()
            .unwrap_or_else(|| (sizes[&id], 0));
        // TODO(model): handle strides that are not a multiple of the bank_Stride.
        let shared_replay_freq = if stride == 0 { 0.0 } else {
            let byte_reply_distance = (gpu.shared_bank_stride * gpu.wrap_size) as u64;
            let hop_replay_distance = integer::lcm(stride, byte_reply_distance);
            stride as f64 / hop_replay_distance as f64
        };
        ThreadDimInfo {
            id: id,
            is_active_thread: is_active_thread,
            size: size as u64,
            stride: stride,
            shared_replay_freq: shared_replay_freq,
        }
    }).collect_vec()
}

fn sort_thread_dims<'a, F>(dims: &'a [ThreadDimInfo], space: &SearchSpace, cost: F)
        -> Vec<&'a ThreadDimInfo> where F: Fn(&ThreadDimInfo) -> f64 {
    dims.iter().sorted_by(|lhs, rhs| {
        if lhs.id == rhs.id { return std::cmp::Ordering::Equal; }
        let nest_order = space.domain().get_order(lhs.id.into(), rhs.id.into());
        let maybe_out = nest_order.intersects(Order::OUTER);
        let maybe_in = nest_order.intersects(Order::INNER);
        let (lhs_cost, rhs_cost) = (cost(lhs), cost(rhs));
        match (maybe_in, maybe_out, lhs.is_active_thread, rhs.is_active_thread) {
            (true, false, true, _) => std::cmp::Ordering::Less,
            (false, true, _, true) => std::cmp::Ordering::Greater,
            _ if lhs_cost < rhs_cost => std::cmp::Ordering::Less,
            _ if lhs_cost > rhs_cost => std::cmp::Ordering::Greater,
            _ if lhs.id < rhs.id => std::cmp::Ordering::Less,
            _ if lhs.id > rhs.id => std::cmp::Ordering::Greater,
            _ => { panic!() }
        }
    })
}

/*
/// Computes the miss ratio for L2 cache.
fn miss_ratios(inst: &ir::Instruction,
               pattern: &ir::AccessPattern,
               space: &SearchSpace,
               gpu: &cuda::Gpu,
               sizes: &HashMap<ir::dim::Id, u32>) -> f64 {
    // Compute MSHR, without taking other accesses into account.
    // (1) Find accesses to the sane memory block.
    let other_accesses = space.ir_instance().insts().filter(|other_inst| {
        let other_mem = other_inst.operator().mem_access_pattern().map(|x| x.mem_block());
        *other_inst != inst && other_mem == Some(pattern.mem_block())
    }).collect_vec();
    // (2) Find the MSHR cache hit ratio on each active dimension.
    let mshr_miss = space.ir_instance().dims().filter(|&dim| {
        let kind = space.domain().get_dim_kind(dim.id());
        space.domain().get_order(dim.bb_id(), inst.bb_id()) == Order::ACTIVE_OUT
            && !(DimKind::BLOCK | DimKind::VECTOR).contains(kind)
    }).map(|dim| {
        // fixme: use other accesses
        let has_other_access = false; /*other_accesses.iter().any(|other| {
            fun.order(other.bb_id(), dim.bb_id()).intersects(Order::INNER)
        });*/
        if has_other_access {
            // TODO(model): better handle other accesses to the same memory block
            0.0
        } else {
            let size = sizes[&dim.id()];
            let stride = eval_stride(pattern, dim.id(), sizes).unwrap_or(0);
            let reuse_distance = reuse_distance(inst, dim, pattern, space, sizes, gpu);
            let mshr_miss = if reuse_distance > gpu.mshr_per_smx {
                1.0
            } else if size == 1 {
                0.0
            } else {
                let num_lines = 1 + (stride*(size as i32-1))/gpu.l1_cache_line as i32;
                f64::min(num_lines as f64/size as f64, 1.0)
            };
            trace!("dim: {:?}, kind: {:?}, reuse_distance: {}, stride: {}, mshr_miss: {}",
                   dim, space.domain().get_dim_kind(dim.id()), reuse_distance, stride, mshr_miss);
            mshr_miss
        }
    }).product();
    // TODO(model): take other accesses into account.
    // TODO(model): compute L2 miss
    // TODO(model): take flags into account.
    // TODO(model): handle block dimensions.
    trace!("Inst {:?} = mshr_miss: {}", inst.id(), mshr_miss);
    // fixme: does not account for reuse in the first iterations
    0.0
}

/// Computes the reuse distance between two iterations of `dim` for the given pattern.
fn reuse_distance(inst: &ir::Instruction,
                  dim: &ir::Dimension,
                  pattern: &ir::AccessPattern,
                  space: &SearchSpace,
                  sizes: &HashMap<ir::dim::Id, u32>,
                  gpu: &cuda::Gpu) -> u32 {
    space.ir_instance().dims().filter(|&other_dim| {
        other_dim.id() != dim.id() &&
        space.domain().get_order(other_dim.bb_id(), inst.bb_id()) == Order::ACTIVE_OUT &&
        dynamic_nesting(dim, other_dim, space) == Some(Ordering::Greater)
    }).map(|other_dim| {
        let stride = eval_stride(pattern, other_dim.id(), sizes).unwrap_or(0) as u32;
        let size = sizes[&other_dim.id()] as u32;
        1 + std::cmp::min(size - 1, stride*(size-1)/gpu.l1_cache_line)
    }).product::<u32>() - 1
}

/// Evaluate the stride of an access pattern of a given dimension.
fn eval_stride(pattern: &ir::AccessPattern,
               dim: ir::dim::Id,
               sizes: &HashMap<ir::dim::Id, u32>) -> ir::Stride {
    match *pattern {
        ir::AccessPattern::Unknown { .. } => ir::Stride::Unknown,
        ir::AccessPattern::Tensor { ref stride, ref dims, .. } => {
            let mut it = dims.iter().skip_while(|other| **other != dim);
            if it.next().is_some() {
                ir::Stride::Int(it.map(|d| sizes[d] as i32).product::<i32>() * stride)
            } else {
                ir::Stride::Int(0)
            }
        },
    }
}

/// Compare the nesting of two dimension in the dynamic schedule. Yeilds a valid partial order.
fn dynamic_nesting(lhs: &ir::Dimension, rhs: &ir::Dimension, space: &SearchSpace)
        -> Option<Ordering> {
    if lhs.id() == rhs.id() { return Some(Ordering::Equal); }
    let order = space.domain().get_order(lhs.bb_id(), rhs.bb_id());
    let lhs_kind = space.domain().get_dim_kind(lhs.id());
    let rhs_kind = space.domain().get_dim_kind(rhs.id());
    let lhs_is_thread = lhs_kind.is(DimKind::THREAD);
    let rhs_is_thread = rhs_kind.is(DimKind::THREAD);
    let lhs_is_vector = lhs_kind.is(DimKind::VECTOR);
    let rhs_is_vector = rhs_kind.is(DimKind::VECTOR);
    match (lhs_is_thread, rhs_is_thread, lhs_is_vector, rhs_is_vector) {
        // Handle ordering with vectors
        (_, _, Trivalent::True, _) => Some(Ordering::Less),
        (_, _, _, Trivalent::True) => Some(Ordering::Greater),
        // Thread/Non-Thread ordering
        (Trivalent::True, Trivalent::False, _, Trivalent::Maybe) => None,
        (Trivalent::True, Trivalent::False, _, Trivalent::False) => Some(Ordering::Less),
        // Non-Thread/Thread ordering
        (Trivalent::False, Trivalent::True, Trivalent::Maybe, _) => None,
        (Trivalent::False, Trivalent::True, Trivalent::False, _) => Some(Ordering::Greater),
        // Non-Thread/Non-Thread and Thread/Thread ordering
        (Trivalent::False, Trivalent::False, _, _) |
        (Trivalent::True, Trivalent::True, _, _) => {
            // Order per nesting order.
            if order.is(Order::INNER).is_true() { Some(Ordering::Less) }
            else if order.is(Order::OUTER).is_true() { Some(Ordering::Greater) }
            else { None }
        },
        // In some cases, we can't say anything.
        (_, Trivalent::Maybe, _, _) |
        (Trivalent::Maybe, _, _, _) => None
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use device::cuda::Gpu;
    use helper;
    use ir;
    use env_logger;

    /// Generates function with a load in two thread dimensions, with non-coalesced
    /// accessed on the first one.
    fn gen_function<'a>(signature: &'a ir::Signature, gpu: &'a Gpu, d0_d1_order: Order)
            -> (SearchSpace<'a>, ir::InstId, HashMap<ir::dim::Id, u32>) {
        let mut builder = helper::Builder::new(&signature, gpu);
        let t = ir::Type::F(32);
        let size = builder.cst_size(gpu.wrap_size);
        let addr_base = builder.cast(&0i64, ir::Type::PtrTo(ir::mem::Id::External(0)));
        let d0 = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let d1 = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let addr = builder.mad(&d0, &(gpu.l1_cache_line as i32), &addr_base);
        let pattern = ir::AccessPattern::Tensor {
            mem_id: ir::mem::Id::External(0),
            stride: gpu.l1_cache_line as i32,
            dims: vec![d0]
        };
        let ld = builder.ld_ex(t, &addr, pattern, InstFlag::MEM_CG);
        builder.order(&d0, &d1, d0_d1_order);

        let mut size_map = HashMap::default();
        size_map.insert(d0, gpu.wrap_size as u32);
        size_map.insert(d1, gpu.wrap_size as u32);
        (builder.get(), ld, size_map)
    }

    /// Generates a dummy signature.
    fn gen_signature() -> ir::Signature {
        ir::Signature { name: String::new(), params: vec![], mem_blocks: 1 }
    }


    /// Tests `MemInfo` for global loads without coalescing.
    #[test]
    fn global_no_coalescing() {
        let _ = env_logger::init();
        let gpu = unwrap!(Gpu::from_name("dummy_cuda_gpu"));
        let base = gen_signature();
        let (space, inst, size_map) = gen_function(&base, &gpu, Order::OUTER);
        let inst = space.ir_instance().inst(inst);
        let inst_info = analyse(&space, &gpu, &inst, &size_map);
        assert_eq!(inst_info.l1_coalescing, 1.0/gpu.wrap_size as f64);
        assert_eq!(inst_info.l2_coalescing, 1.0/gpu.wrap_size as f64);
        assert_eq!(inst_info.replay_factor, 1.0);
    }

    /// Tests `MemInfo` for global loads with full coalescing.
    #[test]
    fn global_full_coalescing() {
        let _ = env_logger::init();
        let gpu = unwrap!(Gpu::from_name("dummy_cuda_gpu"));
        let base = gen_signature();
        let (space, inst, size_map) = gen_function(&base, &gpu, Order::INNER);
        let inst = space.ir_instance().inst(inst);
        let inst_info = analyse(&space, &gpu, &inst, &size_map);
        assert_eq!(inst_info.l1_coalescing, 1.0);
        assert_eq!(inst_info.l2_coalescing, 1.0);
        assert_eq!(inst_info.replay_factor, gpu.wrap_size as f64);
    }
}
