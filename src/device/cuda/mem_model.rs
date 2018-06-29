//! Memory accesses analysis.
use binary_heap_plus::BinaryHeap;
use device::{Context, cuda};
use ir;
use itertools::Itertools;
use model::size;
use num::Integer;
use search_space::{DimKind, Domain, InstFlag, ThreadMapping, SearchSpace};
use std;
use utils::*;

// TODO(model): the pressure changes depending on the list of outer dimensions. Try to
// take this into account be computing the pressure incrementatly when applying levels.

/// Result of the memory analysis for one instruction. Vector instructions are considered
/// as a single instance and predicated dimensions are not considered to compute the
/// average pressure.
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
pub fn analyse(space: &SearchSpace,
               gpu: &cuda::Gpu,
               inst: &ir::Instruction,
               sizes: &HashMap<ir::dim::Id, size::Range>,
               ctx: &Context) -> MemInfo {
    let flag = space.domain().get_inst_flag(inst.id());
    let info = match *inst.operator() {
        ir::Operator::Ld(_, _, ref pattern) |
        ir::Operator::St(_, _, _, ref pattern) => {
            let is_shared = flag.is(InstFlag::MEM_SHARED);
            match pattern {
                _ if flag.intersects(InstFlag::MEM_NC) => unknown_info(is_shared, gpu),
                ir::AccessPattern::Unknown { .. } => unknown_info(is_shared, gpu),
                ir::AccessPattern::Tensor { ref dims, .. } =>
                    info(space, inst, dims, is_shared, gpu, sizes, ctx),
            }
        },
        ir::Operator::TmpLd(..) | ir::Operator::TmpSt(..) => {
            let is_shared = flag.is(InstFlag::MEM_SHARED);
            unknown_info(is_shared, gpu)
        },
        _ => panic!()
    };
    trace!("mem_info for {:?}: {:?}", inst.id(), info);
    info
}

const NO_ACCESS_INFO: MemInfo = MemInfo {
    l2_miss_ratio: std::f64::INFINITY,
    l1_coalescing: std::f64::INFINITY,
    l2_coalescing: std::f64::INFINITY,
    replay_factor: std::f64::INFINITY,
};

/// Computes the `MemInfo` when the access pattern is unknown.
fn unknown_info(is_shared_access: Trivalent, gpu: &cuda::Gpu) -> MemInfo {
    let mut info = NO_ACCESS_INFO;
    if is_shared_access.maybe_true() {
        info.l2_miss_ratio = 0.0;
        info.replay_factor = 1.0;
    }
    if is_shared_access.maybe_false() {
        info.l2_miss_ratio = 0.0;
        info.l1_coalescing = 1.0/f64::from(gpu.wrap_size);
        info.l2_coalescing = 1.0/f64::from(gpu.wrap_size);
        info.replay_factor = 1.0;
    }
    info
}

/// Computes the memory access info for a given memory access.
// TODO(model): The model can decrease if the maximal number decreases: the replay
// assume a full wrap if possible. This is correct as if the wrap is not full the
// waste ratio will repeat the replay factor to achieve the same number. However,
// it makes debugging the performance model harder.
fn info(space: &SearchSpace,
        inst: &ir::Instruction,
        dims: &HashMap<ir::dim::Id, ir::Size>,
        is_shared_access: Trivalent,
        gpu: &cuda::Gpu,
        sizes: &HashMap<ir::dim::Id, size::Range>,
        ctx: &Context) -> MemInfo {
    let mut info = NO_ACCESS_INFO;
    let thread_dims = tensor_thread_dims(space, inst, dims, sizes, ctx);
    trace!("thread dims: {:?}", thread_dims);
    if is_shared_access.maybe_true() {
        let replay = shared_replay_factor(&thread_dims, dims, sizes, space, gpu);
        info.replay_factor = f64::min(replay, info.replay_factor);
        info.l2_miss_ratio = 0.0;
    }
    if is_shared_access.maybe_false() {
        let (l1_coalescing, l2_coalescing, replay) =
            global_coalescing(&thread_dims, space, gpu);
        info.l1_coalescing = f64::min(l1_coalescing, info.l1_coalescing);
        info.l2_coalescing = f64::min(l2_coalescing, info.l2_coalescing);
        info.replay_factor = f64::min(replay, info.replay_factor);
        // TODO(model): compute the miss ratio
        info.l2_miss_ratio = 0.0;
    }
    info
}

#[derive(Debug, Clone, Copy)]
struct ThreadDimInfo {
    id: ir::dim::Id,
    is_active_thread: bool,
    is_partial_dim: bool,
    size: size::Range,
    stride: size::Range,
    stride_factors: size::FactorRange,
}

impl ThreadDimInfo {
    /// Returns part of the dimension size handled by `Self`.
    fn partial_size(&self) -> u64 {
        if self.is_partial_dim { self.size.max - self.size.min } else { self.size.min }
    }
}

/// Returns the size and stride of thread dimensions for a tensor access pattern and
/// sort them in an optimal or better-than-optimal order. For two dimensions `d0`, `d1`
/// such that `d0.stride` < `d1.stride` and `such that, d0` can be nested inside `d1` the
/// order guarantees that `d0 < d1`.
///
/// Dimensions with a non-constrained size are split between a dimension for the minimal
/// size and a partial dimension for the rest.
fn tensor_thread_dims(space: &SearchSpace,
                      inst: &ir::Instruction,
                      tensor_dims: &HashMap<ir::dim::Id, ir::Size>,
                      sizes: &HashMap<ir::dim::Id, size::Range>,
                      ctx: &Context) -> Vec<ThreadDimInfo> {
    let external_dims = external_thread_dims(inst, space);
    let dims = inst.iteration_dims().iter().flat_map(|&dim| {
        match space.domain().get_dim_kind(dim).is(DimKind::THREAD) {
            Trivalent::False => None,
            Trivalent::Maybe => Some((dim, false)),
            Trivalent::True => Some((dim, true)),
        }
    }).chain(external_dims);
    let mut out = Vec::new();
    for (id, is_active_thread) in dims {
        let size = sizes[&id];
        let stride_size = tensor_dims.get(&id);
        let stride = stride_size.map(|s| size::bounds(s, space, ctx))
            .unwrap_or(size::Range::ZERO);
        let stride_factors = stride_size.map(|s| size::factors(s, space, ctx))
            .unwrap_or(size::FactorRange::ZERO);
        let info = ThreadDimInfo {
            is_partial_dim: false,
            stride, id, is_active_thread, stride_factors, size,
        };
        if !size.is_constrained() {
            out.push(ThreadDimInfo { is_partial_dim: true, .. info });
        }
        out.push(info);
    }
    out
}

/// Returns the thread dimensions that are mapped outside an instruction but not active
/// under this instruction. The returned boolean indicates if the thread dimension cannot
/// be mapped to an active dimension and if the dimension is predicated.
fn external_thread_dims<'a>(inst: &'a ir::Instruction, space: &'a SearchSpace)
    -> impl Iterator<Item=(ir::dim::Id, bool)> + 'a
{
    space.ir_instance().thread_dims().flat_map(move |dim| {
        let is_mapped = inst.iteration_dims().iter().map(|&other| {
            if dim.id() == other { return Trivalent::True; }
            // TODO(cc_perf): directly iterate on static dims
            if !space.ir_instance().dim(other).size().is_constant() {
                return Trivalent::False;
            }
            let mapping = space.domain().get_thread_mapping(dim.id(), other);
            mapping.is(ThreadMapping::MAPPED)
        }).fold(Trivalent::False, |l, r| l | r);
        match is_mapped {
            Trivalent::True => None,
            Trivalent::Maybe => Some((dim.id(), false)),
            Trivalent::False => Some((dim.id(), true)),
        }
    })
}

/// Sort thread dimensions in an optimal or better-than-optimal order. The order may not
/// respect dependencies since we don't know the exact order and it would be too costly to
/// explore all of them (exponential). Instead we compute the minimal number of inner
/// thread dimension for each dimension and ensure this amount is respected.
///
/// The `use_gcd` flag sort strides with smaller gdc for pontential values fisrt.
fn sort_thread_dims(dims: &[ThreadDimInfo],
                    use_gcd: bool,
                    space: &SearchSpace,
                    gpu: &cuda::Gpu) -> Vec<ThreadDimInfo> {
    let sure_thread_dims = dims.iter().filter(|d| d.is_active_thread)
        .map(|d| d.id).collect_vec();
    let cmp = |x: &ThreadDimInfo, y: &ThreadDimInfo| cmp_thread_dims(x, y, use_gcd);
    let mut heap = BinaryHeap::with_capacity_by(dims.len(), cmp);
    let mut dim_groups: MultiHashMap<_, _> = dims.into_iter().map(|d| {
        let num_inner = sure_thread_dims.iter().filter(|&&other| {
            if other == d.id { return false; }
            let mapping = space.domain().get_thread_mapping(d.id, other);
            mapping.is(ThreadMapping::MAPPED_OUT).is_true()
        }).count();
        (num_inner, d)
    }).collect();
    heap.extend(dim_groups.remove(&0));
    let mut out = Vec::new();
    let mut total_size = 1;
    while let Some(d) = heap.pop() {
        if d.is_partial_dim {
            total_size = (total_size/d.size.min) * d.size.max
        } else {
            total_size *= d.size.min;
        }
        out.push(d);
        heap.extend(dim_groups.remove(&out.len()));
        if total_size > gpu.wrap_size as u64 { break; }
    }
    out
}

/// Indicates which loop nest order should be considered to minimize replays.
fn cmp_thread_dims(lhs: &ThreadDimInfo, rhs: &ThreadDimInfo, use_gcd: bool)
    -> std::cmp::Ordering
{
    let lhs_val = if use_gcd { lhs.stride_factors.gcd } else { lhs.stride.min };
    let rhs_val = if use_gcd { rhs.stride_factors.gcd } else { rhs.stride.min };
    lhs_val.cmp(&rhs_val).reverse().then(lhs.is_partial_dim.cmp(&rhs.is_partial_dim))
}

/// Returns the offset of memory accesses for each thread in a wrap. The offset is
/// relative to the access of the first thread.
fn wrap_access_offsets(thread_dims: &[ThreadDimInfo],
                       use_gcd: bool,
                       gpu: &cuda::Gpu) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(gpu.wrap_size as usize);
    offsets.push(0);
    let mut indexes = vec![0; thread_dims.len()];
    while offsets.len() < gpu.wrap_size as usize {
        let mut incr = true;
        let offset = thread_dims.iter().enumerate().map(|(i, dim)| {
            if incr { incr = increment_index(i, thread_dims, &mut indexes); }
            if dim.is_partial_dim && indexes[i] > 0 {
                // TODO(cc_perf): save the index of real dimensions instead of recomputing. 
                let real_pos = thread_dims[0..i].iter().position(|d| d.id == dim.id);
                let real_pos = unwrap!(real_pos, "partial dim ordered before its base");
                assert!(!thread_dims[real_pos].is_partial_dim);
                indexes[real_pos] = thread_dims[real_pos].size.min-1;
            }
            let stride = if use_gcd { dim.stride_factors.gcd } else { dim.stride.min };
            indexes[i] * stride
        }).product();
        if incr { break; } // We reached the end of all loops.
        offsets.push(offset);
    }
    offsets
}

/// Increments the index at the given position modulo the dimension size. Indicates if
/// the next index should also be incremented.
fn increment_index(pos: usize, dims: &[ThreadDimInfo], indexes: &mut [u64]) -> bool {
    indexes[pos] += 1;
    if indexes[pos] < dims[pos].partial_size() { false } else {
        indexes[pos] = 0;
        true
    }
}

/// Compute the replay foactor caused by shared memory accesses.
fn shared_replay_factor(
    thread_dims: &[ThreadDimInfo],
    tensor_dims: &HashMap<ir::dim::Id, ir::Size>,
    dim_sizes: &HashMap<ir::dim::Id, size::Range>,
    space: &SearchSpace, gpu: &cuda::Gpu) -> f64
{
    let thread_dims = sort_thread_dims(thread_dims, true, space, gpu);
    // Handle the case where a single thread must access two banks.
    let mut replay = tensor_dims.iter()
        .flat_map(|(&d, stride)| stride.as_fixed().map(|s| (d, s)))
        .filter(|&(d, _)| space.domain().get_dim_kind(d).intersects(DimKind::VECTOR))
        .map(|(d, stride)| dim_sizes[&d].min as u32 * stride)
        .map(|size| div_ceil(size, gpu.shared_bank_stride)).min().unwrap_or(1);
    // Handle replays caused by offsets.
    let mut offsets = vec![wrap_access_offsets(&thread_dims, true, gpu)];
    // Handle the case where the last dimension may not be active. In that case we also
    // try without the dimension as considering it as a thread may increase the pressure.
    // Only the last dimension needs sepcial handling as other dimensions are fully
    // contained into a wrap.
    if thread_dims.last().map(|d| !d.is_active_thread).unwrap_or(false) {
        offsets.push(wrap_access_offsets(&thread_dims[0..thread_dims.len()-1], true, gpu));
    }
    for offsets in &offsets {
        replay = std::cmp::min(replay, offsets_shared_replay_factor(offsets, gpu));
    }
    trace!("shared_replay: {}", replay);
    replay as f64
}

/// Computes the replay factor for a list of shared memory access.
fn offsets_shared_replay_factor(offsets: &[u64], gpu: &cuda::Gpu) -> u32 {
    // We only need to account for hits on the first bank. Other banks will have a smaller
    // replay factor.
    let mut hits: HashSet<_> = std::iter::once(0).collect();
    for &offset in offsets  {
        let num_bank_stride = offset / gpu.shared_bank_stride as u64;
        let (hit_id, rem) = num_bank_stride.div_rem(&(gpu.wrap_size as u64));
        if rem == 0 { hits.insert(hit_id); }
    }
    hits.len() as u32
}

/// Computes the L1, L2 coalescing and replay factor for a global memory access.
fn global_coalescing(thread_dims: &[ThreadDimInfo],
                     space: &SearchSpace,
                     gpu: &cuda::Gpu) -> (f64, f64, f64) {
    let thread_dims = sort_thread_dims(thread_dims, false, space, gpu);
    let offsets = wrap_access_offsets(&thread_dims, true, gpu);
    let (mut l1_coalescing, mut l2_coalescing, mut replay) =
        offsets_global_coalescing(&offsets, gpu);
    if thread_dims.last().map(|d| !d.is_active_thread).unwrap_or(false) {
        let offsets = wrap_access_offsets(&thread_dims[0..thread_dims.len()-1], true, gpu);
        let (l1, l2, r) = offsets_global_coalescing(&offsets, gpu);
        l1_coalescing = f64::min(l1_coalescing, l1);
        l2_coalescing = f64::min(l2_coalescing, l2);
        replay = f64::min(replay, r);
    }
    (l1_coalescing, l2_coalescing, replay)
}

/// Computes the L1, L2 coalescing and replay factor for a global memory access.
fn offsets_global_coalescing(offsets: &[u64], gpu: &cuda::Gpu) -> (f64, f64, f64) {
    let mut l1_lines: HashSet<_> = std::iter::once(0).collect();
    let mut l2_lines: HashSet<_> = std::iter::once(0).collect();
    // Compute the lines accessed by each tread in a wrap.
    for &offset in offsets {
        l1_lines.insert(offset/gpu.l1_cache_line as u64);
        l2_lines.insert(offset/gpu.l2_cache_line as u64);
    }
    trace!("global_replay: {} (size: {})", l1_lines.len(), offsets.len());
    let l1_coalescing = l1_lines.len() as f64 / offsets.len() as f64;
    let l2_coalescing = l2_lines.len() as f64 / offsets.len() as f64;
    (l1_coalescing, l2_coalescing, l1_lines.len() as f64)
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
#[cfg(feature="cuda")]
mod tests {
    use super::*;
    use device::cuda::{self, Gpu};
    use helper;
    use ir;
    use env_logger;
    use search_space::Order;
    use model::size::Range;

    /// Generates function with a load in two thread dimensions, with non-coalesced
    /// accessed on the first one.
    fn gen_function<'a>(signature: &'a ir::Signature, gpu: &'a Gpu, d0_d1_order: Order)
            -> (SearchSpace<'a>, ir::InstId, HashMap<ir::dim::Id, Range>) {
        let mut builder = helper::Builder::new(&signature, gpu);
        let t = ir::Type::F(32);
        let size = builder.cst_size(gpu.wrap_size);
        let addr_base = builder.cast(&0i64, ir::Type::PtrTo(ir::mem::Id::External(0)));
        let d0 = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let d1 = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let addr = builder.mad(&d0, &(gpu.l1_cache_line as i32), &addr_base);
        let stride = ir::Size::new(gpu.l1_cache_line, vec![], 1);
        let pattern = ir::AccessPattern::Tensor {
            mem_id: ir::mem::Id::External(0),
            dims: std::iter::once((d0, stride)).collect(),
        };
        let ld = builder.ld_ex(t, &addr, pattern, InstFlag::MEM_CG);
        builder.order(&d0, &d1, d0_d1_order);

        let mut size_map = HashMap::default();
        let wrap_size = Range { min: gpu.wrap_size as u64, max: gpu.wrap_size as u64 };
        size_map.insert(d0, wrap_size);
        size_map.insert(d1, wrap_size);
        (builder.get(), ld, size_map)
    }

    /// Generates a dummy signature.
    fn gen_signature() -> ir::Signature {
        ir::Signature { name: String::new(), params: vec![], mem_blocks: 1 }
    }


    /// Tests `MemInfo` for global loads without coalescing.
    #[test]
    fn global_no_coalescing() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let ctx = cuda::Context::new(&executor);
        let gpu = cuda::Gpu::from_executor(&executor);
        let base = gen_signature();
        let (space, inst, size_map) = gen_function(&base, &gpu, Order::OUTER);
        let inst = space.ir_instance().inst(inst);
        let inst_info = analyse(&space, &gpu, &inst, &size_map, &ctx);
        assert_eq!(inst_info.l1_coalescing, 1.0/gpu.wrap_size as f64);
        assert_eq!(inst_info.l2_coalescing, 1.0/gpu.wrap_size as f64);
        assert_eq!(inst_info.replay_factor, 1.0);
    }

    /// Tests `MemInfo` for global loads with full coalescing.
    #[test]
    fn global_full_coalescing() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let ctx = cuda::Context::new(&executor);
        let gpu = cuda::Gpu::from_executor(&executor);
        let base = gen_signature();
        let (space, inst, size_map) = gen_function(&base, &gpu, Order::INNER);
        let inst = space.ir_instance().inst(inst);
        let inst_info = analyse(&space, &gpu, &inst, &size_map, &ctx);
        assert_eq!(inst_info.l1_coalescing, 1.0);
        assert_eq!(inst_info.l2_coalescing, 1.0);
        assert_eq!(inst_info.replay_factor, gpu.wrap_size as f64);
    }
}
