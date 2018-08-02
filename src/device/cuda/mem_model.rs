//! Memory accesses analysis.
use binary_heap_plus::BinaryHeap;
use device::{Context, cuda};
use ir;
use itertools::Itertools;
use search_space::{DimKind, Domain, InstFlag, ThreadMapping, SearchSpace};
use std;
use num::Integer;
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
               sizes: &HashMap<ir::DimId, u32>,
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
        dims: &HashMap<ir::DimId, ir::Size>,
        is_shared_access: Trivalent,
        gpu: &cuda::Gpu,
        sizes: &HashMap<ir::DimId, u32>,
        ctx: &Context) -> MemInfo {
    let mut info = NO_ACCESS_INFO;
    let thread_dims = tensor_thread_dims(space, inst, dims, sizes, gpu, ctx);
    trace!("thread dims: {:?}", thread_dims);
    let mut offsets = vec![wrap_access_offsets(&thread_dims, gpu)];
    // Handle the case where the last dimension may not be active. In that case we also
    // try without the dimension as considering it as a thread may increase the pressure.
    // Only the last dimension needs sepcial handling as other dimensions are fully
    // contained into a wrap.
    if thread_dims.last().map(|d| !d.is_active_thread).unwrap_or(false) {
        offsets.push(wrap_access_offsets(&thread_dims[0..thread_dims.len()-1], gpu));
    }
    for offsets in &offsets {
        trace!("wrap offsets: {:?}", offsets);
        if is_shared_access.maybe_true() {
            let replay = shared_replay_factor(offsets, dims, sizes, space, gpu);
            info.replay_factor = f64::min(replay, info.replay_factor);
            info.l2_miss_ratio = 0.0;
        }
        if is_shared_access.maybe_false() {
            let (l1_coalescing, l2_coalescing, replay) = global_coalescing(offsets, gpu);
            info.l1_coalescing = f64::min(l1_coalescing, info.l1_coalescing);
            info.l2_coalescing = f64::min(l2_coalescing, info.l2_coalescing);
            info.replay_factor = f64::min(replay, info.replay_factor);
            // TODO(model): compute the miss ratio
            info.l2_miss_ratio = 0.0;
        }
    }
    info
}

#[derive(Debug)]
struct ThreadDimInfo {
    id: ir::DimId,
    is_active_thread: bool,
    size: u64,
    stride: u64,
}

/// Returns the size and stride of thread dimensions for a tensor access pattern and
/// sort them in an optimal or better-than-optimal order. For two dimensions `d0`, `d1`
/// such that `d0.stride` < `d1.stride` and `such that, d0` can be nested inside `d1` the
/// order guarantees that `d0 < d1`.
fn tensor_thread_dims(space: &SearchSpace,
                      inst: &ir::Instruction,
                      tensor_dims: &HashMap<ir::DimId, ir::Size>,
                      sizes: &HashMap<ir::DimId, u32>,
                      gpu: &cuda::Gpu,
                      ctx: &Context) -> Vec<ThreadDimInfo> {
    let external_dims = external_thread_dims(inst, space);
    let dims = inst.iteration_dims().iter().flat_map(|&dim| {
        match space.domain().get_dim_kind(dim).is(DimKind::THREAD) {
            Trivalent::False => None,
            Trivalent::Maybe => Some((dim, false)),
            Trivalent::True => Some((dim, true)),
        }
    }).chain(external_dims).map(|(id, is_active_thread)| {
        let size = sizes[&id];
        let stride = tensor_dims.get(&id).map(|s| {
            ctx.eval_size(&s.clone().into()) as u64
        }).unwrap_or(0);
        ThreadDimInfo {
            size: u64::from(size),
            stride, id, is_active_thread,
        }
    }).collect_vec();
    sort_thread_dims(dims, space, gpu)
}

/// Returns the thread dimensions that are mapped outside an instruction but not active
/// under this instruction. The returned boolean indicates if the thread dimension cannot
/// be mapped to an active dimension and if the dimension is predicated.
fn external_thread_dims<'a>(inst: &'a ir::Instruction, space: &'a SearchSpace)
    -> impl Iterator<Item=(ir::DimId, bool)> + 'a
{
    space.ir_instance().thread_dims().flat_map(move |dim| {
        let is_mapped = inst.iteration_dims().iter().map(|&other| {
            if dim.id() == other { return Trivalent::True; }
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
/// Because we only support tensor accesses, bigger strides are multiples of smaller
/// strides. Thus smaller stride will lead to less replays.
fn sort_thread_dims(dims: Vec<ThreadDimInfo>,
                    space: &SearchSpace,
                    gpu: &cuda::Gpu) -> Vec<ThreadDimInfo> {
    let sure_thread_dims = dims.iter().filter(|d| d.is_active_thread)
        .map(|d| d.id).collect_vec();
    let cmp = |x: &ThreadDimInfo, y: &ThreadDimInfo| y.stride.cmp(&x.stride);
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
        total_size *= d.size;
        out.push(d);
        heap.extend(dim_groups.remove(&out.len()));
        if total_size > gpu.wrap_size as u64 { break; }
    }
    out
}

/// Returns the offset of memory accesses for each thread in a wrap. The offset is
/// relative to the access of the first thread.
fn wrap_access_offsets(thread_dims: &[ThreadDimInfo], gpu: &cuda::Gpu) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(gpu.wrap_size as usize);
    offsets.push(0);
    let mut indexes = vec![0; thread_dims.len()];
    while offsets.len() < gpu.wrap_size as usize {
        let mut incr = true;
        let mut offset = 0;
        for (idx, dim) in indexes.iter_mut().zip_eq(thread_dims) {
            if incr {
                *idx += 1;
                if *idx == dim.size { *idx = 0; } else { incr = false; }
            }
            offset += *idx * dim.stride;
        }
        if incr { break; } // We reached the end of all loops.
        offsets.push(offset);
    }
    offsets
}

/// Computes the replay factor for a shared memory access.
fn shared_replay_factor(offsets: &[u64],
                        tensor_dims: &HashMap<ir::DimId, ir::Size>,
                        dim_sizes: &HashMap<ir::DimId, u32>,
                        space: &SearchSpace, gpu: &cuda::Gpu) -> f64 {
    // We only need to account for hits on the first bank. Other banks will have a smaller
    // replay factor.
    let mut hits: HashSet<_> = std::iter::once(0).collect();
    for &offset in offsets  {
        let num_bank_stride = offset / gpu.shared_bank_stride as u64;
        let (hit_id, rem) = num_bank_stride.div_rem(&(gpu.wrap_size as u64));
        if rem == 0 { hits.insert(hit_id); }
    }
    // Handle the case where a single thread must access two banks.
    let vector_replay = tensor_dims.iter()
        .flat_map(|(&d, stride)| stride.as_int().map(|s| (d, s)))
        .filter(|&(d, _)| space.domain().get_dim_kind(d).intersects(DimKind::VECTOR))
        .map(|(d, stride)| div_ceil(dim_sizes[&d]*stride as u32, gpu.shared_bank_stride))
        .min().unwrap_or(1);
    let replay_factor = std::cmp::max(hits.len() as u32, vector_replay);
    trace!("shared_replay: {}", replay_factor);
    replay_factor as f64
}

/// Computes the L1, L2 coalescing and replay factor for a global memory access.
fn global_coalescing(offsets: &[u64], gpu: &cuda::Gpu) -> (f64, f64, f64) {
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
               sizes: &HashMap<ir::DimId, u32>) -> f64 {
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
                  sizes: &HashMap<ir::DimId, u32>,
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
               dim: ir::DimId,
               sizes: &HashMap<ir::DimId, u32>) -> ir::Stride {
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

    /// Generates function with a load in two thread dimensions, with non-coalesced
    /// accessed on the first one.
    fn gen_function<'a>(signature: &'a ir::Signature, gpu: &'a Gpu, d0_d1_order: Order)
            -> (SearchSpace<'a>, ir::InstId, HashMap<ir::DimId, u32>) {
        let mut builder = helper::Builder::new(&signature, gpu);
        let t = ir::Type::F(32);
        let size = builder.cst_size(gpu.wrap_size);
        let addr_base = builder.cast(&0i64, ir::Type::PtrTo(ir::MemId::External(0)));
        let d0 = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let d1 = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let addr = builder.mad(&d0, &(gpu.l1_cache_line as i32), &addr_base);
        let stride = ir::Size::new(gpu.l1_cache_line, vec![], 1);
        let pattern = ir::AccessPattern::Tensor {
            mem_id: ir::MemId::External(0),
            dims: std::iter::once((d0, stride)).collect(),
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
