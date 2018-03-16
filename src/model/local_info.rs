//! Compute and represent local information on the different objects representing of the IR.
use device::{Context, Device};
use ir::{self, BasicBlock};
use model::HwPressure;
use search_space::{DimKind, Domain, Order, SearchSpace};
use std::cmp;
use utils::*;

/// Local information on the different objects.
#[derive(Debug)]
pub struct LocalInfo {
    /// The loops inside and outside each BB.
    pub nesting: HashMap<ir::BBId, Nesting>,
    /// The pressure incured by a signle instance of each BB.
    pub hw_pressure: HashMap<ir::BBId, HwPressure>,
    /// The size of each dimensions.
    pub dim_sizes: HashMap<ir::dim::Id, u32>,
    /// The pressure induced by a single iteration of each dimension and the exit latency
    /// of the loop.
    pub dim_overhead: HashMap<ir::dim::Id, (HwPressure, HwPressure)>,
    /// The overhead to initialize a thread.
    pub thread_overhead: HwPressure,
    /// Available parallelism in the kernel.
    pub parallelism: Parallelism,
}

impl LocalInfo {
    /// Compute the local information for the given search space, in the context.
    pub fn compute(space: &SearchSpace, context: &Context) -> Self {
        let dim_sizes = space.ir_instance().dims()
            .map(|d| (d.id(), context.eval_size(d.size()))).collect();
        let nesting = space.ir_instance().blocks()
            .map(|bb| (bb.bb_id(), Nesting::compute(space, bb.bb_id()))).collect();
        let mut hw_pressure = space.ir_instance().blocks().map(|bb| {
            (bb.bb_id(), context.device().hw_pressure(space, &dim_sizes, &nesting, bb))
        }).collect();
        let mut dim_overhead = space.ir_instance().dims().map(|d| {
            let kind = space.domain().get_dim_kind(d.id());
            (d.id(), context.device().loop_iter_pressure(kind))
        }).collect();
        let parallelism = parallelism(space, &nesting, &dim_sizes);
        // Add the pressure induced by induction variables.
        let mut thread_overhead = HwPressure::zero(context.device());
        for (_, var) in space.ir_instance().induction_vars() {
            add_indvar_pressure(context.device(), space, &dim_sizes, var,
                &mut hw_pressure, &mut dim_overhead, &mut thread_overhead);
        }
        LocalInfo {
            dim_sizes, nesting, hw_pressure, dim_overhead, thread_overhead, parallelism
        }
    }
}

fn add_indvar_pressure(device: &Device,
                       space: &SearchSpace,
                       dim_sizes: &HashMap<ir::dim::Id, u32>,
                       indvar: &ir::InductionVar,
                       hw_pressure: &mut HashMap<ir::BBId, HwPressure>,
                       dim_overhead: &mut HashMap<ir::dim::Id, (HwPressure, HwPressure)>,
                       thread_overhead: &mut HwPressure) {
   for &(dim, _) in indvar.dims() {
       let dim_kind = space.domain().get_dim_kind(dim);
       if dim_kind.intersects(DimKind::VECTOR) { continue; }
       let t = device.lower_type(indvar.base().t(), space).unwrap_or(ir::Type::I(32));
       let mut overhead = if dim_kind.intersects(DimKind::UNROLL | DimKind::LOOP) {
           device.additive_indvar_pressure(&t)
       } else {
           device.multiplicative_indvar_pressure(&t)
       };
       // FIXME: do not add the latency if the if it is an additive unrolled dimension that can be
       // precomputed
       let size = dim_sizes[&dim];
       if dim_kind.intersects(DimKind::THREAD | DimKind::BLOCK) {
           thread_overhead.add_parallel(&overhead);
       } else if size > 1 {
           overhead.repeat_parallel((size-1) as f64);
           unwrap!(hw_pressure.get_mut(&dim.into())).add_parallel(&overhead);
           overhead.repeat_parallel(1.0/(size-1) as f64);
           unwrap!(dim_overhead.get_mut(&dim)).0.add_parallel(&overhead);
       }
   }
}

/// Nesting of an object.
#[derive(Debug)]
pub struct Nesting {
    /// Dimensions nested outside the current BB.
    pub inner_dims: VecSet<ir::dim::Id>,
    /// Basic blocks nested inside the current BB.
    pub inner_bbs: VecSet<ir::BBId>,
    /// Dimensions nested inside the current BB.
    pub outer_dims: VecSet<ir::dim::Id>,
    /// Dimensions to be processed before the current BB.
    pub before_self: VecSet<ir::dim::Id>,
    /// Dimensions that should not take the current BB into account when processed.
    pub after_self: VecSet<ir::dim::Id>,
    /// The dimensions that can be merged to this one and have a bigger ID.
    pub bigger_merged_dims: VecSet<ir::dim::Id>,
}

impl Nesting {
    /// Computes the nesting of a `BasicBlock`.
    fn compute(space: &SearchSpace, bb: ir::BBId) -> Self {
        let mut inner_dims = Vec::new();
        let mut inner_bbs = Vec::new();
        let outer_dims = space.ir_instance().block(bb)
            .iteration_dims().iter().cloned().collect();
        let mut before_self = Vec::new();
        let mut after_self = Vec::new();
        let mut bigger_merged_dims = Vec::new();
        for other_bb in space.ir_instance().blocks().map(|bb| bb.bb_id()) {
            if other_bb == bb { continue; }
            let order = space.domain().get_order(other_bb, bb);
            if Order::INNER.contains(order) { inner_bbs.push(other_bb); }
            if let ir::BBId::Dim(id) = other_bb {
                if Order::INNER.contains(order) { inner_dims.push(id); }
                if (Order::INNER | Order::BEFORE).contains(order) { before_self.push(id); }
                if (Order::OUTER | Order::AFTER).contains(order) { after_self.push(id); }
                if order.intersects(Order::MERGED) && other_bb > bb {
                    bigger_merged_dims.push(id);
                }
            }
        }
        Nesting {
            inner_dims: VecSet::new(inner_dims),
            inner_bbs: VecSet::new(inner_bbs),
            outer_dims: VecSet::new(outer_dims),
            before_self: VecSet::new(before_self),
            after_self: VecSet::new(after_self),
            bigger_merged_dims: VecSet::new(bigger_merged_dims),
        }
    }
}

/// Minimum and maximum parallelism in the whole GPU.
#[derive(Debug)]
pub struct Parallelism {
    /// Minimal number of block of threads.
    pub min_num_blocks: u64,
    /// Maximal number of blocks of threads.
    pub max_num_blocks: u64,
    /// Minimal number of threads in the whole GPU.
    pub min_num_threads: u64,
    /// Minimal number of threads per thread blocks.
    pub min_num_threads_per_block: u64,
    /// Maximal number of threads per thread blocks.
    pub max_num_threads_per_block: u64,
}

impl Default for Parallelism {
    fn default() -> Self {
        Parallelism {
            min_num_blocks: 1,
            max_num_blocks: 1,
            min_num_threads: 1,
            min_num_threads_per_block: 1,
            max_num_threads_per_block: 1,
        }
    }
}

/// Computes the minimal and maximal parallelism accross instructions.
fn parallelism(space: &SearchSpace, nesting: &HashMap<ir::BBId, Nesting>,
               dim_sizes: &HashMap<ir::dim::Id, u32>) -> Parallelism {
    space.ir_instance().insts().map(|inst| {
        let mut par = Parallelism::default();
        for &dim in &nesting[&inst.bb_id()].outer_dims {
            let kind = space.domain().get_dim_kind(dim);
            let size = dim_sizes[&dim] as u64;
            if kind == DimKind::BLOCK { par.min_num_blocks *= size; }
            if kind.intersects(DimKind::BLOCK) { par.max_num_blocks *= size; }
            let thread_or_block = DimKind::THREAD | DimKind::BLOCK;
            if thread_or_block.contains(kind) { par.min_num_threads *= size; }
            if kind == DimKind::THREAD { par.min_num_threads_per_block *= size; }
            if DimKind::THREAD.intersects(kind) { par.max_num_threads_per_block *= size; }
        }
        par
    }).fold(Parallelism::default(), |lhs, rhs| Parallelism {
        min_num_blocks: cmp::max(lhs.min_num_blocks, rhs.min_num_blocks),
        max_num_blocks: cmp::max(lhs.max_num_blocks, rhs.max_num_blocks),
        min_num_threads: cmp::max(lhs.min_num_threads, rhs.min_num_threads),
        min_num_threads_per_block: cmp::max(
            lhs.min_num_threads_per_block, rhs.min_num_threads_per_block),
        max_num_threads_per_block: cmp::max(
            lhs.max_num_threads_per_block, rhs.max_num_threads_per_block),
    })
}
