//! Compute and represent local information on the different objects representing of the IR.
use crate::device::{Context, Device};
use crate::ir::{self, Statement};
use crate::model::{size, HwPressure};
use crate::search_space::{DimKind, Domain, Order, SearchSpace, ThreadMapping};
use fxhash::FxHashMap;
use itertools::Itertools;
use num::integer::lcm;

use sym::Range;
use utils::*;

/// Local information on the different objects.
#[derive(Debug)]
pub struct LocalInfo {
    /// The symbolic size of each dimension.
    pub dim_sizes: FxHashMap<ir::DimId, size::SymbolicInt>,
    /// The loops inside and outside each Stmt.
    pub nesting: FxHashMap<ir::StmtId, Nesting>,
    /// The pressure incured by a single instance of each Stmt.
    pub hw_pressure: FxHashMap<ir::StmtId, HwPressure>,
    /// The pressure induced by a single iteration of each dimension and the exit latency
    /// of the loop.
    pub dim_overhead: FxHashMap<ir::DimId, (HwPressure, HwPressure)>,
    /// The overhead to initialize a thread.
    pub thread_overhead: HwPressure,
    /// Available parallelism in the kernel.
    pub parallelism: Parallelism,
}

impl LocalInfo {
    fn dim_sizes(
        space: &SearchSpace,
        context: &dyn Context,
    ) -> FxHashMap<ir::DimId, size::SymbolicInt> {
        struct Builder<'a> {
            space: &'a SearchSpace,
            context: &'a dyn Context,

            /// Internal mapping from dimensin ID to actual size.
            /// The size is either an integer when it is known exactly, or a representation of the
            /// dimension's possible values.
            ///
            /// This only contains information about static dimensions whose size will be exactly known in
            /// a fully specified implementation.
            static_dims: FxHashMap<ir::DimId, Result<u64, size::DimSize>>,
        }

        impl<'a> Builder<'a> {
            fn get_or_create_static_dim(
                &mut self,
                id: ir::DimId,
            ) -> Result<u64, size::DimSize> {
                self.static_dims
                    .entry(id)
                    .or_insert_with({
                        let space = self.space;
                        move || {
                            let size = space.domain().get_size(id);
                            let universe = space
                                .ir_instance()
                                .dim(id)
                                .possible_sizes()
                                .unwrap_or_else(|| panic!("Unknown static dim ?!"));
                            let values = size.list_values(universe).collect::<Vec<_>>();

                            if values.len() == 1 {
                                Ok(u64::from(values[0]))
                            } else {
                                Err(size::DimSize::new(id, values))
                            }
                        }
                    })
                    .clone()
            }

            fn dim(&mut self, size: &ir::PartialSize) -> size::SymbolicInt {
                let (static_factor, param_factors, dim_factors) = size.factors();
                let divisors = size.divisors();

                let mut factor = u64::from(static_factor)
                    * param_factors
                        .iter()
                        .map(|p| {
                            u64::from(
                                self.context.param_as_size(&p.name).unwrap_or_else(
                                    || panic!("Unknown param: {}", p.name),
                                ),
                            )
                        })
                        .product::<u64>();

                let mut numer = Vec::new();
                for &id in dim_factors {
                    match self.get_or_create_static_dim(id) {
                        Ok(x) => factor *= x,
                        Err(d) => numer.push(d),
                    }
                }

                let mut denom = Vec::new();
                for &id in size.divisors() {
                    match self.get_or_create_static_dim(id) {
                        Ok(x) => {
                            assert!(
                                factor % x == 0,
                                "Inconsistent size: denominator does not divide factor."
                            );
                            factor /= x;
                        }
                        Err(d) => denom.push(d),
                    }
                }

                size::SymbolicInt::ratio(factor, numer, denom)
            }
        }

        let mut builder = Builder {
            space,
            context,
            static_dims: FxHashMap::default(),
        };

        space
            .ir_instance()
            .dims()
            .map(|d| (d.id(), builder.dim(d.size())))
            .collect()
    }

    /// Compute the local information for the given search space, in the context.
    pub fn compute(space: &SearchSpace, context: &dyn Context) -> Self {
        let dim_sizes = Self::dim_sizes(space, context);
        let nesting: FxHashMap<_, _> = space
            .ir_instance()
            .statements()
            .map(|stmt| {
                (
                    stmt.stmt_id(),
                    Nesting::compute(space, &dim_sizes, stmt.stmt_id()),
                )
            })
            .collect();
        let parallelism = parallelism(space, &dim_sizes, &nesting, context);
        let mut hw_pressure = space
            .ir_instance()
            .statements()
            .map(|stmt| {
                let is_thread = if let ir::StmtId::Dim(id) = stmt.stmt_id() {
                    space.domain().get_dim_kind(id) == DimKind::THREAD
                } else {
                    false
                };
                // Only keep the pressure of innermost thread dimensions. Otherwise it
                // will be taken multiple times into account.
                let pressure =
                    if is_thread && nesting[&stmt.stmt_id()].has_inner_thread_dims {
                        HwPressure::zero(&*context.device())
                    } else {
                        context
                            .device()
                            .hw_pressure(space, &dim_sizes, &nesting, stmt, context)
                    };
                (stmt.stmt_id(), pressure)
            })
            .collect();
        let mut dim_overhead = space
            .ir_instance()
            .dims()
            .map(|d| {
                let kind = space.domain().get_dim_kind(d.id());
                if kind == DimKind::THREAD && nesting[&d.stmt_id()].has_inner_thread_dims
                {
                    // Only keep the overhead on innermost thread dimensions. Otherwise it
                    // will be taken multiple times into account.
                    let zero = HwPressure::zero(&*context.device());
                    (d.id(), (zero.clone(), zero))
                } else {
                    (d.id(), context.device().loop_iter_pressure(kind))
                }
            })
            .collect();
        // Add the pressure induced by induction variables.
        let mut thread_overhead = HwPressure::zero(&*context.device());
        for (_, var) in space.ir_instance().induction_vars() {
            add_indvar_pressure(
                &*context.device(),
                space,
                &dim_sizes,
                var,
                &mut hw_pressure,
                &mut dim_overhead,
                &mut thread_overhead,
            );
        }
        LocalInfo {
            dim_sizes,
            nesting,
            hw_pressure,
            dim_overhead,
            thread_overhead,
            parallelism,
        }
    }
}

fn add_indvar_pressure(
    device: &dyn Device,
    space: &SearchSpace,
    dim_sizes: &FxHashMap<ir::DimId, size::SymbolicInt>,
    indvar: &ir::InductionVar,
    hw_pressure: &mut FxHashMap<ir::StmtId, HwPressure>,
    dim_overhead: &mut FxHashMap<ir::DimId, (HwPressure, HwPressure)>,
    thread_overhead: &mut HwPressure,
) {
    for &(dim, _) in indvar.dims() {
        let dim_kind = space.domain().get_dim_kind(dim);
        if dim_kind.intersects(DimKind::VECTOR) {
            continue;
        }
        let t = device
            .lower_type(indvar.base().t(), space)
            .unwrap_or(ir::Type::I(32));
        let mut overhead = if dim_kind.intersects(DimKind::UNROLL | DimKind::LOOP) {
            // FIXME: do not add the latency if the induction level can statically computed.
            // This is the case when:
            // - the loop is unrolled
            // - the increment is a constant
            // - both the conditions are also true for an inner dimension.
            device.additive_indvar_pressure(&t)
        } else {
            device.multiplicative_indvar_pressure(&t)
        };
        if dim_kind.intersects(DimKind::THREAD | DimKind::BLOCK) {
            thread_overhead.add_parallel(&overhead);
        } else {
            let size = &dim_sizes[&dim];
            if size.min_value() > 1 {
                unwrap!(dim_overhead.get_mut(&dim))
                    .0
                    .add_parallel(&overhead);
                // TODO(sym): size.to_float() - 1
                overhead.repeat_parallel(&(size - 1u32));
                unwrap!(hw_pressure.get_mut(&dim.into())).add_parallel(&overhead);
            }
        }
    }
}

/// Nesting of an object.
#[derive(Debug)]
pub struct Nesting {
    /// Dimensions nested inside the current Stmt.
    pub inner_dims: VecSet<ir::DimId>,
    /// Basic blocks nested inside the current Stmt.
    pub inner_stmts: VecSet<ir::StmtId>,
    /// Dimensions nested outsidethe current Stmt.
    pub outer_dims: VecSet<ir::DimId>,
    /// Dimensions to be processed before the current Stmt.
    pub before_self: VecSet<ir::DimId>,
    /// Dimensions that should not take the current Stmt into account when processed.
    pub after_self: VecSet<ir::DimId>,
    /// The dimensions that can be merged to this one and have a bigger ID.
    pub bigger_merged_dims: VecSet<ir::DimId>,
    /// Indicates if the block may have thread dimensions nested inside it.
    /// Only consider thread dimensions that are sure to be mapped to threads.
    has_inner_thread_dims: bool,
    /// Number of threads that are not represented in the active dimensions of the block.
    pub num_unmapped_threads: size::SymbolicInt,
    /// Maximal number of threads this block can be in, considering only outer dimensions
    /// (an not mapped out dimensions).
    pub max_threads_per_block: size::SymbolicInt,
}

impl Nesting {
    /// Computes the nesting of a `Statement`.
    fn compute(
        space: &SearchSpace,
        dim_sizes: &FxHashMap<ir::DimId, size::SymbolicInt>,
        stmt: ir::StmtId,
    ) -> Self {
        let mut inner_dims = Vec::new();
        let mut inner_stmts = Vec::new();
        let mut before_self = Vec::new();
        let mut after_self = Vec::new();
        let mut bigger_merged_dims = Vec::new();
        let mut has_inner_thread_dims = false;
        for other_stmt in space.ir_instance().statements() {
            if other_stmt.stmt_id() == stmt {
                continue;
            }
            let order = space.domain().get_order(other_stmt.stmt_id(), stmt);
            if Order::INNER.contains(order) {
                inner_stmts.push(other_stmt.stmt_id());
            }
            if let Some(dim) = other_stmt.as_dim() {
                let other_kind = space.domain().get_dim_kind(dim.id());
                if Order::INNER.contains(order) {
                    inner_dims.push(dim.id());
                }
                if order.intersects(Order::INNER) && other_kind == DimKind::THREAD {
                    has_inner_thread_dims = true;
                }
                if (Order::INNER | Order::BEFORE).contains(order) {
                    before_self.push(dim.id());
                }
                if (Order::OUTER | Order::AFTER).contains(order) {
                    after_self.push(dim.id());
                }
                if order.intersects(Order::MERGED) && other_stmt.stmt_id() > stmt {
                    bigger_merged_dims.push(dim.id());
                }
            }
        }
        let outer_dims = Self::get_iteration_dims(space, stmt);
        let num_unmapped_threads = space
            .ir_instance()
            .thread_dims()
            .filter(|dim| {
                !outer_dims.iter().any(|&other| {
                    if dim.id() == other {
                        return true;
                    }
                    if space.ir_instance().dim(other).possible_sizes().is_none() {
                        return false;
                    }
                    let mapping = space.domain().get_thread_mapping(dim.id(), other);
                    mapping.intersects(ThreadMapping::MAPPED)
                })
            })
            .map(|d| &dim_sizes[&d.id()])
            .product::<size::SymbolicInt>();
        let max_threads_per_block = outer_dims
            .iter()
            .cloned()
            .filter(|&d| space.domain().get_dim_kind(d).intersects(DimKind::THREAD))
            .map(|d| &dim_sizes[&d])
            .product::<size::SymbolicInt>();
        Nesting {
            inner_dims: VecSet::new(inner_dims),
            inner_stmts: VecSet::new(inner_stmts),
            outer_dims,
            before_self: VecSet::new(before_self),
            after_self: VecSet::new(after_self),
            bigger_merged_dims: VecSet::new(bigger_merged_dims),
            has_inner_thread_dims,
            num_unmapped_threads,
            max_threads_per_block,
        }
    }

    /// Computess the list of iteration dimensions of a `Statement`.
    fn get_iteration_dims(space: &SearchSpace, stmt: ir::StmtId) -> VecSet<ir::DimId> {
        let dims = if let ir::StmtId::Inst(inst) = stmt {
            space
                .ir_instance()
                .inst(inst)
                .iteration_dims()
                .iter()
                .cloned()
                .collect()
        } else {
            let mut outer = Vec::new();
            for dim in space.ir_instance().dims().map(|d| d.id()) {
                if stmt == dim.into() {
                    continue;
                }
                let order = space.domain().get_order(dim.into(), stmt);
                if Order::OUTER.contains(order)
                    && outer.iter().cloned().all(|outer: ir::DimId| {
                        let ord = space.domain().get_order(dim.into(), outer.into());
                        !ord.contains(Order::MERGED)
                    })
                {
                    outer.push(dim);
                }
            }
            outer
        };
        VecSet::new(dims)
    }
}

/// Minimum and maximum parallelism in the whole GPU.
#[derive(Debug)]
pub struct Parallelism {
    /// Minimal number of blocks.
    // TODO(sym): Float
    pub min_num_blocks: size::SymbolicInt,
    /// Minimal number of threads per blocks.
    pub min_num_threads_per_blocks: size::SymbolicInt,
    /// Minimal number of threads.
    pub min_num_threads: size::SymbolicInt,
    /// A multiple of the number of blocks.
    pub lcm_num_blocks: size::SymbolicInt,
}

impl Parallelism {
    /// Combines two `Parallelism` summaries computed on different instructions and computes the
    /// `Parallelism` of the union of the instructions.
    fn combine(mut self, rhs: &Self) -> Self {
        self.combine_assign(rhs);
        self
    }

    fn combine_assign(&mut self, rhs: &Self) {
        self.min_num_threads_per_blocks
            .min_assign(&rhs.min_num_threads_per_blocks);
        self.min_num_blocks.min_assign(&rhs.min_num_blocks);
        self.min_num_threads.min_assign(&rhs.min_num_threads);
        self.lcm_num_blocks.lcm_assign(&rhs.lcm_num_blocks);
    }
}

impl Default for Parallelism {
    fn default() -> Self {
        Parallelism {
            // TODO(sym): float
            min_num_blocks: 1u32.into(),
            min_num_threads_per_blocks: 1u32.into(),
            min_num_threads: 1u32.into(),
            lcm_num_blocks: 1u32.into(),
        }
    }
}

/// Computes the minimal and maximal parallelism accross instructions.
fn parallelism(
    space: &SearchSpace,
    dim_sizes: &FxHashMap<ir::DimId, size::SymbolicInt>,
    nesting: &FxHashMap<ir::StmtId, Nesting>,
    ctx: &dyn Context,
) -> Parallelism {
    let size_thread_dims = space
        .ir_instance()
        .thread_dims()
        .map(|d| &dim_sizes[&d.id()])
        .product::<size::SymbolicInt>();
    space
        .ir_instance()
        .insts()
        .map(|inst| {
            let mut min_size_blocks = size::SymbolicInt::one();
            let mut max_size_blocks = size::SymbolicInt::one();
            for &dim in &nesting[&inst.stmt_id()].outer_dims {
                let kind = space.domain().get_dim_kind(dim);
                if kind.intersects(DimKind::BLOCK) {
                    let size = &dim_sizes[&dim];
                    max_size_blocks *= size;
                    if kind == DimKind::BLOCK {
                        min_size_blocks *= size;
                    }
                }
            }

            Parallelism {
                min_num_threads: &min_size_blocks * &size_thread_dims,
                // TODO(sym): min_size_blocks.to_float()
                min_num_blocks: min_size_blocks,
                min_num_threads_per_blocks: size_thread_dims.clone(),
                lcm_num_blocks: max_size_blocks,
            }
        })
        .fold1(|lhs, rhs| lhs.combine(&rhs))
        .unwrap_or_default()
}
