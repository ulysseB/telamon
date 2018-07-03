//! Building Blocks for lower bound performance models.
mod code_point;
mod dependency_map;
mod hw_pressure;
mod level;
mod local_info;

pub mod size;

pub use self::hw_pressure::{Bound, HwPressure, BottleneckLevel};
pub use self::local_info::Nesting;

// TODO(model): we currently take the minimal value of sizes when computing levels size.
// It migh beneficial to consider combination of dimensions instead as the size of loop
// nest is known.
// TODO(model): One some instruction, the latency dependens on the operand position.
// TODO(model): Some instructions are divided into multiple sub-instructions. When adding
//  ordering dependencies, this must be taken into account as the last sub-instruction
//  must be issued for the execution to continue to the next macro-instruction. The
//  latency between the two macro-instructions is thus non-zero.
// TODO(model): latency of induction variables
//  - take the latency to the origin into account
//  - take the latency of add/muls into account
//  - FIXME: should not be taken into account for inner latency of unrolled loops
//    * account for unrolled induction variables only in the hw pressure and not in the
//      iteration overhead => also check regular loops
// TODO(model): take syncthread overhead into account
// TODO(model): the block parallelism is overestimated because we do not account for the
//  number of registers used per threads.
// FIXME: to avoid error, distinguish the issue and consumption of instructions.
//  For example, a loop might issue loads, but the loads can end after the end of the loop
//  is issued. For this, either double the nodes or subtract the size of buffers to the next
//  issue.

use device::{Device, Context};
use ir;
use itertools::Itertools;
use model::code_point::{CodePoint, CodePointDag};
use model::dependency_map::DependencyMap;
use model::level::{Level, RepeatLevel, LevelDag, sum_pressure};
use model::local_info::LocalInfo;
use model::hw_pressure::{FastBound};
use search_space::SearchSpace;
use std::cmp;
use utils::*;

/// Returns a lower bound on the execution time of all the implementation candidates in
/// `space`, when executed in `context`.
pub fn bound(space: &SearchSpace, context: &Context) -> Bound {
    // Build the dependency maps dag.
    let local_info = LocalInfo::compute(space, context);
    trace!("local_info {:?}", local_info);
    let (mut levels, dim_maps) = level::generate(space, context, &local_info);
    let code_points = CodePointDag::build(space, &levels);
    let mut levels_dag = LevelDag::build(
        space, &local_info, &levels, dim_maps, code_points.len(), context);
    trace!("levels {:?}", levels);
    trace!("code_points {:?}", code_points);
    populate(space, context.device(), &local_info, &code_points, &mut levels,
             &mut levels_dag);
    trace!("levels_dag {:?}", levels_dag);
    // Process each level.
    for (from_node, action, to_node) in levels_dag.processing_order(&levels) {
        match action {
            level::DagAction::Repeat(action) => {
                repeat_level(&code_points, &levels, &action, from_node, &to_node,
                             &mut levels_dag)
            },
            level::DagAction::ApplyDimMap(dim_map) => {
                apply_dim_map(context.device(), space, &local_info, &levels,
                    &code_points, &dim_map, from_node, &to_node, &mut levels_dag)
            },
        }
    }
    // Retreive the total latency of a block of threads.
    let root_entry = code_points.ids[&CodePoint::LevelEntry(0)];
    let root_exit = code_points.ids[&CodePoint::LevelExit(0)];
    let block_latency = unwrap!(levels_dag.root().latency(root_entry, root_exit));
    debug!("block latency: {}", block_latency.value());
    // Scale the latency to the block level.
    let block_parallelism = u64::from(context.device().block_parallelism(space));
    let min_num_blocks = local_info.parallelism.min_blocks;
    let lcm_num_blocks = local_info.parallelism.lcm_blocks;
    let latency = block_latency.scale(block_parallelism, min_num_blocks, lcm_num_blocks);
    // Compute the throughput bound at the whole device level.
    let global_pressure = sum_pressure(context, space, &local_info,
                                       BottleneckLevel::Global, &[], &ir::Size::one());
    trace!("global pressure {:?}", global_pressure);
    let device_rates = context.device().total_rates();
    let throughput_bound = global_pressure.bound(BottleneckLevel::Global, &device_rates);
    // Return the biggest bound.
    debug!("full block lat: {}", unwrap!(levels[0].repeated_latency.as_ref()).value());
    let bound = cmp::max(latency, throughput_bound);
    bound.explain(context.device(), &levels, code_points.dag.nodes())
}

/// Populates the dependency maps and the levels with dependency edges and back-edges.
fn populate(space: &SearchSpace,
            device: &Device,
            local_info: &LocalInfo,
            code_points: &CodePointDag,
            levels: &mut [Level],
            level_dag: &mut LevelDag) {
    let thread_rates = device.thread_rates();
    for (point_id, &code_point) in code_points.dag.nodes().iter().enumerate() {
        set_latency(code_points, point_id, &FastBound::zero(), level_dag);
        match code_point {
            CodePoint::Inst(inst_id) => {
                set_data_deps(space, local_info, code_points, &thread_rates, inst_id,
                              point_id, levels, level_dag);
            },
            CodePoint::LevelEntry(id) => {
                let exit = code_points.ids[&CodePoint::LevelExit(id)];
                let latency = &levels[id].latency;
                level_dag.add_dependency_to_all(point_id, exit, latency);
                // Add the latency of all the iterations of the level if present.
                if let Some(ref latency) = levels[id].repeated_latency {
                    if levels[id].dims.is_empty() {
                        level_dag.add_dependency_to_all(point_id, exit, latency);
                    } else {
                        let dims = &levels[id].dims;
                        for &from in code_points.dag.before(point_id) {
                            level_dag.add_if_processed(dims, from, exit, latency);
                        }
                    }
                }
            },
            CodePoint::LevelExit(id) => {
                let latency = &levels[id].end_latency;
                for &from in code_points.dag.before(point_id) {
                    level_dag.add_dependency_to_all(from, point_id, latency);
                }
            },
        }
    }
}

/// Sets the latency from a code point to all its dependencies.
fn set_latency(code_points: &CodePointDag, from: usize, latency: &FastBound,
               level_dag: &mut LevelDag) {
    for &to in code_points.dag.after(from) {
        level_dag.add_dependency_to_all(from, to, latency);
    }
}

/// Updates the dependency maps to account for the data dependencies to an instruction.
// TODO(cleanup): refactor to reduce the number of parameters.
#[cfg_attr(feature="cargo-clippy", allow(clippy))]
fn set_data_deps(space: &SearchSpace,
                 local_info: &LocalInfo,
                 code_points: &CodePointDag,
                 thread_rates: &HwPressure,
                 inst_id: ir::InstId, code_point: usize,
                 levels: &mut [Level],
                 level_dag: &mut LevelDag) {
    for operand in space.ir_instance().inst(inst_id).operands() {
        match *operand {
            ir::Operand::Inst(pred_id, _, ref dim_map, _) => {
                let pred = code_points.ids[&CodePoint::Inst(pred_id)];
                let latency = local_info.hw_pressure[&pred_id.into()]
                    .bound(BottleneckLevel::Thread, thread_rates);
                set_data_dep(space, pred, code_point, dim_map, &latency, level_dag);
            },
            ir::Operand::Reduce(pred_id, _, ref dim_map, ref reduce_dims) => {
                let pred = code_points.ids[&CodePoint::Inst(pred_id)];
                let latency = local_info.hw_pressure[&pred_id.into()]
                    .bound(BottleneckLevel::Thread, thread_rates);
                set_data_dep(space, pred, code_point, dim_map, &latency, level_dag);
                // Add the back-edge in the levels where it is possible.
                let latency = local_info.hw_pressure[&inst_id.into()]
                    .bound(BottleneckLevel::Thread, thread_rates);
                for level in levels.iter_mut() {
                    if level.dims.iter().all(|d| reduce_dims.contains(d)) {
                        level.back_edges.push((code_point, latency.clone()));
                    }
                }
            },
            _ => (),
        }
    }
}

/// Sets a regular data dependency between two instructions.
fn set_data_dep(space: &SearchSpace, from: usize, to: usize, dim_map: &ir::DimMap,
                latency: &FastBound, level_dag: &mut LevelDag) {
    assert!(from < to, "cannot order node {} with node {} (from < to)", from , to);
    let has_src = dim_map.iter().map(|x| x.1).any(|d| level::must_consider_dim(space, d));
    let dst_dims = if has_src {
        dim_map.iter().map(|x| x.1).filter(|&d| {
            level::must_consider_dim(space, d)
        }).collect()
    } else { vec![] };
    level_dag.add_if_processed(&VecSet::new(dst_dims), from, to, latency);
}


/// Applies a `RepeatLevel`.
fn repeat_level(code_points: &CodePointDag,
                levels: &[Level],
                action: &RepeatLevel,
                from_map: level::DagNodeId,
                to_map: &[level::DagNodeId],
                level_dag: &mut LevelDag) {
    // Since we handle separately the first and last iteration, we need at least the
    // first and the last to be present.
    assert!(action.iterations >= 2);
    let level_id = action.level_id;
    let entry_point = code_points.ids[&CodePoint::LevelEntry(action.level_id)];
    let exit_point = code_points.ids[&CodePoint::LevelExit(action.level_id)];
    let (immediate_preds, predecessors, latency_to_exit);
    {
        let dep_map = level_dag.dependencies(from_map);
        // TODO(cc_perf): only predecessors that have an outgoing edge that is not already
        // accounted for by another predecessor in the current dependency map should be
        // considered.
        predecessors = code_points.dag.predecessors(entry_point);
        immediate_preds = dep_map.deps(entry_point).keys().cloned().collect_vec();
        latency_to_exit = dep_map.latency_to(exit_point);
    }
    // Apply the levels repetition factor
    let cycle_lat = unwrap!(latency_to_exit[entry_point].as_ref());
    for pred in immediate_preds {
        // First add the dependency without considering data dependencies from the
        // first and to the last iteration. This reduce the size of the bound
        // explanation when such dependencies are not needed
        let iter_lat = cycle_lat.clone().iterate(action.iterations, level_id);
        let latency = FastBound::zero().chain(entry_point, iter_lat);
        level_dag.add_dependency(to_map, pred, exit_point, &latency);
    }
    for &pred in &predecessors {
        // Then add the bound taking into account data dependencies.
        let init_lat = unwrap!(latency_to_exit[pred].clone());
        let iter_lat = cycle_lat.clone().iterate(action.iterations-2, level_id);
        let latency = init_lat.chain(entry_point, iter_lat);
        level_dag.add_dependency(to_map, pred, entry_point, &latency);
    }
    // Apply back-edges.
    for &(point, ref lat) in &levels[action.level_id].back_edges {
        let latencies = level_dag.dependencies(from_map).latency_to(point);
        for &pred in &predecessors {
            let init_lat_0 = unwrap!(latencies[pred].clone())
                .chain(point, lat.clone());
            let init_lat_1 = unwrap!(latency_to_exit[pred].clone())
                .chain(entry_point, unwrap!(latencies[entry_point].clone()));
            let init_lat = cmp::max(init_lat_0, init_lat_1);
            let latency = init_lat.clone()
                .chain(point, lat.clone().iterate(action.iterations-2, level_id));
            level_dag.add_dependency(to_map, pred, point, &latency);
            if action.iterations >= 3 {
                let exit_lat = unwrap!(latency_to_exit[point].clone());
                let latency = init_lat
                    .chain(point, lat.clone().iterate(action.iterations-3, level_id))
                    .chain(point, exit_lat);
                level_dag.add_dependency(to_map, pred, entry_point, &latency);
            }
        }
    }
}

/// Adds a dependency origination from a dim map.
// TODO(cleanup): refactor to reduce the number of parameters.
#[cfg_attr(feature="cargo-clippy", allow(clippy))]
fn apply_dim_map(device: &Device,
                 space: &SearchSpace,
                 local_info: &LocalInfo,
                 levels: &[Level],
                 code_points: &CodePointDag,
                 dim_map: &level::DimMap,
                 from_map: level::DagNodeId,
                 to_map: &[level::DagNodeId],
                 level_dag: &mut LevelDag) {
    // TODO(cc_perf): only predecessors that have an outgoing edge that is not already
    // accounted for by another predecessor in the current dependency map should be
    // considered.
    let predecessors = code_points.ids.iter()
        .filter(|&(p, _)| p.is_before_dims(space, levels, &dim_map.src_dims))
        .map(|(_, &id)| id);
    let src_point = code_points.ids[&CodePoint::Inst(dim_map.src)];
    let dst_point = code_points.ids[&CodePoint::Inst(dim_map.dst)];
    let latency_to_src = level_dag.dependencies(from_map).latency_to(src_point);
    let src_dst_latency = local_info.hw_pressure[&dim_map.src.into()]
        .bound(BottleneckLevel::Thread, &device.thread_rates());
    for pred in predecessors {
        let latency = unwrap!(latency_to_src[pred].clone())
            .chain(src_point, src_dst_latency.clone());
        level_dag.add_dependency(to_map, pred, dst_point, &latency);
    }
}

#[cfg(feature="cuda")]
#[cfg(test)]
mod cuda_tests {
    use codegen;
    use device::{Context, cuda, EvalMode};
    use env_logger;
    use helper::*;
    use model;
    use search_space::*;
    use super::*;

    #[test]
    fn partial_bound_0() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor);
        let z;
        let signature = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            z = builder.array::<f32>("z", 16).0;
            builder.get()
        };

        let mut builder = Builder::new(&signature, context.device());
        let size = builder.cst_size(4);

        let dim_x = builder.open_dim_ex(size.clone(), DimKind::THREAD);
        let dim_y = builder.open_dim_ex(size.clone(), DimKind::THREAD);
            builder.mov(&0f32);
        builder.close_dim(&dim_y);
        builder.close_dim(&dim_x);

        let dim_z = builder.open_dim_ex(size, DimKind::THREAD);
        let (addr, pattern) = builder.tensor_access(&"z", z, &ir::Type::F(32), &[&dim_z]);
        let st_z = builder.st(&addr, &0f32, pattern);

        builder.order(&dim_x, &dim_z, Order::BEFORE);
        builder.order(&dim_x, &dim_y, Order::OUTER);

        let partial_pressure = {
            let space = builder.get_clone();
            let local_info = LocalInfo::compute(&space, &context);
            trace!("partial nesting: {:?}", local_info.nesting[&st_z.into()]);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(3);

        builder.action(Action::ThreadMapping(dim_z, dim_x, ThreadMapping::MAPPED_OUT));
        let final_pressure = {
            let space = builder.get_clone();
            let local_info = LocalInfo::compute(&space, &context);
            trace!("final nesting: {:?}", local_info.nesting[&st_z.into()]);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(3);

        assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
                final_pressure, partial_pressure);
    }

    #[test]
    fn partial_bound_1() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor); 
        let z;
        let signature = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            z = builder.array::<f32>("z", 256).0;
            builder.get()
        };

        let mut builder = Builder::new(&signature, context.device());
        let size = builder.cst_size(256);
        let dim_x = builder.open_dim_ex(size.clone(), DimKind::THREAD);
            builder.mov(&0f32);
        builder.close_dim(&dim_x);

        let dim_z = builder.open_dim(size);
        let (addr, pattern) = builder.tensor_access(&"z", z, &ir::Type::F(32), &[&dim_z]);
        let st_z = builder.st(&addr, &0f32, pattern);

        let partial_pressure = {
            let space = builder.get_clone();
            let local_info = LocalInfo::compute(&space, &context);
            trace!("partial nesting: {:?}", local_info.nesting[&st_z.into()]);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(5);

        builder.action(Action::DimKind(dim_z, DimKind::THREAD));
        let final_pressure = {
            let space = builder.get();
            let local_info = LocalInfo::compute(&space, &context);
            trace!("final nesting: {:?}", local_info.nesting[&st_z.into()]);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(5);

        assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
                final_pressure, partial_pressure);
    }


    #[test]
    fn partial_bound_2() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor); 

        let (x, y, a);
        let signature = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            builder.scalar("m", 1i32 << 13);
            builder.scalar("n", 1i32 << 13);
            let m_size: tensor::DimSize = "m".into();
            let n_size: tensor::DimSize = "n".into();
            x = builder.tensor::<f32>("x", vec![n_size.clone()], true);
            a = builder.tensor::<f32>("a", vec![m_size.clone(), n_size], true);
            y = builder.tensor::<f32>("y", vec![m_size], false);
            builder.get()
        };

        let m_tiling = &[2];

        let mut builder = Builder::new(&signature, context.device());
        let ld_x = x.load(&[&[]], &mut builder);
        let ld_a = a.load(&[m_tiling, &[]], &mut builder);

        let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
        let init = builder.mov(&0f32);
        let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
        let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
        let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_n], ir::DimMapScope::Global,
                                &mut builder);
        let x_op = ld_x.dim_map(&[&acc_dim_n], ir::DimMapScope::Global, &mut builder);
        let acc = builder.mad(&a_op, &x_op, &Reduce(init));
        builder.close_dim(&acc_dim_n);

        let sum = tensor::VirtualTensor::new(acc, vec![acc_dim_m.clone()]);
        let st_y = sum.store(&y, &mut builder);

        builder.action(Action::DimKind(ld_a[0][1], DimKind::UNROLL));
        builder.action(Action::DimKind(init_dim_m[1], DimKind::UNROLL));
        builder.action(Action::DimKind(st_y[0][1], DimKind::UNROLL));

        builder.order(&acc_dim_n, &st_y.inst(), Order::BEFORE);
        builder.order(&ld_a[0][1], &ld_x.inst(), Order::BEFORE);
        builder.order(&acc_dim_m[1], &ld_x.inst(), Order::AFTER);

        builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CG));
        builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG));
        builder.action(Action::InstFlag(st_y.inst(), InstFlag::MEM_CS));

        let partial_bound = model::bound(&builder.get_clone(), &context);
        builder.action(Action::DimKind(ld_a[0][0], DimKind::BLOCK));

        let final_bound = model::bound(&builder.get(), &context);

        assert!(final_bound.value()*1.001 >= partial_bound.value(), "{} < {}",
                final_bound, partial_bound);

    }

    #[test]
    fn partial_bound_3() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor); 

        let a;
        let signature = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            a = builder.array::<f32>("a", 256).0;
            builder.get()
        };

        let mut builder = Builder::new(&signature, context.device());

        let size_m = builder.cst_size(256);
        let ld_a_dim = builder.open_tiled_dim(size_m, &[4]);
        let (addr, patt) = builder.tensor_access(&"a", a, &ir::Type::F(32), &[&ld_a_dim]);
        builder.ld(ir::Type::F(32), &addr, patt);
        builder.close_dim(&ld_a_dim);

        let size_n = builder.cst_size(4);
        let init_dim_m = builder.open_mapped_dim(&ld_a_dim);
        let init_dim_n = builder.open_dim(size_n);
        builder.mov(&0f32);

        builder.action(Action::DimKind(ld_a_dim[0], DimKind::THREAD));
        builder.action(Action::DimKind(ld_a_dim[1], DimKind::UNROLL));

        builder.action(Action::DimKind(init_dim_m[0], DimKind::THREAD));
        //builder.action(Action::DimKind(init_dim_m[1], DimKind::THREAD));
        builder.action(Action::ThreadMapping(ld_a_dim[0], init_dim_m[0],
                                             ThreadMapping::MAPPED));
        builder.action(Action::ThreadMapping(
                init_dim_m[0], init_dim_m[1], ThreadMapping::MAPPED_IN));
        builder.action(Action::DimKind(init_dim_n, DimKind::THREAD));

        let partial_pressure = {
            let space = builder.get_clone();
            let local_info = LocalInfo::compute(&space, &context);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(4);

        builder.action(Action::ThreadMapping(init_dim_n, ld_a_dim[0],
                                             ThreadMapping::MAPPED_IN));

        let final_pressure = {
            let space = builder.get();
            let local_info = LocalInfo::compute(&space, &context);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(4);
 
        assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
                final_pressure, partial_pressure);
    }

    #[test]
    fn partial_bound_4() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor); 

        let a: tensor::Tensor<f32>;
        let signature = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            a = builder.tensor::<f32>("a", vec![25.into(), 26.into(), 32.into()], true);
            builder.get()
        };

        let mut builder = Builder::new(&signature, context.device());
        let ld_a = a.load(&[&[], &[], &[]], &mut builder);

        builder.action(Action::DimKind(ld_a[0][0], DimKind::THREAD));
        builder.action(Action::DimKind(ld_a[1][0], DimKind::THREAD));
        builder.action(Action::DimKind(ld_a[2][0], DimKind::LOOP));

        builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CS));

        let partial_pressure = {
            let space = builder.get_clone();
            let local_info = LocalInfo::compute(&space, &context);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(3);

        builder.action(Action::ThreadMapping(ld_a[0][0], ld_a[1][0],
                                             ThreadMapping::MAPPED_IN));

        let final_pressure = {
            let space = builder.get();
            let local_info = LocalInfo::compute(&space, &context);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(3);

        assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
                final_pressure, partial_pressure);
    }

    #[test]
    fn partial_bound_5() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor); 

        let (signature, a) = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            let a = tensor::TensorBuilder::new("a", vec![13.into(), 32.into()])
                .stride_dim(1).finish::<f32, _>(&mut builder);
            (builder.get(), a)
        };

        let mut builder = Builder::new(&signature, context.device());

        let ld_a = a.load(&[&[]], &mut builder);
        let dim1 = builder.open_dim_ex(ir::Size::new_const(26), DimKind::THREAD);
        let _ = builder.mov(&0f32);

        builder.order(&ld_a.inst(), &dim1, Order::AFTER);

        let partial_pressure = {
            let space = builder.get_clone();
            let local_info = LocalInfo::compute(&space, &context);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(4);

        builder.action(Action::DimKind(ld_a[0][0], DimKind::UNROLL));

        let final_pressure = {
            let space = builder.get();
            let local_info = LocalInfo::compute(&space, &context);
            sum_pressure(&context, &space, &local_info,
                         BottleneckLevel::Global, &[], &ir::Size::one())
        }.get_bottleneck(4);

        assert!(final_pressure*1.001 >= partial_pressure, "{} < {}",
                final_pressure, partial_pressure);
    }

    #[test]
    fn final_bound_0() {
        let _ = env_logger::try_init();
        let executor = cuda::Executor::init();
        let mut context = cuda::Context::new(&executor);

        let (x, y, z);
        let signature = {
            let mut builder = SignatureBuilder::new("test", &mut context);
            builder.scalar("n", 1 << 25);
            let n_size: tensor::DimSize = "n".into();
            x = builder.tensor::<f32>("x", vec![n_size.clone()], true);
            y = builder.tensor::<f32>("y", vec![n_size.clone()], true);
            z = builder.tensor::<f32>("z", vec![n_size], false);
            builder.get()
        };

        let tiling = &[1024, 4];
        let mut builder = Builder::new(&signature, context.device());

        let ld_x = x.load(&[tiling], &mut builder);
        let ld_y = y.load(&[tiling], &mut builder);
        let mad_dim = builder.open_mapped_dim(&ld_x[0]);
        let x_op = ld_x.dim_map(&[&mad_dim], ir::DimMapScope::Global, &mut builder);
        let y_op = ld_y.dim_map(&[&mad_dim], ir::DimMapScope::Global, &mut builder);
        let mad = tensor::VirtualTensor::new(
            builder.mad(&x_op, &4.33f32, &y_op), vec![mad_dim.clone()]);
        let st_z = mad.store(&z, &mut builder);

        builder.action(Action::DimKind(ld_x[0][2], DimKind::VECTOR));
        builder.action(Action::DimKind(ld_x[0][1], DimKind::THREAD));
        builder.action(Action::DimKind(ld_x[0][0], DimKind::BLOCK));
        builder.action(Action::DimKind(ld_y[0][2], DimKind::VECTOR));
        builder.action(Action::DimKind(ld_y[0][1], DimKind::THREAD));
        builder.action(Action::DimKind(mad_dim[1], DimKind::THREAD));
        builder.action(Action::DimKind(mad_dim[2], DimKind::UNROLL));
        builder.action(Action::DimKind(st_z[0][1], DimKind::THREAD));
        builder.action(Action::DimKind(st_z[0][2], DimKind::VECTOR));
        builder.order(&ld_x[0][2], &ld_y.inst(), Order::BEFORE);
        builder.order(&ld_x[0][1], &ld_y.inst(), Order::BEFORE);
        builder.order(&ld_y[0][1], &mad.inst(), Order::BEFORE);
        builder.order(&mad_dim[1], &st_z.inst(), Order::OUTER);
        builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CS));
        builder.action(Action::InstFlag(ld_y.inst(), InstFlag::MEM_CG));
        builder.action(Action::InstFlag(st_z.inst(), InstFlag::MEM_CG));
        let space = builder.get();
        let bound = model::bound(&space, &context);
        let kernel = codegen::Function::build(&space);
        let eval = unwrap!(context.evaluate(&kernel, EvalMode::TestBound));
        assert!(eval * 1.001 >= bound.value(), "{:.2e} < {}", eval, bound);
    }
}
