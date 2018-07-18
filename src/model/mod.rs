//! Building Blocks for lower bound performance models.
mod code_point;
mod dependency_map;
mod hw_pressure;
mod level;
mod local_info;

mod test_cuda;

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
