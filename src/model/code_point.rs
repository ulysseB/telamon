//! Locations of the code relevant to compute latency.
use ir;
use itertools::Itertools;
use model::Level;
use search_space::{Order, SearchSpace, Domain};
use std::cmp::Ordering;
use utils::*;

/// A poi t of the code that is relevant to compute latency.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum CodePoint {
    /// An instruction.
    Inst(ir::InstId),
    /// The entry in a set of loops, referenced by their index in the loop level order.
    LevelEntry(usize),
    /// The exit of a set of loops, referenced by their index in the loop level order.
    LevelExit(usize),
}

impl CodePoint {
    /// Returns the conditions on which the `CodePoint` is lesser and greater than another.
    fn lesser_greater_conds(&self) -> (Order, Order) {
        match *self {
            CodePoint::Inst(_) => (Order::BEFORE, Order::AFTER),
            CodePoint::LevelEntry(_) => (Order::BEFORE | Order::OUTER, Order::AFTER),
            CodePoint::LevelExit(_) => (Order::BEFORE, Order::AFTER | Order::OUTER),
        }
    }

    /// Returns the basic blocks associated with the code point.
    fn blocks(&self, levels: &[Level]) -> Vec<ir::BBId> {
        match *self {
            CodePoint::Inst(id) => vec![id.into()],
            CodePoint::LevelEntry(id) | CodePoint::LevelExit(id) =>
                levels[id].dims.iter().map(|&id| id.into()).collect(),
        }
    }

    /// Indicates if the code point is before a given set of dimensions.
    pub fn is_before_dims(&self, space: &SearchSpace, levels: &[Level],
                      dims: &[ir::DimId]) -> bool {
        if *self == CodePoint::LevelExit(0) { return false; }
        let lesser_conds = self.lesser_greater_conds().0;
        self.blocks(levels).into_iter().cartesian_product(dims).all(|(lhs, &rhs)| {
            if lhs == rhs.into() { false } else {
                lesser_conds.contains(space.domain().get_order(lhs, rhs.into()))
            }
        })
    }
}

/// Generates the list of code points to consider.
fn generate(space: &SearchSpace, levels: &[Level]) -> Vec<CodePoint> {
    space.ir_instance().insts().map(|x| CodePoint::Inst(x.id()))
        .chain((0..levels.len()).map(CodePoint::LevelEntry))
        .chain((0..levels.len()).map(CodePoint::LevelExit)).collect()
}

/// Exposes data dependencies between code points and provides a valid total order
/// between them.
#[derive(Debug)]
pub struct CodePointDag {
    pub dag: Dag<CodePoint>,
    pub ids: HashMap<CodePoint, usize>
}

impl CodePointDag {
    /// Builds a `CodePointDag` for the given loop levels.
    pub fn build(space: &SearchSpace, levels: &[Level]) -> Self {
        let dag = code_point_dag(space, levels);
        let ids = code_point_ids(&dag);
        CodePointDag { dag, ids }
    }

    /// Returns the number of points in the DAG.
    pub fn len(&self) -> usize { self.ids.len() }
}

/// Generates a directed acyclic graph representing the ordering dependencies between the
/// code points.
fn code_point_dag(space: &SearchSpace, levels: &[Level]) -> Dag<CodePoint> {
    let code_points = generate(space, levels);
    Dag::from_order(code_points, |&lhs, &rhs| {
        use self::CodePoint::*;
        // If the points are not from a common level or fromthe root level, compare the
        // basic blocks that compose them.
        let lhs_conds = lhs.lesser_greater_conds();
        let rhs_conds = rhs.lesser_greater_conds();
        let mut lesser_cond = lhs_conds.0 | rhs_conds.1.inverse();
        let mut greater_cond = lhs_conds.1 | rhs_conds.0.inverse();
        // Check if the points are from the root or a common level.
        let if_equals = match (lhs, rhs) {
            (LevelEntry(0), _) | (_, LevelExit(0)) => return Some(Ordering::Less),
            (LevelExit(0), _) | (_, LevelEntry(0)) => return Some(Ordering::Greater),
            (LevelEntry(_), LevelExit(_)) => {
                lesser_cond.insert(Order::MERGED);
                Ordering::Less
            },
            (LevelExit(_), LevelEntry(_)) => {
                greater_cond.insert(Order::MERGED);
                Ordering::Greater
            },
            _ => Ordering::Equal,
        };
        let lhs_blocks = lhs.blocks(levels).into_iter();
        let out = lhs_blocks.cartesian_product(rhs.blocks(levels)).map(|(lhs, rhs)| {
            convert_order(space, lhs, rhs, lesser_cond, greater_cond, if_equals)
        }).fold1(|x, y| { if x == y { x } else { None } });
        //trace!("ord {:?} {:?} {:?}", lhs, out, rhs);
        unwrap!(out)
    })
}

/// Convert the basic blocks order into `std::cmp::Ordering`. `lesser_cond` and
/// `greater_cond` indicates respectively the conditions for `lhs` to be considered
/// lesser and greater than `rhs`. `if_equals` indicates the order to return if `lhs`
/// and `rhs` are equals.
fn convert_order(space: &SearchSpace,
                 lhs: ir::BBId, rhs: ir::BBId,
                 lesser_cond: Order,
                 greater_cond: Order,
                 if_equals: Ordering) -> Option<Ordering> {
    if lhs == rhs { return Some(if_equals); }
    let order = space.domain().get_order(lhs, rhs);
    if lesser_cond.contains(order) { Some(Ordering::Less) }
    else if greater_cond.contains(order) { Some(Ordering::Greater) }
    else { None }
}

/// Creates a map from code point to code point IDs.
fn code_point_ids(code_points: &Dag<CodePoint>) -> HashMap<CodePoint, usize> {
    code_points.nodes().iter().enumerate().map(|(x, &y)| (y, x)).collect()
}
