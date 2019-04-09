//! Code generation and candidate evaluation for specific targets.
pub mod fake;

mod argument;
mod context;

pub use self::argument::{ArrayArgument, ArrayArgumentExt, ScalarArgument};
pub use self::context::{
    ArgMap, ArgMapExt, AsyncCallback, AsyncEvaluator, Context, EvalMode, KernelEvaluator,
    Stabilizer,
};

use crate::codegen::Function;
use crate::ir;
use crate::model::{self, HwPressure, Nesting};
use crate::search_space::*;
use std::io::Write;
use utils::*;

/// Holds the specifications of a target.
#[allow(clippy::trivially_copy_pass_by_ref)]
pub trait Device: Sync {
    /// Prints the code corresponding to a device `Function`.
    fn print(&self, function: &Function, out: &mut dyn Write);
    /// Indicates if a `Type` can be implemented on the device.
    fn check_type(&self, t: ir::Type) -> Result<(), ir::TypeError>;
    /// Returns the maximal number of block dimensions.
    fn max_block_dims(&self) -> u32;
    /// The maximal size inner block dimensions can have.
    fn max_inner_block_size(&self) -> u32;
    /// Returns the maximal number of threads.
    fn max_threads(&self) -> u32;
    /// Returns the maximal unrolling factor.
    fn max_unrolling(&self) -> u32;
    /// Indicates if the device uses vector registers or has imlicit gathers and scatters
    /// in vector instructions.
    fn has_vector_registers(&self) -> bool;
    /// Indicates if the operator can be vectorized along the dimension.
    fn can_vectorize(&self, dim: &ir::Dimension, op: &ir::Operator) -> bool;
    /// Indicates the maximal vectorization factor for the given operator.
    fn max_vectorization(&self, op: &ir::Operator) -> [u32; 2];
    /// Returns the amount of shared memory available for each thread block.
    fn shared_mem(&self) -> u32;
    /// Indicates the type of the pointer for the given memory space.
    fn pointer_type(&self, mem_space: MemSpace) -> ir::Type;
    /// Indicates the memory flags supported by the operator.
    fn supported_mem_flags(&self, op: &ir::Operator) -> InstFlag;
    /// Returns the name of the device.
    fn name(&self) -> &str;

    /// Returns the pressure cause by a `Statement`. For a dimension, returns the pressure
    /// for the full loop execution.
    fn hw_pressure(
        &self,
        space: &SearchSpace,
        dim_sizes: &FnvHashMap<ir::DimId, model::size::Range>,
        nesting: &FnvHashMap<ir::StmtId, Nesting>,
        bb: &dyn ir::Statement,
        ctx: &dyn Context,
    ) -> HwPressure;
    /// Returns the pressure produced by a single iteration of a loop and the latency
    /// overhead of iterations.
    fn loop_iter_pressure(&self, kind: DimKind) -> (HwPressure, HwPressure);
    /// Returns the processing rates of a single thread, in units/ns
    fn thread_rates(&self) -> HwPressure;
    /// Returns the processing rates of a single block, in units/ns.
    fn block_rates(&self) -> HwPressure;
    /// Returns the processing rates of the whole accelerator un units/ns.
    fn total_rates(&self) -> HwPressure;
    /// Returns the names of potential bottlenecks.
    fn bottlenecks(&self) -> &[&'static str];
    /// Returns the number of blocks that can be executed in parallel on the device.
    fn block_parallelism(&self, space: &SearchSpace) -> u32;
    /// Returns the pressure caused by an additive induction variable level.
    fn additive_indvar_pressure(&self, t: &ir::Type) -> HwPressure;
    /// Returns the pressure caused by a multiplicative induction variable level.
    fn multiplicative_indvar_pressure(&self, t: &ir::Type) -> HwPressure;
    /// Adds the overhead (per instance) due to partial wraps and predicated dimensions to
    /// the pressure. If the instruction is not predicated, `predicated_dims_size` should
    /// be `1`.
    fn add_block_overhead(
        &self,
        max_active_threads: model::size::FactorRange,
        max_threads: model::size::FactorRange,
        predication_factor: model::size::Range,
        pressure: &mut HwPressure,
    );

    /// Lowers a type using the memory space information. Returns `None` if some
    /// information is not yet specified.
    fn lower_type(&self, t: ir::Type, space: &SearchSpace) -> Option<ir::Type>;

    /// Builds and outputs a constrained IR instance.
    fn gen_code(&self, implementation: &SearchSpace, out: &mut dyn Write) {
        let code = Function::build(implementation);
        self.print(&code, out);
    }
}
