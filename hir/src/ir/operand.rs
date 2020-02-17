//! Describes the different kinds of operands an instruction can have.
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use self::Operand::*;
use crate::ir::{self, AccessId, DimMap, InstId, Instruction, Parameter, Type};
use fxhash::FxHashMap;
use itertools::Itertools;
use num::bigint::BigInt;
use num::rational::Ratio;
use num::traits::{Signed, Zero};
use utils::unwrap;

/// Trait for representing integer literals which can be used in the IR.
///
/// This is meant to be used as a trait bound in various functions accepting literals as arguments,
/// and allows to write generic functions accepting both Rust primitive integer types and IR
/// integers.
pub trait IntLiteral<'a> {
    /// Decompose `self` into a big integer and a bit width.
    ///
    /// The resulting value should be understood as having type `Type::I(bit_width)` and thus must
    /// be representable with `bit_width` bits.
    fn decompose(self) -> (Cow<'a, BigInt>, u16);
}

impl<'a> IntLiteral<'a> for (BigInt, u16) {
    fn decompose(self) -> (Cow<'a, BigInt>, u16) {
        (Cow::Owned(self.0), self.1)
    }
}

impl<'a> IntLiteral<'a> for (&'a BigInt, u16) {
    fn decompose(self) -> (Cow<'a, BigInt>, u16) {
        (Cow::Borrowed(self.0), self.1)
    }
}

/// Trait for representing floating-point literals which can be used in the IR.
///
/// This is meant to be used as a trait bound in various functions accepting literals as arguments,
/// and allows to write generic functions accepting both Rust primitive floating-point types and IR
/// floats.
pub trait FloatLiteral<'a> {
    /// Decompose `self` into a big rational and a bit width.
    ///
    /// The resulting value should be understood as having type `Type::F(bit_width)` and thus must
    /// be representable with `bit_width` bits.
    fn decompose(self) -> (Cow<'a, Ratio<BigInt>>, u16);
}

impl<'a> FloatLiteral<'a> for (Ratio<BigInt>, u16) {
    fn decompose(self) -> (Cow<'a, Ratio<BigInt>>, u16) {
        (Cow::Owned(self.0), self.1)
    }
}

impl<'a> FloatLiteral<'a> for (&'a Ratio<BigInt>, u16) {
    fn decompose(self) -> (Cow<'a, Ratio<BigInt>>, u16) {
        (Cow::Borrowed(self.0), self.1)
    }
}

macro_rules! impl_int_literal {
    ($($t:ty),*) => {
        $(impl<'a> IntLiteral<'a> for $t {
            fn decompose(self) -> (Cow<'a, BigInt>, u16) {
                (
                    Cow::Owned(self.into()),
                    8 * std::mem::size_of::<$t>() as u16,
                )
            }
        })*
    };
}

impl_int_literal!(i8, i16, i32, i64);

macro_rules! impl_float_literal {
    ($($t:ty),*) => {
        $(impl<'a> FloatLiteral<'a> for $t {
            fn decompose(self) -> (Cow<'a, Ratio<BigInt>>, u16) {
                (
                    Cow::Owned(Ratio::from_float(self).unwrap()),
                    8 * std::mem::size_of::<$t>() as u16,
                )
            }
        })*
    };
}

impl_float_literal!(f32, f64);

#[derive(Clone, Debug)]
pub struct LoweringMap {
    /// Memory ID to use for the temporary array
    mem_id: ir::MemId,
    /// Instruction ID to use for the `store` instruction when
    /// lowering.
    st_inst: ir::InstId,
    /// Maps the lhs dimensions in `map` to their lowered dimension.
    st_map: FxHashMap<ir::DimId, (ir::DimId, ir::DimMappingId)>,
    /// Instruction ID to use for the `load` instruction when
    /// lowering.
    ld_inst: ir::InstId,
    /// Maps the rhs dimensions in `map` to their lowered dimension.
    ld_map: FxHashMap<ir::DimId, (ir::DimId, ir::DimMappingId)>,
}

impl LoweringMap {
    /// Creates a new lowering map from an existing dimension map and
    /// a counter. This allocates new IDs for the new
    /// dimensions/instructions/memory locations that will be used
    /// when lowering the DimMap.
    pub fn for_dim_map(dim_map: &DimMap, cnt: &mut ir::Counter) -> LoweringMap {
        let mem_id = cnt.next_mem();
        let st_inst = cnt.next_inst();
        let ld_inst = cnt.next_inst();
        let (st_map, ld_map) = dim_map
            .iter()
            .cloned()
            .map(|(src, dst)| {
                let st_dim = cnt.next_dim();
                let ld_dim = cnt.next_dim();
                let st_mapping = cnt.next_dim_mapping();
                let ld_mapping = cnt.next_dim_mapping();
                ((src, (st_dim, st_mapping)), (dst, (ld_dim, ld_mapping)))
            })
            .unzip();

        LoweringMap {
            mem_id,
            st_inst,
            st_map,
            ld_inst,
            ld_map,
        }
    }

    /// Returns lowering information about the dim_map. The returned
    /// `LoweredDimMap` object should not be used immediately: it
    /// refers to fresh IDs that are not activated in the
    /// ir::Function. The appropriate instructions need to be built
    /// and stored with the corresponding IDs.
    pub(crate) fn lower(&self, map: &DimMap) -> ir::LoweredDimMap {
        let (st_dims_mapping, ld_dims_mapping) = map
            .iter()
            .map(|&(src, dst)| {
                let &(st_dim, st_mapping) = unwrap!(self.st_map.get(&src));
                let &(ld_dim, ld_mapping) = unwrap!(self.ld_map.get(&dst));
                ((st_mapping, [src, st_dim]), (ld_mapping, [dst, ld_dim]))
            })
            .unzip();
        ir::LoweredDimMap {
            mem: self.mem_id,
            store: self.st_inst,
            load: self.ld_inst,
            st_dims_mapping,
            ld_dims_mapping,
        }
    }
}

/// Indicates how dimensions can be mapped. The `L` type indicates how
/// to lower mapped dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimMapScope<L> {
    /// The dimensions are mapped within registers, without producing syncthreads.
    Local,
    /// The dimensions are mapped within registers.
    Thread,
    /// The dimensions are mapped, possibly using temporary
    /// memory. The parameter `L` is used to indicate how the mapping
    /// should be lowered. It is `()` when building the function
    /// (lowering is not possible at that time), and a `LoweringMap`
    /// instance when exploring which indicates what IDs to use for
    /// the new objects.
    Global(L),
}

/// Represents an instruction operand.
#[derive(Clone, Debug)]
pub enum Operand<L = LoweringMap> {
    /// An integer constant, on a given number of bits.
    Int(BigInt, u16),
    /// A float constant, on a given number of bits.
    Float(Ratio<BigInt>, u16),
    /// A value produced by an instruction. The boolean indicates if the `DimMap` can be
    /// lowered.
    Inst(InstId, Type, DimMap, DimMapScope<L>),
    /// The current index in a loop.
    Index(ir::DimId),
    /// A parameter of the function.
    Param(Arc<Parameter>),
    /// The address of a memory block.
    Addr(ir::MemId, AccessType),
    /// The value of the current instruction at a previous iteration.
    Reduce(InstId, Type, DimMap, Vec<ir::DimId>),
    /// A variable increased by a fixed amount at every step of some loops.
    InductionVar(ir::IndVarId, Type),
    /// A variable, stored in register.
    Variable(ir::VarId, Type),
    /// Computed access to a memory location.  This type is the type of the address, not of the
    /// value.
    ComputedAddress(AccessId, Type),
}

/// Type of access.  This is used for double-buffering
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccessType {
    Load,
    Store,
}

impl<L> Operand<L> {
    /// Returns the type of the `Operand`.
    pub fn t(&self) -> Type {
        match self {
            Int(_, n_bit) => Type::I(*n_bit),
            Float(_, n_bit) => Type::F(*n_bit),
            Addr(mem, _) => ir::Type::PtrTo(*mem),
            Index(..) => Type::I(32),
            Param(p) => p.t,
            Variable(_, t)
            | Inst(_, t, ..)
            | Reduce(_, t, ..)
            | InductionVar(_, t)
            | ComputedAddress(_, t) => *t,
        }
    }

    /// Create an operand from an instruction.
    pub fn new_inst(
        inst: &Instruction<L>,
        dim_map: DimMap,
        mut scope: DimMapScope<L>,
    ) -> Self {
        // A temporary array can only be generated if the type size is known.
        if let DimMapScope::Global(_) = scope {
            if unwrap!(inst.t()).len_byte().is_none() {
                scope = DimMapScope::Thread;
            }
        }

        Inst(inst.id(), unwrap!(inst.t()), dim_map, scope)
    }

    /// Creates a reduce operand from an instruction and a set of dimensions to reduce on.
    pub fn new_reduce(
        init: &Instruction<L>,
        dim_map: DimMap,
        dims: Vec<ir::DimId>,
    ) -> Self {
        Reduce(init.id(), unwrap!(init.t()), dim_map, dims)
    }

    /// Creates a new Int operand and checks its number of bits.
    pub fn new_int<'a, T: IntLiteral<'a>>(lit: T) -> Self {
        let (val, len) = lit.decompose();
        assert!(num_bits(&val) <= len);

        Int(val.into_owned(), len)
    }

    /// Creates a new Float operand.
    pub fn new_float<'a, T: FloatLiteral<'a>>(lit: T) -> Self {
        let (val, len) = lit.decompose();
        Float(val.into_owned(), len)
    }

    /// Renames a basic block id.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) {
        match *self {
            Inst(_, _, ref mut dim_map, _) | Reduce(_, _, ref mut dim_map, _) => {
                dim_map.merge_dims(lhs, rhs);
            }
            _ => (),
        }
    }

    /// Indicates if a `DimMap` should be lowered if lhs and rhs are not mapped.
    pub fn should_lower_map(&self, lhs: ir::DimId, rhs: ir::DimId) -> bool {
        match *self {
            Inst(_, _, ref dim_map, _) | Reduce(_, _, ref dim_map, _) => dim_map
                .iter()
                .any(|&pair| pair == (lhs, rhs) || pair == (rhs, lhs)),
            _ => false,
        }
    }

    /// If the operand is a reduction, returns the instruction initializing the reduction.
    pub fn as_reduction(&self) -> Option<(InstId, &DimMap, &[ir::DimId])> {
        if let Reduce(id, _, ref dim_map, ref dims) = *self {
            Some((id, dim_map, dims))
        } else {
            None
        }
    }

    /// Indicates if the operand stays constant during the execution.
    pub fn is_constant(&self) -> bool {
        match self {
            Int(..) | Float(..) | Addr(..) | Param(..) => true,
            Index(..) | Inst(..) | Reduce(..) | InductionVar(..) | Variable(..)
            | ComputedAddress(..) => false,
        }
    }

    /// Returns the list of dimensions mapped together by the operand.
    pub fn mapped_dims(&self) -> Option<&DimMap> {
        match self {
            Inst(_, _, dim_map, _) | Reduce(_, _, dim_map, _) => Some(dim_map),
            _ => None,
        }
    }
}

impl Operand<()> {
    pub fn freeze(self, cnt: &mut ir::Counter) -> Operand {
        match self {
            Int(val, len) => Int(val, len),
            Float(val, len) => Float(val, len),
            Inst(id, t, dim_map, DimMapScope::Global(())) => {
                let lowering_map = LoweringMap::for_dim_map(&dim_map, cnt);
                Inst(id, t, dim_map, DimMapScope::Global(lowering_map))
            }
            Inst(id, t, dim_map, DimMapScope::Local) => {
                Inst(id, t, dim_map, DimMapScope::Local)
            }
            Inst(id, t, dim_map, DimMapScope::Thread) => {
                Inst(id, t, dim_map, DimMapScope::Thread)
            }
            Variable(val, t) => Variable(val, t),
            Index(id) => Index(id),
            Param(param) => Param(param),
            Addr(id, at) => Addr(id, at),
            Reduce(id, t, dim_map, dims) => Reduce(id, t, dim_map, dims),
            InductionVar(id, t) => InductionVar(id, t),
            ComputedAddress(address, t) => ComputedAddress(address, t),
        }
    }
}

impl<L> fmt::Display for Operand<L> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Int(val, len) => write!(fmt, "{}u{}", val, len),
            Float(val, len) => write!(fmt, "{}f{}", val, len),
            Inst(id, _t, dim_map, _scope) => write!(fmt, "{:?} [{}]", id, dim_map),
            Index(id) => write!(fmt, "{}", id),
            Param(param) => write!(fmt, "{}", param),
            Addr(id, _at) => write!(fmt, "{}", id),
            Reduce(id, _t, dim_map, dims) => {
                write!(fmt, "reduce({:?}, {:?}) [{}]", id, dims, dim_map)
            }
            InductionVar(_id, _t) => write!(fmt, "ind"),
            Variable(var, t) => write!(fmt, "({}){}", t, var),
            ComputedAddress(aid, _t) => write!(fmt, "{}", aid),
        }
    }
}

impl<L> ir::IrDisplay<L> for Operand<L> {
    fn fmt(&self, fmt: &mut fmt::Formatter, fun: &ir::Function<L>) -> fmt::Result {
        match self {
            Int(val, len) => write!(fmt, "{}u{}", val, len),
            Float(val, len) => write!(fmt, "{}f{}", val, len),
            Inst(id, _t, dim_map, _scope) => {
                let source_dims = fun
                    .inst(*id)
                    .iteration_dims()
                    .iter()
                    .sorted()
                    .collect::<Vec<_>>();
                let mapping = dim_map.iter().cloned().collect::<FxHashMap<_, _>>();

                write!(
                    fmt,
                    "{:?}[{}]",
                    id,
                    source_dims
                        .into_iter()
                        .map(|id| mapping.get(id).unwrap_or(id))
                        .format(", ")
                )
            }
            Index(id) => write!(fmt, "{}", id),
            Param(param) => write!(fmt, "{}", param),
            Addr(id, _at) => write!(fmt, "{}", id),
            Reduce(id, _t, dim_map, dims) => {
                let source_dims = fun
                    .inst(*id)
                    .iteration_dims()
                    .iter()
                    .sorted()
                    .collect::<Vec<_>>();
                let mapping = dim_map.iter().cloned().collect::<FxHashMap<_, _>>();
                write!(
                    fmt,
                    "reduce({:?}[{}], {:?})",
                    id,
                    source_dims
                        .into_iter()
                        .map(|id| mapping.get(id).unwrap_or(id))
                        .format(", "),
                    dims
                )
            }
            InductionVar(id, _t) => {
                write!(fmt, "{}", fun.induction_var(*id).display(fun))
            }
            Variable(var, t) => write!(fmt, "({}){}", t, var),
            ComputedAddress(aid, _t) => {
                write!(fmt, "{}", fun.accesses()[*aid].display(fun))
            }
        }
    }
}

/// Returns the number of bits necessary to encode a `BigInt`.
fn num_bits(val: &BigInt) -> u16 {
    let mut num_bits = if val.is_negative() { 1 } else { 0 };
    let mut rem = val.abs();
    while !rem.is_zero() {
        rem >>= 1;
        num_bits += 1;
    }
    num_bits
}
