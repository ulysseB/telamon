//! Utilities to allocate and operate on tensors.
use device::{ScalarArgument, ArrayArgument, Context, read_array};
use helper::{Builder, DimGroup, MetaDimension};
use ir;
use itertools::Itertools;
use ndarray::{self, ArrayD};
use search_space::{Domain, InstFlag};
use std;

/// A dimension size, before tiling.
#[derive(Copy, Clone)]
pub enum DimSize<'a> { Const(u32), Param(&'a str) }

impl<'a> DimSize<'a> {
    /// Convert the size into the size type used by the IR.
    pub fn into_ir_size<'b>(self, builder: &Builder<'b>) -> ir::Size<'b> {
        match self {
            DimSize::Const(size) => builder.cst_size(size),
            DimSize::Param(p) => builder.param_size(p),
        }
    }

    /// Converts the size into a numerical value for a given context.
    pub fn into_num(self, context: &Context) -> u32 {
        match self {
            DimSize::Const(s) => s,
            DimSize::Param(p) => unwrap!(context.param_as_size(p)),
        }
    }
}

impl<'a> From<u32> for DimSize<'a> {
    fn from(size: u32) -> Self { DimSize::Const(size) }
}

impl<'a> From<&'a str> for DimSize<'a> {
    fn from(param: &'a str) -> Self { DimSize::Param(param) }
}

/// A tensor allocated in main memory.
pub struct Tensor<'a, S: ScalarArgument> {
    name: &'a str,
    mem_id: ir::mem::Id,
    array: std::sync::Arc<ArrayArgument + 'a>,
    dim_sizes: Vec<DimSize<'a>>,
    read_only: bool,
    s: std::marker::PhantomData<S>,
}

impl<'a, S> Tensor<'a, S> where S: ScalarArgument {
    /// Allocates a new `Tensor` in the context.
    pub fn new(name: &'a str,
               dim_sizes: Vec<DimSize<'a>>,
               read_only: bool,
               mem_id: ir::mem::Id,
               array: std::sync::Arc<ArrayArgument + 'a>) -> Self {
        Tensor { name, mem_id, dim_sizes, read_only, array, s: std::marker::PhantomData }
    }

    /// Creates a `VirtualTensor` that contains the values of `self`, loaded in registers.
    pub fn load(&self, tiling: &[&[u32]], builder: &mut Builder) -> VirtualTensor {
        let dims = self.dim_sizes.iter().zip_eq(tiling).map(|(&size, &tiling)| {
            let size = size.into_ir_size(builder);
            builder.open_tiled_dim(size, tiling)
        }).collect_vec();
        let (ptr, pat) = {
            let dims = dims.iter().map(|d| d as &MetaDimension).collect_vec();
            builder.tensor_access(&self.name, self.mem_id, &S::t(), &dims)
        };
        let flag = if self.read_only { InstFlag::ALL } else { InstFlag::MEM_COHERENT };
        let inst = builder.ld_ex(S::t(), &ptr, pat, flag);
        for dim in &dims { builder.close_dim(dim); }
        VirtualTensor { inst, dims }
    }

    /// Reads the tensor value in the context and copies it on the host.
    pub fn read_to_host(&self, context: &Context) -> ArrayD<S> {
        let raw = read_array::<S>(self.array.as_ref());
        let sizes = self.dim_sizes.iter().map(|s| s.into_num(context) as usize)
            .collect_vec();
        unwrap!(ndarray::ArrayBase::from_shape_vec(sizes, raw))
    }
}

/// A tensor loaded in registers.
pub struct VirtualTensor {
    inst: ir::InstId,
    dims: Vec<DimGroup>,
}

impl VirtualTensor {
    /// Creates a new `VirtualTensor`.
    pub fn new(inst: ir::InstId, dims: Vec<DimGroup>) -> Self {
        VirtualTensor { inst, dims }
    }

    /// Creates an operand that yeilds the values of the tensor in the given loop nest.
    pub fn dim_map<'a>(&self,
                       dims: &[&MetaDimension],
                       scope: ir::DimMapScope,
                       builder: &mut Builder<'a>) -> ir::Operand<'a>
    {
        let mapping = self.dims.iter().map(|x| x as &MetaDimension)
            .zip_eq(dims.iter().cloned()).collect_vec();
        builder.dim_map(self.inst, &mapping, scope)
    }

    /// Stores the `VirtualTensor` in memory.
    pub fn store<S>(&self, tensor: &Tensor<S>, builder: &mut Builder) -> VirtualTensor
        where S: ScalarArgument
    {
        assert!(!tensor.read_only);
        let new_dims = self.dims.iter().map(|dim| builder.open_mapped_dim(dim))
            .collect_vec();
        let (ptr, pat) = {
            let dims = new_dims.iter().map(|d| d as &MetaDimension).collect_vec();
            builder.tensor_access(&tensor.name, tensor.mem_id, &S::t(), &dims)
        };
        let inst = builder.st(&ptr, &self.inst, pat);
        for dim in &new_dims { builder.close_dim(dim); }
        VirtualTensor { inst, dims: new_dims }
    }

    /// Returns the underlying instruction.
    pub fn inst(&self) -> ir::InstId { self.inst }
}

impl std::ops::Index<usize> for VirtualTensor {
    type Output = DimGroup;

    fn index(&self, idx: usize) -> &Self::Output { &self.dims[idx] }
}
