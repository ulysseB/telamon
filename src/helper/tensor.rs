//! Utilities to allocate and operate on tensors.
use device::{ArgMap, ScalarArgument, ArrayArgument, Context, read_array};
use helper::{Builder, LogicalDim, MetaDimension, SignatureBuilder};
use ir;
use itertools::Itertools;
use ndarray::{self, ArrayD};
use search_space::InstFlag;
use std;
use utils::*;

/// A dimension size, before tiling.
#[derive(Clone)]
pub struct DimSize<'a> {
    factor: u32,
    params: Vec<&'a str>,
}

impl<'a> DimSize<'a> {
    /// Convert the size into the size type used by the IR.
    pub fn into_ir_size<'b>(&self, builder: &Builder<'b>) -> ir::Size<'b> {
        let params = self.params.iter().map(|p| builder.find_param(p)).collect();
        ir::Size::new(self.factor, params)
    }

    /// Converts the size into a numerical value for a given context.
    pub fn eval(&self, context: &Context) -> u32 {
        self.params.iter().map(|p| unwrap!(context.param_as_size(p)))
            .product::<u32>() * self.factor
    }
}

impl<'a> From<u32> for DimSize<'a> {
    fn from(size: u32) -> Self { DimSize { factor: size, params: vec![] } }
}

impl<'a> From<&'a str> for DimSize<'a> {
    fn from(param: &'a str) -> Self { DimSize { factor: 1, params: vec![param] } }
}

/// An helper to build a tensor.
pub struct TensorBuilder<'a> {
    name: &'a str,
    read_only: bool,
    storage_dims: Vec<DimSize<'a>>,
    exposed_dims: Vec<usize>,
}

impl<'a> BuilderTrait for TensorBuilder<'a> { }

impl<'a> TensorBuilder<'a> {
    /// Start building a `Tensor` with the given logical layout.
    pub fn new(name: &'a str, storage_dims: Vec<DimSize<'a>>) -> Self {
        let exposed_dims = (0..storage_dims.len()).collect();
        TensorBuilder {
            name, storage_dims, exposed_dims,
            read_only: true,
        }
    }

    /// Swap two dimensions in the memory layout of the tensor. Keeps the logical layout
    /// untouched.
    pub fn transpose(&mut self, lhs: usize, rhs: usize) -> &mut Self {
        self.storage_dims.swap(self.exposed_dims[lhs], self.exposed_dims[rhs]);
        self.exposed_dims.swap(lhs, rhs);
        self
    }

    /// Removes a logical dimension but keeps it in the storage.
    pub fn stride_dim(&mut self, dim: usize) -> &mut Self {
        self.exposed_dims.remove(dim);
        self
    }

    /// Allows writing to the tensor.
    pub fn enable_writes(&mut self) -> &mut Self {
        self.read_only = false;
        self
    }

    /// Builds the `Tensor`.
    pub fn finish<S, AM>(&self, builder: &mut SignatureBuilder<AM>) -> Tensor<'a, S>
        where S: ScalarArgument, AM: ArgMap + Context + 'a,
    {
        let size = self.storage_dims.iter().map(|s| s.eval(builder.context()) as usize)
            .product::<usize>();
        let (mem_id, array) = builder.array::<S>(self.name, size);
        let mut stride: DimSize = unwrap!(S::t().len_byte()).into();
        let mut strides = self.storage_dims.iter().rev().map(|s| {
            let cur_stride = stride.clone();
            stride.factor *= s.factor;
            stride.params.extend(s.params.iter().cloned());
            cur_stride
        }).collect_vec();
        strides.reverse();
        let iter_dims = self.exposed_dims.iter().map(|&i| {
            (self.storage_dims[i].clone(), strides[i].clone())
        }).collect();
        Tensor {
            mem_id, array, iter_dims,
            read_only: self.read_only,
            name: self.name,
            s: std::marker::PhantomData,
        }
    }
}

/// A tensor allocated in main memory.
pub struct Tensor<'a, S: ScalarArgument> {
    name: &'a str,
    mem_id: ir::mem::Id,
    array: std::sync::Arc<ArrayArgument + 'a>,
    iter_dims: Vec<(DimSize<'a>, DimSize<'a>)>,
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
        let mut incr: DimSize = unwrap!(S::t().len_byte()).into();
        let mut iter_dims = dim_sizes.into_iter().rev().map(|s| {
            let cur_incr = incr.clone();
            incr.factor *= s.factor;
            incr.params.extend(s.params.iter().cloned());
            (s, cur_incr)
        }).collect_vec();
        iter_dims.reverse();
        Tensor { name, mem_id, iter_dims, read_only, array, s: std::marker::PhantomData }
    }

    /// Creates a `VirtualTensor` that contains the values of `self`, loaded in registers.
    pub fn load(&self, tiling: &[&[u32]], builder: &mut Builder) -> VirtualTensor {
        let mut dims = Vec::new();
        let mut induction_levels = Vec::new();
        for (&(ref size, ref stride), tiling) in self.iter_dims.iter().zip_eq(tiling) {
            let size = size.into_ir_size(builder);
            let dim = builder.open_tiled_dim(size, tiling);
            let mut stride = stride.into_ir_size(builder);
            for d in dim.ids().rev() {
                induction_levels.push((d, stride.clone()));
                stride *= builder.dim_size(d);
            }
            dims.push(dim);
        }
        let pat = ir::AccessPattern::Tensor {
            mem_id: self.mem_id,
            dims: induction_levels.iter().cloned().collect(),
        };
        let ptr = builder.induction_var(&self.name, induction_levels);
        let flag = if self.read_only { InstFlag::ALL } else { InstFlag::MEM_COHERENT };
        let inst = builder.ld_ex(S::t(), &ptr, pat, flag);
        for dim in &dims { builder.close_dim(dim); }
        VirtualTensor { inst, dims }
    }

    /// Reads the tensor value in the context and copies it on the host.
    pub fn read_to_host(&self, context: &Context) -> ArrayD<S> {
        use ndarray::ShapeBuilder;
        let mut raw = read_array::<S>(self.array.as_ref());
        let (sizes, strides): (Vec<_>, _) = self.iter_dims.iter().map(|(l, s)| {
            let s_len = unwrap!(S::t().len_byte());
            (l.eval(context) as usize, (s.eval(context)/s_len) as usize)
        }).unzip();
        let len = unwrap!(sizes.iter().zip_eq(&strides).map(|(&l, &s)| l*s).max());
        raw.split_off(len);
        unwrap!(ndarray::ArrayBase::from_shape_vec(sizes.strides(strides), raw))
    }
}

/// A tensor loaded in registers.
pub struct VirtualTensor {
    inst: ir::InstId,
    dims: Vec<LogicalDim>,
}

impl VirtualTensor {
    /// Creates a new `VirtualTensor`.
    pub fn new(inst: ir::InstId, dims: Vec<LogicalDim>) -> Self {
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
    type Output = LogicalDim;

    fn index(&self, idx: usize) -> &Self::Output { &self.dims[idx] }
}

impl<'a> IntoIterator for &'a VirtualTensor {
    type Item = &'a LogicalDim;
    type IntoIter = std::slice::Iter<'a, LogicalDim>;

    fn into_iter(self) -> Self::IntoIter { self.dims.iter() }
}
