//! Utilities to allocate and operate on tensors.
use crate::device::{ArgMap, ArrayArgument, ArrayArgumentExt, Context, ScalarArgument};
use crate::helper::{Builder, LogicalDim, SignatureBuilder, TilingPattern};
use crate::ir;
use crate::search_space::InstFlag;
use ::ndarray::{self, ArrayD};
use itertools::Itertools;
use std;
use std::sync::Arc;
use utils::*;

/// A dimension size, before tiling.
#[derive(Clone)]
pub struct DimSize<'a> {
    pub factor: u32,
    pub params: Vec<&'a str>,
    pub max_size: u32,
}

impl<'a> DimSize<'a> {
    /// Convert the size into the size type used by the IR.
    pub fn to_ir_size(&self, builder: &Builder) -> ir::Size {
        let params = self
            .params
            .iter()
            .map(|p| Arc::clone(builder.find_param(p)))
            .collect();
        ir::Size::new(self.factor, params, self.max_size)
    }

    /// Converts the size into a numerical value for a given context.
    pub fn eval(&self, context: &dyn Context) -> u32 {
        self.params
            .iter()
            .map(|p| unwrap!(context.param_as_size(p)))
            .product::<u32>()
            * self.factor
    }

    /// Creates a new size equals to the given parameter.
    pub fn new_param(param: &'a str, max_size: u32) -> Self {
        DimSize {
            factor: 1,
            params: vec![param],
            max_size,
        }
    }
}

impl<'a> From<u32> for DimSize<'a> {
    fn from(size: u32) -> Self {
        DimSize {
            factor: size,
            params: vec![],
            max_size: size,
        }
    }
}

/// An helper to build a tensor.
pub struct TensorBuilder<'a> {
    name: &'a str,
    read_only: bool,
    storage_dims: Vec<DimSize<'a>>,
    exposed_dims: Vec<usize>,
}

impl<'a> BuilderTrait for TensorBuilder<'a> {}

impl<'a> TensorBuilder<'a> {
    /// Start building a `Tensor` with the given logical layout.
    pub fn new(name: &'a str, storage_dims: Vec<DimSize<'a>>) -> Self {
        let exposed_dims = (0..storage_dims.len()).collect();
        TensorBuilder {
            name,
            storage_dims,
            exposed_dims,
            read_only: true,
        }
    }

    /// Swap two dimensions in the memory layout of the tensor. Keeps the logical layout
    /// untouched.
    pub fn transpose(&mut self, lhs: usize, rhs: usize) -> &mut Self {
        self.storage_dims
            .swap(self.exposed_dims[lhs], self.exposed_dims[rhs]);
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
    where
        S: ScalarArgument,
        AM: ArgMap<'a> + Context + 'a,
    {
        let size = self
            .storage_dims
            .iter()
            .map(|s| s.eval(builder.context()) as usize)
            .product::<usize>();
        let array = builder.array::<S>(self.name, size);
        let mut stride: DimSize = unwrap!(S::t().len_byte()).into();
        let mut strides = self
            .storage_dims
            .iter()
            .rev()
            .map(|s| {
                let cur_stride = stride.clone();
                stride.factor *= s.factor;
                stride.params.extend(s.params.iter().cloned());
                cur_stride
            })
            .collect_vec();
        strides.reverse();
        let iter_dims = self
            .exposed_dims
            .iter()
            .map(|&i| (self.storage_dims[i].clone(), strides[i].clone()))
            .collect();
        Tensor {
            array,
            iter_dims,
            read_only: self.read_only,
            name: self.name,
            s: std::marker::PhantomData,
        }
    }
}

/// A tensor allocated in main memory.
pub struct Tensor<'a, S: ScalarArgument> {
    name: &'a str,
    array: std::sync::Arc<dyn ArrayArgument + 'a>,
    iter_dims: Vec<(DimSize<'a>, DimSize<'a>)>,
    read_only: bool,
    s: std::marker::PhantomData<S>,
}

impl<'a, S> Tensor<'a, S>
where
    S: ScalarArgument,
{
    /// Allocates a new `Tensor` in the context.
    pub fn new(
        name: &'a str,
        dim_sizes: Vec<DimSize<'a>>,
        read_only: bool,
        array: std::sync::Arc<dyn ArrayArgument + 'a>,
    ) -> Self {
        let mut incr: DimSize = unwrap!(S::t().len_byte()).into();
        let mut iter_dims = dim_sizes
            .into_iter()
            .rev()
            .map(|s| {
                let cur_incr = incr.clone();
                incr.factor *= s.factor;
                incr.params.extend(s.params.iter().cloned());
                (s, cur_incr)
            })
            .collect_vec();
        iter_dims.reverse();
        Tensor {
            name,
            iter_dims,
            read_only,
            array,
            s: std::marker::PhantomData,
        }
    }

    /// Creates a `VirtualTensor` that contains the values of `self`, loaded in registers.
    pub fn load(
        &self,
        tiling: Vec<TilingPattern>,
        builder: &mut Builder,
    ) -> VirtualTensor {
        let dims = self
            .iter_dims
            .iter()
            .zip_eq(tiling)
            .map(|(dim, tiling)| {
                let size = dim.0.to_ir_size(builder);
                builder.open_tiled_dim(size, tiling)
            })
            .collect_vec();
        let (ptr, pattern);
        {
            let increments = dims
                .iter()
                .zip_eq(&self.iter_dims)
                .map(|(dim, (_, stride))| (dim, stride.to_ir_size(builder)))
                .collect_vec();
            ptr = builder.induction_var(&self.name, increments.clone());
            pattern = builder.tensor_access_pattern(None, increments);
        };
        let flag = if self.read_only {
            InstFlag::ALL
        } else {
            InstFlag::COHERENT
        };
        // TODO: take default value as argument
        let predicate = builder
            .predicates_for_logical_dims(&dims)
            .map(|predicates| ir::LoadPredicate::new(predicates, 0f32.as_operand()));
        let inst = builder.predicated_ld_ex(S::t(), &ptr, pattern, flag, predicate);
        for dim in &dims {
            builder.close_dim(dim);
        }
        VirtualTensor { inst, dims }
    }

    /// Reads the tensor value in the context and copies it on the host.
    pub fn read_to_host(&self, context: &dyn Context) -> ArrayD<S> {
        use ndarray::ShapeBuilder;
        let mut raw = self.array.as_ref().read::<S>();
        let (sizes, strides): (Vec<_>, _) = self
            .iter_dims
            .iter()
            .map(|(l, s)| {
                let s_len = unwrap!(S::t().len_byte());
                (l.eval(context) as usize, (s.eval(context) / s_len) as usize)
            })
            .unzip();
        let len = unwrap!(sizes.iter().zip_eq(&strides).map(|(&l, &s)| l * s).max());
        raw.split_off(len);
        unwrap!(ndarray::ArrayBase::from_shape_vec(
            sizes.strides(strides),
            raw
        ))
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
    pub fn dim_map(
        &self,
        dims: &[&LogicalDim],
        scope: ir::DimMapScope<()>,
        builder: &mut Builder,
    ) -> ir::Operand<()> {
        let mapping = self.dims.iter().zip_eq(dims.iter().cloned()).collect_vec();
        builder.dim_map(self.inst, &mapping, scope)
    }

    /// Stores the `VirtualTensor` in memory. Stores contiguously without taking the
    /// layout of the target tensor into account.
    pub fn store<S>(&self, tensor: &Tensor<S>, builder: &mut Builder) -> VirtualTensor
    where
        S: ScalarArgument,
    {
        assert!(!tensor.read_only);
        let new_dims = self
            .dims
            .iter()
            .map(|dim| builder.open_mapped_dim(dim))
            .collect_vec();

        let (ptr, pat) = {
            let new_dims = new_dims.iter().collect_vec();
            builder.tensor_access(&tensor.name, None, S::t(), &new_dims)
        };
        let predicate = builder
            .predicates_for_logical_dims(&new_dims)
            .map(ir::StorePredicate::new);
        let inst = builder.predicated_st(&ptr, &self.inst, pat, predicate);
        for dim in &new_dims {
            builder.close_dim(dim);
        }
        VirtualTensor {
            inst,
            dims: new_dims,
        }
    }

    /// Returns the underlying instruction.
    pub fn inst(&self) -> ir::InstId {
        self.inst
    }

    /// Returns the number of logical dimensions.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// Returns true if the other cirtual tensor has the same number
    /// of dimensions and each dimension has the same size
    pub fn same_shape<T>(&self, other: &Self, function: &ir::Function<T>) -> bool {
        self.num_dims() == other.num_dims()
            && self
                .dims
                .iter()
                .zip(&other.dims)
                .all(|(self_dim, other_dim)| self_dim.size_eq(other_dim, function))
    }

    pub fn iter(&self) -> std::slice::Iter<'_, LogicalDim> {
        self.into_iter()
    }
}

impl std::ops::Index<usize> for VirtualTensor {
    type Output = LogicalDim;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.dims[idx]
    }
}

impl<'a> IntoIterator for &'a VirtualTensor {
    type Item = &'a LogicalDim;
    type IntoIter = std::slice::Iter<'a, LogicalDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.iter()
    }
}
