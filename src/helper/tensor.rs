//! Utilities to allocate and operate on tensors.
use crate::device::{ArgMap, ArrayArgument, ArrayArgumentExt, Context, ScalarArgument};
use crate::helper::{Builder, LogicalDim, SignatureBuilder, TilingPattern};
use crate::ir::{self, IntoIndexExpr as _};
use crate::search_space::InstFlag;
use ::ndarray::{self, ArrayD};
use itertools::Itertools;
use std;
use std::sync::Arc;
use std::{fmt, ops};
use utils::*;

use log::trace;

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

impl<'a, 'b, 'c: 'a> ops::MulAssign<&'b DimSize<'c>> for DimSize<'a> {
    fn mul_assign(&mut self, other: &DimSize<'c>) {
        self.factor *= other.factor;
        self.params.extend(other.params.iter().copied());
        self.max_size *= other.max_size;
    }
}

impl<'a, 'b: 'a> ops::MulAssign<DimSize<'b>> for DimSize<'a> {
    fn mul_assign(&mut self, other: DimSize<'b>) {
        *self *= &other;
    }
}

impl<'a> ops::MulAssign<u32> for DimSize<'a> {
    fn mul_assign(&mut self, other: u32) {
        self.factor *= other;
        self.max_size *= other;
    }
}

impl<'a, 'b, T> ops::Mul<T> for &'b DimSize<'a>
where
    DimSize<'a>: ops::MulAssign<T>,
{
    type Output = DimSize<'a>;

    fn mul(self, other: T) -> Self::Output {
        self.clone() * other
    }
}

impl<'a, T> ops::Mul<T> for DimSize<'a>
where
    DimSize<'a>: ops::MulAssign<T>,
{
    type Output = DimSize<'a>;

    fn mul(mut self, other: T) -> Self::Output {
        self *= other;
        self
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
        let mut stride: DimSize = 1u32.into();
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
    // The size and stride of each dimension in the tensor, in number of elements.
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
        let mut incr: DimSize = 1u32.into();
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

    pub fn load_packed<'b, F, II>(
        &self,
        tilings: II,
        unpacking_fn: F,
        builder: &mut Builder,
    ) -> VirtualTensor
    where
        F: FnOnce(Vec<ir::IndexExpr>) -> (Vec<ir::IndexExpr>, Option<ir::IndexPredicate>),
        II: IntoIterator<Item = (DimSize<'b>, TilingPattern)>,
    {
        let tilings: Vec<_> = tilings
            .into_iter()
            .map(|(size, pattern)| (size.to_ir_size(builder), pattern))
            .collect();
        builder.with_tiled_dims(tilings, move |dims, builder| {
            let (unpacked, predicate) = unpacking_fn(
                dims.iter()
                    .map(|dim| ir::IndexExpr::LogicalDim(dim.id()))
                    .collect(),
            );
            assert_eq!(
                unpacked.len(),
                self.iter_dims.len(),
                "Wrong number of indices after unpacking"
            );

            let ptr = builder.new_access(
                self.name,
                self.iter_dims
                    .iter()
                    .zip(unpacked)
                    .map(|(sizestride, expr)| {
                        // Can't put this in the |..| pattern or Rust gets confused between by-move
                        // and by-ref matching
                        let (_size, stride) = sizestride;
                        (expr, stride.to_ir_size(builder))
                    })
                    .collect(),
                predicate,
            );
            assert_eq!(
                builder.function().accesses()[ptr].base().elem_t,
                Some(S::t()),
                "Tensor access does not match declared type",
            );

            let flag = if self.read_only {
                InstFlag::ALL
            } else {
                InstFlag::COHERENT
            };

            let pattern =
                builder.function().accesses()[ptr].access_pattern(builder.function());

            VirtualTensor {
                inst: builder.ld_ex(S::t(), &ptr, pattern, flag),
                dims: dims.to_vec(),
            }
        })
    }

    /// Creates a `VirtualTensor` that contains the values of `self`, loaded in registers.
    pub fn load(
        &self,
        tiling: Vec<TilingPattern>,
        builder: &mut Builder,
    ) -> VirtualTensor {
        if std::env::var("TELAMON_LOAD_PACKED").is_ok() {
            let sizes = self
                .iter_dims
                .iter()
                .map(|(size, _stride)| size.to_ir_size(builder))
                .collect::<Vec<_>>();

            self.load_packed(
                self.iter_dims
                    .iter()
                    .map(|(size, _stride)| size.clone())
                    .zip_eq(tiling),
                |indices| {
                    (
                        indices.clone(),
                        Some(ir::IndexPredicate::And(
                            indices
                                .into_iter()
                                .zip(sizes.iter().cloned())
                                .map(|(index, size)| index.in_range(0u32.into()..size))
                                .collect(),
                        )),
                    )
                },
                builder,
            )
        } else {
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
                // Stride needs to be converted to bytes
                let bytes = S::t().len_byte().unwrap();
                let increments = dims
                    .iter()
                    .zip_eq(&self.iter_dims)
                    .map(|(dim, (_, stride))| (dim, stride.to_ir_size(builder) * bytes))
                    .collect_vec();
                ptr = builder.induction_var(&self.name, increments.clone());
                pattern = builder.tensor_access_pattern(None, increments);
            };
            let flag = if self.read_only {
                InstFlag::ALL
            } else {
                InstFlag::COHERENT
            };
            let inst = builder.ld_ex(S::t(), &ptr, pattern, flag);
            for dim in &dims {
                builder.close_dim(dim);
            }
            VirtualTensor { inst, dims }
        }
    }

    /// Reads the tensor value in the context and copies it on the host.
    pub fn read_to_host(&self, context: &dyn Context) -> ArrayD<S> {
        use ndarray::ShapeBuilder;
        let mut raw = self.array.as_ref().read::<S>();
        let (sizes, strides): (Vec<_>, _) = self
            .iter_dims
            .iter()
            .map(|(l, s)| (l.eval(context) as usize, s.eval(context) as usize))
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

    /// Permutes the dimensions of a VirtualTensor.
    ///
    /// The `axes` argument follows NumPy's `nd.transpose` API, which is the inverse of the
    /// mathematical notation for permutations.  If the old tensor has shape `shape`, the resulting
    /// tensor has the following shape:
    ///
    ///     [shape[axes[0]], shape[axes[1]], ..., shape[axes[n]]]
    pub fn transpose(&self, axes: &[usize]) -> Self {
        assert_eq!(self.dims.len(), axes.len(), "axes don't match tensor");

        let dims = axes
            .iter()
            .map({
                let mut seen = (0..self.dims.len()).map(|_| false).collect::<Vec<_>>();
                move |&ix| {
                    assert!(
                        ix < self.dims.len(),
                        "axis {} is out of bounds for tensor of rank {}",
                        ix,
                        self.dims.len()
                    );
                    assert!(
                        !std::mem::replace(&mut seen[ix], true),
                        "repeated axis in transpose"
                    );

                    self.dims[ix].clone()
                }
            })
            .collect();

        VirtualTensor {
            inst: self.inst,
            dims,
        }
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

    pub fn store_packed<S, F>(
        &self,
        tensor: &Tensor<S>,
        packed: Vec<Vec<ir::Size>>,
        packing_fn: F,
        builder: &mut Builder,
    ) -> VirtualTensor
    where
        S: ScalarArgument,
        F: FnOnce(Vec<ir::IndexExpr>) -> Vec<ir::IndexExpr>,
    {
        assert!(!tensor.read_only, "Can't write to a read-only tensor");
        assert_eq!(self.dims.len(), packed.len());

        let new_dims = self
            .dims
            .iter()
            .map(|dim| builder.open_mapped_dim(dim))
            .collect::<Vec<_>>();

        let unpacked = new_dims
            .iter()
            .zip(packed)
            .flat_map(|(dim, packed)| dim.id().into_index_expr().delinearize(packed))
            .collect::<Vec<_>>();
        let repacked = packing_fn(unpacked);

        assert_eq!(repacked.len(), tensor.iter_dims.len());

        let ptr = builder.new_access(
            tensor.name,
            tensor
                .iter_dims
                .iter()
                .zip(repacked)
                .map(|(sizestride, expr)| {
                    let (_size, stride) = sizestride;
                    (expr, stride.to_ir_size(builder))
                })
                .collect(),
            None,
        );
        assert_eq!(
            builder.function().accesses()[ptr].base().elem_t,
            Some(S::t()),
            "Tensor store does not match declared type"
        );

        let pattern =
            builder.function().accesses()[ptr].access_pattern(builder.function());

        let inst = builder.st(&ptr, &self.inst, pattern);

        for dim in &new_dims {
            builder.close_dim(dim);
        }

        VirtualTensor {
            inst,
            dims: new_dims,
        }
    }

    /// Stores the `VirtualTensor` in memory. Stores contiguously without taking the
    /// layout of the target tensor into account.
    pub fn store<S>(&self, tensor: &Tensor<S>, builder: &mut Builder) -> VirtualTensor
    where
        S: ScalarArgument,
    {
        assert!(!tensor.read_only);

        let compatible = self.dims.len() == tensor.iter_dims.len()
            && self.dims.iter().zip(tensor.iter_dims.iter()).all(
                |(vt_dim, (size, _stride))| {
                    builder.function().logical_dim(vt_dim.id()).total_size()
                        == &size.to_ir_size(builder)
                },
            );

        trace!(
            "store: {} and {}",
            self.dims
                .iter()
                .map(|dim| builder.function().logical_dim(dim.id()).total_size())
                .format("x"),
            tensor
                .iter_dims
                .iter()
                .map(|(size, _stride)| size.to_ir_size(builder))
                .format("x")
        );

        if !compatible {
            panic!(
                "store: incompatible shapes: {} and {}",
                self.dims
                    .iter()
                    .map(|dim| builder.function().logical_dim(dim.id()).total_size())
                    .format("x"),
                tensor
                    .iter_dims
                    .iter()
                    .map(|(size, _stride)| size.to_ir_size(builder))
                    .format("x")
            );
        }

        let new_dims = self
            .dims
            .iter()
            .map(|dim| builder.open_mapped_dim(dim))
            .collect_vec();
        let (ptr, pat) = {
            let new_dims = new_dims.iter().collect::<Vec<_>>();
            builder.tensor_access(&tensor.name, None, S::t(), &new_dims)
        };
        let inst = builder.st(&ptr, &self.inst, pat);
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

    pub fn shape(&'_ self) -> VirtualTensorShape<'_> {
        VirtualTensorShape { shape: &self.dims }
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

pub struct VirtualTensorShape<'a> {
    shape: &'a [LogicalDim],
}

impl<'a> VirtualTensorShape<'a> {
    pub fn iter(&self) -> std::slice::Iter<'_, LogicalDim> {
        self.shape.iter()
    }

    pub fn len(&self) -> usize {
        self.shape.len()
    }
}

impl ops::Index<usize> for VirtualTensorShape<'_> {
    type Output = LogicalDim;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.shape[idx]
    }
}

impl<'a, 'b> IntoIterator for &'a VirtualTensorShape<'b> {
    type Item = &'a LogicalDim;

    type IntoIter = std::slice::Iter<'a, LogicalDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, L> ir::IrDisplay<L> for VirtualTensorShape<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter, function: &ir::Function<L>) -> fmt::Result {
        write!(
            fmt,
            "({})",
            self.shape.iter().map(|dim| dim.size(function)).format(", ")
        )
    }
}
