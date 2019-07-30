use crate::device::ScalarArgument;
use crate::Scalar;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::fmt;
use telamon::helper::tensor::*;
use telamon::helper::{AutoOperand, Builder, Reduce};
use telamon::ir;

/// Multiplies a matrix `lhs` with a vector `rhs`
pub fn matrix_vector_multiply<'a, S: ScalarArgument>(
    builder: &mut Builder,
    lhs: &VirtualTensor<'_, S>,
    rhs: &VirtualTensor<'_, S>,
) -> VirtualTensor<'a, S> {
    assert!(lhs.num_dims() == 2 && rhs.num_dims() == 1);
    assert!(lhs[lhs.num_dims() - 1].size_eq(&rhs[0], builder.function()));

    // Assume (m x n) . (n) multiplication -> Result: (m)
    let m = &lhs[0];
    let n = &lhs[1];

    // Initialize accumulator
    let accu_init_m = builder.open_mapped_dim(&m);
    let accu_init_instr = builder.mov(&0f32);
    builder.close_dim(&accu_init_m);

    // Map operands and assign accumulator
    let acc_dim_m = builder.open_mapped_dim(&accu_init_m);
    let acc_dim_n = builder.open_mapped_dim(&n);

    let a_operand = lhs.dim_map(
        &[&acc_dim_m, &acc_dim_n],
        ir::DimMapScope::Global(()),
        builder,
    );

    let b_operand = rhs.dim_map(&[&acc_dim_n], ir::DimMapScope::Global(()), builder);

    let acc_instr = builder.mad(&a_operand, &b_operand, &Reduce(accu_init_instr));

    builder.close_dim(&acc_dim_m);
    builder.close_dim(&acc_dim_n);

    VirtualTensor::new(acc_instr, vec![acc_dim_m])
}

/// Multiplies two matrices `lhs` and `rhs`
pub fn matrix_matrix_multiply<'a, S: ScalarArgument>(
    builder: &mut Builder,
    lhs: &VirtualTensor<S>,
    rhs: &VirtualTensor<S>,
) -> VirtualTensor<'a, S> {
    assert!(lhs.num_dims() == 2 && rhs.num_dims() == 2);
    assert!(lhs[lhs.num_dims() - 1].size_eq(&rhs[0], builder.function()));

    // Assume (m x k) . (k x n) multiplication -> Result: (m x n)
    let m = &lhs[0];
    let n = &rhs[1];
    let k = &lhs[1];

    // Initialize accumulator
    let accu_init_m = builder.open_mapped_dim(&m);
    let accu_init_n = builder.open_mapped_dim(&n);

    let accu_init_instr = builder.mov(&0f32);

    builder.close_dim(&accu_init_m);
    builder.close_dim(&accu_init_n);

    // Map operands and assign accumulator
    let acc_dim_m = builder.open_mapped_dim(&accu_init_m);
    let acc_dim_n = builder.open_mapped_dim(&accu_init_n);
    let acc_dim_k = builder.open_mapped_dim(&k);

    let a_operand = lhs.dim_map(
        &[&acc_dim_m, &acc_dim_k],
        ir::DimMapScope::Global(()),
        builder,
    );

    let b_operand = rhs.dim_map(
        &[&acc_dim_k, &acc_dim_n],
        ir::DimMapScope::Global(()),
        builder,
    );

    let acc_instr = builder.mad(&a_operand, &b_operand, &Reduce(accu_init_instr));

    builder.close_dim(&acc_dim_m);
    builder.close_dim(&acc_dim_n);
    builder.close_dim(&acc_dim_k);

    VirtualTensor::new(acc_instr, vec![acc_dim_m, acc_dim_n])
}

/// Adds two tensors `lhs` and `rhs` of the same shape
pub fn tensor_add<'a, S: ScalarArgument>(
    builder: &mut Builder,
    lhs: &VirtualTensor<S>,
    rhs: &VirtualTensor<S>,
) -> VirtualTensor<'a, S> {
    assert!(lhs.same_shape(rhs, builder.function()));

    let dims = lhs
        .iter()
        .map(|dim| builder.open_mapped_dim(dim))
        .collect_vec();

    let a_operand = lhs.dim_map(
        &dims.iter().collect_vec(),
        ir::DimMapScope::Global(()),
        builder,
    );

    let b_operand = rhs.dim_map(
        &dims.iter().collect_vec(),
        ir::DimMapScope::Global(()),
        builder,
    );

    let add_instr = builder.add(&a_operand, &b_operand);

    for dim in &dims {
        builder.close_dim(&dim);
    }

    VirtualTensor::new(add_instr, dims)
}

/// Multiplies all elements of `lhs_mul` with `rhs_mul_operand` and
/// adds the result to the tensor `rhs_add`
pub fn tensor_mad<'a, S: ScalarArgument>(
    builder: &mut Builder,
    lhs_mul: &VirtualTensor<S>,
    rhs_mul_operand: &dyn AutoOperand,
    rhs_add: &VirtualTensor<S>,
) -> VirtualTensor<'a, S> {
    assert!(lhs_mul.same_shape(rhs_add, builder.function()));

    let dims = lhs_mul
        .iter()
        .map(|dim| builder.open_mapped_dim(&dim))
        .collect_vec();

    let lhs_mul_operand = lhs_mul.dim_map(
        &dims.iter().collect_vec(),
        ir::DimMapScope::Global(()),
        builder,
    );

    let rhs_add_operand = rhs_add.dim_map(
        &dims.iter().collect_vec(),
        ir::DimMapScope::Global(()),
        builder,
    );

    let mad_instr = builder.mad(&lhs_mul_operand, rhs_mul_operand, &rhs_add_operand);

    for dim in &dims {
        builder.close_dim(&dim);
    }

    VirtualTensor::new(mad_instr, dims)
}

/// Opens dimensions mapped to the entire set of dimensions of a
/// virtual tensor `a` and calls a function `f` with an operand
/// representing a tensor's element and the builder. All further
/// instructions created by `f` using the builder will be placed in a
/// set of dimensions mapped to the dimensions of the virtual input
/// tensor `a`.
pub fn tensor_map<'a, S: ScalarArgument>(
    builder: &mut Builder,
    a: &VirtualTensor<S>,
    f: impl FnOnce(&ir::Operand<()>, &mut Builder) -> ir::InstId,
) -> VirtualTensor<'a, S> {
    let dims = a
        .iter()
        .map(|dim| builder.open_mapped_dim(&dim))
        .collect_vec();

    let operand = a.dim_map(
        &dims.iter().map(|dim| dim).collect_vec()[..],
        ir::DimMapScope::Global(()),
        builder,
    );

    let res_instr = f(&operand, builder);

    for dim in &dims {
        builder.close_dim(&dim);
    }

    VirtualTensor::new(res_instr, dims)
}

/// Multiplies each element of a virtual tensor `rhs` with a scalar
/// operand `lhs`
pub fn tensor_elementwise_mul<'a, S: ScalarArgument>(
    builder: &mut Builder,
    lhs: &dyn AutoOperand,
    rhs: &VirtualTensor<S>,
) -> VirtualTensor<'a, S> {
    tensor_map(builder, rhs, |tensor_operand, builder| {
        builder.mul(tensor_operand, lhs)
    })
}

/// Applies the `max` function to all elements of a virtual tensor
/// `lhs` with `rhs` as the second argument to `max`
pub fn tensor_elementwise_max<'a, S: ScalarArgument>(
    builder: &mut Builder,
    lhs: &VirtualTensor<S>,
    rhs: &dyn AutoOperand,
) -> VirtualTensor<'a, S> {
    tensor_map(builder, lhs, |tensor_operand, builder| {
        builder.max(tensor_operand, rhs)
    })
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, Debug, Copy)]
pub enum ActivationFunction {
    /// Linear rectifier (i.e., max(0, v))
    ReLU,

    /// Sigmoid activation function (i.e., 1 / (1 + exp(v))
    Sigmoid,
}

impl std::fmt::Display for ActivationFunction {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> fmt::Result {
        match *self {
            ActivationFunction::ReLU => fmt.write_str("relu"),
            ActivationFunction::Sigmoid => fmt.write_str("sigmoid"),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ActivationFunctionParseError {
    token: String,
}

impl fmt::Display for ActivationFunctionParseError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "cannot parse activation function '{}'", self.token)
    }
}

impl ActivationFunction {
    pub fn opt_to_display(
        activation_opt: &Option<ActivationFunction>,
    ) -> OptionDisplay<ActivationFunction> {
        OptionDisplay {
            inner: &activation_opt,
            default: "identity",
        }
    }

    pub fn opt_from_string(
        s: &str,
    ) -> Result<Option<Self>, ActivationFunctionParseError> {
        match s {
            "identity" => Ok(None),
            "relu" => Ok(Some(ActivationFunction::ReLU)),
            "sigmoid" => Ok(Some(ActivationFunction::Sigmoid)),
            _ => Err(ActivationFunctionParseError {
                token: s.to_string(),
            }),
        }
    }
}

pub struct OptionDisplay<'a, T> {
    inner: &'a Option<T>,
    default: &'a str,
}

impl<T: fmt::Display> fmt::Display for OptionDisplay<'_, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(inner) = self.inner {
            fmt::Display::fmt(inner, fmt)
        } else {
            fmt.write_str(self.default)
        }
    }
}

/// Applies an optional activation function element-wise to the
/// virtual tensor `a` and returns a new virtual tensor with the
/// result. If no activation function has been specified, the instance
/// itself is returned.
pub fn tensor_activate<'a, S: Scalar>(
    builder: &mut Builder,
    t: VirtualTensor<'a, S>,
    f: &Option<ActivationFunction>,
) -> VirtualTensor<'a, S> {
    match f {
        Some(ActivationFunction::ReLU) => tensor_elementwise_max(builder, &t, &S::zero()),
        Some(ActivationFunction::Sigmoid) => {
            tensor_map(builder, &t, |operand, builder| {
                let exp = builder.exp(operand);
                let add = builder.add(&S::one(), &exp);
                builder.div(&S::one(), &add)
            })
        }
        None => t,
    }
}

/// Applies an optional activation function element-wise and in place
/// to the Array `a`. If no activation function has been specified,
/// `a` is left unmodified.
pub fn array_activate_inplace<S, D>(
    a: &mut ndarray::Array<S, D>,
    f: &Option<ActivationFunction>,
) where
    S: Scalar,
    D: ndarray::Dimension,
{
    match f {
        Some(ActivationFunction::ReLU) => {
            a.mapv_inplace(|c| c.max(S::zero()));
        }
        Some(ActivationFunction::Sigmoid) => {
            let one = S::one();
            a.mapv_inplace(|c| one / (one + S::exp(c)));
        }
        None => {}
    }
}

/// Applies the softmax function to an array `a` in place, i.e., each
/// element `o_i` of the result has the value `o_i = exp(a_i) /
/// sum(j = 0 to N, exp(a_j))`, where `a_i` is the i-th value of the
/// input array `a` and `N` is the number of elements of `a`.
pub fn array_softmax_inplace<S, D>(
    a: &mut ndarray::Array<S, D>,
) where
    S: Scalar,
    D: ndarray::Dimension,
{
    a.mapv_inplace(|c| S::exp(c));
    let sum = a.scalar_sum();
    a.mapv_inplace(|c| c / sum);
}
