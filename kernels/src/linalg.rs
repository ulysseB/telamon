//! Linera algebra kernels.
#![allow(clippy::many_single_char_names)]
use std::convert::TryFrom;
use std::sync::Arc;

pub use crate::compose::ActivationFunction;
use crate::compose::{
    matrix_matrix_multiply, matrix_vector_multiply, tensor_elementwise_mul, tensor_mad,
};
use crate::kernel::Kernel;
use crate::{build_candidate, check_output, create_size, infer_tiling, Scalar};
use ::ndarray::{Array1, Array2, Array3, Array4, ArrayD};
use rand;
use serde::{Deserialize, Serialize};
use telamon::explorer::Candidate;
use telamon::helper::tensor::*;
use telamon::helper::{self, Builder, SignatureBuilder};
use telamon::ir::DimMapScope::Global as GlobalScope;
use telamon::search_space::*;
use telamon::{device, ir};
use utils::*;

/// Computes `z = alpha*x+y`.
pub struct Axpy<'a, S>
where
    S: Scalar,
{
    n: i32,
    x: Tensor<'a, S>,
    y: Tensor<'a, S>,
    z: Tensor<'a, S>,
}

impl<'a, S> Kernel for Axpy<'a, S>
where
    S: Scalar,
{
    type Parameters = (i32, bool);
    type ExpectedOutput = ArrayD<S>;

    fn name() -> &'static str {
        "axpy"
    }

    fn build_signature<AM>(
        (n, generic): (i32, bool),
        builder: &mut SignatureBuilder<AM>,
    ) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let n_size = create_size(n, "n", generic, builder);
        builder.scalar("alpha", S::one());
        let x = builder.tensor::<S>("x", vec![n_size.clone()], true);
        let y = builder.tensor::<S>("y", vec![n_size.clone()], true);
        let z = builder.tensor::<S>("z", vec![n_size], false);
        Axpy { n, x, y, z }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let tiling = helper::TilingPattern::infer_pattern(self.n as u32, &[1024, 4]);
        let mut builder = Builder::new(signature, ctx.device());

        let x = self.x.load(vec![tiling.clone()], &mut builder);
        let y = self.y.load(vec![tiling], &mut builder);

        let mad = tensor_mad(&mut builder, &x, &"alpha", &y);

        mad.store(&self.z, &mut builder);
        vec![build_candidate(builder.get(), ctx)]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> ArrayD<S> {
        self.x.read_to_host(context) + self.y.read_to_host(context)
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let z = self.z.read_to_host(context);
        if let Err(invalid) = check_output(&z, expected) {
            Err(format!("Invalid axpy output: {}", invalid))
        } else {
            Ok(())
        }
    }
}

/// Computes `y = A.x`.
pub struct MatVec<'a, S>
where
    S: Scalar,
{
    m: i32,
    n: i32,
    x: Tensor<'a, S>,
    a: Tensor<'a, S>,
    y: Tensor<'a, S>,
}

impl<'a, S> Kernel for MatVec<'a, S>
where
    S: Scalar,
{
    type Parameters = (i32, i32, bool);
    type ExpectedOutput = Array1<S>;

    fn name() -> &'static str {
        "mv"
    }

    fn build_signature<AM>(
        (m, n, generic): (i32, i32, bool),
        builder: &mut SignatureBuilder<AM>,
    ) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let m_size = create_size(m, "m", generic, builder);
        let n_size = create_size(n, "n", generic, builder);
        let x = builder.tensor::<S>("x", vec![n_size.clone()], true);
        let a = builder.tensor::<S>("a", vec![m_size.clone(), n_size], true);
        let y = builder.tensor::<S>("y", vec![m_size], false);
        MatVec { m, n, x, a, y }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let m_tiling = helper::TilingPattern::infer_pattern(self.m as u32, &[128, 16]);
        let n_tiling = helper::TilingPattern::infer_pattern(self.n as u32, &[128]);
        let mut builder = Builder::new(signature, ctx.device());
        let x = self.x.load(vec![n_tiling.clone()], &mut builder);
        let a = self.a.load(vec![m_tiling, n_tiling], &mut builder);

        let ax = matrix_vector_multiply(&mut builder, &a, &x);
        ax.store(&self.y, &mut builder);

        vec![build_candidate(builder.get(), ctx)]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> Array1<S> {
        let a_shape = (self.m as usize, self.n as usize);
        self.a
            .read_to_host(context)
            .into_shape(a_shape)
            .unwrap()
            .dot(
                &self
                    .x
                    .read_to_host(context)
                    .into_shape(self.n as usize)
                    .unwrap(),
            )
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let y = self
            .y
            .read_to_host(context)
            .into_shape(self.m as usize)
            .unwrap();
        if let Err(invalid) = check_output(&y, expected) {
            Err(format!("Invalid mv output: {}", invalid))
        } else {
            Ok(())
        }
    }
}

/// Computes `y = (alpha*A + beta*B).x`.
pub struct Gesummv<'a, S: Scalar> {
    m: i32,
    n: i32,
    alpha: S,
    beta: S,
    a: Tensor<'a, S>,
    b: Tensor<'a, S>,
    x: Tensor<'a, S>,
    y: Tensor<'a, S>,
}

impl<'a, S: Scalar> Kernel for Gesummv<'a, S> {
    type Parameters = (i32, i32, bool);
    type ExpectedOutput = Array1<S>;

    fn name() -> &'static str {
        "gesummv"
    }

    fn build_signature<AM>(
        (m, n, generic): (i32, i32, bool),
        builder: &mut SignatureBuilder<AM>,
    ) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let m_size = create_size(m, "m", generic, builder);
        let n_size = create_size(n, "n", generic, builder);
        let mut rng = rand::thread_rng();
        let alpha = S::gen_random(&mut rng);
        let beta = S::gen_random(&mut rng);
        builder.scalar("alpha", alpha);
        builder.scalar("beta", beta);
        Gesummv {
            m,
            n,
            alpha,
            beta,
            x: builder.tensor::<S>("x", vec![n_size.clone()], true),
            a: builder.tensor::<S>("a", vec![m_size.clone(), n_size.clone()], true),
            b: builder.tensor::<S>("b", vec![m_size.clone(), n_size], true),
            y: builder.tensor::<S>("y", vec![m_size], false),
        }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let m_tiling = helper::TilingPattern::infer_pattern(self.m as u32, &[128, 16]);
        let n_tiling = helper::TilingPattern::infer_pattern(self.n as u32, &[128]);
        let ab_tiling = vec![m_tiling, n_tiling.clone()];

        let mut builder = helper::Builder::new(signature, ctx.device());

        let x = self.x.load(vec![n_tiling], &mut builder);
        let a = self.a.load(ab_tiling.clone(), &mut builder);
        let b = self.b.load(ab_tiling, &mut builder);

        let ax = matrix_vector_multiply(&mut builder, &a, &x);
        let aax = tensor_elementwise_mul(&mut builder, &"alpha", &ax);

        let bx = matrix_vector_multiply(&mut builder, &b, &x);

        let aaxpbbx = tensor_mad(&mut builder, &bx, &"beta", &aax);

        aaxpbbx.store(&self.y, &mut builder);

        vec![build_candidate(builder.get(), ctx)]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> Array1<S> {
        let (m, n) = (self.m as usize, self.n as usize);
        let a = unwrap!(self.a.read_to_host(context).into_shape((m, n)));
        let b = unwrap!(self.b.read_to_host(context).into_shape((m, n)));
        let x = unwrap!(self.x.read_to_host(context).into_shape(m));
        a.dot(&x) * self.alpha + b.dot(&x) * self.beta
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let y = unwrap!(self.y.read_to_host(context).into_shape(self.m as usize));
        if let Err(invalid) = check_output(&y, expected) {
            Err(format!("Invalid gesummv output: {}", invalid))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct FusedMMP {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub a_stride: u32,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub generic: bool,
    pub m_tiling: Option<helper::TilingPattern>,
    pub n_tiling: Option<helper::TilingPattern>,
    pub k_tiling: Option<helper::TilingPattern>,
    pub activation_fun: Option<ActivationFunction>,
}

impl FusedMMP {
    pub fn new(m: i32, n: i32, k: i32) -> Self {
        FusedMMP {
            m,
            n,
            k,
            a_stride: 1,
            transpose_a: false,
            transpose_b: false,
            generic: true,
            m_tiling: None,
            n_tiling: None,
            k_tiling: None,
            activation_fun: None,
        }
    }

    pub fn transpose_a(mut self) -> Self {
        self.transpose_a = true;
        self
    }

    pub fn transpose_b(mut self) -> Self {
        self.transpose_b = true;
        self
    }

    pub fn stride_a(mut self, stride: u32) -> Self {
        self.a_stride = stride;
        self
    }

    pub fn activation_fun<F>(mut self, fun: F) -> Self
    where
        F: Into<Option<ActivationFunction>>,
    {
        self.activation_fun = fun.into();
        self
    }

    /// Inline the sizes in the generated code.
    pub fn static_sizes(mut self) -> Self {
        self.generic = false;
        self
    }
}

/// Computes `C = A.B` and applies an activation function to each
/// element of C.
pub struct FusedMM<'a, S: Scalar> {
    pub params: FusedMMP,
    a: Tensor<'a, S>,
    b: Tensor<'a, S>,
    c: Tensor<'a, S>,
}

impl<'a, S: Scalar> Kernel for FusedMM<'a, S> {
    type Parameters = FusedMMP;
    type ExpectedOutput = Array2<S>;

    fn name() -> &'static str {
        "fused_mm"
    }

    fn build_signature<AM>(params: FusedMMP, builder: &mut SignatureBuilder<AM>) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let m_size = create_size(params.m, "m", params.generic, builder);
        let n_size = create_size(params.n, "n", params.generic, builder);
        let k_size = create_size(params.k, "k", params.generic, builder);
        let a_dims = vec![m_size.clone(), k_size.clone(), params.a_stride.into()];
        let a = TensorBuilder::new("a", a_dims)
            .doif(params.transpose_a, |b| b.transpose(0, 1))
            .stride_dim(2)
            .finish(builder);
        let b = TensorBuilder::new("b", vec![k_size, n_size.clone()])
            .doif(params.transpose_b, |b| b.transpose(0, 1))
            .finish(builder);
        let c = builder.tensor::<S>("c", vec![m_size, n_size], false);
        FusedMM { params, a, b, c }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let m_tiling = infer_tiling(self.params.m, &self.params.m_tiling, &[32, 4]);
        let n_tiling = infer_tiling(self.params.n, &self.params.n_tiling, &[32, 4]);
        let k_tiling = infer_tiling(self.params.k, &self.params.k_tiling, &[32, 4]);

        let mut builder = helper::Builder::new(signature, ctx.device());

        let a = self.a.load(vec![m_tiling, k_tiling.clone()], &mut builder);
        let b = self.b.load(vec![k_tiling, n_tiling], &mut builder);
        // let h = p + r - pad;
        // let hok = h.in_range(0..H);
        // let w = q + s - pad;
        // let wok = w.in_range(0..W);
        // ([n, c, h, w], hok && wok)

        let ab = matrix_matrix_multiply(&mut builder, &a, &b);

        if let Some(activation_fun) = &self.params.activation_fun {
            let res = activation_fun.apply::<S>(&mut builder, &ab);
            res.store(&self.c, &mut builder);
        } else {
            ab.store(&self.c, &mut builder);
        }

        vec![build_candidate(builder.get(), ctx)]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> Array2<S> {
        let a_shape = (self.params.m as usize, self.params.k as usize);
        let b_shape = (self.params.k as usize, self.params.n as usize);
        let a = unwrap!(self.a.read_to_host(context).into_shape(a_shape));
        let b = unwrap!(self.b.read_to_host(context).into_shape(b_shape));
        let mut res = a.dot(&b);

        match self.params.activation_fun {
            Some(ActivationFunction::ReLU) => {
                res.mapv_inplace(|c| c.max(S::zero()));
            }

            Some(ActivationFunction::Sigmoid) => {
                let one = S::one();
                res.mapv_inplace(|c| one / (one + S::exp(c)));
            }

            None => {}
        };

        res
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let c_shape = (self.params.m as usize, self.params.n as usize);
        let c = unwrap!(self.c.read_to_host(context).into_shape(c_shape));
        if let Err(invalid) = check_output(&c, expected) {
            Err(format!("Invalid fused_mm output: {}", invalid))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum DataFormat {
    Nchw,
    Nhwc,
}

impl DataFormat {
    pub fn convert_from_nchw<T>(self, n: T, c: T, h: T, w: T) -> [T; 4] {
        match self {
            DataFormat::Nchw => [n, c, h, w],
            DataFormat::Nhwc => [n, h, w, c],
        }
    }
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFormat::Nchw => fmt.write_str("NCHW"),
            DataFormat::Nhwc => fmt.write_str("NHWC"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDataFormatError {
    source: String,
}

impl std::fmt::Display for ParseDataFormatError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "invalid data format: {:?}", self.source)
    }
}

impl std::error::Error for ParseDataFormatError {}

impl std::str::FromStr for DataFormat {
    type Err = ParseDataFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match &s.trim().to_uppercase()[..] {
            "NCHW" => DataFormat::Nchw,
            "NHWC" => DataFormat::Nhwc,
            _ => {
                return Err(ParseDataFormatError {
                    source: s.to_string(),
                })
            }
        })
    }
}

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub enum FilterFormat {
    Kcrs,
    Krsc,
}

impl FilterFormat {
    pub fn default_for_data_format(format: DataFormat) -> Self {
        match format {
            DataFormat::Nchw => FilterFormat::Kcrs,
            DataFormat::Nhwc => FilterFormat::Krsc,
        }
    }

    pub fn convert_to_crs<T>(self, crs: [T; 3]) -> [T; 3] {
        match (self, crs) {
            (FilterFormat::Kcrs, [c, r, s]) => [c, r, s],
            (FilterFormat::Krsc, [r, s, c]) => [c, r, s],
        }
    }

    pub fn convert_from_crs<T>(self, c: T, r: T, s: T) -> [T; 3] {
        match self {
            FilterFormat::Kcrs => [c, r, s],
            FilterFormat::Krsc => [r, s, c],
        }
    }

    pub fn convert_from_kcrs<T>(self, k: T, c: T, r: T, s: T) -> [T; 4] {
        match self {
            FilterFormat::Kcrs => [k, c, r, s],
            FilterFormat::Krsc => [k, r, s, c],
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum PadMode {
    Valid,
    Same,
    Full,
}

impl PadMode {
    pub fn value(self, f: i32) -> i32 {
        match self {
            PadMode::Valid => 0,
            PadMode::Same => f / 2,
            PadMode::Full => f - 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePadModeError {
    source: String,
}

impl std::fmt::Display for ParsePadModeError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "invalid padding mode: {:?}", self.source,)
    }
}

impl std::error::Error for ParsePadModeError {}

impl std::str::FromStr for PadMode {
    type Err = ParsePadModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match &s.trim().to_uppercase()[..] {
            "VALID" => PadMode::Valid,
            "SAME" => PadMode::Same,
            "FULL" => PadMode::Full,
            _ => {
                return Err(ParsePadModeError {
                    source: s.to_string(),
                })
            }
        })
    }
}

impl std::fmt::Display for PadMode {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PadMode::Valid => fmt.write_str("VALID"),
            PadMode::Same => fmt.write_str("SAME"),
            PadMode::Full => fmt.write_str("FULL"),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Conv2dP {
    pub batch: i32,
    pub in_channels: i32,
    pub in_height: i32,
    pub in_width: i32,
    pub out_channels: i32,
    pub filter_width: i32,
    pub filter_height: i32,
    pub pad_mode: PadMode,
    pub data_format: DataFormat,
}

impl Conv2dP {
    pub fn pad_h(&self) -> i32 {
        self.pad_mode.value(self.filter_height)
    }

    pub fn pad_w(&self) -> i32 {
        self.pad_mode.value(self.filter_width)
    }

    pub fn out_height(&self) -> i32 {
        self.in_height - self.filter_height + 1 + 2 * self.pad_h()
    }

    pub fn out_width(&self) -> i32 {
        self.in_width - self.filter_width + 1 + 2 * self.pad_w()
    }

    pub fn data_format(&self) -> DataFormat {
        self.data_format
    }

    pub fn filter_format(&self) -> FilterFormat {
        FilterFormat::default_for_data_format(self.data_format())
    }

    fn filter_shape(&self) -> [usize; 4] {
        self.filter_format().convert_from_kcrs(
            self.out_channels as usize,
            self.in_channels as usize,
            self.filter_height as usize,
            self.filter_width as usize,
        )
    }

    fn input_shape(&self) -> [usize; 4] {
        self.data_format().convert_from_nchw(
            self.batch as usize,
            self.in_channels as usize,
            self.in_height as usize,
            self.in_width as usize,
        )
    }

    fn output_shape(&self) -> [usize; 4] {
        self.data_format().convert_from_nchw(
            self.batch as usize,
            self.out_channels as usize,
            self.out_height() as usize,
            self.out_width() as usize,
        )
    }
}

pub struct Conv2d<'a, S: Scalar> {
    pub params: Conv2dP,
    input: Tensor<'a, S>,
    filter: Tensor<'a, S>,
    output: Tensor<'a, S>,
    n: DimSize<'a>,
    c: DimSize<'a>,
    h: DimSize<'a>,
    w: DimSize<'a>,
    k: DimSize<'a>,
    r: DimSize<'a>,
    s: DimSize<'a>,
    p: DimSize<'a>,
    q: DimSize<'a>,
}

impl<'a, S: Scalar> Kernel for Conv2d<'a, S> {
    type Parameters = Conv2dP;
    type ExpectedOutput = Array4<S>;

    fn name() -> &'static str {
        "conv2d"
    }

    fn build_signature<AM>(
        params: Self::Parameters,
        builder: &mut SignatureBuilder<AM>,
    ) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let n = create_size(params.batch, "n", true, builder);
        let c = create_size(params.in_channels, "c", true, builder);
        let h = create_size(params.in_height, "h", true, builder);
        let w = create_size(params.in_width, "w", true, builder);
        let k = create_size(params.out_channels, "k", true, builder);
        let r = create_size(params.filter_height, "r", false, builder);
        let s = create_size(params.filter_width, "s", false, builder);
        let p = create_size(params.out_height(), "p", true, builder);
        let q = create_size(params.out_width(), "q", true, builder);

        let input_dims = params.data_format().convert_from_nchw(
            n.clone(),
            c.clone(),
            h.clone(),
            w.clone(),
        );
        let input = TensorBuilder::new("input", input_dims.to_vec()).finish(builder);
        let filter_dims = [k.clone(), &c * &r * &s];
        let filter = TensorBuilder::new("filter", filter_dims.to_vec()).finish(builder);
        let output_dims = params.data_format().convert_from_nchw(
            n.clone(),
            k.clone(),
            p.clone(),
            q.clone(),
        );
        let output = builder.tensor::<S>("output", output_dims.to_vec(), false);

        Conv2d {
            params,
            input,
            filter,
            output,
            n,
            c,
            h,
            w,
            k,
            r,
            s,
            p,
            q,
        }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let npq = self.params.batch * self.params.out_height() * self.params.out_width();
        let crs = self.params.in_channels
            * self.params.filter_height
            * self.params.filter_width;

        let npq_tiling = infer_tiling(npq, &None, &[32, 4]);
        let crs_tiling = infer_tiling(crs, &None, &[32]);
        let k_tiling = infer_tiling(self.params.out_channels, &None, &[32, 4]);

        let mut builder = helper::Builder::new(signature, ctx.device());

        let n_size = self.n.to_ir_size(&mut builder);
        let p_size = self.p.to_ir_size(&mut builder);
        let q_size = self.q.to_ir_size(&mut builder);
        let c_size = self.c.to_ir_size(&mut builder);
        let k_size = self.k.to_ir_size(&mut builder);
        let r_size = self.r.to_ir_size(&mut builder);
        let s_size = self.s.to_ir_size(&mut builder);
        let h_size = self.h.to_ir_size(&mut builder);
        let w_size = self.w.to_ir_size(&mut builder);
        let a = self.input.load_packed(
            [
                (&self.n * &self.p * &self.q, npq_tiling.clone()),
                (&self.c * &self.r * &self.s, crs_tiling.clone()),
            ]
            .to_vec(),
            |args| {
                if let [npq, crs] = &args[..] {
                    let npq = npq.clone().delinearize(vec![
                        n_size.clone(),
                        p_size.clone(),
                        q_size.clone(),
                    ]);
                    if let [n, p, q] = &npq[..] {
                        let crs = self.params.filter_format().convert_to_crs(
                            <&'_ [_; 3]>::try_from(
                                &crs.clone().delinearize(
                                    self.params
                                        .filter_format()
                                        .convert_from_crs(
                                            c_size.clone(),
                                            r_size.clone(),
                                            s_size.clone(),
                                        )
                                        .to_vec(),
                                )[..],
                            )
                            .unwrap()
                            .clone(),
                        );
                        if let [c, r, s] = &crs[..] {
                            let h = p + r + (-self.params.pad_h());
                            let w = q + s + (-self.params.pad_w());

                            let predicate = h.clone().in_range(0u32.into()..h_size);
                            let predicate =
                                predicate & w.clone().in_range(0u32.into()..w_size);

                            (
                                self.params
                                    .data_format()
                                    .convert_from_nchw(n.clone(), c.clone(), h, w)
                                    .to_vec(),
                                Some(predicate),
                            )
                        } else {
                            unreachable!()
                        }
                    } else {
                        unreachable!()
                    }
                } else {
                    unreachable!()
                }
            },
            &mut builder,
        );
        let b = self.filter.load_packed(
            [
                (&self.c * &self.r * &self.s, crs_tiling.clone()),
                (self.k.clone(), k_tiling.clone()),
            ]
            .to_vec(),
            |args| {
                if let [crs, k] = &args[..] {
                    ([k.clone(), crs.clone()].to_vec(), None)
                } else {
                    unreachable!()
                }
            },
            &mut builder,
        );
        //.load(vec![k_tiling, crs_tiling], &mut builder)
        //.transpose(&[1, 0]);

        let ab = matrix_matrix_multiply(&mut builder, &a, &b);

        // ab.store(&self.output, &mut builder);
        ab.store_packed(
            &self.output,
            vec![vec![n_size, p_size, q_size], vec![k_size]],
            |args| {
                if let [n, p, q, k] = &args[..] {
                    self.params
                        .data_format()
                        .convert_from_nchw(n.clone(), k.clone(), p.clone(), q.clone())
                        .to_vec()
                } else {
                    unreachable!()
                }
            },
            &mut builder,
        );

        vec![build_candidate(builder.get(), ctx)]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> Self::ExpectedOutput {
        let input_shape = self.params.input_shape();
        let filter_shape = self.params.filter_shape();
        let output_shape = self.params.output_shape();

        let input = self.input.read_to_host(context);
        assert_eq!(input.shape(), &input_shape);
        let input = input.into_shape(input_shape).unwrap();
        let filter = self.filter.read_to_host(context);
        assert_eq!(
            filter.shape(),
            &[
                self.params.out_channels as usize,
                (self.params.in_channels
                    * self.params.filter_width
                    * self.params.filter_height) as usize
            ]
        );
        let filter = filter.into_shape(filter_shape).unwrap();

        let mut res = Array4::<S>::zeros(output_shape);
        for n in 0..self.params.batch as usize {
            for k in 0..self.params.out_channels as usize {
                for p in 0..self.params.out_height() as usize {
                    for q in 0..self.params.out_width() as usize {
                        let mut x = S::zero();
                        for r in 0..self.params.filter_height as usize {
                            for s in 0..self.params.filter_width as usize {
                                for c in 0..self.params.in_channels as usize {
                                    let h = (p + r) as i32 - self.params.pad_h();
                                    let w = (q + s) as i32 - self.params.pad_w();
                                    if h >= 0
                                        && w >= 0
                                        && h < self.params.in_height
                                        && w < self.params.in_width
                                    {
                                        x += input[self
                                            .params
                                            .data_format()
                                            .convert_from_nchw(
                                                n, c, h as usize, w as usize,
                                            )]
                                            * filter[self
                                                .params
                                                .filter_format()
                                                .convert_from_kcrs(k, c, r, s)];
                                    }
                                }
                            }
                        }
                        res[self.params.data_format().convert_from_nchw(n, k, p, q)] = x;
                    }
                }
            }
        }

        res
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let output_shape = self.params.output_shape();
        let output = self.output.read_to_host(context);
        assert_eq!(output.shape(), &output_shape);

        let output = output.into_shape(output_shape).unwrap();
        if let Err(invalid) = check_output(&output, expected) {
            Err(format!("Invalid conv2d output: {}", invalid))
        } else {
            Ok(())
        }
    }
}

/// Batch transposed matrix-matrix multiplication.
pub struct BatchMM<'a, S>
where
    S: Scalar,
{
    params: BatchMMP,
    a: Tensor<'a, S>,
    b: Tensor<'a, S>,
    c: Tensor<'a, S>,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct BatchMMP {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub batch: i32,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub batch_b: bool,
    pub generic: bool,
}

impl BatchMMP {
    pub fn new(batch: i32, m: i32, n: i32, k: i32) -> Self {
        BatchMMP {
            m,
            n,
            k,
            batch,
            transpose_a: false,
            transpose_b: false,
            batch_b: true,
            generic: true,
        }
    }

    pub fn transpose_a(mut self) -> Self {
        self.transpose_a = true;
        self
    }

    pub fn transpose_b(mut self) -> Self {
        self.transpose_b = true;
        self
    }

    /// Generate code that is onyl valid for the given sizes. The batch size is still
    /// generic.
    pub fn static_sizes(mut self) -> Self {
        self.generic = false;
        self
    }

    /// Reuse the `B` matrix across the batch.
    pub fn reuse_b(mut self) -> Self {
        self.batch_b = false;
        self
    }
}

impl<'a, S: Scalar> Kernel for BatchMM<'a, S> {
    type Parameters = BatchMMP;
    type ExpectedOutput = Array3<S>;

    fn name() -> &'static str {
        "batch_mm"
    }

    fn build_signature<AM>(params: BatchMMP, builder: &mut SignatureBuilder<AM>) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let m_size = create_size(params.m, "m", params.generic, builder);
        let n_size = create_size(params.n, "n", params.generic, builder);
        let k_size = create_size(params.k, "k", params.generic, builder);
        let batch = create_size(params.batch, "batch", true, builder);
        let a_dims = vec![batch.clone(), m_size.clone(), k_size.clone()];
        let a = TensorBuilder::new("a", a_dims)
            .doif(params.transpose_a, |b| b.transpose(1, 2))
            .finish(builder);
        let b = TensorBuilder::new("b", vec![batch.clone(), k_size, n_size.clone()])
            .doif(params.transpose_b, |b| b.transpose(1, 2))
            .doif(!params.batch_b, |b| b.stride_dim(0))
            .finish(builder);
        let c = builder.tensor::<S>("c", vec![batch, m_size, n_size], false);
        BatchMM { params, a, b, c }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let m_tiling = helper::TilingPattern::infer_pattern(self.params.m as u32, &[64]);
        let n_tiling = helper::TilingPattern::infer_pattern(self.params.n as u32, &[64]);
        let k_tiling = helper::TilingPattern::infer_pattern(self.params.k as u32, &[64]);
        let batch_tiling =
            helper::TilingPattern::infer_pattern(self.params.batch as u32, &[128]);
        let mut builder = helper::Builder::new(signature, ctx.device());
        let a_tiling = vec![batch_tiling.clone(), m_tiling, k_tiling.clone()];
        let ld_a = self.a.load(a_tiling, &mut builder);
        let b_tiling = if self.params.batch_b {
            vec![batch_tiling, k_tiling, n_tiling]
        } else {
            vec![k_tiling, n_tiling]
        };
        let ld_b = self.b.load(b_tiling, &mut builder);

        let init_batch = builder.open_mapped_dim(&ld_a[0]);
        let init_dim_m = builder.open_mapped_dim(&ld_a[1]);
        let dim_n = &ld_b[if self.params.batch_b { 2 } else { 1 }];
        let init_dim_n = builder.open_mapped_dim(dim_n);
        let acc_init = builder.mov(&0f32);
        let acc_batch = builder.open_mapped_dim(&init_batch);
        let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
        let acc_dim_n = builder.open_mapped_dim(&init_dim_n);
        let acc_dim_k = builder.open_mapped_dim(&ld_a[2]);
        let a_op = ld_a.dim_map(
            &[&acc_batch, &acc_dim_m, &acc_dim_k],
            GlobalScope(()),
            &mut builder,
        );
        let b_op = {
            let b_dims = [&acc_batch, &acc_dim_k, &acc_dim_n];
            let b_dims = if self.params.batch_b {
                &b_dims
            } else {
                &b_dims[1..]
            };
            ld_b.dim_map(b_dims, GlobalScope(()), &mut builder)
        };
        let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
        builder.close_dim(&acc_dim_k);

        let acc = VirtualTensor::new(acc, vec![acc_batch, acc_dim_m, acc_dim_n]);
        let st_c = acc.store(&self.c, &mut builder);

        // Order for correctness.
        builder.order(&st_c.inst(), &acc_dim_k, Order::AFTER);
        vec![build_candidate(builder.get(), ctx)]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> Array3<S> {
        let batch = self.params.batch as usize;
        let m = self.params.m as usize;
        let n = self.params.n as usize;
        let k = self.params.k as usize;
        let a = self
            .a
            .read_to_host(context)
            .into_shape((batch, m, k))
            .unwrap();
        let b = self
            .b
            .read_to_host(context)
            .into_shape((batch, k, n))
            .unwrap();
        let mut c = Array3::zeros((batch, m, n));
        for (mut c, (a, b)) in c.outer_iter_mut().zip(a.outer_iter().zip(b.outer_iter()))
        {
            c.assign(&a.dot(&b));
        }
        c
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let batch = self.params.batch as usize;
        let c_shape = (batch, self.params.m as usize, self.params.n as usize);
        let c = self.c.read_to_host(context).into_shape(c_shape).unwrap();
        if let Err(invalid) = check_output(&c, expected) {
            Err(format!("Invalid batched_gemm output: {}", invalid))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Fused2MMP {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub p: i32,
    pub alpha: f32,
    pub beta: f32,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub transpose_c: bool,
    pub transpose_d: bool,
    pub generic: bool,
    pub m_tiling: Option<helper::TilingPattern>,
    pub n_tiling: Option<helper::TilingPattern>,
    pub k_tiling: Option<helper::TilingPattern>,
    pub p_tiling: Option<helper::TilingPattern>,
    pub activation_fun: Option<ActivationFunction>,
}

impl Fused2MMP {
    pub fn new(m: i32, n: i32, k: i32, p: i32, alpha: f32, beta: f32) -> Self {
        Fused2MMP {
            m,
            n,
            k,
            p,
            alpha,
            beta,
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            transpose_d: false,
            generic: true,
            m_tiling: None,
            n_tiling: None,
            k_tiling: None,
            p_tiling: None,
            activation_fun: None,
        }
    }

    pub fn transpose_a(mut self) -> Self {
        self.transpose_a = true;
        self
    }

    pub fn transpose_b(mut self) -> Self {
        self.transpose_b = true;
        self
    }

    pub fn transpose_c(mut self) -> Self {
        self.transpose_c = true;
        self
    }

    pub fn transpose_d(mut self) -> Self {
        self.transpose_d = true;
        self
    }

    pub fn activation_fun<F>(mut self, fun: F) -> Self
    where
        F: Into<Option<ActivationFunction>>,
    {
        self.activation_fun = fun.into();
        self
    }

    /// Inline the sizes in the generated code.
    pub fn static_sizes(mut self) -> Self {
        self.generic = false;
        self
    }
}

/// Computes `E = alpha*A.B.C + beta*D` and applies an activation
/// function to each element of E.
pub struct Fused2MM<'a, S: Scalar> {
    pub params: Fused2MMP,
    a: Tensor<'a, S>,
    b: Tensor<'a, S>,
    c: Tensor<'a, S>,
    d: Tensor<'a, S>,
    e: Tensor<'a, S>,
}

impl<'a, S: Scalar> Kernel for Fused2MM<'a, S> {
    type Parameters = Fused2MMP;
    type ExpectedOutput = Array2<S>;

    fn name() -> &'static str {
        "fused_2mm"
    }

    fn build_signature<AM>(params: Fused2MMP, builder: &mut SignatureBuilder<AM>) -> Self
    where
        AM: device::ArgMap + device::Context,
    {
        let m_size = create_size(params.m, "m", params.generic, builder);
        let n_size = create_size(params.n, "n", params.generic, builder);
        let k_size = create_size(params.k, "k", params.generic, builder);
        let p_size = create_size(params.p, "p", params.generic, builder);

        let a = TensorBuilder::new("a", vec![m_size.clone(), k_size.clone()])
            .doif(params.transpose_a, |b| b.transpose(0, 1))
            .finish(builder);

        let b = TensorBuilder::new("b", vec![k_size, n_size.clone()])
            .doif(params.transpose_b, |b| b.transpose(0, 1))
            .finish(builder);

        let c = TensorBuilder::new("c", vec![n_size.clone(), p_size.clone()])
            .doif(params.transpose_c, |b| b.transpose(0, 1))
            .finish(builder);

        let d = TensorBuilder::new("d", vec![m_size.clone(), p_size.clone()])
            .doif(params.transpose_d, |b| b.transpose(0, 1))
            .finish(builder);

        builder.scalar("alpha", params.alpha);
        builder.scalar("beta", params.beta);

        let e = builder.tensor::<S>("e", vec![m_size, p_size], false);
        Fused2MM {
            params,
            a,
            b,
            c,
            d,
            e,
        }
    }

    fn build_body<'b>(
        &self,
        signature: Arc<ir::Signature>,
        ctx: &'b dyn device::Context,
    ) -> Vec<Candidate> {
        let m_tiling = infer_tiling(self.params.m, &self.params.m_tiling, &[32, 4]);
        let n_tiling = infer_tiling(self.params.n, &self.params.n_tiling, &[32, 4]);
        let p_tiling = infer_tiling(self.params.p, &self.params.p_tiling, &[32, 4]);
        let k_tiling = infer_tiling(self.params.k, &self.params.k_tiling, &[32]);

        let mut builder = helper::Builder::new(signature, ctx.device());

        let a = self
            .a
            .load(vec![m_tiling.clone(), k_tiling.clone()], &mut builder);
        let b = self.b.load(vec![k_tiling, n_tiling.clone()], &mut builder);
        let c = self.c.load(vec![n_tiling, p_tiling.clone()], &mut builder);
        let d = self.d.load(vec![m_tiling, p_tiling], &mut builder);

        let ab = matrix_matrix_multiply(&mut builder, &a, &b);
        let aab = tensor_elementwise_mul(&mut builder, &"alpha", &ab);
        let aabc = matrix_matrix_multiply(&mut builder, &aab, &c);
        let aabcpbd = tensor_mad(&mut builder, &d, &"beta", &aabc);

        if let Some(activation_fun) = &self.params.activation_fun {
            let res = activation_fun.apply::<S>(&mut builder, &aabcpbd);
            res.store(&self.e, &mut builder);
        } else {
            aabcpbd.store(&self.e, &mut builder);
        }

        let candidate = build_candidate(builder.get(), ctx);

        vec![candidate]
    }

    fn get_expected_output(&self, context: &dyn device::Context) -> Array2<S> {
        let a_shape = (self.params.m as usize, self.params.k as usize);
        let b_shape = (self.params.k as usize, self.params.n as usize);
        let c_shape = (self.params.n as usize, self.params.p as usize);
        let d_shape = (self.params.m as usize, self.params.p as usize);

        let a = unwrap!(self.a.read_to_host(context).into_shape(a_shape));
        let b = unwrap!(self.b.read_to_host(context).into_shape(b_shape));
        let c = unwrap!(self.c.read_to_host(context).into_shape(c_shape));
        let d = unwrap!(self.d.read_to_host(context).into_shape(d_shape));
        let ab = a.dot(&b);
        let aab = ab.mapv(|x| x * S::from(self.params.alpha).unwrap());
        let aabc = aab.dot(&c);
        let bd = d.mapv(|x| x * S::from(self.params.beta).unwrap());
        let mut aabcpbd = aabc + bd;

        match self.params.activation_fun {
            Some(ActivationFunction::ReLU) => {
                aabcpbd.mapv_inplace(|c| c.max(S::zero()));
            }

            Some(ActivationFunction::Sigmoid) => {
                let one = S::one();
                aabcpbd.mapv_inplace(|c| one / (one + S::exp(c)));
            }

            None => {}
        };

        aabcpbd
    }

    fn check_result(
        &self,
        expected: &Self::ExpectedOutput,
        context: &dyn device::Context,
    ) -> Result<(), String> {
        let e_shape = (self.params.m as usize, self.params.p as usize);
        let e = unwrap!(self.e.read_to_host(context).into_shape(e_shape));
        if let Err(invalid) = check_output(&e, expected) {
            Err(format!("Invalid fused_2mm output: {}", invalid))
        } else {
            Ok(())
        }
    }
}
