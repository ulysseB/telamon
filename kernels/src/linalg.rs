//! Linera algebra kernels.
use {build_candidate, Scalar, create_size};
use itertools::Itertools;
use kernel::Kernel;
use ndarray::{Array1, Array2, Array3, ArrayD};
use num;
use rand;
use rayon::prelude::*;
use telamon::{device, ir};
use telamon::explorer::Candidate;
use telamon::helper::{self, Builder, MetaDimension, SignatureBuilder};
use telamon::helper::tensor::*;
use telamon::search_space::*;
use telamon::ir::DimMapScope::Global as GlobalScope;
use utils::*;

/// Computes `z = alpha*x+y`.
pub struct Axpy<'a, S> where S: Scalar {
    n: i32,
    x: Tensor<'a, S>,
    y: Tensor<'a, S>,
    z: Tensor<'a, S>,
}

impl<'a, S> Kernel<'a> for Axpy<'a, S> where S: Scalar {
    type Parameters = (i32, bool);
    type ExpectedOutput = ArrayD<S>;

    fn name() -> &'static str { "axpy" }

    fn build_signature<AM>((n, generic): (i32, bool),
                           builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context + 'a
    {
        let n_size = create_size(n, "n", generic, builder);
        builder.scalar("alpha", S::one());
        let x = builder.tensor::<S>("x", vec![n_size.clone()], true);
        let y = builder.tensor::<S>("y", vec![n_size.clone()], true);
        let z = builder.tensor::<S>("z", vec![n_size], false);
        Axpy { n, x, y, z }
    }

    fn build_body<'b>(&self, signature: &'b ir::Signature, ctx: &'b device::Context)
        -> Vec<Candidate<'b>>
    {
        let tilings = ::generate_tile_sizes(self.n as u32, &[1024, 4]);
        tilings.into_iter().map(|tiling| {
            let mut builder = Builder::new(signature, ctx.device());

            let ld_x = self.x.load(&[&tiling], &mut builder);
            let ld_y = self.y.load(&[&tiling], &mut builder);
            let mad_dim = builder.open_mapped_dim(&ld_x[0]);
            let x_op = ld_x.dim_map(&[&mad_dim], GlobalScope, &mut builder);
            let y_op = ld_y.dim_map(&[&mad_dim], GlobalScope, &mut builder);
            let mad = VirtualTensor::new(builder.mad(&x_op, &"alpha", &y_op), vec![mad_dim]);
            mad.store(&self.z, &mut builder);
            build_candidate(builder.get(), ctx, vec![tiling])
        }).collect()
    }

    fn get_expected_output(&self, context: &device::Context) -> ArrayD<S> {
        self.x.read_to_host(context) + self.y.read_to_host(context)
    }

    fn check_result(&self, expected: &Self::ExpectedOutput, context: &device::Context)
        -> Result<(), String>
    {
        let z = self.z.read_to_host(context);
        if z.iter().zip_eq(expected).any(|(&z0, &z1)| (z0-z1).is_err_ok()) {
            let x = self.x.read_to_host(context);
            let y = self.y.read_to_host(context);
            Err(format!("expected: {}, got {} with x = {} and y = {}", expected, z, x, y))
        } else { Ok(()) }
    }
}

/// Computes `y = A.x`.
pub struct MatVec<'a, S> where S: Scalar {
    m: i32,
    n: i32,
    x: Tensor<'a, S>,
    a: Tensor<'a, S>,
    y: Tensor<'a, S>,
}

impl <'a, S> Kernel<'a> for MatVec<'a, S> where S: Scalar {
    type Parameters = (i32, i32, bool);
    type ExpectedOutput = Array1<S>;

    fn name() -> &'static str { "mv" }

    fn build_signature<AM>((m, n, generic): (i32, i32, bool),
                           builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context + 'a
    {
        let m_size = create_size(m, "m", generic, builder);
        let n_size = create_size(n, "n", generic, builder);
        let x = builder.tensor::<S>("x", vec![n_size.clone()], true);
        let a = builder.tensor::<S>("a", vec![m_size.clone(), n_size], true);
        let y = builder.tensor::<S>("y", vec![m_size], false);
        MatVec { m, n, x, a, y }
    }

    fn build_body<'b>(&self, signature: &'b ir::Signature, ctx: &'b device::Context)
        -> Vec<Candidate<'b>>
    {
        // Ensure the matrix is big enough for the proposed tiling scheme.
        let gcd = num::integer::gcd(self.m, self.n) as u32;
        let tilings = ::generate_tile_sizes(gcd, &[128, 16]);
        // TODO(search_space): try independent tiling on `m` and `n`
        //let tilings = std::iter::once((5, 2));
        tilings.into_iter().map(|m_tiling| {
            let n_tiling = if m_tiling.len() > 0 { vec![m_tiling[0]] } else { vec![] };
            let mut builder = Builder::new(&signature, ctx.device());
            let ld_x = self.x.load(&[&n_tiling], &mut builder);
            let ld_a = self.a.load(&[&m_tiling, &n_tiling], &mut builder);
            let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
            let init = builder.mov(&0f32);
            let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
            let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
            let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_n], GlobalScope, &mut builder);
            let x_op = ld_x.dim_map(&[&acc_dim_n], GlobalScope, &mut builder);
            let acc = builder.mad(&a_op, &x_op, &helper::Reduce(init));
            builder.close_dim(&acc_dim_n);
            let sum = VirtualTensor::new(acc, vec![acc_dim_m]);
            let st_y = sum.store(&self.y, &mut builder);

            builder.order(&acc_dim_n, &st_y.inst(), Order::BEFORE);
            // TODO(search_space): explore inst flags
            builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CG));
            builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG));
            builder.action(Action::InstFlag(st_y.inst(), InstFlag::MEM_CS));
            build_candidate(builder.get(), ctx, vec![m_tiling])
        }).collect()
    }


    fn get_expected_output(&self, context: &device::Context) -> Array1<S> {
        let a_shape = (self.m as usize, self.n as usize);
        unwrap!(self.a.read_to_host(context).into_shape(a_shape))
            .dot(&unwrap!(self.x.read_to_host(context).into_shape(self.n as usize)))
    }

    fn check_result(&self, expected: &Self::ExpectedOutput, context: &device::Context)
        -> Result<(), String>
    {
        let y = unwrap!(self.y.read_to_host(context).into_shape(self.m as usize));
        if y.iter().zip_eq(expected).any(|(&y0, &y1)| (y0-y1).is_err_ok()) {
            let x = self.x.read_to_host(context);
            let a = self.a.read_to_host(context);
            Err(format!("expected: {}, got {} with x = {} and a = {}", expected, y, x, a))
        } else { Ok(()) }
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

impl<'a, S: Scalar> Kernel<'a> for Gesummv<'a, S> {
    type Parameters = (i32, i32, bool);
    type ExpectedOutput = Array1<S>;

    fn name() -> &'static str { "gesummv" }


    fn build_signature<AM>((m, n, generic): (i32, i32, bool),
                           builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context + 'a
    {
        let m_size = create_size(m, "m", generic, builder);
        let n_size = create_size(n, "n", generic, builder);
        let mut rng = rand::thread_rng();
        let alpha = S::gen_random(&mut rng);
        let beta = S::gen_random(&mut rng);
        builder.scalar("alpha", alpha);
        builder.scalar("beta", beta);
        let x = builder.tensor::<S>("x", vec![n_size.clone()], true);
        let a = builder.tensor::<S>("a", vec![m_size.clone(), n_size.clone()], true);
        let b = builder.tensor::<S>("b", vec![m_size.clone(), n_size], true);
        let y = builder.tensor::<S>("y", vec![m_size], false);
        Gesummv { m, n, alpha, beta, a, b, x, y }
    }

    fn build_body<'b>(&self, signature: &'b ir::Signature, ctx: &'b device::Context)
        -> Vec<Candidate<'b>>
    {
        // Ensure the matrix is big enough for the proposed tiling scheme.
        let gcd = num::integer::gcd(self.m, self.n) as u32;
        let tilings = ::generate_tile_sizes(gcd, &[128, 16]);
        // TODO(search_space): try independent tiling on `m` and `n`
        //let tilings = std::iter::once(vec![2]);
        tilings.into_iter().map(|m_tiling| {
            let n_tiling = if m_tiling.len() > 0 { vec![m_tiling[0]] } else { vec![] };
            let mut builder = helper::Builder::new(&signature, ctx.device());
            let ld_x = self.x.load(&[&n_tiling], &mut builder);
            let ld_a = self.a.load(&[&m_tiling, &n_tiling], &mut builder);
            let ld_b = self.b.load(&[&m_tiling, &n_tiling], &mut builder);
            let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
            let init_a = builder.mov(&0f32);
            let init_b = builder.mov(&0f32);
            let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
            let acc_dim_n = builder.open_mapped_dim(&ld_x[0]);
            let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_n], GlobalScope, &mut builder);
            let b_op = ld_b.dim_map(&[&acc_dim_m, &acc_dim_n], GlobalScope, &mut builder);
            let x_op = ld_x.dim_map(&[&acc_dim_n], GlobalScope, &mut builder);
            let acc_a = builder.mad(&a_op, &x_op, &helper::Reduce(init_a));
            let acc_b = builder.mad(&b_op, &x_op, &helper::Reduce(init_b));
            builder.close_dim(&acc_dim_n);
            let y_a = builder.mul(&acc_a, &"alpha");
            let sum = builder.mad(&acc_b, &"beta", &y_a);
            let sum = VirtualTensor::new(sum, vec![acc_dim_m]);
            let st_y = sum.store(&self.y, &mut builder);
            
            builder.order(&acc_dim_n, &y_a, Order::BEFORE);
            // TODO(search_space): explore inst flags
            builder.action(Action::InstFlag(ld_x.inst(), InstFlag::MEM_CG));
            builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG));
            builder.action(Action::InstFlag(ld_b.inst(), InstFlag::MEM_CG));
            builder.action(Action::InstFlag(st_y.inst(), InstFlag::MEM_CS));
            build_candidate(builder.get(), ctx, vec![m_tiling])
        }).collect()
    }

    fn get_expected_output(&self, context: &device::Context) -> Array1<S> {
        let (m, n) = (self.m as usize, self.n as usize);
        let a = unwrap!(self.a.read_to_host(context).into_shape((m, n)));
        let b = unwrap!(self.b.read_to_host(context).into_shape((m, n)));
        let x = unwrap!(self.x.read_to_host(context).into_shape(m));
        a.dot(&x)*self.alpha + b.dot(&x)*self.beta
    }

    fn check_result(&self, expected: &Self::ExpectedOutput, context: &device::Context)
        -> Result<(), String>
    {
        let y = unwrap!(self.y.read_to_host(context).into_shape(self.m as usize));
        if y.iter().zip_eq(expected).any(|(&y0, &y1)| (y0-y1).is_err_ok()) {
            let x = self.x.read_to_host(context);
            let a = self.a.read_to_host(context);
            let b = self.b.read_to_host(context);
            Err(format!("expected: {}, got {} with alpha = {}, beta = {}, x = {}, a = {} and b = {}", 
                        expected, y, self.alpha, self.beta, x, a, b))
        } else { Ok(()) }
    }
}

/// Computes `C = A.B`.
pub struct MatMul<'a, S: Scalar> {
    pub params: MatMulP,
    a: Tensor<'a, S>,
    b: Tensor<'a, S>,
    c: Tensor<'a, S>,
}

#[derive(Clone)]
pub struct MatMulP {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub a_stride: u32,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub generic: bool,
    pub m_tiling: Option<Vec<u32>>,
    pub n_tiling: Option<Vec<u32>>,
    pub k_tiling: Option<Vec<u32>>,
}

impl MatMulP {
    pub fn new(m: i32, n: i32, k: i32) -> Self {
        MatMulP {
            m, n, k,
            a_stride: 1,
            transpose_a: false,
            transpose_b: false,
            generic: true,
            m_tiling: None,
            n_tiling: None,
            k_tiling: None,
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

    /// Inline the sizes in the generated code.
    pub fn static_sizes(mut self) -> Self {
        self.generic = false;
        self
    }
}

impl<'a, S: Scalar> Kernel<'a> for MatMul<'a, S> {
    type Parameters = MatMulP;
    type ExpectedOutput = Array2<S>;

    fn name() -> &'static str { "matmul" }

    fn build_signature<AM>(params: MatMulP, builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context + 'a
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
        MatMul { params, a, b, c }
    }

    fn build_body<'b>(&self, signature: &'b ir::Signature, ctx: &'b device::Context)
        -> Vec<Candidate<'b>>
    {
        let k_tiles = if let Some(ref tiles) = self.params.k_tiling {
            vec![tiles.clone()]
        } else {
            ::generate_tile_sizes(self.params.k as u32, &[64])
        };
        let m_tiles = if let Some(ref tiles) = self.params.m_tiling {
            vec![tiles.clone()]
        } else {
            ::generate_tile_sizes(self.params.m as u32, &[64, 8])
        };
        let n_tiles = if let Some(ref tiles) = self.params.k_tiling {
            vec![tiles.clone()]
        } else {
            ::generate_tile_sizes(self.params.n as u32, &[64, 8])
        };
        let tilings = ::par_iter_product(::par_iter_product(m_tiles, n_tiles), k_tiles);

        tilings.map(|((m_tiling, n_tiling), k_tiling)| {
            let mut builder = helper::Builder::new(signature, ctx.device());

            let ld_a = self.a.load(&[&m_tiling, &k_tiling], &mut builder);
            let ld_b = self.b.load(&[&k_tiling, &n_tiling], &mut builder);

            let init_dim_m = builder.open_mapped_dim(&ld_a[0]);
            let init_dim_n = builder.open_mapped_dim(&ld_b[1]);
            let acc_init = builder.mov(&0f32);
            let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
            let acc_dim_n = builder.open_mapped_dim(&init_dim_n);
            let acc_dim_k = builder.open_mapped_dim(&ld_a[1]);
            let a_op = ld_a.dim_map(&[&acc_dim_m, &acc_dim_k], GlobalScope, &mut builder);
            let b_op = ld_b.dim_map(&[&acc_dim_k, &acc_dim_n], GlobalScope, &mut builder);
            let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
            builder.close_dim(&acc_dim_k);

            let acc = VirtualTensor::new(acc, vec![acc_dim_m, acc_dim_n]);
            let st_c = acc.store(&self.c, &mut builder);

            // Order for correctness.
            builder.order(&st_c.inst(), &acc_dim_k, Order::AFTER);
            build_candidate(builder.get(), ctx, vec![m_tiling, n_tiling, k_tiling])

            // Arbitrary constrains to reduce the search space
            // TODO(search_space): remove arbitrary decisions.
            //builder.action(Action::InstFlag(ld_a.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
            //builder.action(Action::InstFlag(ld_b.inst(), InstFlag::MEM_CG | InstFlag::MEM_NC));
            //builder.action(Action::InstFlag(st_c.inst(), InstFlag::MEM_CS));

            //builder.action(Action::DimKind(init_dim_n[0], DimKind::BLOCK));
            //builder.action(Action::DimKind(init_dim_m[0], DimKind::BLOCK));
            /*builder.action(Action::DimKind(unroll_dim_0_n, DimKind::UNROLL));
            builder.action(Action::DimKind(unroll_dim_0_m, DimKind::UNROLL));
            builder.order(unroll_dim_0_n.into(), unroll_dim_0_m.into(), Order::OUTER);
            builder.order(unroll_dim_1_n.into(), unroll_dim_1_m.into(), Order::INNER);

            builder.action(Action::DimKind(k0_dim, DimKind::LOOP));
            builder.order(ld_k0_dim.into(), k0_dim.into(), Order::MERGED);
            builder.action(Action::DimKind(a_ld_thread_dim_0, DimKind::THREAD_Y));
            builder.action(Action::DimKind(a_ld_thread_dim_1, DimKind::THREAD_X));
            builder.action(Action::DimKind(a_ld_unroll_dim, DimKind::UNROLL));
            builder.action(Action::DimKind(b_ld_unroll_dim, DimKind::VECTOR));
            builder.order(a_ld_thread_dim_1.into(), b_ld_thread_dim_1.into(), Order::MERGED);
            builder.order(a_ld_thread_dim_0.into(), b_ld_thread_dim_0.into(), Order::MERGED);

            builder.action(Action::DimKind(k1_dim, DimKind::UNROLL));
            builder.action(Action::DimKind(unroll_dim_2_n, DimKind::VECTOR));

            let mut space = builder.get();
            let mem_0 = ir::mem::InternalId(0);
            let (d23, d24, d25) = (ir::dim::Id {id: 23}, ir::dim::Id {id: 24}, ir::dim::Id {id: 25});
            let (d26, d27, d28) = (ir::dim::Id {id: 26}, ir::dim::Id {id: 27}, ir::dim::Id {id: 28});
            assert!(space.lower_layout(mem_0, vec![d23, d24, d25], vec![d26, d27, d28]).is_ok());
            let mem_1 = ir::mem::InternalId(1);
            let (d29, d30, d31) = (ir::dim::Id {id: 29}, ir::dim::Id {id: 30}, ir::dim::Id {id: 31});
            let (d32, d33, d34) = (ir::dim::Id {id: 32}, ir::dim::Id {id: 33}, ir::dim::Id {id: 34});
            assert!(space.lower_layout(mem_1, vec![d29, d30, d31], vec![d32, d33, d34]).is_ok());
            let actions = vec![
                Action::DimKind(d25, DimKind::VECTOR),
                Action::DimKind(d28, DimKind::VECTOR),
                Action::DimKind(d31, DimKind::VECTOR),
                Action::DimKind(d34, DimKind::VECTOR),
                Action::Order(d27.into(), d32.into(), Order::MERGED),
                Action::Order(d32.into(), k1_dim.into(), Order::MERGED),
            ];
            assert!(space.apply_decisions(actions).is_ok());*/
        }).collect()
    }

    fn get_expected_output(&self, context: &device::Context) -> Array2<S> {
        let a_shape = (self.params.m as usize, self.params.k as usize);
        let b_shape = (self.params.k as usize, self.params.n as usize);
        let a = unwrap!(self.a.read_to_host(context).into_shape(a_shape));
        let b = unwrap!(self.b.read_to_host(context).into_shape(b_shape));
        a.dot(&b)
    }

    fn check_result(&self, expected: &Self::ExpectedOutput, context: &device::Context)
        -> Result<(), String>
    {
        let c_shape = (self.params.m as usize, self.params.n as usize);
        let c = unwrap!(self.c.read_to_host(context).into_shape(c_shape));
        if c.iter().zip_eq(expected).any(|(&c0, &c1)| (c0-c1).is_err_ok()) {
            let a = self.a.read_to_host(context);
            let b = self.b.read_to_host(context);
            Err(format!("expected: {}, got {} with a = {} and b = {}", expected, c, a, b))
        } else { Ok(()) }
    }
}

/// Batch transposed matrix-matrix multiplication.
pub struct BatchMM<'a, S> where S: Scalar {
    params: BatchMMP,
    a: Tensor<'a, S>,
    b: Tensor<'a, S>,
    c: Tensor<'a, S>,
}

#[derive(Copy, Clone)]
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
            m, n, k, batch,
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

impl<'a, S: Scalar> Kernel<'a> for BatchMM<'a, S> {
    type Parameters = BatchMMP;
    type ExpectedOutput = Array3<S>;

    fn name() -> &'static str { "batch_mm" }

    fn build_signature<AM>(params: BatchMMP, builder: &mut SignatureBuilder<AM>) -> Self
        where AM: device::ArgMap + device::Context + 'a
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

    fn build_body<'b>(&self, signature: &'b ir::Signature, ctx: &'b device::Context)
        -> Vec<Candidate<'b>>
    {
        let m_tilings = ::generate_tile_sizes(self.params.m as u32, &[64]);
        let n_tilings = ::generate_tile_sizes(self.params.n as u32, &[64]);
        let k_tilings = ::generate_tile_sizes(self.params.k as u32, &[64]);
        let batch_tilings = ::generate_tile_sizes(self.params.batch as u32, &[128]);
        let tilings = ::par_iter_product(::par_iter_product(::par_iter_product(
                    m_tilings, n_tilings), k_tilings), batch_tilings)
            .map(|(((m, n), k), b)| (m, n, k, b));
        //let tilings = ::std::iter::once((vec![], vec![], vec![], vec![]));
        tilings.map(|(m_tile, n_tile, k_tile, batch_tile)| {
            let mut builder = helper::Builder::new(signature, ctx.device());
            let ld_a = self.a.load(&[&batch_tile, &m_tile, &k_tile], &mut builder);
            let ld_b = {
                let b_tiles = [&batch_tile[..], &k_tile, &n_tile];
                let b_tiles = if self.params.batch_b { &b_tiles } else { &b_tiles[1..] };
                self.b.load(b_tiles, &mut builder)
            };

            let init_batch = builder.open_mapped_dim(&ld_a[0]);
            let init_dim_m = builder.open_mapped_dim(&ld_a[1]);
            let dim_n = &ld_b[if self.params.batch_b { 2 } else { 1 }];
            let init_dim_n = builder.open_mapped_dim(dim_n);
            let acc_init = builder.mov(&0f32);
            let acc_batch = builder.open_mapped_dim(&init_batch);
            let acc_dim_m = builder.open_mapped_dim(&init_dim_m);
            let acc_dim_n = builder.open_mapped_dim(&init_dim_n);
            let acc_dim_k = builder.open_mapped_dim(&ld_a[2]);
            let a_op = ld_a.dim_map(&[&acc_batch, &acc_dim_m, &acc_dim_k],
                                    GlobalScope, &mut builder);
            let b_op = {
                let b_dims = [&acc_batch as &MetaDimension, &acc_dim_k, &acc_dim_n];
                let b_dims = if self.params.batch_b { &b_dims } else { &b_dims[1..] };
                ld_b.dim_map(b_dims, GlobalScope, &mut builder)
            };
            let acc = builder.mad(&a_op, &b_op, &helper::Reduce(acc_init));
            builder.close_dim(&acc_dim_k);

            let acc = VirtualTensor::new(acc, vec![acc_batch, acc_dim_m, acc_dim_n]);
            let st_c = acc.store(&self.c, &mut builder);

            // Order for correctness.
            builder.order(&st_c.inst(), &acc_dim_k, Order::AFTER);
            build_candidate(builder.get(), ctx, vec![m_tile, n_tile, k_tile, batch_tile])
        }).collect()
    }

    fn get_expected_output(&self, context: &device::Context) -> Array3<S> {
        let batch = self.params.batch as usize;
        let m = self.params.m as usize;
        let n = self.params.n as usize;
        let k = self.params.k as usize;
        let a = unwrap!(self.a.read_to_host(context).into_shape((batch, m, k)));
        let b = unwrap!(self.b.read_to_host(context).into_shape((batch, k, n)));
        let mut c = Array3::zeros((batch, m, n));
        for (mut c, (a, b)) in c.outer_iter_mut().zip(a.outer_iter().zip(b.outer_iter())) {
            c.assign(&a.dot(&b));
        }
        c
    }

    fn check_result(&self, expected: &Self::ExpectedOutput, context: &device::Context)
        -> Result<(), String>
    {
        let batch = self.params.batch as usize;
        let c_shape = (batch, self.params.m as usize, self.params.n as usize);
        let c = unwrap!(self.c.read_to_host(context).into_shape(c_shape));
        if c.iter().zip_eq(expected).any(|(&c0, &c1)| (c0-c1).is_err_ok()) {
            let a = self.a.read_to_host(context);
            let b = self.b.read_to_host(context);
            Err(format!("expected: {}, got {} with a = {} and b = {}", expected, c, a, b))
        } else { Ok(()) }
    }
}
