//! Tests the memory model.
use telamon::device::{ArgMap, Context};
use telamon::helper::{SignatureBuilder, Builder, DimGroup, Reduce};
use telamon::ir;
use telamon::search_space::{Action, DimKind, InstFlag, Order};
use PerfModelTest;

/// Tests the model in presence of global access replay.
pub struct L1LinesPressure;

impl PerfModelTest for L1LinesPressure {
    fn name() -> &'static str { "l1_lines_pressure" }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("n", 100i32);
        builder.array::<f32>("array", 128*32*32*32);
        builder.array::<f32>("out", 1);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        const UNROLL: u32 = 128;
        const THREAD_X: u32 = 32;
        const THREAD_Y: u32 = 32;
        const STRIDE: u32 = 32;

        let t = ir::Type::F(32);
        let size_n = builder.param_size("n");
        let d1_0 = builder.open_dim_ex(ir::Size::new(THREAD_Y, vec![], 1), DimKind::THREAD);
        let d2_0 = builder.open_dim_ex(ir::Size::new(THREAD_X, vec![], 1), DimKind::THREAD);
        let init = builder.mov(&0f32);

        let d0 = builder.open_dim_ex(size_n.clone(), DimKind::LOOP);
        let d1_1 = builder.open_mapped_dim(&d1_0)[0];
        let d2_1 = builder.open_mapped_dim(&d2_0)[0];
        let d3 = builder.open_dim_ex(ir::Size::new(UNROLL, vec![], 1), DimKind::UNROLL);
        let strides = vec![
            (d3, ir::Size::new(THREAD_Y*THREAD_X*32*4, vec![], 1)),
            (d1_1, ir::Size::new(THREAD_X*32*4, vec![], 1)),
            (d2_1, ir::Size::new(STRIDE*4, vec![], 1)),
        ];
        let pattern = ir::AccessPattern::Tensor {
            mem_id: ir::mem::Id::External(0),
            dims: strides.iter().cloned().collect(),
        };
        let addr = builder.induction_var(&"array", strides);
        let val = builder.ld_ex(t, &addr, pattern, InstFlag::MEM_CG);
        let acc = builder.add(&val, &Reduce(init));
        builder.close_dim(&d0);
        builder.close_dim(&d3);

        let d1_2 = builder.open_mapped_dim(&d1_1)[0];
        let d2_2 = builder.open_mapped_dim(&d2_1)[0];
        let out_pattern = builder.unknown_access_pattern(ir::mem::Id::External(1));
        builder.st_ex(&"out", &acc, true, out_pattern, InstFlag::MEM_CS);

        builder.order(&d1_0, &d2_0, Order::OUTER);
        builder.order(&d1_0, &d0, Order::BEFORE);
        builder.order(&d0, &d1_1, Order::OUTER);
        builder.order(&d1_1, &d2_1, Order::OUTER);
        builder.order(&d0, &d1_2, Order::BEFORE);
        builder.order(&d1_2, &d2_2, Order::OUTER);
        L1LinesPressure
    }
}

/// Tests the model in presence of global access replay.
pub struct L2LinesPressure;

impl PerfModelTest for L2LinesPressure {
    fn name() -> &'static str { "l2_lines_pressure" }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("n", 100i32);
        builder.array::<f32>("array", 128*32*32*8);
        builder.array::<f32>("out", 1);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        const UNROLL: u32 = 128;
        const THREAD_X: u32 = 32;
        const THREAD_Y: u32 = 32;
        const STRIDE: u32 = 8;

        let t = ir::Type::F(32);
        let size_n = builder.param_size("n");
        let d1_0 = builder.open_dim_ex(ir::Size::new(THREAD_Y, vec![], 1), DimKind::THREAD);
        let d2_0 = builder.open_dim_ex(ir::Size::new(THREAD_X, vec![], 1), DimKind::THREAD);
        let init = builder.mov(&0f32);

        let d0 = builder.open_dim_ex(size_n.clone(), DimKind::LOOP);
        let d1_1 = builder.open_mapped_dim(&d1_0)[0];
        let d2_1 = builder.open_mapped_dim(&d2_0)[0];
        let d3 = builder.open_dim_ex(ir::Size::new(UNROLL, vec![], 1), DimKind::UNROLL);
        let strides = vec![
            (d3, ir::Size::new(THREAD_Y*THREAD_X*8*4, vec![], 1)),
            (d1_1, ir::Size::new(THREAD_X*8*4, vec![], 1)),
            (d2_1, ir::Size::new(STRIDE*4, vec![], 1)),
        ];
        let pattern = ir::AccessPattern::Tensor {
            mem_id: ir::mem::Id::External(0),
            dims: strides.iter().cloned().collect(),
        };
        let addr = builder.induction_var(&"array", strides);
        let val = builder.ld_ex(t, &addr, pattern, InstFlag::MEM_CG);
        let acc = builder.add(&val, &Reduce(init));
        builder.close_dim(&d0);
        builder.close_dim(&d3);

        let d1_2 = builder.open_mapped_dim(&d1_1)[0];
        let d2_2 = builder.open_mapped_dim(&d2_1)[0];
        let out_pattern = builder.unknown_access_pattern(ir::mem::Id::External(1));
        builder.st_ex(&"out", &acc, true, out_pattern, InstFlag::MEM_CS);

        builder.order(&d1_0, &d2_0, Order::OUTER);
        builder.order(&d1_0, &d0, Order::BEFORE);
        builder.order(&d0, &d1_1, Order::OUTER);
        builder.order(&d1_1, &d2_1, Order::OUTER);
        builder.order(&d0, &d1_2, Order::BEFORE);
        builder.order(&d1_2, &d2_2, Order::OUTER);
        L2LinesPressure
    }
}

pub struct SharedLoad {
    d0: ir::dim::Id,
    d1: ir::dim::Id,
    d2: ir::dim::Id,
    d3: ir::dim::Id,
}

impl PerfModelTest for SharedLoad {
    fn name() -> &'static str { "shared_load" }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("n", 1000i32);
        builder.scalar("arg_zero", 0i32);
        builder.array::<f32>("out", 1);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let size_0 = builder.cst_size(32);
        let size_1 = builder.cst_size(32);
        let size_2 = builder.param_size("n");
        let mem = builder.allocate_shared(8*32*32*4);
        let d0 = builder.open_dim_ex(size_0, DimKind::THREAD);
        let d1 = builder.open_dim_ex(size_1, DimKind::THREAD);
        let ptr_to_mem_type = builder.type_of(&mem);
        let ptr_zero = builder.cast(&"arg_zero", ptr_to_mem_type);
        let idx = builder.mad(&d1, &32i32, &d0);
        let ptr_0 = builder.mad(&idx, &8i32, &mem);
        let acc_0 = builder.mov(&0f32);
        let d2 = builder.open_dim_ex(size_2, DimKind::LOOP);
        let d3_size = builder.cst_size(100);
        let d3 = builder.open_dim_ex(d3_size, DimKind::UNROLL);
        let ptr = builder.add(&Reduce(ptr_0), &ptr_zero);
        let pattern = builder.tensor_access_pattern(
            mem.into(), &ir::Type::F(64), &[&d1, &d0]);
        let ld = builder.ld(ir::Type::F(32), &ptr, pattern);
        let acc = builder.add(&Reduce(acc_0), &ld);
        builder.close_dim(&d2);
        builder.close_dim(&d3);
        let out_pattern = builder.unknown_access_pattern(ir::mem::Id::External(0));
        builder.st_ex(&"out", &acc, true, out_pattern, InstFlag::MEM_CS);
        builder.order(&ptr_zero, &idx, Order::BEFORE);
        SharedLoad { d0, d1, d2, d3 }
    }

    fn get_actions(&self) -> Vec<Action> {
        vec![Action::Order(self.d0.into(), self.d1.into(), Order::OUTER),
             Action::Order(self.d1.into(), self.d2.into(), Order::OUTER),
             Action::Order(self.d2.into(), self.d3.into(), Order::OUTER)]
    }
}

pub struct VectorSharedLoad {
    d0: ir::dim::Id,
    d1: ir::dim::Id,
    d2: ir::dim::Id,
    d3: ir::dim::Id,
}

impl PerfModelTest for VectorSharedLoad {
    fn name() -> &'static str { "vector_shared_load" }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("n", 1000i32);
        builder.scalar("arg_zero", 0i32);
        builder.array::<f32>("out", 1);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let size_0 = builder.cst_size(32);
        let size_1 = builder.cst_size(32);
        let size_2 = builder.param_size("n");
        let mem = builder.allocate_shared(64*4*4);
        let d0 = builder.open_dim_ex(size_0, DimKind::THREAD);
        let d1 = builder.open_dim_ex(size_1, DimKind::THREAD);
        let acc_0 = builder.mov(&0f32);
        let d2 = builder.open_dim_ex(size_2, DimKind::LOOP);
        let d3 = builder.open_dim_ex(ir::Size::new(64, vec![], 1), DimKind::UNROLL);
        let d4 = builder.open_dim_ex(ir::Size::new(4, vec![], 1), DimKind::VECTOR);
        let (addr, pattern) = builder.tensor_access(
            &mem, mem.into(), &ir::Type::F(32), &[&d3, &d4]);
        let ld = builder.ld(ir::Type::F(32), &addr, pattern);
        let d4_2 = builder.open_mapped_dim(&d4)[0];
        let acc = builder.add(&Reduce(acc_0), &ld);
        builder.close_dim(&DimGroup::new(vec![d2, d3, d4_2]));
        let out_pattern = builder.unknown_access_pattern(ir::mem::Id::External(0));
        builder.st_ex(&"out", &acc, true, out_pattern, InstFlag::MEM_CS);

        VectorSharedLoad { d0, d1, d2, d3 }
    }

    fn get_actions(&self) -> Vec<Action> {
        vec![Action::Order(self.d0.into(), self.d1.into(), Order::OUTER),
             Action::Order(self.d1.into(), self.d2.into(), Order::OUTER),
             Action::Order(self.d2.into(), self.d3.into(), Order::OUTER)]
    }
}

pub struct SharedReplay;

impl PerfModelTest for SharedReplay {
    fn name() -> &'static str { "shared_replay" }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("n", 1000i32);
        builder.scalar("arg_zero", 0i32);
        builder.array::<f32>("out", 1);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let size_0 = builder.cst_size(32);
        let size_1 = builder.cst_size(32);
        let size_2 = builder.param_size("n");
        let mem = builder.allocate_shared(8*32*32*4);
        let d0 = builder.open_dim_ex(size_0, DimKind::THREAD);
        let d1 = builder.open_dim_ex(size_1, DimKind::THREAD);
        let ptr_to_mem_type = builder.type_of(&mem);
        let ptr_zero = builder.cast(&"arg_zero", ptr_to_mem_type);
        let init = builder.mov(&0f32);
        let idx = builder.mad(&d0, &32i32, &d1);
        let addr_0 = builder.mad(&idx, &8i32, &mem);
        let pattern = builder.tensor_access_pattern(
            mem.into(), &ir::Type::F(64), &[&d0, &d1]);
        let d2 = builder.open_dim_ex(size_2, DimKind::LOOP);
        let d4 = builder.open_dim_ex(ir::Size::new(32, vec![], 1), DimKind::UNROLL);
        let d3_0 = builder.open_dim_ex(ir::Size::new(4, vec![], 1), DimKind::UNROLL);
        let addr = builder.add(&Reduce(addr_0), &ptr_zero);
        let val = builder.ld(ir::Type::F(32), &addr, pattern);
        let d3_1 = builder.open_mapped_dim(&d3_0)[0];
        let acc = builder.add(&val, &Reduce(init));
        builder.close_dim(&DimGroup::new(vec![d2, d4, d3_1]));
        let out_pattern = builder.unknown_access_pattern(ir::mem::Id::External(0));

        builder.st_ex(&"out", &acc, true, out_pattern, InstFlag::MEM_CS);
        builder.order(&d0, &d1, Order::OUTER);
        builder.order(&d1, &d2, Order::OUTER);
        builder.order(&d4, &d3_0, Order::OUTER);
        builder.order(&d3_0, &d3_1, Order::BEFORE);
        builder.order(&ptr_zero, &init, Order::BEFORE);
        SharedReplay
    }
}

pub struct VectorSharedReplay;

impl PerfModelTest for VectorSharedReplay {
    fn name() -> &'static str { "vector_shared_replay" }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("n", 1000i32);
        builder.scalar("arg_zero", 0i32);
        builder.array::<f32>("out", 1);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let size_0 = builder.cst_size(32);
        let size_1 = builder.cst_size(32);
        let size_2 = builder.param_size("n");
        let mem_size = 8*32*32*4;
        let mem = builder.allocate_shared(mem_size);
        let d0 = builder.open_dim_ex(size_0, DimKind::THREAD);
        let d1 = builder.open_dim_ex(size_1, DimKind::THREAD);
        let ptr_to_mem_type = builder.type_of(&mem);
        let ptr_zero = builder.cast(&"arg_zero", ptr_to_mem_type);
        let init = builder.mov(&0f32);
        let idx = builder.mad(&d0, &32i32, &d1);
        let addr_0 = builder.mad(&idx, &16i32, &mem);
        let d2 = builder.open_dim_ex(size_2, DimKind::LOOP);
        let d4 = builder.open_dim_ex(ir::Size::new(32, vec![], 1), DimKind::UNROLL);
        let addr = builder.add(&Reduce(addr_0), &ptr_zero);
        let d3_0 = builder.open_dim_ex(ir::Size::new(4, vec![], 1), DimKind::VECTOR);
        let pattern = builder.tensor_access_pattern(
            mem.into(), &ir::Type::F(32), &[&d0, &d1, &d3_0]);
        let val = builder.ld(ir::Type::F(32), &addr, pattern);
        let d3_1 = builder.open_mapped_dim(&d3_0)[0];
        let acc = builder.add(&val, &Reduce(init));
        builder.close_dim(&DimGroup::new(vec![d2, d4, d3_1]));
        let out_pattern = builder.unknown_access_pattern(ir::mem::Id::External(0));

        builder.st_ex(&"out", &acc, true, out_pattern, InstFlag::MEM_CS);
        builder.order(&d0, &d1, Order::OUTER);
        builder.order(&d1, &d2, Order::OUTER);
        builder.order(&d4, &d3_0, Order::OUTER);
        builder.order(&d3_0, &d3_1, Order::BEFORE);
        builder.order(&ptr_zero, &init, Order::BEFORE);
        VectorSharedReplay
    }
}

// TODO(test): mutlidimentsional global tensor without pressure.
// TODO(test): test RAM bandwidth.
