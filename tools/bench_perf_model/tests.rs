use telamon::device::{ArgMap, Context};
use telamon::helper::*;
use telamon::ir;
use telamon::search_space::*;
use PerfModelTest;

pub struct Test0;

impl Test0 {
    const M: i32 = 1024;

    const N: i32 = 1024;

    const K: i32 = 1024;

    const TILE_1: i32 = 32;

    const TILE_2: i32 = 4;
}

impl PerfModelTest for Test0 {
    fn name() -> &'static str {
        "test_0"
    }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("m", Self::M / (Self::TILE_1 * Self::TILE_2));
        builder.scalar("n", Self::N / (Self::TILE_1 * Self::TILE_2));
        builder.scalar("k", Self::K / Self::TILE_1);
        builder.array::<f32>("a", (Self::M * Self::K) as usize);
        builder.array::<f32>("b", (Self::K * Self::N) as usize);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let tile_1_size = builder.cst_size(Self::TILE_1 as u32);
        let tile_2_size = builder.cst_size(Self::TILE_2 as u32);
        let tmp_mem_size = Self::TILE_1 * Self::TILE_1 * Self::TILE_2;
        let a_tmp_mem = builder.allocate_shared(ir::Type::F(32), tmp_mem_size as u32);
        let b_tmp_mem = builder.allocate_shared(ir::Type::F(32), tmp_mem_size as u32);
        // Configure dimension sizes
        let m_tiled = builder.param_size("m", Self::M as u32);
        let n_tiled = builder.param_size("n", Self::N as u32);
        let k_tiled = builder.param_size("k", Self::K as u32);

        let b0 = builder.open_dim_ex(n_tiled, DimKind::BLOCK);
        let b1 = builder.open_dim_ex(m_tiled, DimKind::BLOCK);
        // Compute AxB in acc.
        let k0_dim = builder.open_dim_ex(k_tiled, DimKind::LOOP);
        let thread_dim_0 = builder.open_dim_ex(tile_1_size.clone(), DimKind::THREAD);
        let thread_dim_1 = builder.open_dim_ex(tile_1_size.clone(), DimKind::THREAD);
        // Load A from global memory
        let a_ld_unroll_dim = builder.open_dim_ex(tile_2_size.clone(), DimKind::UNROLL);
        let (a_addr, a_pattern) = builder.tensor_access(
            &"a",
            None,
            ir::Type::F(32),
            &[&b1, &thread_dim_1, &a_ld_unroll_dim, &k0_dim, &thread_dim_0],
        );
        let a_ld =
            builder.ld_ex(ir::Type::F(32), &a_addr, a_pattern, InstFlag::CACHE_GLOBAL);
        builder.close_dim(&a_ld_unroll_dim);
        // Load B from global memory
        let b_ld_unroll_dim = builder.open_dim_ex(tile_2_size.clone(), DimKind::VECTOR);
        let (b_addr, b_pattern) = builder.tensor_access(
            &"b",
            None,
            ir::Type::F(32),
            &[&k0_dim, &thread_dim_1, &b0, &thread_dim_0, &b_ld_unroll_dim],
        );
        let b_ld =
            builder.ld_ex(ir::Type::F(32), &b_addr, b_pattern, InstFlag::CACHE_GLOBAL);
        builder.close_dim(&b_ld_unroll_dim);
        // Store A in shared memory.
        let a_st_tmp_unroll_dim = builder.open_mapped_dim(&a_ld_unroll_dim);
        let (a_tmp_addr, a_tmp_st_pattern) = builder.tensor_access(
            &a_tmp_mem,
            a_tmp_mem.into(),
            ir::Type::F(32),
            &[&thread_dim_1, &thread_dim_0, &a_st_tmp_unroll_dim],
        );
        builder.st(&a_tmp_addr, &a_ld, a_tmp_st_pattern);
        builder.close_dim(&a_st_tmp_unroll_dim);
        // Store B in shared memory.
        let b_st_tmp_unroll_dim = builder.open_mapped_dim(&b_ld_unroll_dim);
        let (b_tmp_addr, b_tmp_st_pattern) = builder.tensor_access(
            &b_tmp_mem,
            b_tmp_mem.into(),
            ir::Type::F(32),
            &[&thread_dim_1, &thread_dim_0, &b_st_tmp_unroll_dim],
        );
        builder.st(&b_tmp_addr, &b_ld, b_tmp_st_pattern);

        builder.order(&b0, &b1, Order::OUTER);
        builder.order(&k0_dim, &thread_dim_0, Order::OUTER);
        builder.order(&thread_dim_0, &thread_dim_1, Order::OUTER);
        builder.order(&a_ld_unroll_dim, &b_ld_unroll_dim, Order::BEFORE);
        builder.order(&b_ld_unroll_dim, &a_st_tmp_unroll_dim, Order::BEFORE);
        builder.order(&a_st_tmp_unroll_dim, &b_st_tmp_unroll_dim, Order::BEFORE);

        builder.action(Action::DimKind(a_st_tmp_unroll_dim[0], DimKind::VECTOR));
        builder.action(Action::DimKind(b_st_tmp_unroll_dim[0], DimKind::VECTOR));

        Test0
    }
}

pub struct Test1;

impl Test1 {
    const K: i32 = 1024;
}

impl PerfModelTest for Test1 {
    fn name() -> &'static str {
        "test_1"
    }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("k", Self::K);
        builder.array::<f32>("out", 4 * 32 * 32 * 4 as usize);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let tile_1 = 8;
        let tile_2 = 4;
        let tile_1_size = builder.cst_size(tile_1);
        let tile_2_size = builder.cst_size(tile_2);
        let tmp_mem_size = tile_1 * tile_2;
        let a_tmp_mem = builder.allocate(ir::Type::F(32), tmp_mem_size, true);

        // Configure dimension sizes
        let thread_dim_1_0 = builder.open_dim_ex(tile_1_size.clone(), DimKind::THREAD);
        let unroll_dim_0_0 = builder.open_dim_ex(tile_2_size.clone(), DimKind::UNROLL);
        let acc_init = builder.mov(&0f32);
        builder.close_dim(&unroll_dim_0_0);

        let k_size = builder.param_size("k", Self::K as u32);
        let k_dim = builder.open_dim_ex(k_size, DimKind::LOOP);
        // Load A
        let unroll_dim_a = builder.open_dim_ex(tile_2_size.clone(), DimKind::VECTOR);
        let (addr, pattern) = builder.tensor_access(
            &a_tmp_mem,
            a_tmp_mem.into(),
            ir::Type::F(32),
            &[&thread_dim_1_0, &unroll_dim_a],
        );
        let a_val =
            builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::CACHE_GLOBAL);
        builder.close_dim(&unroll_dim_a);
        // Mad a and b
        let unroll_dims_1 = builder.open_mapped_dim(&unroll_dim_0_0);
        let a_op = builder.dim_map(
            a_val,
            &[(&unroll_dim_a, &unroll_dims_1)],
            ir::DimMapScope::Thread,
        );
        let fby = builder.create_fby_variable(acc_init, &[&k_dim]);
        let acc = builder.mad(&a_op, &2f32, &fby);
        builder.set_loop_carried_variable(fby, acc);
        builder.close_dim(&k_dim);

        let _ = builder.open_mapped_dim(&unroll_dims_1);
        let (addr, pattern) = builder.tensor_access(&"out", None, ir::Type::F(32), &[]);
        let _ = builder.st_ex(
            &addr,
            &Last(acc, &[&k_dim]),
            true,
            pattern,
            InstFlag::NO_CACHE,
        );

        builder.order(&k_dim, &thread_dim_1_0, Order::INNER);
        builder.order(&unroll_dim_a, &unroll_dims_1[0], Order::BEFORE);

        Test1
    }
}

pub struct Test2;

impl Test2 {
    const M: i32 = 1024;

    const N: i32 = 1024;

    const K: i32 = 1024;

    const TILE_1: i32 = 32;

    const TILE_2: i32 = 4;
}

impl PerfModelTest for Test2 {
    fn name() -> &'static str {
        "test_2"
    }

    fn gen_signature<AM: ArgMap + Context>(builder: &mut SignatureBuilder<AM>) {
        builder.scalar("m", Self::M / (Self::TILE_1 * Self::TILE_2));
        builder.scalar("n", Self::N / (Self::TILE_1 * Self::TILE_2));
        builder.scalar("k", Self::K);
        builder.array::<f32>("out", 4 * 32 * 32 * 4 as usize);
    }

    fn gen_function(builder: &mut Builder) -> Self {
        let tile_1_size = builder.cst_size(Self::TILE_1 as u32);
        let tile_2_size = builder.cst_size(Self::TILE_2 as u32);
        let tmp_mem_size = Self::TILE_1 * Self::TILE_1 * Self::TILE_2;
        let a_tmp_mem = builder.allocate(ir::Type::F(32), tmp_mem_size as u32, true);
        let b_tmp_mem = builder.allocate(ir::Type::F(32), tmp_mem_size as u32, true);

        // Configure dimension sizes
        let m_tiled = builder.param_size("m", Self::M as u32);
        let n_tiled = builder.param_size("n", Self::N as u32);
        let b0 = builder.open_dim_ex(n_tiled, DimKind::BLOCK);
        let b1 = builder.open_dim_ex(m_tiled, DimKind::BLOCK);
        builder.order(&b0, &b1, Order::OUTER);

        let thread_dim_0_0 = builder.open_dim_ex(tile_1_size.clone(), DimKind::THREAD);
        let thread_dim_1_0 = builder.open_dim_ex(tile_1_size.clone(), DimKind::THREAD);
        let unroll_dim_0_0 = builder.open_dim_ex(tile_2_size.clone(), DimKind::UNROLL);
        let unroll_dim_1_0 = builder.open_dim_ex(tile_2_size.clone(), DimKind::UNROLL);
        let acc_init = builder.mov(&0f32);
        builder.close_dim(&unroll_dim_0_0);
        builder.close_dim(&unroll_dim_1_0);

        let k_size = builder.param_size("k", Self::K as u32);
        let k_dim = builder.open_dim_ex(k_size, DimKind::LOOP);
        let thread_dims_0_1 = builder.open_mapped_dim(&thread_dim_0_0);
        let thread_dims_1_1 = builder.open_mapped_dim(&thread_dim_1_0);
        // Load A
        let unroll_dim_a = builder.open_dim_ex(tile_2_size.clone(), DimKind::VECTOR);
        let (addr, pattern) = builder.tensor_access(
            &a_tmp_mem,
            a_tmp_mem.into(),
            ir::Type::F(32),
            &[&thread_dims_0_1, &unroll_dim_a],
        );
        let a_val =
            builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::CACHE_GLOBAL);
        builder.close_dim(&unroll_dim_a);
        // Load B
        let unroll_dim_b = builder.open_dim_ex(tile_2_size.clone(), DimKind::VECTOR);
        let (addr, pattern) = builder.tensor_access(
            &b_tmp_mem,
            b_tmp_mem.into(),
            ir::Type::F(32),
            &[&thread_dims_1_1, &unroll_dim_b],
        );
        let b_val = builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::NO_CACHE);
        builder.action(Action::MemSpace(b_tmp_mem, MemSpace::SHARED));
        builder.close_dim(&unroll_dim_b);
        // Mad a and b
        let unroll_dims_0_1 = builder.open_mapped_dim(&unroll_dim_0_0);
        let unroll_dims_1_1 = builder.open_mapped_dim(&unroll_dim_1_0);
        let a_op = builder.dim_map(
            a_val,
            &[(&unroll_dim_a, &unroll_dims_0_1)],
            ir::DimMapScope::Thread,
        );
        let b_op = builder.dim_map(
            b_val,
            &[(&unroll_dim_b, &unroll_dims_1_1)],
            ir::DimMapScope::Thread,
        );
        let fby = builder.create_fby_variable(acc_init, &[&k_dim]);
        let acc = builder.mad(&a_op, &b_op, &fby);
        builder.set_loop_carried_variable(fby, acc);
        builder.close_dim(&k_dim);

        let thread_dims_0_2 = builder.open_mapped_dim(&thread_dims_0_1);
        let thread_dims_1_2 = builder.open_mapped_dim(&thread_dims_1_1);
        let unroll_dims_0_2 = builder.open_mapped_dim(&unroll_dims_0_1);
        let unroll_dims_1_2 = builder.open_mapped_dim(&unroll_dims_1_1);
        let (addr, pattern) = builder.tensor_access(
            &"out",
            None,
            ir::Type::F(32),
            &[
                &thread_dims_0_2,
                &unroll_dims_0_2,
                &thread_dims_1_2,
                &unroll_dims_1_2,
            ],
        );
        let _ = builder.st_ex(
            &addr,
            &Last(acc, &[&k_dim]),
            true,
            pattern,
            InstFlag::NO_CACHE,
        );

        builder.order(&k_dim, &thread_dims_0_1, Order::OUTER);
        builder.order(&thread_dim_0_0, &thread_dim_1_0, Order::OUTER);
        builder.order(&unroll_dim_0_0, &unroll_dim_1_0, Order::OUTER);
        builder.order(&unroll_dims_0_1, &unroll_dims_1_1, Order::OUTER);
        builder.order(&unroll_dim_a, &unroll_dim_b, Order::BEFORE);
        builder.order(&unroll_dim_b, &unroll_dims_0_1, Order::BEFORE);

        for id in &unroll_dims_1_2 {
            builder.action(Action::DimKind(id, DimKind::VECTOR));
        }

        Test2
    }
}
