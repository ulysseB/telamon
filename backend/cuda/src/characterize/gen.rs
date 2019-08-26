//! Builds code for micro benchmarks.
use std::sync::Arc;

use crate::characterize::Table;
use crate::{Context, Gpu, Kernel, PerfCounterSet};
use itertools::Itertools;
use log::*;
use num::Zero;
use telamon::codegen;
use telamon::device::{ArgMapExt, Device, ScalarArgument};
use telamon::explorer;
use telamon::helper::tensor::DimSize;
use telamon::helper::{AutoOperand, Builder, Reduce};
use telamon::ir::{self, Signature};
use telamon::search_space::*;
use utils::*;

/// Generates a function base with the given arguments.
pub fn base(params: &[(&str, ir::Type)], arrays: &[&str], gpu: &Gpu) -> Signature {
    let mut signature = Signature::new("bench".to_owned());
    for &(name, t) in params {
        signature.add_scalar(name.to_owned(), t);
    }
    for &name in arrays {
        signature.add_array(gpu, name.to_owned(), ir::Type::I(8));
    }
    signature
}

/// Binds a parameter to a value in the given context.
pub fn bind_scalar<T: ScalarArgument>(name: &str, val: T, context: &mut Context) {
    let p = ir::Parameter {
        t: T::t(),
        name: name.to_string(),
        elem_t: None,
    };
    context.bind_scalar(&p, val);
}

/// Binds a parameter to a value in the given context.
pub fn bind_array<'a, T: 'a>(name: &str, len: usize, context: &mut Context<'a>) {
    let array = std::sync::Arc::new(context.executor().allocate_array::<T>(len));
    context.bind_param(name.to_string(), array.clone());
}

/// Generates a kernel with two nested empty loops.
pub fn two_empty_loops(
    base: Arc<Signature>,
    device: Arc<dyn Device>,
    outer: &DimSize,
    inner: &DimSize,
) -> SearchSpace {
    let mut builder = Builder::new(base, device);
    let outer_size = outer.to_ir_size(&builder);
    let inner_size = inner.to_ir_size(&builder);
    let d0 = builder.open_dim_ex(outer_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(inner_size, DimKind::LOOP);
    builder.mov(&0i32);
    builder.order(&d0, &d1, Order::OUTER);
    builder.get()
}

/// Generates a kernel with chained adds in a loop.
pub fn loop_chained_adds(
    base: Arc<Signature>,
    device: Arc<dyn Device>,
    loop_size: &DimSize,
    chained: u32,
    out: &str,
) -> SearchSpace {
    let mut builder = Builder::new(base, device);
    let init = builder.mov(&0f32);
    let loop_size = loop_size.to_ir_size(&builder);
    let unroll_size = builder.cst_size(chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let acc = builder.add(&Reduce(init), &2f32);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern = ir::AccessPattern::Unknown(None);
    builder.st_ex(&out, &acc, true, pattern, InstFlag::CACHE_GLOBAL);
    builder.get()
}

/// A function that produce a single instruction using the first argument on one of its
/// operands. The second argument may be used for other operands.
pub type InstGenerator = dyn Fn(&dyn AutoOperand, &&str, &mut Builder) -> ir::InstId;

/// Generates a single thread with a loop containing chained instructions.
///
/// * `T`: the type of the instructions.
/// * `inst_gen`: function that genrates a single instruction.
/// * `n_iter`: the number of loop iteration.
/// * `n_chained`: the number of chained instructions in each loop iteration.
/// * `arg`: a value that my be used as an operand by instruction.
/// * `out`: an array to store the computation result.
pub fn inst_chain<T>(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    inst_gen: &InstGenerator,
    n_iter: &DimSize,
    n_chained: u32,
    arg: &str,
    out: &str,
) -> SearchSpace
where
    T: ScalarArgument + Zero,
{
    let mut builder = Builder::new(signature, device);
    let init = builder.mov(&T::zero());
    let loop_size = n_iter.to_ir_size(&builder);
    let unroll_size = builder.cst_size(n_chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let acc = inst_gen(&Reduce(init), &arg, &mut builder);
    builder.order(&d0, &d1, Order::OUTER);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern = ir::AccessPattern::Unknown(None);
    builder.st_ex(&out, &acc, true, pattern, InstFlag::NO_CACHE);
    builder.get()
}

/// Generates a function that initializes an array with addresses pointing to the same
/// array, `stride` cells further.
pub fn init_stride_array(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    array: &str,
    n: u32,
    stride: i32,
) -> SearchSpace {
    let byte_stride = stride * 8;
    let mut builder = Builder::new(signature, device);
    let size = builder.cst_size(n);
    let (dim, addr) = if n > 1 {
        let dim = builder.open_dim_ex(size, DimKind::LOOP);
        let addr = builder.mad(&dim, &byte_stride, &array);
        (Some(dim), addr)
    } else {
        (None, builder.mov(&array))
    };
    let next_addr = builder.mad(&byte_stride, &1i32, &addr);
    let pattern0 = ir::AccessPattern::Unknown(None);
    builder.st_ex(&addr, &next_addr, true, pattern0, InstFlag::CACHE_GLOBAL);
    if let Some(dim) = dim.as_ref() {
        builder.close_dim(dim);
    }
    let last_addr = builder.mad(&byte_stride, &(n as i32 - 1), &array);

    let pattern1 = ir::AccessPattern::Unknown(None);
    builder.st_ex(&last_addr, &array, true, pattern1, InstFlag::CACHE_GLOBAL);
    if let Some(dim) = dim.as_ref() {
        builder.order(dim, &last_addr, Order::BEFORE);
    }
    builder.get()
}

/// Generates a function that performs chained loads.
#[allow(clippy::too_many_arguments)]
pub fn load_chain(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    n_threads: u32,
    n_iter: &DimSize,
    n_chained: u32,
    array: &str,
    out: &str,
) -> SearchSpace {
    let mut builder = Builder::new(signature, device);
    let init = builder.mov(&array);
    let loop_size = n_iter.to_ir_size(&builder);
    let unroll_size = builder.cst_size(n_chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    if n_threads != 1 {
        let d = builder.open_dim_ex(ir::Size::new_const(n_threads), DimKind::THREAD);
        builder.order(&d, &d0, Order::OUTER);
    }
    let pattern0 = ir::AccessPattern::Unknown(None);
    let ptr = builder.ld_ex(
        ir::Type::I(64),
        &Reduce(init),
        pattern0,
        InstFlag::CACHE_GLOBAL,
    );
    builder.order(&d0, &d1, Order::OUTER);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern1 = ir::AccessPattern::Unknown(None);
    builder.st_ex(&out, &ptr, true, pattern1, InstFlag::NO_CACHE);
    builder.get()
}

/// Generates chained loads from shared memory.
pub fn shared_load_chain(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    n_iter: &DimSize,
    n_chained: u32,
    array_size: u32,
    out: &str,
) -> SearchSpace {
    let mut builder = Builder::new(signature, device);
    let array_dim_size = builder.cst_size(array_size);
    let array = builder.allocate_shared(4 * array_size);
    let init_dim = builder.open_dim_ex(array_dim_size.clone(), DimKind::LOOP);
    let init_addr = builder.mad(&init_dim, &4i32, &array);
    let increment = builder.cast(&4i32, ir::Type::PtrTo(array));
    let next_addr = builder.add(&init_addr, &increment);
    let array_pattern = ir::AccessPattern::Unknown(Some(array));
    builder.st(&init_addr, &next_addr, array_pattern.clone());
    builder.close_dim(&init_dim);
    let last_st = builder.st(&init_addr, &array, array_pattern.clone());

    let addr_init = builder.mov(&array);
    let loop_size = n_iter.to_ir_size(&builder);
    let unroll_size = builder.cst_size(n_chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let addr = builder.ld(ir::Type::I(32), &Reduce(addr_init), array_pattern);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let out_pattern = ir::AccessPattern::Unknown(None);
    builder.st_ex(&out, &addr, true, out_pattern, InstFlag::CACHE_GLOBAL);

    builder.order(&last_st, &addr_init, Order::BEFORE);
    builder.order(&d0, &d1, Order::OUTER);
    builder.get()
}

/// Generates many parallel loads in a single block.
#[allow(clippy::too_many_arguments)]
pub fn parallel_load(
    signature: Arc<Signature>,
    gpu: Arc<Gpu>,
    num_blocks: &DimSize,
    n: &DimSize,
    n_chained: u32,
    n_unroll: u32,
    num_wraps: u32,
    stride: u32,
    array: &str,
    out: &str,
) -> SearchSpace {
    assert!(stride * 4 <= gpu.l1_cache_line);
    let mut builder = Builder::new(signature, Arc::<Gpu>::clone(&gpu));
    let block_size = num_blocks.to_ir_size(&builder);
    let _ = builder.open_dim_ex(block_size, DimKind::BLOCK);
    // Initialize the result
    let d1_0_a = if num_wraps > 1 {
        Some(builder.open_dim_ex(ir::Size::new_const(num_wraps), DimKind::THREAD))
    } else {
        None
    };
    let d1_0_b = builder.open_dim_ex(ir::Size::new_const(gpu.wrap_size), DimKind::THREAD);
    builder.order(&d1_0_a, &d1_0_b, Order::OUTER);

    let init = builder.mov(&0f32);
    // Sum in the result.
    let loop_size = n.to_ir_size(&builder);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1_1_a = d1_0_a
        .as_ref()
        .map(|d1_0_a| builder.open_mapped_dim(&d1_0_a));
    let d1_1_b = builder.open_mapped_dim(&d1_0_b);
    builder.order(&d1_1_a, &d1_1_b, Order::OUTER);

    let d3 = builder.open_dim_ex(ir::Size::new_const(n_chained), DimKind::UNROLL);
    let d4_0 = builder.open_dim_ex(ir::Size::new_const(n_unroll), DimKind::UNROLL);
    let pattern = ir::AccessPattern::Unknown(None);
    let wrap_stride = gpu.wrap_size * gpu.l1_cache_line;
    let mut strides = vec![
        (&d3, ir::Size::new_const(n_unroll * num_wraps * wrap_stride)),
        (&d4_0, ir::Size::new_const(num_wraps * wrap_stride)),
        (&d1_1_b, ir::Size::new_const(stride * 4)),
    ];
    if let Some(ref d1_1_a) = d1_1_a {
        strides.push((d1_1_a, ir::Size::new_const(wrap_stride)));
    }
    let addr = builder.induction_var(&array, strides);
    let val = builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::CACHE_GLOBAL);
    let d4_1 = builder.open_mapped_dim(&d4_0);
    let acc = builder.add(&val, &Reduce(init));
    builder.close_dim(&d0);
    builder.close_dim(&d3);
    builder.close_dim(&d4_1);
    // Write the result
    let d1_2_a = d1_1_a
        .as_ref()
        .map(|d1_1_a| builder.open_mapped_dim(&d1_1_a));
    let d1_2_b = builder.open_mapped_dim(&d1_1_b);
    builder.order(&d1_2_a, &d1_2_b, Order::OUTER);
    let out_pattern = ir::AccessPattern::Unknown(None);
    builder.st_ex(&out, &acc, true, out_pattern, InstFlag::NO_CACHE);

    builder.order(&d1_0_a, &d0, Order::BEFORE);
    builder.order(&d1_0_b, &d0, Order::BEFORE);
    builder.order(&d0, &d1_1_a, Order::OUTER);
    builder.order(&d0, &d1_1_b, Order::OUTER);
    builder.order(&d0, &d1_2_a, Order::BEFORE);
    builder.order(&d0, &d1_2_b, Order::BEFORE);
    builder.order(&d3, &d4_0, Order::OUTER);
    builder.order(&d3, &d4_1, Order::OUTER);
    builder.order(&d4_0, &d4_1, Order::BEFORE);

    builder.get()
}

/// Generates many parallel stores.
#[allow(clippy::too_many_arguments)]
pub fn parallel_store(
    signature: Arc<Signature>,
    gpu: Arc<Gpu>,
    num_blocks: &DimSize,
    n: &DimSize,
    n_chained: u32,
    n_unroll: u32,
    num_wraps: u32,
    stride: u32,
    array: &str,
) -> SearchSpace {
    assert!(stride * 4 <= gpu.l1_cache_line);
    let mut builder = Builder::new(signature, Arc::<Gpu>::clone(&gpu));
    let block_size = num_blocks.to_ir_size(&builder);
    let _ = builder.open_dim_ex(block_size, DimKind::BLOCK);
    let loop_size = n.to_ir_size(&builder);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);

    let d1_0 = if num_wraps > 1 {
        Some(builder.open_dim_ex(ir::Size::new_const(num_wraps), DimKind::THREAD))
    } else {
        None
    };
    let d1_1 = builder.open_dim_ex(ir::Size::new_const(gpu.wrap_size), DimKind::THREAD);
    builder.order(&d1_0, &d1_1, Order::OUTER);

    let d3 = builder.open_dim_ex(ir::Size::new_const(n_chained), DimKind::UNROLL);
    let d4 = builder.open_dim_ex(ir::Size::new_const(n_unroll), DimKind::UNROLL);
    let pattern = ir::AccessPattern::Unknown(None);
    let wrap_stride = gpu.wrap_size * gpu.l1_cache_line;
    let mut strides = vec![
        (&d3, ir::Size::new_const(n_unroll * num_wraps * wrap_stride)),
        (&d4, ir::Size::new_const(num_wraps * wrap_stride)),
        (&d1_1, ir::Size::new_const(stride * 4)),
    ];
    if let Some(ref d1_0) = d1_0 {
        strides.push((d1_0, ir::Size::new_const(wrap_stride)));
    }
    let addr = builder.induction_var(&array, strides);
    builder.st_ex(&addr, &42f32, true, pattern, InstFlag::CACHE_GLOBAL);

    builder.order(&d0, &d1_0, Order::OUTER);
    builder.order(&d0, &d1_1, Order::OUTER);
    builder.order(&d1_0, &d3, Order::OUTER);
    builder.order(&d1_1, &d3, Order::OUTER);
    builder.order(&d3, &d4, Order::OUTER);

    builder.get()
}

/// Generates a wrap of syncthreads separated by a single instruction.
pub fn syncthread(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    n_iter: &DimSize,
    n_chained: u32,
    wrap_size: u32,
) -> SearchSpace {
    let mut builder = Builder::new(signature, device);
    let loop_size = n_iter.to_ir_size(&builder);
    let unroll_size = builder.cst_size(n_chained);
    let thread_size = builder.cst_size(wrap_size);

    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let d2 = builder.open_dim_ex(thread_size, DimKind::THREAD);
    let _ = builder.mov(&0i32);

    builder.order(&d0, &d1, Order::OUTER);
    builder.order(&d0, &d2, Order::OUTER);

    let mut kernel = builder.get();
    kernel
        .domain_mut()
        .set_order(d1[0].into(), d2[0].into(), Order::OUTER);
    kernel
}

/// Generates a wrap of syncthreads separated by a single instruction.
#[allow(clippy::too_many_arguments)]
pub fn chain_in_syncthread(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    n_iter: &DimSize,
    sync_chained: u32,
    add_chained: u32,
    wrap_size: u32,
    out: &str,
) -> SearchSpace {
    let mut builder = Builder::new(signature, device);
    let loop_size = n_iter.to_ir_size(&builder);
    let sync_unroll_size = builder.cst_size(sync_chained);
    let thread_size = builder.cst_size(wrap_size);
    let add_unroll_size = builder.cst_size(add_chained);

    let d0 = builder.open_dim_ex(thread_size, DimKind::THREAD);
    let init = builder.mov(&0f32);

    let d1 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d2 = builder.open_dim_ex(sync_unroll_size, DimKind::UNROLL);
    let d3 = builder.open_mapped_dim(&d0);
    let d4 = builder.open_dim_ex(add_unroll_size, DimKind::UNROLL);
    let acc = builder.add(&Reduce(init), &2f32);
    builder.close_dim(&d1);
    builder.close_dim(&d2);
    builder.close_dim(&d3);
    builder.close_dim(&d4);

    let d5 = builder.open_mapped_dim(&d0);
    let pattern = ir::AccessPattern::Unknown(None);
    builder.st_ex(&out, &acc, true, pattern, InstFlag::CACHE_GLOBAL);

    builder.order(&d1, &d2, Order::OUTER);
    builder.order(&d1, &d3, Order::OUTER);
    builder.order(&d2, &d4, Order::OUTER);
    builder.order(&d0, &d1, Order::BEFORE);
    builder.order(&d1, &d5, Order::BEFORE);
    builder.action(Action::ThreadMapping(d0[0], d3[0], ThreadMapping::MAPPED));
    builder.action(Action::ThreadMapping(d0[0], d5[0], ThreadMapping::MAPPED));

    let mut kernel = builder.get();
    kernel
        .domain_mut()
        .set_order(d1[0].into(), d2[0].into(), Order::OUTER);
    kernel
}

/// Generates a global memory load in a loop.
pub fn load_in_loop(
    signature: Arc<Signature>,
    device: Arc<dyn Device>,
    k_size: &DimSize,
    threads: u32,
    out: &str,
) -> SearchSpace {
    let mut builder = Builder::new(signature, device);
    let size_4 = builder.cst_size(4);
    let tmp_mem_size = 4 * 4 * threads;
    let tmp_mem = builder.allocate(tmp_mem_size, true);

    // Configure dimension sizes
    let threads_size = builder.cst_size(threads);
    let thread_dim_1_0 = builder.open_dim_ex(threads_size, DimKind::THREAD);
    let unroll_dim_0_0 = builder.open_dim_ex(size_4.clone(), DimKind::UNROLL);
    let acc_init = builder.mov(&0f32);
    builder.close_dim(&unroll_dim_0_0);

    let k_size = k_size.to_ir_size(&builder);
    let k_dim = builder.open_dim_ex(k_size, DimKind::LOOP);
    // Load A
    let unroll_dim_a = builder.open_dim_ex(size_4.clone(), DimKind::VECTOR);
    let (addr, pattern) = builder.tensor_access(
        &tmp_mem,
        tmp_mem.into(),
        ir::Type::F(32),
        &[&thread_dim_1_0, &unroll_dim_a],
    );
    let a_val = builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::CACHE_GLOBAL);
    builder.close_dim(&unroll_dim_a);
    // Mad a and b
    let unroll_dims_1 = builder.open_mapped_dim(&unroll_dim_0_0);
    let a_op = builder.dim_map(
        a_val,
        &[(&unroll_dim_a, &unroll_dims_1)],
        ir::DimMapScope::Thread,
    );
    let acc = builder.mad(&a_op, &2f32, &Reduce(acc_init));
    builder.close_dim(&k_dim);

    let _ = builder.open_mapped_dim(&unroll_dims_1);
    let (addr, pattern) = builder.tensor_access(&out, None, ir::Type::F(32), &[]);
    let _ = builder.st_ex(&addr, &acc, true, pattern, InstFlag::NO_CACHE);

    builder.order(&k_dim, &thread_dim_1_0, Order::INNER);
    builder.order(&unroll_dim_a, &unroll_dims_1[0], Order::BEFORE);
    builder.get()
}

/// Instruments a kernel and stores the results in a `Table`.
///
/// * `args_range`: the arguments that must vary, with their range.
/// * `perf_counters`: the CUDA performance counters to monitor.
/// * `result`: the table in which to store the results.
pub fn run(
    context: &mut Context,
    space: &SearchSpace,
    args_range: &[(&str, &[i32])],
    counters: &PerfCounterSet,
    result_prefix: &[u64],
    result: &mut Table<u64>,
) {
    if let Some(choice) = explorer::choice::default_list(space).next() {
        panic!("The benchmark is not completely scheduled: {:?}", choice);
    }
    let dev_fun = codegen::Function::build(space);
    let kernel = Kernel::compile(&dev_fun, context.gpu(), context.executor(), 1);
    for &(arg, range) in args_range {
        bind_scalar(arg, range[0], context);
    }
    kernel.instrument(context, counters);
    let args_range_len = args_range.iter().map(|&(_, x)| x.len()).collect_vec();
    for index in NDRange::new(&args_range_len) {
        let mut entry = result_prefix.iter().cloned().collect_vec();
        let mut arg_values = vec![];
        for (i, &(arg, arg_range)) in index.into_iter().zip(args_range.iter()) {
            bind_scalar(arg, arg_range[i], context);
            entry.push(arg_range[i] as u64);
            arg_values.push(arg_range[i]);
        }
        // Flush the cache
        trace!("Running with params: {:?}", arg_values);
        entry.append(&mut kernel.instrument(context, counters));
        result.add_entry(entry);
    }
}
