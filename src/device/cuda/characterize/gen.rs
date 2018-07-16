//! Builds code for micro benchmarks.
use num::Zero;
use codegen;
use device::{ArgMap, ScalarArgument, Device};
use device::cuda::{Gpu, Context, Kernel, PerfCounterSet};
use device::cuda::characterize::Table;
use explorer;
use helper::{AutoOperand, Builder, DimGroup, Reduce};
use ir::{self, Signature};
use search_space::*;
use itertools::Itertools;
use std;
use utils::*;

/// Generates a function base with the given arguments.
pub fn base(params: &[(&str, ir::Type)], arrays: &[&str])
    -> (Signature, Vec<ir::mem::Id>)
{
    let mut p = params.iter().map(|&(name, t)| {
        ir::Parameter { name: name.to_string(), t }
    }).collect_vec();
    let mut mem_blocks = 0;
    let mem_ids = arrays.iter().map(|name| {
        let id = ir::mem::Id::External(mem_blocks);
        mem_blocks += 1;
        p.push(ir::Parameter { name: name.to_string(), t: ir::Type::PtrTo(id) });
        id
    }).collect();
    (Signature {
        name: "bench".to_string(),
        params: p,
        mem_blocks,
    }, mem_ids)
}

/// Binds a parameter to a value in the given context.
pub fn bind_scalar<T: ScalarArgument>(name: &str, val: T, context: &mut Context) {
    let p = ir::Parameter { t: T::t(), name: name.to_string() };
    context.bind_scalar(&p, val);
}

/// Binds a parameter to a value in the given context.
pub fn bind_array<'a, T: 'a>(name: &str, len: usize, context: &mut Context<'a>) {
    let array = std::sync::Arc::new(context.executor().allocate_array::<T>(len));
    context.bind_param(name.to_string(), array.clone());
}

/// Generates a kernel with two nested empty loops.
pub fn two_empty_loops<'a>(base: &'a Signature, device: &'a Device,
                          outer: &str, inner: &str) -> SearchSpace<'a> {
    let mut builder = Builder::new(base, device);
    let outer_size = builder.param_size(outer);
    let inner_size = builder.param_size(inner);
    let d0 = builder.open_dim_ex(outer_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(inner_size, DimKind::LOOP);
    builder.mov(&0i32);
    builder.order(&d0, &d1, Order::OUTER);
    builder.get()
}

/// Generates a kernel with chained adds in a loop.
pub fn loop_chained_adds<'a>(base: &'a Signature, device: &'a Device,
                             size: &str, chained: u32,
                             out: &str, out_id: ir::mem::Id) -> SearchSpace<'a> {
    let mut builder = Builder::new(base, device);
    let init = builder.mov(&0f32);
    let loop_size = builder.param_size(size);
    let unroll_size = builder.cst_size(chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let acc = builder.add(&Reduce(init), &2f32);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern = builder.unknown_access_pattern(out_id);
    builder.st_ex(&out, &acc, true, pattern, InstFlag::MEM_CG);
    builder.get()
}

/// A function that produce a single instruction using the first argument on one of its
/// operands. The second argument may be used for other operands.
pub type InstGenerator = Fn(&AutoOperand<'static>, &&str, &mut Builder) -> ir::InstId;

/// Generates a single thread with a loop containing chained instructions.
///
/// * `T`: the type of the instructions.
/// * `inst_gen`: function that genrates a single instruction.
/// * `n_iter`: the number of loop iteration.
/// * `n_chained`: the number of chained instructions in each loop iteration.
/// * `arg`: a value that my be used as an operand by instruction.
/// * `out`: an array to store the computation result.
pub fn inst_chain<'a, T>(signature: &'a Signature, device: &'a Device,
                         inst_gen: &InstGenerator,
                         n_iter: &str, n_chained: u32, arg: &str, out: &str
                        ) -> SearchSpace<'a> where T: ScalarArgument + Zero {
    let mut builder = Builder::new(signature, device);
    let init = builder.mov(&T::zero());
    let loop_size = builder.param_size(n_iter);
    let unroll_size = builder.cst_size(n_chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let acc = inst_gen(&Reduce(init), &arg, &mut builder);
    builder.order(&d0, &d1, Order::OUTER);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern = builder.unknown_access_pattern(ir::mem::Id::External(0));
    builder.st_ex(&out, &acc, true, pattern, InstFlag::MEM_CS);
    builder.get()
}

/// Generates a function that initializes an array with addresses pointing to the same
/// array, `stride` cells further.
pub fn init_stride_array<'a>(signature: &'a Signature, device: &'a Device,
                             mem_id: ir::mem::Id, array: &str, n: u32, stride: i32
                            ) -> SearchSpace<'a> {
    let byte_stride = stride * 8;
    let mut builder = Builder::new(signature, device);
    let size = builder.cst_size(n);
    let (dim, addr) = if n > 1 {
        let dim = builder.open_dim_ex(size, DimKind::LOOP);
        let addr = builder.mad(&dim, &byte_stride, &array);
        (DimGroup::new(vec![dim]), addr)
    } else { (DimGroup::default(), builder.mov(&array)) };
    let next_addr = builder.mad(&byte_stride, &1i32, &addr);
    let pattern0 = builder.unknown_access_pattern(mem_id);
    builder.st_ex(&addr, &next_addr, true, pattern0, InstFlag::MEM_CG);
    builder.close_dim(&dim);
    let last_addr = builder.mad(&byte_stride, &(n as i32 -1), &array);

    let pattern1 = builder.unknown_access_pattern(mem_id);
    builder.st_ex(&last_addr, &array, true, pattern1, InstFlag::MEM_CG);
    builder.order(&dim, &last_addr, Order::BEFORE);
    builder.get()
}

/// Generates a function that performs chained loads.
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn load_chain<'a>(signature: &'a Signature, device: &'a Device,
                      n_threads: u32, n_iter: &str, n_chained: u32,
                      mem_id: ir::mem::Id, array: &str, out_id: ir::mem::Id, out: &str
                     ) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let init = builder.mov(&array);
    let loop_size = builder.param_size(n_iter);
    let unroll_size = builder.cst_size(n_chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    if n_threads != 1 {
        let d = builder.open_dim_ex(ir::Size::new(n_threads, vec![], 1), DimKind::THREAD);
        builder.order(&d, &d0, Order::OUTER);
    }
    let pattern0 = builder.unknown_access_pattern(mem_id);
    let ptr = builder.ld_ex(ir::Type::I(64), &Reduce(init), pattern0, InstFlag::MEM_CG);
    builder.order(&d0, &d1, Order::OUTER);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern1 = builder.unknown_access_pattern(out_id);
    builder.st_ex(&out, &ptr, true, pattern1, InstFlag::MEM_CS);
    builder.get()
}

/// Generates chained loads from shared memory.
pub fn shared_load_chain<'a>(signature: &'a Signature, device: &'a Device,
                             n_iter: &str, n_chained: u32,
                             array_size: u32, out_id: ir::mem::Id, out: &str
                            ) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let array_dim_size = builder.cst_size(array_size);
    let array = builder.allocate_shared(ir::Size::new(4*array_size, vec![], 1));
    let init_dim = builder.open_dim_ex(array_dim_size.clone(), DimKind::LOOP);
    let init_addr = builder.mad(&init_dim, &4i32, &array);
    let increment = builder.cast(&4i32, ir::Type::PtrTo(array.into()));
    let next_addr = builder.add(&init_addr, &increment);
    let pattern0 = builder.unknown_access_pattern(array.into());
    builder.st(&init_addr, &next_addr, pattern0);
    builder.close_dim(&init_dim);
    let pattern1 = builder.unknown_access_pattern(array.into());
    let last_st = builder.st(&init_addr, &array, pattern1);

    let addr_init = builder.mov(&array);
    let loop_size = builder.param_size(n_iter);
    let unroll_size = builder.cst_size(n_chained);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let pattern2 = builder.unknown_access_pattern(array.into());
    let addr = builder.ld(ir::Type::I(32), &Reduce(addr_init), pattern2);
    builder.close_dim(&d0);
    builder.close_dim(&d1);
    let pattern3 = builder.unknown_access_pattern(out_id);
    builder.st_ex(&out, &addr, true, pattern3, InstFlag::MEM_CG);

    builder.order(&last_st, &addr_init, Order::BEFORE);
    builder.order(&d0, &d1, Order::OUTER);
    builder.get()
}

/// Generates many parallel loads in a single block.
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn parallel_load<'a>(signature: &'a Signature, gpu: &'a Gpu, num_blocks: &str,
                         n: &str, n_chained: u32, n_unroll: u32, num_wraps: u32, stride: u32,
                         mem_id: ir::mem::Id, array: &str,
                         out_id: ir::mem::Id, out: &str) -> SearchSpace<'a> {
    assert!(stride*4 <= gpu.l1_cache_line);
    let mut builder = Builder::new(signature, gpu);
    let block_size = builder.param_size(num_blocks);
    let _ = builder.open_dim_ex(block_size, DimKind::BLOCK);
    // Initialize the result
    let init_size = builder.cst_size(num_wraps * gpu.wrap_size);
    let thread_tilling = if num_wraps == 1 { vec![] } else { vec![gpu.wrap_size] };
    let d1_0 = builder.open_tiled_dim(init_size, &thread_tilling);
    for d in &d1_0 { builder.action(Action::DimKind(d, DimKind::THREAD)); }
    for (x, y) in d1_0.iter().tuple_windows() { builder.order(&x, &y, Order::OUTER); }
    let init = builder.mov(&0f32);
    // Sum in the result.
    let loop_size = builder.param_size(n);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1_1 = builder.open_mapped_dim(&d1_0);
    for (x, y) in d1_1.iter().tuple_windows() { builder.order(&x, &y, Order::OUTER); }
    let d3 = builder.open_dim_ex(ir::Size::new(n_chained, vec![], 1), DimKind::UNROLL);
    let d4_0 = builder.open_dim_ex(ir::Size::new(n_unroll, vec![], 1), DimKind::UNROLL);
    let pattern = builder.unknown_access_pattern(mem_id);
    let mut strides = vec![
        (d3, ir::Size::new(n_unroll*num_wraps*gpu.wrap_size*gpu.l1_cache_line, vec![], 1)),
        (d4_0, ir::Size::new(num_wraps*gpu.wrap_size*gpu.l1_cache_line, vec![], 1)),
    ];
    if stride != 0 {
        let i = if num_wraps == 1 { 0 } else {
            strides.push((d1_1[0], ir::Size::new(gpu.wrap_size*gpu.l1_cache_line, vec![], 1)));
            1
        };
        strides.push((d1_1[i], ir::Size::new(stride*4, vec![], 1)));
    };
    let addr = builder.induction_var(&array, strides);
    let val = builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::MEM_CG);
    let d4_1 = builder.open_mapped_dim(&d4_0)[0];
    let acc = builder.add(&val, &Reduce(init));
    builder.close_dim(&DimGroup::new(vec![d0, d3, d4_1]));
    // Write the result
    let d1_2 = builder.open_mapped_dim(&d1_1);
    for (x, y) in d1_2.iter().tuple_windows() { builder.order(&x, &y, Order::OUTER); }
    let out_pattern = builder.unknown_access_pattern(out_id);
    builder.st_ex(&out, &acc, true, out_pattern, InstFlag::MEM_CS);

    builder.order(&d1_0, &d0, Order::BEFORE);
    builder.order(&d0, &d1_1, Order::OUTER);
    builder.order(&d0, &d1_2, Order::BEFORE);
    builder.order(&d3, &d4_0, Order::OUTER);
    builder.order(&d3, &d4_1, Order::OUTER);
    builder.order(&d4_0, &d4_1, Order::BEFORE);

    builder.get()
}

/// Generates many parallel stores.
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn parallel_store<'a>(signature: &'a Signature, gpu: &'a Gpu, num_blocks: &str,
                          n: &str, n_chained: u32, n_unroll: u32, num_wraps: u32, stride: u32,
                          mem_id: ir::mem::Id, array: &str) -> SearchSpace<'a> {
    assert!(stride*4 <= gpu.l1_cache_line);
    let mut builder = Builder::new(signature, gpu);
    let block_size = builder.param_size(num_blocks);
    let _ = builder.open_dim_ex(block_size, DimKind::BLOCK);
    let loop_size = builder.param_size(n);
    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);

    let thread_tilling = if num_wraps == 1 { vec![] } else { vec![gpu.wrap_size] };
    let thread_size = builder.cst_size(num_wraps * gpu.wrap_size);
    let d1 = builder.open_tiled_dim(thread_size, &thread_tilling);
    for d in &d1 { builder.action(Action::DimKind(d, DimKind::THREAD)); }
    for (x, y) in d1.iter().tuple_windows() { builder.order(&x, &y, Order::OUTER); }

    let d3 = builder.open_dim_ex(ir::Size::new(n_chained, vec![], 1), DimKind::UNROLL);
    let d4 = builder.open_dim_ex(ir::Size::new(n_unroll, vec![], 1), DimKind::UNROLL);
    let pattern = builder.unknown_access_pattern(mem_id);
    let mut strides = vec![
        (d3, ir::Size::new(n_unroll*num_wraps*gpu.wrap_size*gpu.l1_cache_line, vec![], 1)),
        (d4, ir::Size::new(num_wraps*gpu.wrap_size*gpu.l1_cache_line, vec![], 1)),
    ];
    if stride != 0 {
        let i = if num_wraps == 1 { 0 } else {
            strides.push((d1[0], ir::Size::new(gpu.wrap_size*gpu.l1_cache_line, vec![], 1)));
            1
        };
        strides.push((d1[i], ir::Size::new(stride*4, vec![], 1)));
    };
    let addr = builder.induction_var(&array, strides);
    builder.st_ex(&addr, &42f32, true, pattern, InstFlag::MEM_CG);
    builder.close_dim(&DimGroup::new(vec![d0, d3, d4]));

    builder.order(&d0, &d1, Order::OUTER);
    builder.order(&d1, &d3, Order::OUTER);
    builder.order(&d3, &d4, Order::OUTER);

    builder.get()
}

/// Generates a wrap of syncthreads separated by a single instruction.
pub fn syncthread<'a>(signature: &'a Signature, device: &'a Device,
                      n_iter: &str, n_chained: u32, wrap_size: u32) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let loop_size = builder.param_size(n_iter);
    let unroll_size = builder.cst_size(n_chained);
    let thread_size = builder.cst_size(wrap_size);

    let d0 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d1 = builder.open_dim_ex(unroll_size, DimKind::UNROLL);
    let d2 = builder.open_dim_ex(thread_size, DimKind::THREAD);
    let _ = builder.mov(&0i32);

    builder.order(&d0, &d1, Order::OUTER);
    builder.order(&d0, &d2, Order::OUTER);

    let mut kernel = builder.get();
    kernel.domain_mut().set_order(d1.into(), d2.into(), Order::OUTER);
    kernel
}

/// Generates a wrap of syncthreads separated by a single instruction.
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn chain_in_syncthread<'a>(signature: &'a Signature, device: &'a Device, n_iter: &str,
                               sync_chained: u32, add_chained: u32, wrap_size: u32,
                               out: &str, out_id: ir::mem::Id) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let loop_size = builder.param_size(n_iter);
    let sync_unroll_size = builder.cst_size(sync_chained);
    let thread_size = builder.cst_size(wrap_size);
    let add_unroll_size = builder.cst_size(add_chained);

    let d0 = builder.open_dim_ex(thread_size, DimKind::THREAD);
    let init = builder.mov(&0f32);

    let d1 = builder.open_dim_ex(loop_size, DimKind::LOOP);
    let d2 = builder.open_dim_ex(sync_unroll_size, DimKind::UNROLL);
    let d3 = builder.open_mapped_dim(&d0)[0];
    let d4 = builder.open_dim_ex(add_unroll_size, DimKind::UNROLL);
    let acc = builder.add(&Reduce(init), &2f32);
    builder.close_dim(&DimGroup::new(vec![d1, d2, d3, d4]));

    let d5 = builder.open_mapped_dim(&d0)[0];
    let pattern = builder.unknown_access_pattern(out_id);
    builder.st_ex(&out, &acc, true, pattern, InstFlag::MEM_CG);

    builder.order(&d1, &d2, Order::OUTER);
    builder.order(&d1, &d3, Order::OUTER);
    builder.order(&d2, &d4, Order::OUTER);
    builder.order(&d0, &d1, Order::BEFORE);
    builder.order(&d1, &d5, Order::BEFORE);
    builder.action(Action::ThreadMapping(d0, d3, ThreadMapping::MAPPED));
    builder.action(Action::ThreadMapping(d0, d5, ThreadMapping::MAPPED));

    let mut kernel = builder.get();
    kernel.domain_mut().set_order(d1.into(), d2.into(), Order::OUTER);
    kernel
}

/// Generates a global memory load in a loop.
pub fn load_in_loop<'a>(signature: &'a Signature, device: &'a Device, threads: u32,
                        out: &str, out_id: ir::mem::Id) -> SearchSpace<'a> {
        let mut builder = Builder::new(signature, device);
        let size_4 = builder.cst_size(4);
        let tmp_mem_size = builder.cst_size(4*4*threads);
        let tmp_mem = builder.allocate(tmp_mem_size.clone(), true);

        // Configure dimension sizes
        let threads_size = builder.cst_size(threads);
        let thread_dim_1_0 = builder.open_dim_ex(threads_size, DimKind::THREAD);
        let unroll_dim_0_0 = builder.open_dim_ex(size_4.clone(), DimKind::UNROLL);
        let acc_init = builder.mov(&0f32);
        builder.close_dim(&unroll_dim_0_0);

        let k_size = builder.param_size("k");
        let k_dim = builder.open_dim_ex(k_size, DimKind::LOOP);
        // Load A
        let unroll_dim_a = builder.open_dim_ex(size_4.clone(), DimKind::VECTOR);
        let (addr, pattern) = builder.tensor_access(
            &tmp_mem, tmp_mem.into(), &ir::Type::F(32), &[&thread_dim_1_0, &unroll_dim_a]);
        let a_val = builder.ld_ex(ir::Type::F(32), &addr, pattern, InstFlag::MEM_CG);
        builder.close_dim(&unroll_dim_a);
        // Mad a and b
        let unroll_dims_1 = builder.open_mapped_dim(&unroll_dim_0_0);
        let a_dim_map = ir::dim::Map::new(vec![(unroll_dim_a, unroll_dims_1[0])]);
        let a_op = ir::Operand::Inst(
            a_val, ir::Type::F(32), a_dim_map, ir::DimMapScope::Thread);
        let acc = builder.mad(&a_op, &2f32, &Reduce(acc_init));
        builder.close_dim(&k_dim);

        let _ = builder.open_mapped_dim(&unroll_dims_1);
        let (addr, pattern) = builder.tensor_access(&out, out_id, &ir::Type::F(32), &[]);
        let _ = builder.st_ex(&addr, &acc, true, pattern, InstFlag::MEM_CS);

        builder.order(&k_dim, &thread_dim_1_0, Order::INNER);
        builder.order(&unroll_dim_a, &unroll_dims_1[0], Order::BEFORE);
        builder.get()
    }

/// Instruments a kernel and stores the results in a `Table`.
///
/// * `args_range`: the arguments that must vary, with their range.
/// * `perf_counters`: the CUDA performance counters to monitor.
/// * `result`: the table in which to store the results.
pub fn run(context: &mut Context, space: &SearchSpace,
           args_range: &[(&str, &[i32])], counters: &PerfCounterSet,
           result_prefix: &[u64], result: &mut Table<u64>
          ) {
    if let Some(choice) = explorer::choice::list(space).next() {
        panic!("The benchmark is not completely scheduled: {:?}", choice);
    }
    let dev_fun = codegen::Function::build(space);
    let kernel = Kernel::compile(&dev_fun, context.gpu(), context.executor(), 1);
    for &(arg, range) in args_range { bind_scalar(arg, range[0], context); }
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
