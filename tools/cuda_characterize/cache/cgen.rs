//! Exhaust IR generators.
use telamon::ir;
use telamon::device::Device;
use telamon::helper::{Builder, Reduce};
use telamon::search_space::{DimKind, InstFlag, Order, SearchSpace};

/// Increment the values stored in an array.
pub fn incr_array<'a>(signature: &'a ir::Signature,
                      device: &'a Device,
                      ld_flag: InstFlag,
                      st_flag: InstFlag,
                      n_outer: &str,
                      n_inner: &str,
                      stride: &str,
                      tab_id: ir::mem::Id,
                      tab: &str) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let (_, inner_dim) = two_nested_loop(n_outer, n_inner, &mut builder);
    let ptr = builder.mad(&inner_dim, &stride, &tab);
    let pattern = builder.unknown_access_pattern(tab_id);
    let val = builder.ld_ex(ir::Type::F(32), &ptr, pattern.clone(), ld_flag);
    let res = builder.add(&val, &1f32);
    builder.st_ex(&ptr, &res, true, pattern, st_flag);
    builder.get()
}

/// Load two successive values, add them and store the results at the two locations. Every
/// location is used in two consecutive iterations of the loop.
pub fn add_successive<'a>(signature: &'a ir::Signature,
                          device: &'a Device,
                          ld_flag: InstFlag,
                          st_flag: InstFlag,
                          n_outer: &str,
                          n_inner: &str,
                          stride: &str,
                          tab_id: ir::mem::Id,
                          tab: &str) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let (_, inner_dim) = two_nested_loop(n_outer, n_inner, &mut builder);
    let ptr = builder.mad(&inner_dim, &stride, &tab);
    let ptr2 = builder.mad(&stride, &1i32, &ptr);
    let pattern = builder.unknown_access_pattern(tab_id);
    let val = builder.ld_ex(ir::Type::F(32), &ptr, pattern.clone(), ld_flag);
    let val2 = builder.ld_ex(ir::Type::F(32), &ptr2, pattern.clone(), ld_flag);
    let res = builder.add(&val, &val2);
    builder.st_ex(&ptr, &res, true, pattern.clone(), st_flag);
    builder.st_ex(&ptr2, &res, true, pattern, st_flag);
    builder.get()
}

/// Accumulate the values stored in an array.
pub fn acc_array<'a>(signature: &'a ir::Signature,
                     device: &'a Device,
                     ld_flag: InstFlag,
                     n_outer: &str,
                     n_inner: &str,
                     stride: &str,
                     tab_id: ir::mem::Id,
                     tab: &str) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let acc_init = builder.mov(&0f32);
    let (_, inner_dim) = two_nested_loop(n_outer, n_inner, &mut builder);

    // Compute the pointer to the data accessed by the loop iteration
    let ptr = builder.mad(&inner_dim, &stride, &tab);
    // Set the number instructions executed for 'mad' operation
    // Load v, add it to the accumulator
    let pattern = builder.unknown_access_pattern(tab_id);
    let val = builder.ld_ex(ir::Type::F(32), &ptr, pattern.clone(), ld_flag);
    let acc = builder.add(&val, &Reduce(acc_init));
    builder.close_dim(&inner_dim);
    // After the loop has ended, store the accumulated value
    builder.st_ex(&ptr, &acc, true, pattern, InstFlag::MEM_CS);
    builder.get()
}

/// Stores a value every `stride` in `tab`.
pub fn write_array<'a>(signature: &'a ir::Signature,
                       device: &'a Device,
                       st_flag: InstFlag,
                       n_outer: &str,
                       n_inner: &str,
                       stride: &str,
                       tab_id: ir::mem::Id,
                       tab: &str) -> SearchSpace<'a> {
    let mut builder = Builder::new(signature, device);
    let (_, inner_dim) = two_nested_loop(n_outer, n_inner, &mut builder);
    let ptr = builder.mad(&inner_dim, &stride, &tab);
    let pattern = builder.unknown_access_pattern(tab_id);
    builder.st_ex(&ptr, &1f32, true, pattern, st_flag);
    builder.get()
}

/// Adds to nested loops to the builder. Returns ths IDs of the loops.
fn two_nested_loop(n_outer: &str, n_inner: &str, builder: &mut Builder
                   ) -> (ir::dim::Id, ir::dim::Id) {
    let outer_size = builder.param_size(n_outer);
    let inner_size = builder.param_size(n_inner);
    let outer_dim = builder.open_dim_ex(outer_size, DimKind::LOOP);
    let inner_dim = builder.open_dim_ex(inner_size, DimKind::LOOP);
    builder.order(&outer_dim, &inner_dim, Order::OUTER);
    (outer_dim, inner_dim)
}
