use telamon_x86 as x86;
use telamon::helper;
use telamon::ir;

#[cfg(test)]
fn basic_test() {
    let device = x86::Cpu::dummy_cpu();
    let signature = ir::Signature::new("test".to_string());
    let builder = helper::Builder::new(&signature, &device);
    // This code builds the following function:
    // ```pseudocode
    // for i in 0..16:
    //   for j in 0..16:
    //      src[i] = 0;
    // for i in 0..16:
    //   dst = src[i]
    // ```
    // where all loops are unrolled.
    let dim0 = builder.open_dim_ex(ir::Size::new_const(4), DimKind::UNROLL);
    let dim1 = builder.open_dim_ex(ir::Size::new_const(8), DimKind::UNROLL);
    let src = builder.mov(&0i32);
    let src_var = builder.get_inst_variable(src);
    builder.close_dim(&dim1);
    let last_var = builder.create_last_variable(src_var, &[&dim1]);
    let dim2 = builder.open_mapped_dim(&dim0);
    builder.action(Action::DimKind(dim2[0], DimKind::UNROLL));
    builder.order(&dim0, &dim2, Order::BEFORE);
    let mapped_var = builder.create_dim_map_variable(last_var, &[(&dim0, &dim2)]);
}
