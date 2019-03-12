use telamon_x86 as x86;
use telamon::helper;
use telamon::ir;

#[cfg(test)]
fn tutu() {
    let device = x86::Cpu::dummy_cpu();
    let signature = ir::Signature::new("test".to_string());
    let builder = helper::Builder::new(&signature, &device);
}
