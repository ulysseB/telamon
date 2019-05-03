use std::rc::Rc;

use sym::*;

fn main() {
    let x: Float<f64> = Float::from_numeric(0.3);
    let y: Float<f64> = Float::from_numeric(0.7);
    let z: Float<f64> = Rc::new(FloatInner::Atom(Clovis)).into();
    println!("{:?}", x.apply(AddOp, &y));
    println!("{:?}", x.apply(AddOp, &z));
    println!("{:?}", x.inner.apply(AddOp, &y.inner));
    println!("{:?}", x.inner.apply(AddOp, &z.inner));
}
