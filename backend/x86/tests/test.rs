use bincode;
use telamon_x86 as x86;
use telamon::codegen;
use telamon::device::{self, Context};
use telamon::helper;
use telamon::ir;
use telamon::model;
use telamon::search_space::*;
use telamon::explorer::{self, local_selection};
use std::{fs::File, io::Write};
use std::io::Error as IoError;

#[derive(Debug)]
enum Error {
    Io(std::io::Error),
    Telamon,
}

impl From<IoError> for Error {
    fn from(err: IoError) -> Self {
        Error::Io(err)
    }
}

#[test]
fn basic_test() -> Result<(), Error> {
    let context = x86::Context::default();
    let signature = ir::Signature::new("test".to_string());
    let mut builder = helper::Builder::new(&signature, context.device());
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
    builder.mov(&mapped_var);

    let space = builder.get();
    let bound = model::bound(&space, &context);
    let mut candidate = explorer::Candidate::new(space, bound);

    // Building a candidate
    let order = explorer::config::NewNodeOrder::WeightedRandom;
    let ordering = explorer::config::ChoiceOrdering::default();
    loop {
            // We don't care about the cut as we have no performance model
        let cand_clone = candidate.clone();
        let leaf =
            local_selection::descend(&ordering, order, &context, cand_clone, 1.0f64);
        if let Some(leaf) = leaf {
            let device_fn = codegen::Function::build(&leaf.space);
            context.evaluate(&device_fn, device::EvalMode::FindBest).map_err(|_| Error::Telamon)?;
            candidate = leaf;
            break;
        } else {
            return Err(Error::Telamon);
        }
    }
    let dump = bincode::serialize(&candidate.actions).unwrap();
    let mut file = File::create("cand_dump.log")?;
    file.write_all(&dump)?;
    Ok(())
}
