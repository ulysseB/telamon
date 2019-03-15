use bincode;
use rpds::List;
use telamon_x86 as x86;
use telamon::codegen;
use telamon::device::{self, Context};
use telamon::helper;
use telamon::ir;
use telamon::model;
use telamon::search_space::*;
use telamon::explorer::{self, Candidate, choice::ActionEx, local_selection};
use std::{convert::AsRef, fs::File, io::{Error as IoError, Write}, path::Path};

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

fn reload_cand<P: AsRef<Path>>(candidate: Candidate, context: &Context, dump_path: P) -> Result<(), Error> {

    // Getting the action taken from log file
    let cand_bytes = std::fs::read(dump_path)?;
    let action_list: List<ActionEx> = bincode::deserialize(&cand_bytes).unwrap();
    let implem = action_list.iter().fold(candidate, |cand, action| cand.apply_decision(&context, action.clone()).expect("Could not apply some action"));

    // Running candidate
    let device_fn = codegen::Function::build(&implem.space);
    context.evaluate(&device_fn, device::EvalMode::FindBest).map_err(|_| Error::Telamon)?;
    Ok(())
}

#[test]
fn basic_test() -> Result<(), Error> {
    let context = x86::Context::default();

    let signature = ir::Signature::new("test".to_string());
    let space = build_cand(&signature, &context);
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
    generate_dump(candidate, "cand_dump.log")?;
    Ok(())
}

#[test]
/// Reload a candidate from a dull kernel using the dump
/// This function always execute the exact same implementation
fn reload_plain_cand() -> Result<(), Error> {
    let context = x86::Context::default();

    // Building back the same candidate we used before
    let signature = ir::Signature::new("test".to_string());
    let space = build_cand(&signature, &context);
    let bound = model::bound(&space, &context);
    let candidate = explorer::Candidate::new(space, bound);

    reload_cand(candidate, &context, "cand_dump.log")?;
    Ok(())
}
