//! Contains integration tests for Exhaust.
#![allow(dead_code)]
extern crate env_logger;

pub mod fake;

use telamon::device::{Context, EvalMode};
use telamon::{explorer, ir, codegen};
use telamon::search_space::SearchSpace;
use std::io::sink;

/// Returns an empty function base.
pub fn empty_signature(num_block: u32) -> ir::Signature {
    ir::Signature {
        name: "empty".to_string(),
        params: vec![],
        mem_blocks: num_block,
    }
}

/// Find the best candidate for a function and outputs it.
pub fn gen_best(context: &Context, space: SearchSpace) {
    let mut config = explorer::Config::read();
    config.num_workers = 1;
    let best = explorer::find_best(&config, context, vec![space]).unwrap();
    context.device().gen_code(&best, &mut sink());
}

/// Checks the result of all valid candidates.
pub fn check_candidates<F>(space: SearchSpace, ctx: &Context, mut check: F)
        where F: FnMut() {
    explorer::gen_space(ctx, space, |_| (), |candidate| {
        debug!("testing candidate with actions {:?}", candidate.actions);
        let fun = codegen::Function::build(&candidate.space);
        ctx.evaluate(&fun, EvalMode::FindBest).unwrap();
        check();
    });
}
