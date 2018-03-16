/// Function shared among examples.
extern crate getopts;
extern crate itertools;

use telamon::device::Context;
use telamon::explorer;
use telamon::search_space::SearchSpace;
use std;

/// Generates the code for the best candidate in the search space.
pub fn gen_best<'a, T>(search_space: Vec<SearchSpace>, context: &'a T) where T: Context<'a> {
    let conf = explorer::Config::read();
    let begin_time = std::time::Instant::now();
    let best = explorer::find_best(&conf, context, search_space).unwrap();
    let duration = std::time::Instant::now() - begin_time;
    warn!("best candidate found in {}s", duration.as_secs());
    context.device().gen_code(&best, &mut std::io::stdout());
}
