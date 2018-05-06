/// Function shared among examples.
use itertools::Itertools;
use telamon::device::Context;
use telamon::{explorer, ir};
use telamon::search_space::SearchSpace;
use std;

/// Generates the code for the best candidate in the search space.
pub fn gen_best<'a>(search_space: Vec<SearchSpace>,
                    context: &'a Context,
                    out: &str) {
    let conf = explorer::Config::read();
    let begin_time = std::time::Instant::now();
    let best_opt = explorer::find_best(&conf, context, search_space);
    let duration = std::time::Instant::now() - begin_time;
    warn!("Search completed in {}s", duration.as_secs());
    match best_opt {
        Some(best) => {
            let mut file = std::fs::File::create(out).unwrap();
            context.device().gen_code(&best, &mut file)
        }
        None => println!("Did not find any well suited candidate before timeout"),
    }
}

/// Generate a name for the output file.
pub fn file_name(name: &str,
                 _: ir::Type,
                 sizes: &[i32],
                 instantiated: bool) -> String {
    const PATH: &str = "examples/out/";
    std::fs::create_dir_all(PATH).unwrap();
    let sizes = sizes.iter().format_with("", |i, f| f(&format_args!("_{}", i)));
    format!("{}{}_{}{}.c", PATH, name, instantiated, sizes)
}
