/// Function shared among examples.
use rayon::prelude::*;
use itertools::Itertools;
use telamon::device::Context;
use telamon::{explorer, ir};
use telamon::helper::SignatureBuilder;
use telamon::helper::tensor::DimSize;
use telamon::search_space::SearchSpace;
use std;

/// Generates the code for the best candidate in the search space.
pub fn gen_best<'a, T>(search_space: Vec<SearchSpace>,
                       context: &'a T,
                       out: &str) where T: Context<'a> {
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

/// Creates a `DimSize`. If the instantiate flag is true, it uses a constant size,
/// otherwise it creates a parameter with the given name.
pub fn create_size<'a>(value: i32, name: &'a str,
                       is_generic: bool,
                       builder: &mut SignatureBuilder) -> DimSize<'a> {
    if is_generic {
        builder.param(name, value);
        DimSize::Param(name)
    } else { DimSize::Const(value as u32) }
}

/// Removes tiles of size 1.
pub fn cleanup_tiling(tiling: &[u32]) -> Vec<u32> {
    tiling.iter().cloned().filter(|&t| t > 1).collect()
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

pub fn par_iter_product<I1, I2>(i1: I1, i2: I2)
    -> impl ParallelIterator<Item=(I1::Item, I2::Item)>
where
    I1: IntoParallelIterator, I1::Item: Clone + Sync,
    I2: IntoParallelIterator + Clone + Send + Sync
{
    i1.into_par_iter().flat_map(move |x| {
        i2.clone().into_par_iter().map(move |y| (x.clone(), y))
    })
}
