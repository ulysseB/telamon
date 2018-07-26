extern crate env_logger;
extern crate telamon;
extern crate serde_json;
#[macro_use]
extern crate telamon_utils;

use telamon::device::cuda;

fn main() {
    env_logger::init();
    let executor = cuda::Executor::init();
    let gpu = cuda::characterize::characterize(&executor);
    unwrap!(serde_json::to_writer_pretty(std::io::stdout(), &gpu));
    //instruction::print_smx_bandwidth(&gpu, &executor);
    //instruction::print_smx_store_bandwidth(&gpu, &executor);*/
}
