/// Tool that generates constraints from stdin to stdout.
extern crate telamon_gen;
extern crate env_logger;

fn main() {
    env_logger::init();
    telamon_gen::process(&mut std::io::stdin(), &mut std::io::stdout(), true);
}
