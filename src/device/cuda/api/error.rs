/// Error created when initializing the Executor.

use failure::Fail;

#[derive(Debug, Fail)]
pub enum InitError {
    #[fail(display = "must be compiled with --feature=cuda to use cuda")]
    NeedsCudaFeature,
}
