/// Error created when initializing the Executor.
#[derive(Debug, Fail)]
pub enum InitError {
    #[fail(display="must be compiled with --feature=cuda to use cuda")]
    NeedsCudaFeature,
}
