use explorer::candidate::Candidate;
use device::Context;
use explorer::config::Config;

/// A Trait defining a structure containing the candidates, meant to explore the
/// search space
pub trait Store<'a> : Sync  {
    /// Type that could be needed to update correctly the search space upon
    /// execution
    type PayLoad : Send;
    /// Updates the value that will be used to prune the search space
    fn update_cut(&self, new_cut: f64);
    /// Commit the result of an evaluation back to Store
    fn commit_evaluation(&self, config: &Config, payload: Self::PayLoad, eval: f64);
    /// Retrieve a Candidate for evaluation
    fn explore(&self, config: &Config, context: &Context) 
        -> Option<(Candidate<'a>, Self::PayLoad)>;
}
