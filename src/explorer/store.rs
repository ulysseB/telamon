use explorer::candidate::Candidate;
use device::Context;

/// A Trait defining a structure containing the candidates, meant to explore the
/// search space
pub trait Store<'a> : Sync  {
    /// Transmits the information needed to update the store after a `Candidate` is
    /// evaluated.
    type PayLoad: Send;
    /// Updates the value that will be used to prune the search space
    fn update_cut(&self, new_cut: f64);
    /// Commit the result of an evaluation back to Store
    fn commit_evaluation(&self, payload: Self::PayLoad, eval: f64);
    /// Retrieve a Candidate for evaluation, returns `None` if no candidate remains.
    fn explore(&self, context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)>;
    /// Displays statistics about the candidate store.
    fn print_stats(&self) {}
}
