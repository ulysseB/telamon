trait KernelParameters<'a>: Sized {
    type Kernel: kernel::Kernel<'a, Parameters = Self>;

    fn with_candidates<T, F, C>(&self, context: &mut C, body: F) -> T
        where
            F: FnOnce(Vec<Candidate<'_>>, &C) -> T,
            C: ArgMap + Context + 'a,
    {
    }

trait Kernel<'a, T, F, C>
where
    F: FnOnce(Vec<Candidate<'_>>, &C) -> T,
    C: ArgMap + Context + 'a,
{
    fn with_candidates(&self, context: &mut C, body: F) -> T;
}

impl<'a, P, T, F, C> Kernel<'a, T, F, C> for P
where
    P: KernelParameters<'a>,
