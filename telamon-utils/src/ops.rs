use std::ops::Deref;

/// A variant of the Deref trait which is allowed to fail and return `None`.
///
/// This is meant to allow cases where the underlying object is not available at the time
pub trait TryDeref {
    type Target: ?Sized;

    fn try_deref(&self) -> Option<&Self::Target>;
}

impl<T> TryDeref for Option<T>
where
    T: Deref,
{
    type Target = <T as Deref>::Target;

    fn try_deref(&self) -> Option<&Self::Target> {
        self.as_ref().map(Deref::deref)
    }
}
