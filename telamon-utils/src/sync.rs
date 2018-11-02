pub use self::lazy::Lazy;
pub use self::thunk::Thunk;

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub trait Pointer: Deref {
    fn new(value: Self::Target) -> Self
    where
        Self::Target: Sized;

    fn into_raw(this: Self) -> usize;

    unsafe fn from_raw(ptr: usize) -> Self;
}

pub trait OwningPointer: Pointer + DerefMut {
    fn into_inner(this: Self) -> Self::Target
    where
        Self::Target: Sized;
}

impl<T> Pointer for Box<T> {
    fn new(value: T) -> Self {
        Box::new(value)
    }

    fn into_raw(this: Self) -> usize {
        Box::into_raw(this) as usize
    }

    unsafe fn from_raw(ptr: usize) -> Self {
        Box::from_raw(ptr as *mut T)
    }
}

impl<T> OwningPointer for Box<T> {
    fn into_inner(this: Self) -> T
    where
        T: Sized,
    {
        *this
    }
}

impl<T> Pointer for Arc<T> {
    fn new(value: T) -> Self
    where
        T: Sized,
    {
        Arc::new(value)
    }

    fn into_raw(this: Self) -> usize {
        Arc::into_raw(this) as usize
    }

    unsafe fn from_raw(ptr: usize) -> Self {
        Arc::from_raw(ptr as *const T)
    }
}

pub mod lazy {
    use std::marker::PhantomData;
    use std::mem;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::{OwningPointer, Pointer};

    pub struct Lazy<
        T,
        I = (),
        PT: OwningPointer<Target = T> = Box<T>,
        PI: Pointer<Target = I> = Arc<I>,
    > {
        val: AtomicUsize,
        init: AtomicUsize,
        _marker: PhantomData<(T, PT, I, PI)>,
    }

    impl<T, I, PT: OwningPointer<Target = T>, PI: Pointer<Target = I>> Lazy<T, I, PT, PI> {
        pub fn new(init: PI) -> Self {
            Lazy {
                val: AtomicUsize::new(0),
                init: AtomicUsize::new(PI::into_raw(init)),
                _marker: PhantomData,
            }
        }

        pub fn from_val(val: T) -> Self {
            Lazy {
                val: AtomicUsize::new(PT::into_raw(PT::new(val))),
                init: AtomicUsize::new(0),
                _marker: PhantomData,
            }
        }

        pub fn get(&self) -> Option<&T> {
            unsafe {
                let val = self.val.load(Ordering::Relaxed) as *mut T;
                if val.is_null() {
                    None
                } else {
                    Some(&*val)
                }
            }
        }

        pub fn force<F: FnOnce(PI) -> T>(&self, fun: F) -> &T {
            unsafe {
                self.get().unwrap_or_else(|| {
                    let init = self.init.swap(0, Ordering::Relaxed);

                    if init == 0 {
                        // Another thread is currently forcing the
                        // value; wait until they are done.
                        loop {
                            if let Some(val) = self.get() {
                                break val;
                            }
                        }
                    } else {
                        let val = fun(PI::from_raw(init));
                        let val = PT::into_raw(PT::new(val));
                        self.val.store(val, Ordering::Relaxed);
                        &*(val as *const T)
                    }
                })
            }
        }

        pub fn into_inner<F: FnOnce(PI) -> T>(self, fun: F) -> T {
            unsafe {
                let val = self.val.swap(0, Ordering::Relaxed);
                if val == 0 {
                    let init = self.init.swap(0, Ordering::Relaxed);
                    assert!(init != 0);

                    fun(PI::from_raw(init))
                } else {
                    PT::into_inner(PT::from_raw(val))
                }
            }
        }
    }

    impl<T, I, PT: OwningPointer<Target = T>, PI: Pointer<Target = I>> Drop
        for Lazy<T, I, PT, PI>
    {
        fn drop(&mut self) {
            // Since we have &mut self, no other thread can have a
            // reference to the objets pointed to by the val or init
            // fields.
            unsafe {
                let val = mem::replace(self.val.get_mut(), 0);
                let init = mem::replace(self.init.get_mut(), 0);

                if val != 0 {
                    mem::drop(PT::from_raw(val));
                }

                if init != 0 {
                    mem::drop(PI::from_raw(init));
                }
            }
        }
    }
}

pub mod thunk {
    use super::lazy::Lazy;
    use ops::TryDeref;

    pub struct Thunk<T, F: FnOnce() -> T> {
        lazy: Lazy<T, F, Box<T>, Box<F>>,
    }

    impl<T, F: FnOnce() -> T> Thunk<T, F> {
        pub fn new(fun: F) -> Self {
            Thunk {
                lazy: Lazy::new(Box::new(fun)),
            }
        }

        pub fn unwrap(thunk: Self) -> T {
            thunk.lazy.into_inner(|f| (*f)())
        }
    }

    impl<'a, T, F: FnOnce() -> Option<T>> TryDeref for Thunk<Option<T>, F> {
        type Target = T;

        fn try_deref(&self) -> Option<&T> {
            self.lazy.force(|f| (*f)()).as_ref()
        }
    }

    impl<'a, T: 'a, F: FnOnce() -> Option<T> + 'a> TryDeref for &'a Thunk<Option<T>, F> {
        type Target = T;

        fn try_deref(&self) -> Option<&T> {
            Thunk::try_deref(self)
        }
    }
}
