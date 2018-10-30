pub use self::lazy::Lazy;
pub use self::thunk::Thunk;

pub mod lazy {
    use std::marker::PhantomData;
    use std::mem;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    pub struct Lazy<T, I = ()> {
        val: AtomicUsize,
        init: AtomicUsize,
        _marker: PhantomData<(Box<T>, Arc<I>)>,
    }

    impl<T, I> Lazy<T, I> {
        pub fn new(init: Arc<I>) -> Self {
            Lazy {
                val: AtomicUsize::new(0),
                init: AtomicUsize::new(Arc::into_raw(init) as usize),
                _marker: PhantomData,
            }
        }

        pub fn from_val(val: T) -> Self {
            Lazy {
                val: AtomicUsize::new(Box::into_raw(Box::new(val)) as usize),
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

        pub fn force<F: FnOnce(Arc<I>) -> T>(&self, fun: F) -> &T {
            unsafe {
                self.get().unwrap_or_else(|| {
                    let init = self.init.swap(0, Ordering::Relaxed) as *const I;

                    if init.is_null() {
                        // Another thread is currently forcing the
                        // value; wait until they are done.
                        loop {
                            if let Some(val) = self.get() {
                                break val;
                            }
                        }
                    } else {
                        let val = fun(Arc::from_raw(init));
                        let val = Box::into_raw(Box::new(val)) as *const T;
                        self.val.store(val as usize, Ordering::Relaxed);
                        &*val
                    }
                })
            }
        }

        pub fn into_inner<F: FnOnce(Arc<I>) -> T>(self, fun: F) -> T {
            unsafe {
                let val = self.val.swap(0, Ordering::Relaxed) as *mut T;
                if val.is_null() {
                    let init = self.init.swap(0, Ordering::Relaxed) as *const I;
                    assert!(!init.is_null());

                    fun(Arc::from_raw(init))
                } else {
                    *Box::from_raw(val)
                }
            }
        }
    }

    impl<T, I> Drop for Lazy<T, I> {
        fn drop(&mut self) {
            // Since we have &mut self, no other thread can have a
            // reference to the objets pointed to by the val or init
            // fields.
            unsafe {
                let val = mem::replace(self.val.get_mut(), 0) as *mut T;
                let init = mem::replace(self.init.get_mut(), 0) as *const I;

                if !val.is_null() {
                    mem::drop(Box::from_raw(val));
                }

                if !init.is_null() {
                    mem::drop(Arc::from_raw(init));
                }
            }
        }
    }
}

pub mod thunk {
    use std::sync::Arc;

    use super::lazy::Lazy;
    use ops::TryDeref;

    pub struct Thunk<T, F: FnOnce() -> T> {
        lazy: Lazy<T, F>,
    }

    impl<T, F: FnOnce() -> T> Thunk<T, F> {
        pub fn new(fun: F) -> Self {
            Thunk {
                lazy: Lazy::new(Arc::new(fun)),
            }
        }

        pub fn unwrap(thunk: Self) -> T {
            thunk
                .lazy
                .into_inner(|f| Arc::try_unwrap(f).ok().unwrap()())
        }
    }

    impl<T, F: FnOnce() -> Option<T>> Thunk<Option<T>, F> {
        pub fn as_ref(&self) -> ThunkRef<'_, T, F> {
            ThunkRef(self)
        }
    }

    impl<'a, T, F: FnOnce() -> Option<T>> TryDeref for Thunk<Option<T>, F> {
        type Target = T;

        fn try_deref(&self) -> Option<&T> {
            self.lazy
                .force(|f| Arc::try_unwrap(f).ok().unwrap()())
                .as_ref()
        }
    }

    pub struct ThunkRef<'a, T: 'a, F: FnOnce() -> Option<T> + 'a>(
        &'a Thunk<Option<T>, F>,
    );

    impl<'a, T: 'a, F: FnOnce() -> Option<T> + 'a> TryDeref for ThunkRef<'a, T, F> {
        type Target = T;

        fn try_deref(&self) -> Option<&T> {
            self.0.try_deref()
        }
    }
}
