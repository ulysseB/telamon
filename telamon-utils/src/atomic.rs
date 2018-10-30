use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct AtomicF64(AtomicUsize);

impl fmt::Debug for AtomicF64 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.load())
    }
}

impl AtomicF64 {
    pub fn new(val: f64) -> Self {
        AtomicF64(AtomicUsize::new(val.to_bits() as usize))
    }

    pub fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed) as u64)
    }

    pub fn try_add(&self, val: f64) -> Result<f64, ()> {
        let cur = self.0.load(Ordering::Relaxed);
        let new = (f64::from_bits(cur as u64) + val).to_bits() as usize;
        if self.0.compare_and_swap(cur, new, Ordering::Relaxed) == cur {
            Ok(f64::from_bits(cur as u64))
        } else {
            Err(())
        }
    }

    pub fn add(&self, val: f64) {
        while self.try_add(val).is_err() {}
    }
}

impl Default for AtomicF64 {
    fn default() -> Self {
        0.0.into()
    }
}

impl From<f64> for AtomicF64 {
    fn from(value: f64) -> Self {
        AtomicF64::new(value)
    }
}
