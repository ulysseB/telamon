//! Framework to run benchmarks.
#![allow(dead_code)]
use pbr;
use std;
use std::time::{Instant, Duration};

lazy_static! {
    static ref FILTER: Option<String> = ::std::env::args().skip(1).next();
}

/// Ouputs of a benchmark.
#[derive(Clone, Debug, Default)]
pub struct Bencher {
    name: &'static str,
    num_samples: usize,
}

impl Bencher {
    /// Creates a new Bencher.
    pub fn new(name: &'static str, num_samples: usize) -> Self {
        Bencher { name, num_samples }
    }

    /// Returns the targeted number of samples.
    pub fn num_samples(&self) -> usize { self.num_samples }

    /// Runs `f` `samples` times to estimate its average execution time.
    pub fn bench<T, F: FnMut() -> T>(&self, mut f: F) {
        let progress = Progress::new(self.name, self.num_samples);
        let t = Instant::now();
        for _ in 0..self.num_samples {
            f();
            progress.incr();
        }
        progress.finish(&self.format(Instant::now() - t, self.num_samples));
    }

    /// Runs `f` to estimate the average execution time of an iteration. `f` must run `n`
    /// iterations.
    pub fn bench_single<T, F: FnOnce(&Progress) -> T>(&self, f: F) {
        let progress = Progress::new(self.name, self.num_samples);
        let t = Instant::now();
        f(&progress);
        let n = progress.value();
        progress.finish(&self.format(Instant::now() - t, n));
    }

    /// Formats the benchmark results.
    pub fn format(&self, t: Duration, samples: usize) -> String {
        let total = (t.as_secs()*1000) as usize + (t.subsec_nanos()/1000000) as usize;
        let per_iter = total/samples;
        format!("{}: {}ms/iter ({} ms / {} iter)", self.name, per_iter, total, samples)
    }
}

/// Displays the current progress of the benchmark.
pub struct Progress {
    bar: std::sync::Mutex<pbr::ProgressBar<std::io::Stdout>>
}

impl Progress {
    /// Displays an empty progress bar.
    fn new(name: &str, samples: usize) -> Self {
        let msg = format!("{}: ", name);
        let mut pb = pbr::ProgressBar::new(samples as u64);
        pb.show_percent = false;
        pb.message(&msg);
        pb.set_max_refresh_rate(Some(Duration::from_millis(200)));
        Progress { bar: std::sync::Mutex::new(pb) }
    }

    /// Indicates that an iteration is finished.
    pub fn incr(&self) { self.bar.lock().unwrap().inc(); }

    /// Displays the given message instead of the progress bar.
    fn finish(mut self, msg: &str) {
        self.bar.get_mut().unwrap().finish_print(msg);
    }

    /// Returns the current value of the progress bar.
    fn value(&self) -> usize { self.bar.lock().unwrap().total as usize }
}

/// Indicates if the filter should be run.
pub fn is_enabled(name: &str) -> bool {
    ::bencher::FILTER.as_ref().map(|filter| name.contains(filter)).unwrap_or(true)
}

/// Runs a benchmark.
macro_rules! benchmark {
    ($name:ident, $samples:expr$(,$args:expr)*) => {
        let str_name = stringify!($name);
        if is_enabled(str_name) {
            let bencher = Bencher::new(str_name, $samples);
            $name(&bencher, $($args),*)
        }
    }
}
