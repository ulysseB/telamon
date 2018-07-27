//! Exploration of the search space.
pub use explorer::candidate::Candidate;

use device::Context;
use explorer::choice;
use explorer::store::Store;
use interval_heap::IntervalHeap;
use rpds::List;
use std;
use std::f64;

impl<'a> Store<'a> for ParallelCandidateList<'a> {
    type PayLoad = ();

    type Event = ();

    fn update_cut(&self, new_cut: f64) {
        self.lock().0.update_cut(new_cut);
    }

    fn commit_evaluation(
        &self,
        _actions: &List<choice::ActionEx>,
        (): Self::PayLoad,
        _: f64)
    { }

    fn explore(&self, context: &Context) -> Option<(Candidate<'a>, Self::PayLoad)> {
        loop {
            if let Some(candidate) = self.pop() {
                let choice_opt = choice::list(&candidate.space).next();
                if let Some(choice) = choice_opt {
                    self.insert_many(candidate.apply_choice(context, choice));
                } else {
                    return Some((candidate, ()));
                }
            }
            else {return None;}
        }
    }
}

/// A `CandidateList` that can be accessed by multiple threads.
pub struct ParallelCandidateList<'a> {
    mutex: std::sync::Mutex<(CandidateList<'a>, usize)>,
    wakeup: std::sync::Condvar,
}

impl<'a> ParallelCandidateList<'a> {
    /// Creates a new `ParallelCandidateList` that can be accessed by num_worker threads.
    pub fn new(num_worker: usize) -> Self {
        ParallelCandidateList {
            mutex: std::sync::Mutex::new((CandidateList::new(), num_worker)),
            wakeup: std::sync::Condvar::new(),
        }
    }

    /// Insert a candidate to process.
    pub fn insert(&self, candidate: Candidate<'a>) {
        self.lock().0.insert(candidate);
        self.wakeup.notify_all();
    }

    /// Insert multiple candidates to process.
    pub fn insert_many(&self, candidates: Vec<Candidate<'a>>) {
        let mut lock = self.lock();
        for candidate in candidates { lock.0.insert(candidate); }
        self.wakeup.notify_all();
    }


    /// Returns a candidate to process or `None` if the queue has been entirely processed.
    pub fn pop(&self) -> Option<Candidate<'a>> {
        let mut lock = self.lock();
        loop {
            if let Some(candidate) = lock.0.pop() {
                return Some(candidate);
            }
            if lock.1 == 1 {
                lock.1 -= 1;
                self.wakeup.notify_all();
                return None;
            }
            lock.1 -= 1;
            lock = unwrap!(self.wakeup.wait(lock));
            lock.1 += 1;
        }
    }

    /// Acquire the lock to the candidate list
    fn lock(&self) -> std::sync::MutexGuard<(CandidateList<'a>, usize)> {
        unwrap!(self.mutex.lock())
    }
}

pub struct CandidateList<'a> {
    /// The maximum value over which we drop candidates
    cut: f64,
    /// The queue of candidates to evaluate.
    queue: IntervalHeap<Candidate<'a>>,
    /// The number of leaf encountered.
    n_leaf: usize,
    /// The number of candidate encountered.
    n_candidate: usize,
    /// The number of candidate dropped.
    n_dropped: usize,
}

impl<'a> CandidateList<'a> {
    /// Create an empty candidate list.
    pub fn new() -> Self {
        CandidateList {
            cut: std::f64::INFINITY,
            queue: IntervalHeap::new(),
            n_leaf: 0,
            n_candidate: 0,
            n_dropped: 0,
        }
    }

    /// Inserts a candidate to process.
    pub fn insert(&mut self, candidate: Candidate<'a>) {
        self.n_candidate += 1;
        if candidate.bound.value() < self.cut {
            let bound = candidate.bound.value();
            info!("inserting candidate {}: {:.4e}ns < {:.4e}ns, {} in queue",
                  self.n_candidate, bound, self.cut, self.queue.len());
            self.queue.push(candidate);
        } else {
            self.drop_candidate(candidate);
        }
    }

    pub fn update_cut(&mut self, new_cut: f64) {
        self.cut = new_cut;
        while self.queue.max().map_or(false, |x| x.bound.value() >= new_cut) {
            let candidate = unwrap!(self.queue.pop_max());
            self.drop_candidate(candidate);
        }
    }



    /// Returns a candidate to process.
    pub fn pop(&mut self) -> Option<Candidate<'a>> {
        let candidate = self.queue.pop_min();
        if let Some(ref x) = candidate {
            warn!("candidate {}, depth {}, bound {:.4e}ns, cut {:.4e}ns, queue {}, leaves {}",
                  self.n_candidate, x.depth, x.bound.value(), self.cut,
                  self.queue.len(), self.n_leaf);
        }
        candidate
    }


    /// Drops a candiate.
    fn drop_candidate(&mut self, candidate: Candidate<'a>) {
        info!("dropping candidate: {:.4e}ns >= {:.4e}ns.",
              candidate.bound.value(), self.cut);
        self.n_dropped += 1;
    }
}
