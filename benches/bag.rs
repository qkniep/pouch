//! Bag benchmarks — `Bag` vs `SortedSet` / `UnsortedSet`, all over `Vec<u64>`.
//!
//! A bag keeps duplicates; the sets dedup. To make the comparison apples-to-apples
//! the inputs are **distinct** keys, so all three end up holding `n` elements — the
//! `build` rows then measure *the cost of the set discipline a bag skips* (the
//! sort/dedup/per-insert shift), not a difference in result size. `contains` shows
//! the other side of the trade: a bag scans linearly like `UnsortedSet`, so it is
//! not a membership structure — reach for `SortedSet` (binary search) when lookup
//! is the workload.
//!
//! Run: `cargo bench --bench bag` (filter with `-- build`, `-- contains`, …).

use divan::counter::ItemsCount;
use divan::{black_box, Bencher};
use pouch::{Bag, SortedSet, UnsortedSet};

mod common;
use common::keys;

fn main() {
    divan::main();
}

/// Element counts, from the small-collection sweet spot up into the regime where
/// the sets' O(n) per-insert dedup / shift starts to bite.
const SIZES: [usize; 5] = [4, 16, 64, 256, 1024];

/// The surface common to every contender. `Sync` because divan may drive a benched
/// closure from multiple threads.
trait Collection: Sync {
    fn build(keys: &[u64]) -> Self;
    fn contains(&self, key: u64) -> bool;
    fn sum(&self) -> u64;
}

// A bag's idiomatic bulk build is a bare append — `O(n)`, no dedup, no sort.
struct PouchBag(Bag<Vec<u64>>);
impl Collection for PouchBag {
    fn build(keys: &[u64]) -> Self {
        PouchBag(Bag::try_from_iter(keys.iter().copied()).unwrap())
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn sum(&self) -> u64 {
        self.0
            .as_slice()
            .iter()
            .copied()
            .fold(0u64, u64::wrapping_add)
    }
}

// SortedSet's idiomatic bulk build: `try_from_iter` — append, then sort + dedup
// once (`O(n log n)`). Membership is `binary_search` (`O(log n)`).
struct PouchSorted(SortedSet<Vec<u64>>);
impl Collection for PouchSorted {
    fn build(keys: &[u64]) -> Self {
        PouchSorted(SortedSet::try_from_iter(keys.iter().copied()).unwrap())
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn sum(&self) -> u64 {
        self.0
            .as_slice()
            .iter()
            .copied()
            .fold(0u64, u64::wrapping_add)
    }
}

// UnsortedSet has no faster dedup without `Ord`, so its build is an insert loop
// scanning the kept elements per item (`O(n²)`). Membership is a linear scan.
struct PouchUnsorted(UnsortedSet<Vec<u64>>);
impl Collection for PouchUnsorted {
    fn build(keys: &[u64]) -> Self {
        let mut s = UnsortedSet::new();
        for &k in keys {
            s.insert(k);
        }
        PouchUnsorted(s)
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn sum(&self) -> u64 {
        self.0
            .as_slice()
            .iter()
            .copied()
            .fold(0u64, u64::wrapping_add)
    }
}

/// Build from keys arriving in pseudo-random order.
#[divan::bench(types = [PouchBag, PouchSorted, PouchUnsorted], args = SIZES)]
fn build_random<C: Collection>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.random.as_slice();
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| C::build(black_box(src)));
}

/// Build from already-ascending keys.
#[divan::bench(types = [PouchBag, PouchSorted, PouchUnsorted], args = SIZES)]
fn build_sorted<C: Collection>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.sorted.as_slice();
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| C::build(black_box(src)));
}

/// Append `n` elements one at a time — the accumulate-as-you-go path (per-key event
/// logs, group-by). A bag pushes in `O(1)`; the sets pay `O(n)` per insert.
#[divan::bench(args = SIZES)]
fn push_loop_bag(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.random.as_slice();
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut b = Bag::<Vec<u64>>::new();
        for &x in black_box(src) {
            b.push(x);
        }
        b
    });
}

#[divan::bench(args = SIZES)]
fn push_loop_sorted(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.random.as_slice();
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut s = SortedSet::<Vec<u64>>::new();
        for &x in black_box(src) {
            s.insert(x);
        }
        s
    });
}

#[divan::bench(args = SIZES)]
fn push_loop_unsorted(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.random.as_slice();
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut s = UnsortedSet::<Vec<u64>>::new();
        for &x in black_box(src) {
            s.insert(x);
        }
        s
    });
}

/// `n` membership tests, all present. A bag is not a membership structure — this
/// row exists to show it tracks `UnsortedSet` (linear) and loses to `SortedSet`.
#[divan::bench(types = [PouchBag, PouchSorted, PouchUnsorted], args = SIZES)]
fn contains_hit<C: Collection>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let c = C::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.random {
            hits += c.contains(black_box(key)) as u64;
        }
        hits
    });
}

/// `n` membership tests, none present.
#[divan::bench(types = [PouchBag, PouchSorted, PouchUnsorted], args = SIZES)]
fn contains_miss<C: Collection>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let c = C::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.misses {
            hits += c.contains(black_box(key)) as u64;
        }
        hits
    });
}

/// Sum every element (full iteration). All three are a contiguous slice scan, so
/// this should tie — a bag gives up nothing on read.
#[divan::bench(types = [PouchBag, PouchSorted, PouchUnsorted], args = SIZES)]
fn iter<C: Collection>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let c = C::build(&k.random);
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| black_box(c.sum()));
}
