//! Set benchmarks — pouch `SortedSet` / `UnsortedSet` vs `sorted-vec`,
//! `vecmap-rs`, and `std::collections::{BTreeSet, HashSet}` (plus FxHash).
//! `litemap` is map-only, so it doesn't appear here. The `Hash` baseline uses
//! std's default (SipHash) hasher; `FxHash` is the fast-hash baseline.
//!
//! Run: `cargo bench --bench set` (filter with `-- contains`, `-- Sorted`, …).

use std::collections::{BTreeSet, HashSet};

use divan::counter::ItemsCount;
use divan::{black_box, Bencher};
use pouch::{SortedSet, UnsortedSet};
use rustc_hash::FxHashSet;
use sorted_vec::SortedSet as SortedVecSet;
use vecmap::VecSet;

mod common;
use common::keys;

fn main() {
    divan::main();
}

/// Element counts, spanning the small-collection sweet spot up into the regime
/// where O(n) shifting / scanning starts to bite.
const SIZES: [usize; 5] = [4, 16, 64, 256, 1024];

/// The set surface common to every contender.
/// `Sync` because divan may drive a benched closure from multiple threads.
trait Set: Sync {
    fn build(keys: &[u64]) -> Self;
    fn contains(&self, key: u64) -> bool;
    fn remove(&mut self, key: u64);
    fn sum(&self) -> u64;
}

struct PouchSorted(SortedSet<Vec<u64>>);
impl Set for PouchSorted {
    fn build(keys: &[u64]) -> Self {
        // Idiomatic bulk build: sort-once `try_from_iter`, not an insert-per-element
        // loop. The `construct` benches break the strategies out head to head.
        PouchSorted(SortedSet::try_from_iter(keys.iter().copied()).unwrap())
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn remove(&mut self, key: u64) {
        self.0.remove(&key);
    }
    fn sum(&self) -> u64 {
        self.0
            .as_slice()
            .iter()
            .copied()
            .fold(0u64, u64::wrapping_add)
    }
}

struct PouchUnsorted(UnsortedSet<Vec<u64>>);
impl Set for PouchUnsorted {
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
    fn remove(&mut self, key: u64) {
        self.0.remove(&key);
    }
    fn sum(&self) -> u64 {
        self.0
            .as_slice()
            .iter()
            .copied()
            .fold(0u64, u64::wrapping_add)
    }
}

struct BTree(BTreeSet<u64>);
impl Set for BTree {
    fn build(keys: &[u64]) -> Self {
        BTree(keys.iter().copied().collect())
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn remove(&mut self, key: u64) {
        self.0.remove(&key);
    }
    fn sum(&self) -> u64 {
        self.0.iter().copied().fold(0u64, u64::wrapping_add)
    }
}

struct Hash(HashSet<u64>);
impl Set for Hash {
    fn build(keys: &[u64]) -> Self {
        Hash(keys.iter().copied().collect())
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn remove(&mut self, key: u64) {
        self.0.remove(&key);
    }
    fn sum(&self) -> u64 {
        self.0.iter().copied().fold(0u64, u64::wrapping_add)
    }
}

// `std::HashSet` with FxHash instead of SipHash — a fair fast-hash baseline.
struct FxHash(FxHashSet<u64>);
impl Set for FxHash {
    fn build(keys: &[u64]) -> Self {
        FxHash(keys.iter().copied().collect())
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn remove(&mut self, key: u64) {
        self.0.remove(&key);
    }
    fn sum(&self) -> u64 {
        self.0.iter().copied().fold(0u64, u64::wrapping_add)
    }
}

// sorted-vec: the maintained sorted-`Vec` set — the direct `SortedSet`
// competitor. Membership goes through `binary_search` (O(log n)), not the
// slice's linear scan.
struct SortedVec(SortedVecSet<u64>);
impl Set for SortedVec {
    fn build(keys: &[u64]) -> Self {
        let mut s = SortedVecSet::default();
        for &k in keys {
            s.find_or_insert(k);
        }
        SortedVec(s)
    }
    fn contains(&self, key: u64) -> bool {
        self.0.binary_search(&key).is_ok()
    }
    fn remove(&mut self, key: u64) {
        self.0.remove_item(&key);
    }
    fn sum(&self) -> u64 {
        self.0.iter().copied().fold(0u64, u64::wrapping_add)
    }
}

// vecmap-rs: an unsorted `Vec`-backed set (linear scan), the std-side twin of
// `UnsortedSet`.
struct VecSetRs(VecSet<u64>);
impl Set for VecSetRs {
    fn build(keys: &[u64]) -> Self {
        let mut s = VecSet::new();
        for &k in keys {
            s.insert(k);
        }
        VecSetRs(s)
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains(&key)
    }
    fn remove(&mut self, key: u64) {
        self.0.remove(&key);
    }
    fn sum(&self) -> u64 {
        self.0.iter().copied().fold(0u64, u64::wrapping_add)
    }
}

/// Build from keys arriving in pseudo-random order (worst case for a sorted
/// `Vec`).
#[divan::bench(types = [PouchSorted, PouchUnsorted, SortedVec, VecSetRs, BTree, Hash, FxHash], args = SIZES)]
fn build_random<S: Set>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.random.as_slice();
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| S::build(black_box(src)));
}

/// Build from already-ascending keys (favorable case for a sorted `Vec`).
#[divan::bench(types = [PouchSorted, PouchUnsorted, SortedVec, VecSetRs, BTree, Hash, FxHash], args = SIZES)]
fn build_sorted<S: Set>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.sorted.as_slice();
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| S::build(black_box(src)));
}

/// `n` membership tests, all present.
#[divan::bench(types = [PouchSorted, PouchUnsorted, SortedVec, VecSetRs, BTree, Hash, FxHash], args = SIZES)]
fn contains_hit<S: Set>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let s = S::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.random {
            hits += s.contains(black_box(key)) as u64;
        }
        hits
    });
}

/// `n` membership tests, none present.
#[divan::bench(types = [PouchSorted, PouchUnsorted, SortedVec, VecSetRs, BTree, Hash, FxHash], args = SIZES)]
fn contains_miss<S: Set>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let s = S::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.misses {
            hits += s.contains(black_box(key)) as u64;
        }
        hits
    });
}

/// Remove every element. Each sample rebuilds a fresh set (unmeasured) via
/// `with_inputs`, then the timed closure drains it.
#[divan::bench(types = [PouchSorted, PouchUnsorted, SortedVec, VecSetRs, BTree, Hash, FxHash], args = SIZES)]
fn remove<S: Set>(bencher: Bencher, n: usize) {
    let k = keys(n);
    bencher
        .counter(ItemsCount::new(n))
        .with_inputs(|| S::build(&k.random))
        .bench_values(|mut s| {
            for &key in &k.random {
                s.remove(black_box(key));
            }
            s
        });
}

/// Sum every element (full iteration).
#[divan::bench(types = [PouchSorted, PouchUnsorted, SortedVec, VecSetRs, BTree, Hash, FxHash], args = SIZES)]
fn iter<S: Set>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let s = S::build(&k.random);
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| black_box(s.sum()));
}

// ---------------------------------------------------------------------------
// Construction strategies for `SortedSet<Vec<_>>`: the payoff of the bulk
// constructors over a repeated-`insert` loop, head to head at each size.
//   * `insert_loop`      — O(n²): binary-search + tail shift per element.
//   * `try_from_iter`    — O(n log n): append all, then sort + dedup once.
//   * `from_sorted_iter` — O(n): ascending input, append-only, no sort or search.
// ---------------------------------------------------------------------------
mod construct {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn insert_loop(bencher: Bencher, n: usize) {
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
    fn try_from_iter(bencher: Bencher, n: usize) {
        let k = keys(n);
        let src = k.random.as_slice();
        bencher.counter(ItemsCount::new(n)).bench(|| {
            SortedSet::<Vec<u64>>::try_from_iter(black_box(src).iter().copied()).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn from_sorted_iter(bencher: Bencher, n: usize) {
        let k = keys(n);
        let src = k.sorted.as_slice();
        bencher
            .counter(ItemsCount::new(n))
            .bench(|| SortedSet::<Vec<u64>>::from_sorted_iter(black_box(src).iter().copied()));
    }
}

// ---------------------------------------------------------------------------
// Fixed-capacity, no_std competitors, instantiated at a capacity matching the
// element count via `consts = FIXED_SIZES` (see map.rs for the rationale). All
// store inline: pouch's `UnsortedSet`/`SortedSet` over a `heapless::Vec`
// against `micromap` (the "fastest set under 20 keys"). `heapless` ships no
// linear set, so it appears only on the map side.
// ---------------------------------------------------------------------------

const FIXED_SIZES: [usize; 4] = [4, 16, 64, 256];

macro_rules! fixed_cap_set {
    ($modname:ident, $ctor:expr, $insert:ident, $contains:ident) => {
        mod $modname {
            use super::*;

            #[divan::bench(consts = FIXED_SIZES)]
            fn build<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut s = $ctor;
                    for &x in &k.random {
                        let _ = s.$insert(x);
                    }
                    s
                });
            }

            #[divan::bench(consts = FIXED_SIZES)]
            fn contains_hit<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                let mut s = $ctor;
                for &x in &k.random {
                    let _ = s.$insert(x);
                }
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut hits = 0u64;
                    for &x in &k.random {
                        hits += s.$contains(black_box(&x)) as u64;
                    }
                    hits
                });
            }

            #[divan::bench(consts = FIXED_SIZES)]
            fn contains_miss<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                let mut s = $ctor;
                for &x in &k.random {
                    let _ = s.$insert(x);
                }
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut hits = 0u64;
                    for &x in &k.misses {
                        hits += s.$contains(black_box(&x)) as u64;
                    }
                    hits
                });
            }
        }
    };
}

fixed_cap_set!(
    fixedcap_pouch_unsorted,
    UnsortedSet::<heapless::Vec<u64, N>>::new(),
    try_insert,
    contains
);
fixed_cap_set!(
    fixedcap_pouch_sorted,
    SortedSet::<heapless::Vec<u64, N>>::new(),
    try_insert,
    contains
);
fixed_cap_set!(
    fixedcap_micromap,
    micromap::Set::<u64, N>::new(),
    insert,
    contains
);
