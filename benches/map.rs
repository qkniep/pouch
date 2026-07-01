//! Map benchmarks — pouch `SortedMap` / `UnsortedMap` / `UnsortedColumnMap` vs
//! `litemap::LiteMap`, plus `std::collections::{BTreeMap, HashMap}` for
//! reference.
//!
//! `SortedMap` is the direct shape-match for `litemap` (a flat, key-sorted
//! `Vec`). The benched surface is **build** and **get** — the operations common
//! to every contender. `UnsortedColumnMap` is the struct-of-arrays unsorted map; this
//! file pits it at `V = u64` against the whole field, while the value-size
//! sweep that shows off its denser scan (the `sizeof(V)/sizeof(K)` axis) lives
//! in `benches/soa.rs`. The `HashMap` baseline uses the std default (SipHash)
//! hasher; a faster hasher would narrow its gap at larger `n`.
//!
//! Run: `cargo bench --bench map` (filter with `-- build`, `-- Lite`, …).

use std::collections::{BTreeMap, HashMap};

use divan::counter::ItemsCount;
use divan::{black_box, Bencher};
use litemap::LiteMap;
use pouch::{SortedMap, UnsortedColumnMap, UnsortedMap};
use rustc_hash::FxHashMap;
use vecmap::VecMap;

mod common;
use common::keys;

fn main() {
    divan::main();
}

/// Element counts, spanning the small-collection sweet spot up into the regime
/// where O(n) shifting starts to bite.
const SIZES: [usize; 5] = [4, 16, 64, 256, 1024];

/// The map surface common to every contender. Values mirror their keys.
/// `Sync` because divan may drive a benched closure from multiple threads.
trait Map: Sync {
    fn build(keys: &[u64]) -> Self;
    fn get(&self, key: u64) -> bool;
}

struct PouchSorted(SortedMap<Vec<(u64, u64)>>);
impl Map for PouchSorted {
    fn build(keys: &[u64]) -> Self {
        // Idiomatic bulk build: sort-once `try_from_iter`, not an insert-per-entry
        // loop. The `construct` benches break the strategies out head to head.
        PouchSorted(SortedMap::try_from_iter(keys.iter().map(|&k| (k, k))).unwrap())
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
}

struct PouchUnsorted(UnsortedMap<Vec<(u64, u64)>>);
impl Map for PouchUnsorted {
    fn build(keys: &[u64]) -> Self {
        let mut m = UnsortedMap::new();
        for &k in keys {
            let _ = m.try_insert(k, k);
        }
        PouchUnsorted(m)
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
}

// The struct-of-arrays unsorted map: keys and values in separate `Vec`s. Same
// O(n) scan as `PouchUnsorted` but over a dense `[u64]` key column.
struct PouchColumn(UnsortedColumnMap<Vec<u64>, Vec<u64>>);
impl Map for PouchColumn {
    fn build(keys: &[u64]) -> Self {
        let mut m = UnsortedColumnMap::new();
        for &k in keys {
            let _ = m.try_insert(k, k);
        }
        PouchColumn(m)
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
}

struct Lite(LiteMap<u64, u64>);
impl Map for Lite {
    fn build(keys: &[u64]) -> Self {
        let mut m = LiteMap::new();
        for &k in keys {
            m.insert(k, k);
        }
        Lite(m)
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
}

struct BTree(BTreeMap<u64, u64>);
impl Map for BTree {
    fn build(keys: &[u64]) -> Self {
        BTree(keys.iter().map(|&k| (k, k)).collect())
    }
    fn get(&self, key: u64) -> bool {
        self.0.contains_key(&key)
    }
}

struct Hash(HashMap<u64, u64>);
impl Map for Hash {
    fn build(keys: &[u64]) -> Self {
        Hash(keys.iter().map(|&k| (k, k)).collect())
    }
    fn get(&self, key: u64) -> bool {
        self.0.contains_key(&key)
    }
}

// `std::HashMap` with FxHash instead of SipHash — a fair fast-hash baseline.
struct FxHash(FxHashMap<u64, u64>);
impl Map for FxHash {
    fn build(keys: &[u64]) -> Self {
        FxHash(keys.iter().map(|&k| (k, k)).collect())
    }
    fn get(&self, key: u64) -> bool {
        self.0.contains_key(&key)
    }
}

// vecmap-rs: an unsorted `Vec`-backed map (linear scan), the std-side twin of
// `UnsortedMap`. Built with per-element `insert` (its O(n) dedup), like the
// others.
struct VecMapRs(VecMap<u64, u64>);
impl Map for VecMapRs {
    fn build(keys: &[u64]) -> Self {
        let mut m = VecMap::new();
        for &k in keys {
            m.insert(k, k);
        }
        VecMapRs(m)
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
}

/// Build from keys arriving in pseudo-random order — the common case, and the
/// worst case for a sorted `Vec` (every insert shifts a tail).
#[divan::bench(types = [PouchSorted, PouchUnsorted, PouchColumn, Lite, VecMapRs, BTree, Hash, FxHash], args = SIZES)]
fn build_random<M: Map>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.random.as_slice();
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| M::build(black_box(src)));
}

/// Build from already-ascending keys — the favorable case: a sorted `Vec` then
/// only ever appends at the tail (no shift), so binary-search-insert is O(n log
/// n).
#[divan::bench(types = [PouchSorted, PouchUnsorted, PouchColumn, Lite, VecMapRs, BTree, Hash, FxHash], args = SIZES)]
fn build_sorted<M: Map>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let src = k.sorted.as_slice();
    bencher
        .counter(ItemsCount::new(n))
        .bench(|| M::build(black_box(src)));
}

/// `n` successful lookups (every key present).
#[divan::bench(types = [PouchSorted, PouchUnsorted, PouchColumn, Lite, VecMapRs, BTree, Hash, FxHash], args = SIZES)]
fn get_hit<M: Map>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let m = M::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.random {
            hits += m.get(black_box(key)) as u64;
        }
        hits
    });
}

/// `n` failed lookups (no key present).
#[divan::bench(types = [PouchSorted, PouchUnsorted, PouchColumn, Lite, VecMapRs, BTree, Hash, FxHash], args = SIZES)]
fn get_miss<M: Map>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let m = M::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.misses {
            hits += m.get(black_box(key)) as u64;
        }
        hits
    });
}

// ---------------------------------------------------------------------------
// Construction strategies for `SortedMap<Vec<_>>`: the payoff of the bulk
// constructors over a repeated-`try_insert` loop, head to head at each size.
//   * `insert_loop`      — O(n²): binary-search + tail shift per entry.
//   * `try_from_iter`    — O(n log n): append all, sort by key, reject dup keys.
//   * `from_sorted_iter` — O(n): ascending input, append-only, no sort or search.
// All key sets here are distinct, so the strict builders never raise
// DuplicateKey.
// ---------------------------------------------------------------------------
mod construct {
    use super::*;

    #[divan::bench(args = SIZES)]
    fn insert_loop(bencher: Bencher, n: usize) {
        let k = keys(n);
        let src = k.random.as_slice();
        bencher.counter(ItemsCount::new(n)).bench(|| {
            let mut m = SortedMap::<Vec<(u64, u64)>>::new();
            for &x in black_box(src) {
                let _ = m.try_insert(x, x);
            }
            m
        });
    }

    #[divan::bench(args = SIZES)]
    fn try_from_iter(bencher: Bencher, n: usize) {
        let k = keys(n);
        let src = k.random.as_slice();
        bencher.counter(ItemsCount::new(n)).bench(|| {
            SortedMap::<Vec<(u64, u64)>>::try_from_iter(black_box(src).iter().map(|&x| (x, x)))
                .unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn from_sorted_iter(bencher: Bencher, n: usize) {
        let k = keys(n);
        let src = k.sorted.as_slice();
        bencher.counter(ItemsCount::new(n)).bench(|| {
            SortedMap::<Vec<(u64, u64)>>::try_from_sorted_iter(
                black_box(src).iter().map(|&x| (x, x)),
            )
            .unwrap()
        });
    }
}

// ---------------------------------------------------------------------------
// Fixed-capacity, no_std competitors. These are const-generic over capacity, so
// they can't join the `types = [...]` matrix above; instead `n = FIXED_SIZES`
// instantiates each at a capacity exactly matching the element count (no
// oversized inline storage to skew the result). All store inline —
// `UnsortedMap` over a `heapless::Vec` is pouch's own fixed-cap path, benched
// against `heapless::LinearMap` (unsorted scan) and `micromap` (the "fastest
// map under 20 keys").
// ---------------------------------------------------------------------------

/// Capped at 256: beyond ~64 these linear-scan maps are out of their regime,
/// and a large inline `N` would make the build's return-move dominate the
/// measurement.
const FIXED_SIZES: [usize; 4] = [4, 16, 64, 256];

macro_rules! fixed_cap_map {
    ($modname:ident, $ctor:expr, $insert:ident) => {
        mod $modname {
            use super::*;

            #[divan::bench(consts = FIXED_SIZES)]
            fn build<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut m = $ctor;
                    for &x in &k.random {
                        let _ = m.$insert(x, x);
                    }
                    m
                });
            }

            #[divan::bench(consts = FIXED_SIZES)]
            fn get_hit<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                let mut m = $ctor;
                for &x in &k.random {
                    let _ = m.$insert(x, x);
                }
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut hits = 0u64;
                    for &x in &k.random {
                        hits += m.get(black_box(&x)).is_some() as u64;
                    }
                    hits
                });
            }

            #[divan::bench(consts = FIXED_SIZES)]
            fn get_miss<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                let mut m = $ctor;
                for &x in &k.random {
                    let _ = m.$insert(x, x);
                }
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut hits = 0u64;
                    for &x in &k.misses {
                        hits += m.get(black_box(&x)).is_some() as u64;
                    }
                    hits
                });
            }
        }
    };
}

fixed_cap_map!(
    fixedcap_pouch_heapless,
    UnsortedMap::<heapless::Vec<(u64, u64), N>>::new(),
    try_insert
);
fixed_cap_map!(
    fixedcap_heapless_linear,
    heapless::LinearMap::<u64, u64, N>::new(),
    insert
);
fixed_cap_map!(
    fixedcap_micromap,
    micromap::Map::<u64, u64, N>::new(),
    insert
);
