//! Value-size sweep — `UnsortedMap` (array-of-structs) vs `UnsortedColumnMap`
//! (struct-of-arrays), the one axis `benches/map.rs` doesn't cover.
//!
//! Both are pouch's *real* unsorted maps; the only difference is layout.
//! `UnsortedMap<Vec<(K, V)>>` interleaves keys and values, so its O(n) lookup
//! scan reads `.0` at a `sizeof((K, V))` stride and drags every `V` through
//! cache. `UnsortedColumnMap<Vec<K>, Vec<V>>` keeps keys in their own dense `Vec`, so
//! the scan is contiguous, branch-predictable, and auto-vectorizable, and never
//! touches the values. Theory says the gap widens with `sizeof(V)/sizeof(K)`
//! and is largest for misses (which always scan the whole array); this sweep
//! measures it.
//!
//! `K = u64` throughout; `V` sweeps 8 → 32 → 64 bytes (ratio 1 → 4 → 8). The
//! head-to-head at `V = u64` against the rest of the field (litemap, vecmap-rs,
//! the hashmaps) lives in `benches/map.rs`; this file isolates the `V`-size
//! effect.
//!
//! The `scan_*` benches use `get` (index lookup); `membership_miss` uses
//! `contains_key` (boolean query). Both fold the dense key scan to branchless
//! compares — `get` via the fixed-trip `chunked_position` reduction,
//! `contains_key` via the stdlib boolean `contains` — so the SoA win lands on
//! the index lookup too, not just membership.
//!
//! The `sorted_*` benches add the **sorted** pair on the same value-size axis —
//! `SortedMap` (AoS, `binary_search` over `(K, V)`) vs `SortedColumnMap` (SoA,
//! `binary_search` over a dense `[K]`). The lookup is `O(log n)`, not a scan,
//! so the win has a different shape: it tracks `sizeof(V)/sizeof(K)` (large
//! values let the dense key search skip value cache lines), and for word-sized
//! values a *hit* can even lose at small `n` — the value sits in a separate
//! column, a second cache line, where the AoS map has it beside the key. Misses
//! (no value load) favor SoA more broadly.
//!
//! Run: `cargo bench --bench soa` (filter with `-- scan_miss`, `-- membership`,
//! `-- sorted`, …).

use divan::counter::ItemsCount;
use divan::{black_box, Bencher};
use pouch::{SortedColumnMap, SortedMap, UnsortedColumnMap, UnsortedMap};

mod common;
use common::keys;

fn main() {
    divan::main();
}

/// Element counts for the linear scan. Lookup is O(n), so a miss sweep over `n`
/// keys is O(n²) — kept to the unsorted map's small-`n` regime and a bit past.
const SCAN_SIZES: [usize; 5] = [16, 64, 256, 1024, 4096];

/// Value payloads of three sizes. With `K = u64` (8 bytes) fixed, these set the
/// `sizeof(V)/sizeof(K)` ratio: 1, 4, 8.
type V8 = u64; // ratio 1:1
type V32 = [u64; 4]; // ratio 4:1
type V64 = [u64; 8]; // ratio 8:1 — one value fills a cache line

/// A value derivable from its key, so `build` can synthesize payloads. `lo`
/// reads one word back *out* of the value, so a hit bench can force the value
/// load (the cost that separates AoS from SoA on word-sized hits).
trait Payload: Copy + Sync {
    fn from_key(k: u64) -> Self;
    fn lo(&self) -> u64;
}
impl Payload for V8 {
    fn from_key(k: u64) -> Self {
        k
    }
    fn lo(&self) -> u64 {
        *self
    }
}
impl Payload for V32 {
    fn from_key(k: u64) -> Self {
        [k; 4]
    }
    fn lo(&self) -> u64 {
        self[0]
    }
}
impl Payload for V64 {
    fn from_key(k: u64) -> Self {
        [k; 8]
    }
    fn lo(&self) -> u64 {
        self[0]
    }
}

/// The lookup surface under test. `Sync` because divan may drive the closure
/// from multiple threads. `get` is the index-returning lookup
/// (`get(k).is_some()`); `contains` is the boolean-membership query
/// (`contains_key`) — the same scan over the same data, differing only in
/// whether an index must be produced.
trait Map: Sync {
    fn build(keys: &[u64]) -> Self;
    fn get(&self, key: u64) -> bool;
    fn contains(&self, key: u64) -> bool;
}

/// Array-of-structs: the interleaved-`(K, V)` unsorted map.
struct Aos<V>(UnsortedMap<Vec<(u64, V)>>);
impl<V: Payload> Map for Aos<V> {
    fn build(keys: &[u64]) -> Self {
        let mut m = UnsortedMap::new();
        for &k in keys {
            let _ = m.try_insert(k, V::from_key(k));
        }
        Aos(m)
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains_key(&key)
    }
}

/// Struct-of-arrays: the column map (dense `[u64]` key scan, values untouched).
struct Soa<V>(UnsortedColumnMap<Vec<u64>, Vec<V>>);
impl<V: Payload> Map for Soa<V> {
    fn build(keys: &[u64]) -> Self {
        let mut m = UnsortedColumnMap::new();
        for &k in keys {
            let _ = m.try_insert(k, V::from_key(k));
        }
        Soa(m)
    }
    fn get(&self, key: u64) -> bool {
        self.0.get(&key).is_some()
    }
    fn contains(&self, key: u64) -> bool {
        self.0.contains_key(&key)
    }
}

/// The sorted-lookup surface. Unlike [`Map`]'s presence check, `lookup` reads a
/// word *out of the value* on a hit (`0` on a miss) — so the hit path pays to
/// fetch the value from wherever its layout keeps it. That fetch is the whole
/// AoS/SoA hit tradeoff for word-sized values: co-located with the key in AoS,
/// a *separate* cache line (the value column) in SoA. A miss reads no value, so
/// it isolates the search.
trait SortedLookup: Sync {
    fn build(keys: &[u64]) -> Self;
    fn lookup(&self, key: u64) -> u64;
}

/// Sorted array-of-structs: `SortedMap`, `binary_search` over interleaved `(K,
/// V)`.
struct SortedAos<V>(SortedMap<Vec<(u64, V)>>);
impl<V: Payload> SortedLookup for SortedAos<V> {
    fn build(keys: &[u64]) -> Self {
        let mut m = SortedMap::new();
        for &k in keys {
            let _ = m.try_insert(k, V::from_key(k));
        }
        SortedAos(m)
    }
    fn lookup(&self, key: u64) -> u64 {
        self.0.get(&key).map_or(0, Payload::lo)
    }
}

/// Sorted struct-of-arrays: `SortedColumnMap`, `binary_search` over a dense
/// `[u64]` key column — the search never touches values; only a *hit* then
/// loads one (from the separate value column).
struct SortedSoa<V>(SortedColumnMap<Vec<u64>, Vec<V>>);
impl<V: Payload> SortedLookup for SortedSoa<V> {
    fn build(keys: &[u64]) -> Self {
        let mut m = SortedColumnMap::new();
        for &k in keys {
            let _ = m.try_insert(k, V::from_key(k));
        }
        SortedSoa(m)
    }
    fn lookup(&self, key: u64) -> u64 {
        self.0.get(&key).map_or(0, Payload::lo)
    }
}

/// Linear-scan hits — found on average halfway through the scan.
#[divan::bench(
    types = [Aos<V8>, Soa<V8>, Aos<V32>, Soa<V32>, Aos<V64>, Soa<V64>],
    args = SCAN_SIZES,
)]
fn scan_hit<M: Map>(bencher: Bencher, n: usize) {
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

/// Linear-scan misses — every lookup scans the whole array, where the dense,
/// vectorizable SoA scan should dominate and the `V`-size effect is sharpest.
#[divan::bench(
    types = [Aos<V8>, Soa<V8>, Aos<V32>, Soa<V32>, Aos<V64>, Soa<V64>],
    args = SCAN_SIZES,
)]
fn scan_miss<M: Map>(bencher: Bencher, n: usize) {
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

/// Misses via `contains_key` — the boolean-membership query, the
/// originally-vectorized SoA path. Now that `Soa::get` folds its index scan too
/// (`chunked_position`), the contrast with `scan_miss` no longer isolates "the
/// cost of returning the index": both `get` and `contains_key` reduce over the
/// dense key column, the boolean query being simply the cheaper of the two.
/// `Aos` is strided either way, so both SoA queries pull ahead of it.
#[divan::bench(
    types = [Aos<V8>, Soa<V8>, Aos<V32>, Soa<V32>, Aos<V64>, Soa<V64>],
    args = SCAN_SIZES,
)]
fn membership_miss<M: Map>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let m = M::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut hits = 0u64;
        for &key in &k.misses {
            hits += m.contains(black_box(key)) as u64;
        }
        hits
    });
}

/// Sorted-map hits that **read the value** (`O(log n)` search + one value
/// load). SoA pulls ahead as `sizeof(V)/sizeof(K)` grows (the dense key search
/// skips value cache lines); for word-sized values it can *lose* at small `n`,
/// because the hit's value load is a second cache line in SoA but free-riding
/// the key's line in AoS.
#[divan::bench(
    types = [SortedAos<V8>, SortedSoa<V8>, SortedAos<V32>, SortedSoa<V32>, SortedAos<V64>, SortedSoa<V64>],
    args = SCAN_SIZES,
)]
fn sorted_get_hit<M: SortedLookup>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let m = M::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut acc = 0u64;
        for &key in &k.random {
            acc = acc.wrapping_add(m.lookup(black_box(key)));
        }
        acc
    });
}

/// Sorted-map misses — `O(log n)` search, no value load (the key is absent), so
/// this isolates the search: the dense-key SoA wins a bit more broadly, and
/// earlier in `n`, than on value-loading hits.
#[divan::bench(
    types = [SortedAos<V8>, SortedSoa<V8>, SortedAos<V32>, SortedSoa<V32>, SortedAos<V64>, SortedSoa<V64>],
    args = SCAN_SIZES,
)]
fn sorted_get_miss<M: SortedLookup>(bencher: Bencher, n: usize) {
    let k = keys(n);
    let m = M::build(&k.random);
    bencher.counter(ItemsCount::new(n)).bench(|| {
        let mut acc = 0u64;
        for &key in &k.misses {
            acc = acc.wrapping_add(m.lookup(black_box(key)));
        }
        acc
    });
}

// `UnsortedColumnMap::get`/`remove` need the matching *index*, not just a yes/no, so
// they can't ride the already-vectorized boolean `contains`. They scan the
// dense key column with `chunked_position` (src/column_map.rs): a fixed-trip
// OR-reduction LLVM folds to branchless compares (one chunk-level branch per
// `LANES` keys), where a plain `iter().position` stays scalar and mispredicts a
// branch per element. This A/B quantifies and guards that win — `scalar_*`
// forces the old early-exit `position`, `chunked_*` is the shipped `get` — on
// hits *and* misses (a miss scans the whole column, so the gap is largest
// there). Both sum the located value, so the index can't be elided back down to
// a `contains`.
mod locate {
    use pouch::UnsortedColumnMap;

    use super::*;

    fn build(src: &[u64]) -> UnsortedColumnMap<Vec<u64>, Vec<u64>> {
        let mut m = UnsortedColumnMap::new();
        for &k in src {
            let _ = m.try_insert(k, k);
        }
        m
    }

    /// The pre-change baseline: first-match early-exit scalar scan (does not
    /// vectorize), kept to quantify what the shipped `chunked_position` buys.
    fn scalar_get<'a>(keys: &[u64], values: &'a [u64], key: u64) -> Option<&'a u64> {
        keys.iter().position(|&k| k == key).map(|i| &values[i])
    }

    #[divan::bench(args = SCAN_SIZES)]
    fn scalar_hit(bencher: Bencher, n: usize) {
        let k = keys(n);
        let m = build(&k.random);
        let (ks, vs) = (m.keys(), m.values());
        bencher.counter(ItemsCount::new(n)).bench(|| {
            let mut s = 0u64;
            for &key in &k.random {
                s = s.wrapping_add(scalar_get(ks, vs, black_box(key)).copied().unwrap_or(0));
            }
            s
        });
    }

    #[divan::bench(args = SCAN_SIZES)]
    fn scalar_miss(bencher: Bencher, n: usize) {
        let k = keys(n);
        let m = build(&k.random);
        let (ks, vs) = (m.keys(), m.values());
        bencher.counter(ItemsCount::new(n)).bench(|| {
            let mut s = 0u64;
            for &key in &k.misses {
                s = s.wrapping_add(scalar_get(ks, vs, black_box(key)).copied().unwrap_or(0));
            }
            s
        });
    }

    /// The shipped path: `UnsortedColumnMap::get` routes through `chunked_position`.
    #[divan::bench(args = SCAN_SIZES)]
    fn chunked_hit(bencher: Bencher, n: usize) {
        let k = keys(n);
        let m = build(&k.random);
        bencher.counter(ItemsCount::new(n)).bench(|| {
            let mut s = 0u64;
            for &key in &k.random {
                s = s.wrapping_add(m.get(&black_box(key)).copied().unwrap_or(0));
            }
            s
        });
    }

    #[divan::bench(args = SCAN_SIZES)]
    fn chunked_miss(bencher: Bencher, n: usize) {
        let k = keys(n);
        let m = build(&k.random);
        bencher.counter(ItemsCount::new(n)).bench(|| {
            let mut s = 0u64;
            for &key in &k.misses {
                s = s.wrapping_add(m.get(&black_box(key)).copied().unwrap_or(0));
            }
            s
        });
    }
}
