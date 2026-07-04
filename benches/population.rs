//! Population benchmark — the *inline building block* thesis.
//!
//! The regime that separates pouch from litemap / sorted-vec / std isn't one
//! standalone collection; it's a `Vec` of **many small collections** (adjacency
//! lists, inverted-index postings, per-key buckets, quorum/vote sets). There,
//! heap-backed inners cost one allocation *per non-empty collection* and
//! scatter the data across cache-cold blocks; an inline backend keeps each
//! small inner in place, collapsing the population to ~one allocation with
//! everything contiguous.
//!
//! Sizes are heavy-tailed (power-law-ish): ~99% of inners hold 1–4 elements,
//! ~1% are "hubs" holding 64–1024 — so the inline backend must spill gracefully
//! for the tail, which is why `SmallVec` (inline → heap) is the backend under
//! test, not a fixed-capacity one.
//!
//! divan's [`AllocProfiler`] is the global allocator here, so each row also
//! reports **allocation count and bytes** — the headline metric, since the win
//! is allocation and memory, not per-op throughput.
//!
//! Run: `cargo bench --bench population`.

use std::collections::{BTreeSet, HashSet};

use divan::counter::ItemsCount;
use divan::{black_box, Bencher};
use pouch::SortedSet;
use smallvec::SmallVec;
use thincollections::thin_set::ThinSet;

mod common;
use common::splitmix64;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

/// Number of inner collections in the population.
const POP: usize = 10_000;

/// A heavy-tailed population: each inner's element list. ~99% tiny (1–4), ~1%
/// hubs (64–1024). Deterministic. `sorted` returns each inner's elements
/// ascending.
fn population(sorted: bool) -> Vec<Vec<u64>> {
    let mut state = 0u64;
    let mut pop = Vec::with_capacity(POP);
    for _ in 0..POP {
        let r = splitmix64(&mut state);
        let size = if r.is_multiple_of(100) {
            64 + (splitmix64(&mut state) % 961) as usize // tail: 64..=1024
        } else {
            1 + (r % 4) as usize // body: 1..=4
        };
        let mut elems: Vec<u64> = (0..size).map(|_| splitmix64(&mut state)).collect();
        if sorted {
            elems.sort_unstable();
        }
        pop.push(elems);
    }
    pop
}

/// A flat list of `(inner_index, key)` lookups: half present, half absent.
fn queries(pop: &[Vec<u64>]) -> Vec<(usize, u64)> {
    let mut state = 0x5DEE_CE66u64;
    let mut q = Vec::with_capacity(pop.len() * 2);
    for (i, elems) in pop.iter().enumerate() {
        if !elems.is_empty() {
            let hit = elems[(splitmix64(&mut state) as usize) % elems.len()];
            q.push((i, hit)); // present
        }
        q.push((i, splitmix64(&mut state) | 1)); // (almost surely) absent
    }
    q
}

/// One small inner collection. `build_from` inserts an element list (the order
/// is the caller's: random or pre-sorted); `contains` is membership. (`Sync` is
/// added only on the `lookup` bench, which holds the built population across
/// divan's threads — `thincollections::ThinSet` isn't `Sync`, so it sits out
/// that one.)
trait Inner {
    fn build_from(elems: &[u64]) -> Self;
    fn contains(&self, x: u64) -> bool;
}

struct StdHash(HashSet<u64>);
impl Inner for StdHash {
    fn build_from(elems: &[u64]) -> Self {
        let mut s = HashSet::new();
        for &x in elems {
            s.insert(x);
        }
        StdHash(s)
    }
    fn contains(&self, x: u64) -> bool {
        self.0.contains(&x)
    }
}

struct StdBTree(BTreeSet<u64>);
impl Inner for StdBTree {
    fn build_from(elems: &[u64]) -> Self {
        let mut s = BTreeSet::new();
        for &x in elems {
            s.insert(x);
        }
        StdBTree(s)
    }
    fn contains(&self, x: u64) -> bool {
        self.0.contains(&x)
    }
}

// thincollections: a thin hash set built for nesting. One word inline, but
// still a hash table — so it allocates *per non-empty inner*, like `HashSet`,
// just smaller.
struct Thin(ThinSet<u64>);
impl Inner for Thin {
    fn build_from(elems: &[u64]) -> Self {
        let mut s = ThinSet::new();
        for &x in elems {
            s.insert(x);
        }
        Thin(s)
    }
    fn contains(&self, x: u64) -> bool {
        self.0.contains(&x)
    }
}

// pouch over `Vec` — one heap allocation per non-empty inner (≈ a sorted-vec /
// litemap of sets). Isolates what the inline backend adds.
struct PouchHeap(SortedSet<Vec<u64>>);
impl Inner for PouchHeap {
    fn build_from(elems: &[u64]) -> Self {
        let mut s = SortedSet::new();
        for &x in elems {
            s.insert(x);
        }
        PouchHeap(s)
    }
    fn contains(&self, x: u64) -> bool {
        self.0.contains(&x)
    }
}

// pouch over `SmallVec` — the thesis: inline while small (no allocation),
// spills to the heap only for the rare hub. Two `N`s bracket the size knob:
// `N=4` is tuned to the 1–4 body (minimal inline waste); `N=16` is deliberately
// too large, to show `size_of` (and thus total memory) scaling with `N ·
// size_of::<T>()`.
struct PouchInline4(SortedSet<SmallVec<[u64; 4]>>);
impl Inner for PouchInline4 {
    fn build_from(elems: &[u64]) -> Self {
        let mut s = SortedSet::new();
        for &x in elems {
            s.insert(x);
        }
        PouchInline4(s)
    }
    fn contains(&self, x: u64) -> bool {
        self.0.contains(&x)
    }
}

struct PouchInline16(SortedSet<SmallVec<[u64; 16]>>);
impl Inner for PouchInline16 {
    fn build_from(elems: &[u64]) -> Self {
        let mut s = SortedSet::new();
        for &x in elems {
            s.insert(x);
        }
        PouchInline16(s)
    }
    fn contains(&self, x: u64) -> bool {
        self.0.contains(&x)
    }
}

fn build_pop<I: Inner>(pop: &[Vec<u64>]) -> Vec<I> {
    let mut v = Vec::with_capacity(pop.len());
    for elems in pop {
        v.push(I::build_from(elems));
    }
    v
}

/// Build the whole population from randomly-ordered inner elements. The alloc
/// count / bytes columns are the headline. (`thincollections::ThinSet` isn't
/// `Sync`, which divan's `types = [...]` registry requires, so it's benched in
/// `mod thin`.)
#[divan::bench(types = [StdHash, StdBTree, PouchHeap, PouchInline4, PouchInline16])]
fn build_random<I: Inner>(bencher: Bencher) {
    let pop = population(false);
    bencher
        .counter(ItemsCount::new(POP))
        .bench(|| build_pop::<I>(black_box(&pop)));
}

/// Build from pre-sorted inner elements — the build-once case. For the
/// sorted-Vec backends this is the cheap path (append, no shift); shows the
/// tail need not "fall over" without promote-to-tree machinery.
#[divan::bench(types = [StdHash, StdBTree, PouchHeap, PouchInline4, PouchInline16])]
fn build_sorted<I: Inner>(bencher: Bencher) {
    let pop = population(true);
    bencher
        .counter(ItemsCount::new(POP))
        .bench(|| build_pop::<I>(black_box(&pop)));
}

// `Thin` benched on its own (not `Sync`, so it can't join the generic
// registry). It's a hash set, so its lookup tracks `StdHash`; the point here is
// its build allocation profile — one table per non-empty inner, like `HashSet`.
mod thin {
    use super::*;

    #[divan::bench]
    fn build_random(bencher: Bencher) {
        let pop = population(false);
        bencher
            .counter(ItemsCount::new(POP))
            .bench(|| build_pop::<Thin>(black_box(&pop)));
    }

    #[divan::bench]
    fn build_sorted(bencher: Bencher) {
        let pop = population(true);
        bencher
            .counter(ItemsCount::new(POP))
            .bench(|| build_pop::<Thin>(black_box(&pop)));
    }
}

/// Membership across the whole population (half hits, half misses). Times the
/// cache behavior: contiguous inline vs a pointer-chase of scattered
/// allocations. (`Thin` is omitted — `ThinSet` isn't `Sync`; its lookup tracks
/// `HashSet`.)
#[divan::bench(types = [StdHash, StdBTree, PouchHeap, PouchInline4, PouchInline16])]
fn lookup<I: Inner + Sync>(bencher: Bencher) {
    let pop = population(false);
    let q = queries(&pop);
    let built = build_pop::<I>(&pop);
    bencher.counter(ItemsCount::new(q.len())).bench(|| {
        let mut hits = 0u64;
        for &(i, k) in &q {
            hits += built[i].contains(black_box(k)) as u64;
        }
        hits
    });
}
