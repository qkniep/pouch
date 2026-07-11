<p align="center">
  <img src="assets/banner.webp" alt="pouch" width="640">
</p>

# pouch

[![CI](https://github.com/qkniep/pouch/actions/workflows/rust.yml/badge.svg)](https://github.com/qkniep/pouch/actions/workflows/rust.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Allocation-avoiding flat sets and maps for Rust, built for the case most
collection crates ignore: **many small collections nested in a larger structure**
— a `Vec` of adjacency lists, inverted-index postings, per-key buckets, quorum /
vote / share sets. The default `Set` / `Map` keep their elements **inline** until
they outgrow `N`, so a population of thousands of small sets costs roughly *one*
heap allocation instead of one per set.

> On a `Vec` of 10 000 heavy-tailed small sets, the inline default builds in **105
> allocations** where `Vec<HashSet>` / `Vec<BTreeSet>` take 10 000–18 000 — **~95×
> fewer**, with the lowest memory and ~5× faster lookups. See [Benchmarks](#benchmarks).

Under the hood every collection is **backend-generic**: the same set/map logic runs
over a `Vec`, `SmallVec`, `TinyVec`, `ArrayVec`, `heapless::Vec`, or a **borrowed**
slice or buffer (`&[T]`, `ScratchVec`) — heap, inline, hybrid, or borrowed — and
stores compose: `Capped` adds a runtime bound to any store, `Spill` chains two
tiers. And since the core never touches an allocator, the same collections run
unchanged on `no_std` and embedded targets.

> [!NOTE]
> **Early days — the API is not yet stable.** The store traits are still
> settling; expect breaking changes between 0.x releases. The collection layer
> is filling in: bulk constructors, an `Entry` API, borrowed-key lookups, set
> algebra, `Hash`/`Ord` on the sorted flavors, and `serde` behind a feature.
> Comparators are next.

## Design

Three concerns that other small-collection crates usually fuse are kept
orthogonal, so you mix them freely:

- **storage** — *where* elements live (heap / inline / hybrid): the `Store` trait
  family, implemented once per backend.
- **bound** — the maximum logical element count, reported by
  `Store::max_capacity() -> Option<usize>`. A runtime bound is added with the
  `Capped<S>` wrapper rather than per backend.
- **ordering** — sorted (`SortedSet`/`SortedMap`, `O(log n)` lookup) vs unsorted
  (`UnsortedSet`/`UnsortedMap`, `O(n)` lookup, `O(1)` structural mutation, needs only
  `Eq`). This lives in the collection layer; the stores are ordering-agnostic. See
  [Complexity](#complexity).

The default `Set` / `Map` fix the combination this crate is tuned for — sorted,
`SmallVec`-backed (inline), unbounded — so the nested-population win is the path of
least resistance. Swap any axis (a `Vec` for one big collection, `heapless` for
`no_std`, unsorted when elements aren't `Ord`) when your case differs. A fourth,
invariant-free shape rounds out the lineup: `Bag`, a `Vec`-like sequence
(duplicates kept, no ordering, no element bounds) that gives *composed* stores —
`Bag<Capped<Vec<T>>>` is a capped vector — an ergonomic push/pop API.

**Struct-of-arrays layout (`soa` feature: `UnsortedColumnMap` / `SortedColumnMap`).**
A map can instead keep keys and values in *two* parallel stores, so a lookup scans
(`UnsortedColumnMap`) or binary-searches (`SortedColumnMap`) a dense key column
without dragging values through cache. Niche enough that the array-of-structs
`UnsortedMap` / `SortedMap` stay the default; reach for a column map when lookups
dominate, especially with big values. See [Benchmarks](#benchmarks).

## Complexity

The **ordering** axis sets the asymptotics; they are *backend-independent* — every
store is a contiguous array, so the backend changes only the constant factor (see
[Benchmarks](#benchmarks)), never the order. `n` is the element count.

| Operation                   | Sorted     | Unsorted   |
| --------------------------- | ---------- | ---------- |
| lookup — `contains` / `get` | `O(log n)` | `O(n)`     |
| insert — `try_insert`       | `O(n)` ¹   | `O(n)` ²   |
| remove — by value           | `O(n)` ³   | `O(n)` ⁴   |
| iterate — `as_slice`        | `O(n)`     | `O(n)`     |

1. `O(log n)` binary search for the slot, then `O(n)` shift to keep order.
2. `O(n)` duplicate scan, then `O(1)` append — the structural cost is `O(1)`; the
   scan is the membership check. (For a no-dedup sequence — `O(1)` push, duplicates
   kept — use `Bag`.)
3. `O(log n)` search, then `O(n)` shift.
4. `O(n)` find, then `O(1)` swap-remove (does not preserve order).

In short: **sorted** wins lookups; **unsorted** has `O(1)` structural mutation and
needs only `Eq`, so it wins when `n` is small or elements aren't `Ord`. Pick by the
operation mix, then pick a backend below.

## Backends

Storage and bound are orthogonal to ordering (above) — any backend pairs with
either flavor. Choose by where memory should live and whether the size is bounded;
the asymptotics don't change.

| Backend               | Storage              | Capacity       | `no_std` | Infallible `insert` | Feature *(default ✅)* | Reach for it when…           |
| --------------------- | -------------------- | -------------- | :------: | :-----------------: | ---------------------- | ---------------------------- |
| `Vec<T>`              | heap                 | unbounded      |    —     |         ✅          | `alloc`                | one big collection; `N` unpredictable |
| `SmallVec<[T; N]>`    | inline `N` → heap    | unbounded      |    —     |         ✅          | `smallvec` ✅          | **the default (`Set`/`Map`)** — many small / nested |
| `TinyVec<[T; N]>`     | inline `N` → heap    | unbounded      |    —     |         ✅          | `tinyvec`              | same, 100% safe (`Elem: Default`) |
| `ArrayVec<T, N>`      | inline `N`           | `N` (fixed)    |    ✅    |   — (`try_insert`)  | `arrayvec`             | hard cap, no allocator |
| `heapless::Vec<T, N>` | inline `N`           | `N` (fixed)    |    ✅    |   — (`try_insert`)  | `heapless`             | hard cap, no allocator (embedded) |
| `&[T]` / `&[T; N]`    | borrowed, read-only  | = `len` (full) |    ✅    |    — (read-only)    | — *(always on)*        | `static` sorted tables in flash: `SliceSet` / `SliceMap` |
| `ScratchVec<T>`       | borrowed `&mut [T]`  | buffer `len`   |    ✅    |   — (`try_insert`)  | — *(always on)*        | alloc-free scratch space; `Spill`'s overflow tier |
| `Spill<A, B>`         | tier `A` → tier `B`  | = `B`'s        | = tiers  |       = `B`'s       | — *(always on)*        | custom two-tier spill, e.g. inline → borrowed buffer |
| `Capped<S>`           | wraps any store `S`  | runtime cap    |  = `S`   |   — (`try_insert`)  | — *(always on)*        | enforce a limit / backpressure |

`try_insert` is always available and returns the rejected element on a bounded
store via `CapacityError<T>`. When the backing store is genuinely unbounded
(`Vec`, `SmallVec`, `TinyVec`), an infallible `insert` is also available.

**When you care about nanoseconds:** hybrid stores (`SmallVec`, `TinyVec`,
`Spill`) pay a well-predicted "inline or heap?" branch on every access;
`SortedSet<ArrayVec<T, N>>` skips it for a read-hot collection with a hard small
bound, and `SliceSet`/`SliceMap` over a `static` sorted table is unbeatable for
build-once-query-forever lookups. If you haven't measured, the default
`Set`/`Map` are right.

## Example

```rust
use pouch::Set;

// `Set`/`Map` keep small contents inline (no allocation), spilling past `N`.
let mut s: Set<u64> = Set::default();
s.insert(5);
s.insert(1);
s.insert(5); // duplicate, ignored
assert_eq!(s.as_slice(), &[1, 5]); // sorted, inline
assert!(s.contains(&1));

// The point: a population of small sets is ~one allocation, not one per set.
let mut adjacency: Vec<Set<u32>> = (0..1000).map(|_| Set::default()).collect();
adjacency[0].insert(7);
adjacency[0].insert(3);
```

Insert-or-update a map value in a single lookup with the `Entry` API, instead of a
separate `get` then `insert`:

```rust
use pouch::Map;

let mut counts: Map<&str, u32> = Map::default();
for word in ["a", "b", "a", "a"] {
    *counts.entry(word).or_insert(0) += 1; // one search per word, not two
}
assert_eq!(counts.get("a"), Some(&3));
```

Fixed-capacity backends (and any store wrapped in `Capped`) make insertion
fallible instead of allocating without bound:

```rust
use arrayvec::ArrayVec;
use pouch::SortedSet;

let mut s: SortedSet<ArrayVec<u64, 3>> = SortedSet::new();
assert_eq!(s.try_insert(5), Ok(true));
assert_eq!(s.try_insert(1), Ok(true));
assert_eq!(s.try_insert(9), Ok(true));
// At capacity: the new element is handed back instead of inserted.
assert!(s.try_insert(2).is_err());
```

Also available: bulk constructors (`try_from_iter` sorts + dedups once,
`from_sorted_iter` skips the sort), merge-based set algebra (`union` /
`intersection` / `is_subset`, `O(n + m)` and cross-backend), and the `Unsorted`
variants (`O(1)` append + swap-remove for elements cheap to scan or not `Ord`).

## Benchmarks

Apple M4 Max, rustc 1.96, `cargo bench` — illustrative, re-run on your own hardware.
**Bold** = best. Full matrix (population, sets, maps, fixed-cap, backend sweep) in
[BENCHMARKS.md](BENCHMARKS.md).

**The headline — a `Vec` of 10 000 small sets** (heavy-tailed: ~99% hold 1–4
elements, ~1% are hubs of 64–1024). Building the whole population, `peak allocations`
and memory from divan's allocation profiler:

| inner set                  | allocations | memory      | lookup |
| -------------------------- | ----------: | ----------: | -----: |
| `pouch::Set` (inline, N=4) | **105**     | **1.10 MB** | 28 µs  |
| pouch over `Vec`           | 10 001      | 1.18 MB     | **24 µs** |
| `HashSet`                  | 10 001      | 1.93 MB     | 139 µs |
| `BTreeSet`                 | 17 980      | 2.20 MB     | 70 µs  |
| `thincollections::ThinSet` | 10 001      | 3.02 MB     | —      |

~95× fewer allocations, the lowest memory, and ~5× faster lookups than `HashSet`.
Two honest caveats: `N` is a memory knob — `N=4` (tuned to the 1–4 body) is shown;
`N=16` keeps the 105 allocations but uses 2.06 MB, and the default `Set` is `N=8`,
between. And the lookup win is the *sorted-small-set* property (both pouch backends
have it), not inline specifically — inline's unique, decisive win is allocation count.

Beyond the headline, [BENCHMARKS.md](BENCHMARKS.md) covers single-collection maps,
set iteration, the SoA column maps, bulk-construction strategies, and a backend
sweep. The short version:

- **Nested populations are the win** (table above): inline storage collapses a
  population of small sets to ~one allocation, which `Vec<HashSet>` /
  thincollections can't — they allocate per inner set regardless.
- **Parity with litemap** on the shared sorted-`Vec` design — the backend-generic
  layer costs nothing; flat binary search beats `BTreeMap` and SipHash `HashMap` on
  lookups, while a fast hasher (`FxHashMap`) overtakes past `n ≈ 16`.
- **Bulk construction** (`try_from_iter` / `from_sorted_iter`) beats an insert-loop
  by ~8× / ~58× at `n = 1024`, and **iteration** over contiguous memory runs
  ~10–50× faster than the tree/hash maps.

## Crate features

Defaults are deliberately lean — `std` + `smallvec`, just enough for the blessed
`Set` / `Map` aliases; every other backend is opt-in.

- `std` *(default)* — implies `alloc`; provides `std::error::Error` for the error types.
- `alloc` — the heap-backed `Vec` backend.
- `smallvec` *(default)* — the `SmallVec` backend behind `Set`/`Map` (implies `alloc`).
- `tinyvec` — the `TinyVec` backend; 100% safe, requires `Elem: Default` (implies `alloc`).
- `arrayvec` — the fixed-capacity `ArrayVec` backend (alloc-free).
- `heapless` — the fixed-capacity `heapless::Vec` backend (alloc-free).
- `soa` — the struct-of-arrays column maps (`UnsortedColumnMap` / `SortedColumnMap`);
  backend-agnostic, pulls in no dependency.
- `serde` — `Serialize`/`Deserialize` for every collection (sets/bags as sequences,
  maps as maps). Deserialization enforces the bulk-build policy: sets dedup, **maps
  reject duplicate keys**, and a bounded store that fills is a data error — so
  deserializing into a fixed-capacity collection is input validation for free.

**MSRV:** Rust 1.87.

## `no_std`

The crate is `#![no_std]`. Build with `--no-default-features` and enable only the
backends you need: `arrayvec` and `heapless` stay allocator-free (`core` only),
while `Vec`, `smallvec`, and `tinyvec` pull in `alloc`. The borrowed stores need
no feature at all — a `SliceSet` lookup table, a `ScratchVec` over a stack
buffer, or a `Spill` composing them all work under `--no-default-features`.

Code size scales with what you instantiate, not with the crate: a single
fixed-capacity collection compiles to a few hundred bytes of `.text`, on par with
hand-rolling the equivalent `Vec`-based logic — you pay only for the backend and
collection combinations you actually use. See [Binary size](BENCHMARKS.md#binary-size-embedded).

Note that *logical* capacity (a fixed backend's `N`, or a `Capped` cap) is a
recoverable `CapacityError`, distinct from *allocator* OOM — a growable backend
that cannot grow aborts, and even a `Capped<Vec<_>>` can OOM below its cap.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
