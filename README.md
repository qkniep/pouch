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
over a `Vec`, `SmallVec`, `TinyVec`, `ArrayVec`, or `heapless::Vec` — heap, inline, or
hybrid — optionally bounded by a runtime cap. `no_std`-first.

> [!NOTE]
> **Early days.** The collection layer (`SortedSet`/`SortedMap`,
> `UnsortedSet`/`UnsortedMap`) is deliberately thin while the store traits settle.
> An `Entry` API has landed (`map.entry(k)`); comparators are next, and the API is
> **not yet stable**.

## Design

Three concerns that other small-collection crates usually fuse are kept
orthogonal, so you mix them freely:

- **storage** — *where* elements live (heap / inline / hybrid): the `Store` trait
  family, implemented once per backend.
- **bound** — the maximum logical element count, reported by
  `Store::capacity() -> Option<usize>`. A runtime bound is added with the
  `Capped<S>` wrapper rather than per backend.
- **ordering** — sorted (`SortedSet`/`SortedMap`, `O(log n)` lookup) vs unsorted
  (`UnsortedSet`/`UnsortedMap`, `O(n)` lookup, `O(1)` structural mutation, needs only
  `Eq`). This lives in the collection layer; the stores are ordering-agnostic. See
  [Complexity](#complexity).

The default `Set` / `Map` fix the combination this crate is tuned for — sorted,
`SmallVec`-backed (inline), unbounded — so the nested-population win is the path of
least resistance. Swap any axis (a `Vec` for one big collection, `heapless` for
`no_std`, unsorted when elements aren't `Ord`) when your case differs.

**Struct-of-arrays layout (`ColumnMap` / `SortedColumnMap`).** A map can instead keep
keys and values in *two* parallel stores, so a lookup scans (`ColumnMap`) or
binary-searches (`SortedColumnMap`) a dense key column without dragging values through
cache. `ColumnMap`'s scan also *vectorizes* — `get` and `contains_key` fold to
branchless compares the strided `(K, V)` scan can't manage, a ~2× edge on misses and
long scans at **any** value size — and for **large values** the skipped value-column
cache traffic stacks on top (a saving `SortedColumnMap`'s binary search shares, though
it has no scan to vectorize). Still niche enough that the array-of-structs `UnsortedMap`
/ `SortedMap` stay the default; reach for a column map when lookups dominate, especially
with big values. See [Benchmarks](#benchmarks).

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
   scan is the membership check. (A no-dedup bulk builder is planned; see the note above.)
3. `O(log n)` search, then `O(n)` shift.
4. `O(n)` find, then `O(1)` swap-remove (does not preserve order).

In short: **sorted** wins lookups; **unsorted** has `O(1)` structural mutation and
needs only `Eq`, so it wins when `n` is small or elements aren't `Ord`. Pick by the
operation mix, then pick a backend below.

## Backends

Storage and bound are orthogonal to ordering (above) — any backend pairs with
either flavor. Choose by where memory should live and whether the size is bounded;
the asymptotics don't change.

| Backend               | Storage              | Capacity      | `no_std` | Infallible `insert` | Feature *(default ✅)* | Reach for it when…           |
| --------------------- | -------------------- | ------------- | :------: | :-----------------: | ---------------------- | ---------------------------- |
| `Vec<T>`              | heap                 | unbounded     |    —     |         ✅          | `alloc`                | one big collection; `N` unpredictable |
| `SmallVec<[T; N]>`    | inline `N` → heap    | unbounded     |    —     |         ✅          | `smallvec` ✅          | **the default (`Set`/`Map`)** — many small / nested |
| `TinyVec<[T; N]>`     | inline `N` → heap    | unbounded     |    —     |         ✅          | `tinyvec` ✅           | same, 100% safe (`Elem: Default`) |
| `ArrayVec<T, N>`      | inline `N`           | `N` (fixed)   |    ✅    |   — (`try_insert`)  | `arrayvec` ✅          | embedded; hard cap, no alloc |
| `heapless::Vec<T, N>` | inline `N`           | `N` (fixed)   |    ✅    |   — (`try_insert`)  | `heapless` ✅          | embedded; hard cap, no alloc |
| `Capped<S>`           | wraps any store `S`  | runtime cap   |  = `S`   |   — (`try_insert`)  | —                      | enforce a limit / backpressure |

`try_insert` is always available and returns the rejected element on a bounded
store via `CapacityError<T>`. When the backing store is genuinely unbounded
(`Vec`, `SmallVec`, `TinyVec`), an infallible `insert` is also available.

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

Build a sorted collection in bulk instead of inserting one at a time — `O(n log n)`,
or `O(n)` from already-sorted input:

```rust
use pouch::VecSet;

let s = VecSet::try_from_iter([3, 1, 2, 3, 1]).unwrap(); // sorts + dedups once
assert_eq!(s.as_slice(), &[1, 2, 3]);
```

Insert-or-update a map value in a single lookup with the `Entry` API, instead of a
separate `get` then `insert`:

```rust
use pouch::Map;

let mut counts: Map<&str, u32> = Map::default();
for word in ["a", "b", "a", "a"] {
    *counts.entry(word).or_insert(0) += 1; // one search per word, not two
}
assert_eq!(counts.get(&"a"), Some(&3));
```

Fixed-capacity backends (and any store wrapped in `Capped`) make insertion
fallible instead of allocating without bound:

```rust
use pouch::{ArraySet, SortedSet};

let mut s: ArraySet<u64, 3> = SortedSet::new();
assert_eq!(s.try_insert(5), Ok(true));
assert_eq!(s.try_insert(1), Ok(true));
assert_eq!(s.try_insert(9), Ok(true));
// At capacity: the new element is handed back instead of inserted.
assert!(s.try_insert(2).is_err());
```

Use the unsorted variants for `O(1)` insert/delete (append + swap-remove) when
elements are cheap to scan or aren't `Ord`:

```rust
use pouch::{UnsortedSet, UnsortedVecSet};

let mut s: UnsortedVecSet<&str> = UnsortedSet::new();
s.insert("hello");
s.insert("world");
assert!(s.remove(&"hello")); // swap-remove; order is not preserved
```

## Benchmarks

Apple M4 Max, rustc 1.96, `cargo bench` — illustrative, re-run on your own hardware.
**Bold** = best. Full matrix (population, sets, maps, fixed-cap, backend sweep) in
[BENCHMARKS.md](BENCHMARKS.md).

**The headline — a `Vec` of 10 000 small sets** (heavy-tailed: ~99% hold 1–4
elements, ~1% are hubs of 64–1024). Building the whole population, `peak allocations`
and memory from divan's allocation profiler:

| inner set                  | allocations | memory      | lookup |
| -------------------------- | ----------: | ----------: | -----: |
| `pouch::Set` (inline, N=4) | **105**     | **1.10 MB** | 25 µs  |
| pouch over `Vec`           | 10 001      | 1.18 MB     | **23 µs** |
| `HashSet`                  | 10 001      | 1.93 MB     | 137 µs |
| `BTreeSet`                 | 17 980      | 2.20 MB     | 73 µs  |
| `thincollections::ThinSet` | 10 001      | 3.02 MB     | —      |

~95× fewer allocations, the lowest memory, and ~5× faster lookups than `HashSet`.
Two honest caveats: `N` is a memory knob — `N=4` (tuned to the 1–4 body) is shown;
`N=16` keeps the 105 allocations but uses 2.06 MB, and the default `Set` is `N=8`,
between. And the lookup win is the *sorted-small-set* property (both pouch backends
have it), not inline specifically — inline's unique, decisive win is allocation count.

Single-collection view — map, `n = 64`, `u64` keys:

| op           | pouch Sorted | litemap | BTreeMap | HashMap | FxHashMap |
| ------------ | ------------ | ------- | -------- | ------- | --------- |
| build random | 385 ns       | 1.34 µs | 546 ns   | 552 ns  | **234 ns**|
| get (hit)    | 186 ns       | 184 ns  | 227 ns   | 369 ns  | **62 ns** |
| get (miss)   | 184 ns       | 184 ns  | 248 ns   | 356 ns  | **92 ns** |

Set iteration (sum) — contiguous storage is the standout:

| n    | pouch Sorted | BTreeSet | HashSet |
| ---- | ------------ | -------- | ------- |
| 256  | **12 ns**    | 791 ns   | 118 ns  |
| 1024 | **56 ns**    | 3.19 µs  | 546 ns  |

Struct-of-arrays — `SortedColumnMap` (SoA, dense key column) vs `SortedMap` (AoS),
`u64` keys, median for a batch of `n` lookups (`SortedMap` / **`SortedColumnMap`**):

| op           | value | n = 16            | n = 4096            |
| ------------ | ----- | ----------------- | ------------------- |
| `get` hit    | 8 B   | **24 ns** / 33 ns | 28.8 µs / **23.0 µs** |
| `get` hit    | 64 B  | **28 ns** / 33 ns | 40.1 µs / **26.2 µs** |
| `get` miss   | 8 B   | 25 ns / **22 ns** | 28.9 µs / **21.3 µs** |
| `get` miss   | 64 B  | 30 ns / **24 ns** | 38.5 µs / **20.9 µs** |

The split wins at scale and on misses — the search skips the value column entirely
(up to ~1.8× for 64-byte misses). The catch is small-`n` *hits*: fetching the value
from its separate column is a second cache line, so `SortedMap` (value beside the key)
leads there. Hence `SortedColumnMap` is for large values + key-ordered iteration +
lookup-heavy; otherwise `SortedMap`.

`build random` uses the bulk `try_from_iter` constructor; the strategy breakdown
(insert-loop vs `try_from_iter` vs `from_sorted_iter`) is in [BENCHMARKS.md](BENCHMARKS.md).

What the numbers say:

- **Nested populations are the win** (table above): inline storage collapses a
  population of small sets to ~one allocation, which `Vec<HashSet>` / thincollections
  can't — they allocate per inner set regardless.
- **Parity with litemap** on the shared sorted-Vec design — the backend-generic
  layer costs nothing.
- **vs std:** flat binary search beats `BTreeMap` and SipHash `HashMap` on lookups;
  a fast hasher (`FxHashMap`) overtakes it past `n ≈ 16` on lookups and on random-order
  `build`, but the bulk constructors close most of the build gap — and from
  already-sorted input pouch builds *fastest* at large `n`. A sorted `Vec` is the
  small-`n`, iteration-heavy, or `no_std` choice.
- **Bulk construction:** `try_from_iter` (sort once) and `from_sorted_iter` (no sort)
  beat an `insert`-per-element loop by ~6× and ~40× at `n = 1024` — a 1024-key sorted
  set builds in ~0.89 µs from sorted input vs ~35 µs one at a time.
- **Iteration** over contiguous memory runs ~10–50× faster than the tree/hash maps.
- **Backend choice moves only the constant:** at `n = 16` an inline `ArrayVec` /
  `heapless` builds a sorted set in ~135 ns vs `Vec`'s ~187 ns (the heap allocation);
  by `n = 256` they converge.

## Crate features

- `std` *(default)* — implies `alloc`; provides `std::error::Error for CapacityError`.
- `alloc` — the heap-backed `Vec` backend and `Capped` over growable stores.
- `smallvec` *(default)* — the `SmallVec` backend (implies `alloc`).
- `tinyvec` *(default)* — the `TinyVec` backend; 100% safe, requires `Elem: Default` (implies `alloc`).
- `arrayvec` *(default)* — the fixed-capacity `ArrayVec` backend (alloc-free).
- `heapless` *(default)* — the fixed-capacity `heapless::Vec` backend (alloc-free).

**MSRV:** Rust 1.87.

## `no_std`

The crate is `#![no_std]`. Build with `--no-default-features` and enable only the
backends you need: `arrayvec` and `heapless` stay allocator-free (`core` only),
while `Vec`, `smallvec`, and `tinyvec` pull in `alloc`.

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
