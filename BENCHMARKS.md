# Benchmarks

Median wall-clock, lower is better. **Bold** = fastest in the row. Keys are distinct
`u64` from a SplitMix64 stream; each row times a batch of `n` operations.

- **Machine:** Apple M4 Max, macOS 26.3.1
- **Toolchain:** rustc 1.96.0, `--release`, [divan](https://docs.rs/divan)
- **Date:** 2026-07-04, all wall-clock tables from one run (binary size: 2026-06-27)
- **Reproduce:** `cargo bench` (or `cargo bench --bench map|set|soa|backend`)

These are constant factors on one machine — re-run on your own hardware. The
machine-independent asymptotics are in the README complexity table.

Contenders: pouch `Sorted*`/`Unsorted*` (over `Vec`), [litemap](https://crates.io/crates/litemap),
[sorted-vec](https://crates.io/crates/sorted-vec), [vecmap-rs](https://crates.io/crates/vecmap-rs),
std `BTree*`/`Hash*` (SipHash), and [FxHash](https://crates.io/crates/rustc-hash).

## Nested population — `Vec` of 10 000 small sets (the headline)

Heavy-tailed sizes: ~99% hold 1–4 elements, ~1% are hubs of 64–1024. `peak
allocations` and `peak bytes` are live highs from divan's allocation profiler.
This is the regime the crate is positioned for; the standalone tables below are
the single-collection view.

### `build_random` — build the whole population (random insert order)

| inner collection | build time | peak allocations | peak bytes |
|---|--:|--:|--:|
| `HashSet` | 1.74 ms | 10001 | 1.925 MB |
| `BTreeSet` | 2.35 ms | 17980 | 2.201 MB |
| thincollections | 1.44 ms | 10001 | 3.021 MB |
| pouch / `Vec` | 2.32 ms | 10001 | 1.177 MB |
| pouch / `SmallVec<[_;4]>` | 2.28 ms | **105** | **1.1 MB** |
| pouch / `SmallVec<[_;16]>` | 2.30 ms | **105** | 2.06 MB |

### `build_sorted` — build from pre-sorted inner elements (build-once)

| inner collection | build time | peak allocations | peak bytes |
|---|--:|--:|--:|
| `HashSet` | 1.79 ms | 10001 | 1.925 MB |
| `BTreeSet` | 937.6 µs | 19852 | 2.438 MB |
| thincollections | 1.43 ms | 10001 | 3.021 MB |
| pouch / `Vec` | 454.4 µs | 10001 | 1.177 MB |
| pouch / `SmallVec<[_;4]>` | 467.1 µs | **105** | **1.1 MB** |
| pouch / `SmallVec<[_;16]>` | 494.1 µs | **105** | 2.06 MB |

### `lookup` — membership across the population (half hit / half miss)

| inner collection | lookup time |
|---|--:|
| `HashSet` | 139.1 µs |
| `BTreeSet` | 69.8 µs |
| pouch / `Vec` | 23.6 µs |
| pouch / `SmallVec<[_;4]>` | 27.5 µs |
| pouch / `SmallVec<[_;16]>` | 27.5 µs |

Takeaways:

- **Allocation count is categorical:** the inline backend builds the whole
  population in **105** allocations vs **10 001** (`Vec`/`HashSet`/thincollections)
  or **17 980** (`BTreeSet`) — ~95× fewer.
- **Memory needs `N` matched to the body:** `SmallVec<[_;4]>` (fits the 1–4 body)
  is the lowest-memory option at **1.10 MB**; `SmallVec<[_;16]>` keeps the alloc
  win but nearly doubles bytes (**2.06 MB**) — `size_of` scales with `N`.
- **Lookup** is ~5× faster than `HashSet` and ~2.5–3× faster than `BTreeSet` for
  both pouch backends — that's the sorted-small-set property, not inline
  specifically (inline is a touch slower than `Vec` here; its cold-cache / churn
  edge isn't captured by build-then-read timing).
- **thincollections** optimizes pointer footprint, not allocation count: it still
  allocates per non-empty inner (10 001) and used the most memory (3.02 MB).

## Maps (growable, over `Vec`)

### `build_random` — build from keys in random order

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 13 ns | 9.7 ns | 34 ns | **9.5 ns** | 46 ns | 47 ns | 23 ns |
| 16 | **56 ns** | 103 ns | 199 ns | 104 ns | 119 ns | 147 ns | 97 ns |
| 64 | 250 ns | 823 ns | 1.27 µs | 781 ns | 510 ns | 531 ns | **229 ns** |
| 256 | 1.27 µs | 10.20 µs | 7.79 µs | 9.33 µs | 1.81 µs | 2.04 µs | **791 ns** |
| 1024 | 5.87 µs | 128.8 µs | 64.3 µs | 130.2 µs | 8.87 µs | 8.29 µs | **2.79 µs** |

### `build_sorted` — build from keys already ascending

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 13 ns | **10 ns** | 18 ns | 10 ns | 28 ns | 40 ns | 19 ns |
| 16 | **27 ns** | 92 ns | 157 ns | 95 ns | 71 ns | 136 ns | 88 ns |
| 64 | **73 ns** | 781 ns | 640 ns | 802 ns | 169 ns | 489 ns | 260 ns |
| 256 | **270 ns** | 9.87 µs | 2.54 µs | 10.08 µs | 541 ns | 1.98 µs | 713 ns |
| 1024 | **937 ns** | 132.1 µs | 7.92 µs | 133.4 µs | 2.02 µs | 7.71 µs | 2.33 µs |

### `get_hit` — lookup, all keys present

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 3.4 ns | 3.4 ns | 4.8 ns | **3.3 ns** | 4.4 ns | 23 ns | 3.6 ns |
| 16 | 28 ns | 46 ns | 37 ns | 46 ns | 38 ns | 95 ns | **15 ns** |
| 64 | 183 ns | 619 ns | 213 ns | 614 ns | 224 ns | 398 ns | **73 ns** |
| 256 | 1.04 µs | 9.58 µs | 1.09 µs | 8.75 µs | 1.50 µs | 1.51 µs | **250 ns** |
| 1024 | 5.75 µs | 129.8 µs | 5.73 µs | 128.4 µs | 7.08 µs | 6.08 µs | **1.04 µs** |

### `get_miss` — lookup, no keys present

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | **3.4 ns** | 6.0 ns | 4.9 ns | 6.1 ns | 4.6 ns | 20 ns | 5.8 ns |
| 16 | 28 ns | 84 ns | 37 ns | 82 ns | 49 ns | 82 ns | **23 ns** |
| 64 | 183 ns | 989 ns | 213 ns | 1.02 µs | 259 ns | 317 ns | **102 ns** |
| 256 | 1.05 µs | 16.66 µs | 1.11 µs | 16.77 µs | 1.67 µs | 1.60 µs | **393 ns** |
| 1024 | 5.75 µs | 257.1 µs | 5.79 µs | 258.9 µs | 7.71 µs | 6.04 µs | **1.71 µs** |

## Struct-of-arrays maps — value-size sweep (`UnsortedColumnMap` / `SortedColumnMap`)

The same map logic with keys and values in *two* parallel stores instead of one
`(K, V)` store, so a lookup touches a dense key column and skips the value payloads.
`K = u64`; `V` sweeps `u64` (8 B) → `[u64; 8]` (64 B). Median for a batch of `n`
lookups; **bold** = faster layout *for that value size* (the array-of-structs
`SortedMap`/`UnsortedMap` vs its column twin).

### Sorted — `SortedColumnMap` vs `SortedMap` (binary search, `O(log n)`)

`get_hit` (reads the value):

| n | AoS 8 B | SoA 8 B | AoS 64 B | SoA 64 B |
|--:|--:|--:|--:|--:|
| 16 | **27 ns** | 33 ns | **31 ns** | 33 ns |
| 64 | 179 ns | **165 ns** | 209 ns | **165 ns** |
| 256 | 1.05 µs | **812 ns** | 1.21 µs | **854 ns** |
| 1024 | 5.83 µs | **4.37 µs** | 6.87 µs | **4.58 µs** |
| 4096 | 29.5 µs | **22.8 µs** | 40.1 µs | **26.0 µs** |

`get_miss` (no value load):

| n | AoS 8 B | SoA 8 B | AoS 64 B | SoA 64 B |
|--:|--:|--:|--:|--:|
| 16 | 28 ns | **21 ns** | 30 ns | **23 ns** |
| 64 | 175 ns | **131 ns** | 201 ns | **131 ns** |
| 256 | 1.02 µs | **739 ns** | 1.18 µs | **739 ns** |
| 1024 | 5.71 µs | **4.00 µs** | 6.71 µs | **4.00 µs** |
| 4096 | 29.0 µs | **20.8 µs** | 42.1 µs | **20.8 µs** |

The column split wins at scale and on misses (the search never touches the value
column) — ~2× for 64-byte misses at `n = 4096`. The exception is small-`n` **hits**:
the value load is a second cache line for SoA but rides the key's line in AoS, so
`SortedMap` leads at `n = 16`. Net: `SortedColumnMap` pays off for large values with
lookup-heavy, key-ordered workloads; `SortedMap` is the default.

### Unsorted — `UnsortedColumnMap` vs `UnsortedMap` (linear scan, `O(n)`)

Both queries scan the dense key column as a folded reduction — `contains_key` via the
boolean `chunked_contains` fold (the crate's mirror of the standard library's
specialized `slice::contains`, whose `&K` needle borrowed-key lookups can't supply),
`get` via the index-recovering `chunked_position` — chunked OR-reductions LLVM lowers
to branchless compares, so both pull far ahead of the strided AoS scan, ≈
value-size-independent. For large values a cache-bandwidth effect (the scan never
touches the value column) stacks on top. Misses (whole-array scan), median batch of
`n`:

| n | `contains_key` AoS 64 B | SoA 64 B | `get` AoS 64 B | SoA 64 B |
|--:|--:|--:|--:|--:|
| 64 | 968 ns | **289 ns** | 1.01 µs | **567 ns** |
| 256 | 16.6 µs | **4.62 µs** | 16.7 µs | **8.58 µs** |
| 4096 | 8.64 ms | **1.19 ms** | 8.57 ms | **2.02 ms** |

`contains_key` is ~3.4–7× faster on the column layout and `get` ~1.8–4.2× — the win
holds down to small `n` on misses (the dense scan's edge is value-size-independent), and
the `get` win covers word-sized values too, where it was previously a wash. The one
spot the column map doesn't lead is word-sized **hits** at `n ≲ 64`: the match is found
early — blunting the scan advantage — and the value sits in a second cache line, so AoS
(value beside the key) is ~par or a hair faster there. `n = 16` is omitted as
timer-noise-dominated (batches of tens of nanoseconds against ~41 ns precision). See
`benches/soa.rs`.

## Sets (growable, over `Vec`)

### `build_random` — build from keys in random order

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 17 ns | 16 ns | 16 ns | **9.9 ns** | 39 ns | 38 ns | 19 ns |
| 16 | **55 ns** | 91 ns | 173 ns | 113 ns | 89 ns | 130 ns | 114 ns |
| 64 | **213 ns** | 333 ns | 989 ns | 802 ns | 418 ns | 468 ns | 216 ns |
| 256 | 1.10 µs | 2.58 µs | 5.25 µs | 10.02 µs | 1.44 µs | 2.80 µs | **968 ns** |
| 1024 | 5.06 µs | 37.1 µs | 39.0 µs | 133.0 µs | 6.58 µs | 8.12 µs | **3.69 µs** |

### `build_sorted` — build from keys already ascending

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 15 ns | 14 ns | 12 ns | **11 ns** | 29 ns | 43 ns | 19 ns |
| 16 | **25 ns** | 90 ns | 114 ns | 107 ns | 63 ns | 152 ns | 87 ns |
| 64 | **78 ns** | 330 ns | 560 ns | 781 ns | 146 ns | 526 ns | 281 ns |
| 256 | **281 ns** | 2.58 µs | 1.48 µs | 9.98 µs | 468 ns | 1.98 µs | 948 ns |
| 1024 | **1.03 µs** | 37.0 µs | 5.33 µs | 134.9 µs | 1.77 µs | 7.81 µs | 1.83 µs |

### `contains_hit` — membership, all present

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | **3.2 ns** | 3.7 ns | 4.7 ns | 3.5 ns | 4.5 ns | 21 ns | 3.6 ns |
| 16 | 23 ns | 16 ns | 22 ns | 52 ns | 37 ns | 94 ns | **15 ns** |
| 64 | 139 ns | 165 ns | 136 ns | 591 ns | 226 ns | 370 ns | **73 ns** |
| 256 | 791 ns | 2.62 µs | 791 ns | 8.71 µs | 1.50 µs | 1.51 µs | **229 ns** |
| 1024 | 4.25 µs | 46.3 µs | 4.29 µs | 126.2 µs | 7.00 µs | 6.08 µs | **958 ns** |

### `contains_miss` — membership, none present

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | **4.1 ns** | 4.2 ns | 4.8 ns | 6.0 ns | 4.7 ns | 20 ns | 5.9 ns |
| 16 | 23 ns | **22 ns** | 22 ns | 84 ns | 48 ns | 83 ns | 24 ns |
| 64 | 140 ns | 289 ns | 139 ns | 1.20 µs | 258 ns | 375 ns | **112 ns** |
| 256 | 797 ns | 4.58 µs | 791 ns | 18.5 µs | 1.67 µs | 1.51 µs | **401 ns** |
| 1024 | 4.33 µs | 73.7 µs | 4.31 µs | 259.5 µs | 7.67 µs | 6.08 µs | **1.64 µs** |

### `remove` — remove every element

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 8.9 ns | **3.9 ns** | 9.0 ns | 8.6 ns | 21 ns | 33 ns | 12 ns |
| 16 | 114 ns | **28 ns** | 114 ns | 41 ns | 153 ns | 164 ns | 80 ns |
| 64 | 823 ns | 328 ns | 838 ns | 261 ns | 708 ns | 750 ns | **228 ns** |
| 256 | 4.50 µs | 4.98 µs | 4.62 µs | 2.69 µs | 3.37 µs | 3.07 µs | **823 ns** |
| 1024 | 37.9 µs | 73.5 µs | 37.9 µs | 37.6 µs | 14.7 µs | 12.2 µs | **2.96 µs** |

### `iter` — sum every element

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | **1.0 ns** | 1.0 ns | 1.0 ns | 1.3 ns | 6.9 ns | 1.4 ns | 1.4 ns |
| 16 | **0.96 ns** | 1.0 ns | 0.99 ns | 1.0 ns | 32 ns | 6.6 ns | 7.2 ns |
| 64 | 3.7 ns | 3.6 ns | 3.6 ns | **3.4 ns** | 198 ns | 29 ns | 32 ns |
| 256 | 13 ns | 13 ns | 13 ns | **13 ns** | 820 ns | 130 ns | 139 ns |
| 1024 | 58 ns | 57 ns | 58 ns | **57 ns** | 3.28 µs | 562 ns | 536 ns |

## Construction strategy (`SortedMap` / `SortedSet` over `Vec`)

The same final contents (distinct keys) built three ways — the payoff of the bulk
constructors over a repeated-`try_insert` loop:

- `insert_loop` — `try_insert` per entry, random input: `O(n²)`, binary-search + tail
  shift each time.
- `try_from_iter` — same random input, append all then one `sort_unstable` + dedup:
  `O(n log n)`. This is what the `build_random` tables above use.
- `from_sorted_iter` — ascending input, append-only, no sort or search: `O(n)`.

Both bulk builders now `reserve` up front from the iterator's `size_hint`, so the
append pays one allocation instead of `log n` growth spikes — visible below as a
~25% drop for `from_sorted_iter` at `n = 1024` versus the previous measurement.

### Map (`SortedMap<Vec<(u64, u64)>>`)

| n | insert_loop | try_from_iter | from_sorted_iter |
|--:|--:|--:|--:|
| 4 | 17 ns | 12 ns | **11 ns** |
| 16 | 182 ns | 46 ns | **19 ns** |
| 64 | 1.19 µs | 252 ns | **55 ns** |
| 256 | 7.37 µs | 1.27 µs | **187 ns** |
| 1024 | 61.1 µs | 5.96 µs | **698 ns** |

### Set (`SortedSet<Vec<u64>>`)

| n | insert_loop | try_from_iter | from_sorted_iter |
|--:|--:|--:|--:|
| 4 | 25 ns | 15 ns | **14 ns** |
| 16 | 166 ns | 39 ns | **20 ns** |
| 64 | 979 ns | 189 ns | **52 ns** |
| 256 | 5.25 µs | 1.07 µs | **187 ns** |
| 1024 | 38.8 µs | 4.96 µs | **666 ns** |

At `n = 1024` the bulk constructors beat the insert-per-element loop by **~8×**
(`try_from_iter`) and **~58×** (`from_sorted_iter`) for the set — ~10× / ~88× for the
map. A 1024-key sorted set builds in ~0.67 µs from already-sorted input versus ~39 µs
one at a time.

## Fixed-capacity / `no_std` (capacity matched to `n`)

Inline storage: pouch over `heapless::Vec` vs [micromap](https://crates.io/crates/micromap)
and `heapless::LinearMap`. These small fixed-cap monomorphizations show the most
run-to-run codegen/layout variance of any table here (unchanged third-party
contenders moved ~2× between measurement sessions on the same toolchain) — treat
single cells as ±2× ballpark and re-measure for your own build.

### Maps

**`build`**

| n | pouch Unsorted/heapless | heapless::LinearMap | micromap |
|--:|--:|--:|--:|
| 4 | 5.6 ns | 4.8 ns | **4.0 ns** |
| 16 | **48 ns** | 55 ns | 52 ns |
| 64 | 677 ns | 651 ns | **398 ns** |
| 256 | 9.04 µs | 8.79 µs | **5.71 µs** |

**`get_hit`**

| n | pouch Unsorted/heapless | heapless::LinearMap | micromap |
|--:|--:|--:|--:|
| 4 | 4.5 ns | 3.3 ns | **3.1 ns** |
| 16 | 78 ns | 50 ns | **48 ns** |
| 64 | **1.02 µs** | 1.24 µs | 1.25 µs |
| 256 | 15.9 µs | 27.0 µs | **8.92 µs** |

**`get_miss`**

| n | pouch Unsorted/heapless | heapless::LinearMap | micromap |
|--:|--:|--:|--:|
| 4 | 7.9 ns | 6.2 ns | **4.3 ns** |
| 16 | 125 ns | **77 ns** | 84 ns |
| 64 | 1.92 µs | 1.05 µs | **1.02 µs** |
| 256 | 30.7 µs | **16.5 µs** | 16.8 µs |

### Sets

**`build`**

| n | pouch Unsorted/heapless | pouch Sorted/heapless | micromap |
|--:|--:|--:|--:|
| 4 | 3.7 ns | 8.2 ns | **3.6 ns** |
| 16 | **25 ns** | 125 ns | 28 ns |
| 64 | **208 ns** | 958 ns | 458 ns |
| 256 | **2.96 µs** | 5.25 µs | 12.5 µs |

**`contains_hit`**

| n | pouch Unsorted/heapless | pouch Sorted/heapless | micromap |
|--:|--:|--:|--:|
| 4 | 5.4 ns | 3.3 ns | **3.2 ns** |
| 16 | **18 ns** | 22 ns | 46 ns |
| 64 | 172 ns | **144 ns** | 666 ns |
| 256 | 2.69 µs | **807 ns** | 16.0 µs |

**`contains_miss`**

| n | pouch Unsorted/heapless | pouch Sorted/heapless | micromap |
|--:|--:|--:|--:|
| 4 | 7.0 ns | **3.3 ns** | 4.5 ns |
| 16 | 23 ns | **22 ns** | 87 ns |
| 64 | 302 ns | **144 ns** | 1.10 µs |
| 256 | 4.75 µs | **817 ns** | 34.9 µs |

## Backend sweep — same `SortedSet` op, every backend

Big-O is identical across backends (each store is a contiguous array); only the
constant moves. `Vec` pays an allocation that inline stores skip at small `n`, and
the gap closes as `n` grows.

### `build` — sorted insert, random order

| n | Vec | SmallVec | ArrayVec | heapless::Vec |
|--:|--:|--:|--:|--:|
| 16 | 195 ns | 174 ns | 147 ns | **139 ns** |
| 64 | 1.11 µs | 1.12 µs | 1.13 µs | **1.07 µs** |
| 256 | **5.92 µs** | 6.33 µs | 6.37 µs | 6.00 µs |

### `contains_hit` — membership, all present

| n | Vec | SmallVec | ArrayVec | heapless::Vec |
|--:|--:|--:|--:|--:|
| 16 | **24 ns** | 29 ns | 27 ns | 25 ns |
| 64 | **156 ns** | 167 ns | 163 ns | 161 ns |
| 256 | **885 ns** | 890 ns | 916 ns | 911 ns |

## Binary size (embedded)

Flash footprint rather than wall-clock — the concern for the `no_std` audience.
Cross-compiled to `thumbv7em-none-eabihf` (Cortex-M4F), `opt-level = "z"` + fat
LTO, `K = V = u32`, fixed capacity 64. Each number is the marginal `.text` a
collection's full API (insert + lookup + remove) adds over a bare `#![no_std]`
baseline (panic handler only), measured by diffing defined symbols with
`llvm-nm` so shared `core` / `compiler_builtins` code is excluded. Code is emitted
per monomorphization, so you pay only for the `(collection × backend × element
type)` combinations you actually instantiate.

| collection (`heapless::Vec`, `u32`) | `.text` | + entry API |
|---|--:|--:|
| `SortedSet` | 300 B | — |
| `UnsortedSet` | 312 B | — |
| `SortedMap` | 332 B | +538 B |
| `UnsortedMap` | 348 B | +514 B |
| `SortedColumnMap` | 478 B | +390 B |
| `UnsortedColumnMap` | 526 B | +534 B |
| **all six together** | **1990 B** | |

All six together (1990 B) cost less than their independent sum (2296 B): the
per-element-type helpers (`binary_search`, panic glue) are shared, so adding more
collection *types* of the same element type is cheap.

The **`+ entry API`** column is the *additional* `.text` the `entry`-based methods
(`or_try_insert`, an `and_modify` update, and removal through the entry) add on top
of the collection's own insert/get/remove — and only if you instantiate `.entry()`
at all (it is generic, so a binary that never calls it pays nothing, and the basic
column is unchanged either way). Sets have no entry API. The few-hundred-byte figure
is genuinely *new* code: the slot lookup is shared with the basic API, but a vacant
insert and the `and_modify` closure are distinct monomorphizations. The `or_insert`
family (infallible) is `Unbounded`-gated and so unreachable on a fixed-cap
`heapless::Vec`; on a growable backend it adds a little more.

For context, same setup: a `SortedSet` hand-rolled over a raw `heapless::Vec` is
320 B, `heapless::LinearMap` 268 B, `heapless::FnvIndexMap` 680 B. pouch's generic
layer is zero-overhead — its `SortedSet` (300 B) matches both the hand-rolled
version and the `ArrayVec` backend (also 300 B). Numbers are toolchain-, target-,
and `opt-level`-dependent; treat them as ballpark and re-measure for your build.
(Binary-size figures date from 2026-06-27, before the borrowed-key lookup and
`reserve` work; the fixed-cap insert/lookup/remove paths they measure are
unchanged, but re-measure if the bytes matter to you.)
