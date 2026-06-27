# Benchmarks

Median wall-clock, lower is better. **Bold** = fastest in the row. Keys are distinct
`u64` from a SplitMix64 stream; each row times a batch of `n` operations.

- **Machine:** Apple M4 Max, macOS 26.3.1
- **Toolchain:** rustc 1.96.0, `--release`, [divan](https://docs.rs/divan)
- **Date:** 2026-06-26 (struct-of-arrays, and the `build_*` / construction sections: 2026-06-27, same setup)
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
| `HashSet` | 1.72 ms | 10001 | 1.925 MB |
| `BTreeSet` | 2.142 ms | 17980 | 2.201 MB |
| thincollections | 1.376 ms | 10001 | 3.021 MB |
| pouch / `Vec` | 2.124 ms | 10001 | 1.177 MB |
| pouch / `SmallVec<[_;4]>` | 2.158 ms | **105** | **1.1 MB** |
| pouch / `SmallVec<[_;16]>` | 2.161 ms | **105** | 2.06 MB |

### `build_sorted` — build from pre-sorted inner elements (build-once)

| inner collection | build time | peak allocations | peak bytes |
|---|--:|--:|--:|
| `HashSet` | 1.81 ms | 10001 | 1.925 MB |
| `BTreeSet` | 902.2 µs | 19852 | 2.438 MB |
| thincollections | 1.359 ms | 10001 | 3.021 MB |
| pouch / `Vec` | 419.1 µs | 10001 | 1.177 MB |
| pouch / `SmallVec<[_;4]>` | 460.8 µs | **105** | **1.1 MB** |
| pouch / `SmallVec<[_;16]>` | 466.9 µs | **105** | 2.06 MB |

### `lookup` — membership across the population (half hit / half miss)

| inner collection | lookup time |
|---|--:|
| `HashSet` | 136.3 µs |
| `BTreeSet` | 73.1 µs |
| pouch / `Vec` | 23.29 µs |
| pouch / `SmallVec<[_;4]>` | 25.31 µs |
| pouch / `SmallVec<[_;16]>` | 26.66 µs |

Takeaways:

- **Allocation count is categorical:** the inline backend builds the whole
  population in **105** allocations vs **10 001** (`Vec`/`HashSet`/thincollections)
  or **17 980** (`BTreeSet`) — ~95× fewer.
- **Memory needs `N` matched to the body:** `SmallVec<[_;4]>` (fits the 1–4 body)
  is the lowest-memory option at **1.10 MB**; `SmallVec<[_;16]>` keeps the alloc
  win but nearly doubles bytes (**2.06 MB**) — `size_of` scales with `N`.
- **Lookup** is ~5× faster than `HashSet` and ~3× faster than `BTreeSet` for both
  pouch backends — that's the sorted-small-set property, not inline specifically
  (inline is a touch slower than `Vec` here; its cold-cache / churn edge isn't
  captured by build-then-read timing).
- **thincollections** optimizes pointer footprint, not allocation count: it still
  allocates per non-empty inner (10 001) and used the most memory (3.02 MB).

## Maps (growable, over `Vec`)

### `build_random` — build from keys in random order

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 14 ns | 12 ns | 19 ns | **10 ns** | 36 ns | 43 ns | 21 ns |
| 16 | 107 ns | 107 ns | 196 ns | 86 ns | 104 ns | 121 ns | **86 ns** |
| 64 | 385 ns | 739 ns | 1.06 µs | 661 ns | 453 ns | 437 ns | **235 ns** |
| 256 | 1.50 µs | 10.24 µs | 7.21 µs | 9.12 µs | 1.62 µs | 1.71 µs | **740 ns** |
| 1024 | 6.50 µs | 126.20 µs | 55.20 µs | 128.10 µs | 7.75 µs | 7.62 µs | **2.50 µs** |

### `build_sorted` — build from keys already ascending

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 14 ns | **9.7 ns** | 14 ns | 12 ns | 25 ns | 41 ns | 19 ns |
| 16 | **68 ns** | 88 ns | 148 ns | 95 ns | 69 ns | 139 ns | 87 ns |
| 64 | 190 ns | 661 ns | 604 ns | 750 ns | **159 ns** | 494 ns | 255 ns |
| 256 | **500 ns** | 9.00 µs | 2.17 µs | 9.21 µs | 505 ns | 1.96 µs | 729 ns |
| 1024 | **1.46 µs** | 125.60 µs | 7.50 µs | 130.50 µs | 1.90 µs | 7.04 µs | 2.35 µs |

### `get_hit` — lookup, all keys present

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 3.3 ns | 3.5 ns | 3.4 ns | **2.9 ns** | 4.2 ns | 23 ns | 3.5 ns |
| 16 | 27 ns | 47 ns | 28 ns | 43 ns | 39 ns | 96 ns | **15 ns** |
| 64 | 186 ns | 630 ns | 184 ns | 604 ns | 227 ns | 369 ns | **62 ns** |
| 256 | 1.05 µs | 9.71 µs | 1.07 µs | 8.83 µs | 1.46 µs | 1.50 µs | **252 ns** |
| 1024 | 5.83 µs | 131.60 µs | 6.00 µs | 124.50 µs | 7.21 µs | 6.10 µs | **1.04 µs** |

### `get_miss` — lookup, no keys present

| n | pouch Sorted | pouch Unsorted | litemap | vecmap-rs | BTreeMap | HashMap | FxHashMap |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 3.4 ns | 7.1 ns | **3.3 ns** | 5.9 ns | 4.4 ns | 20 ns | 5.7 ns |
| 16 | 27 ns | 84 ns | 27 ns | 74 ns | 47 ns | 83 ns | **23 ns** |
| 64 | 184 ns | 1.03 µs | 184 ns | 979 ns | 248 ns | 356 ns | **92 ns** |
| 256 | 1.04 µs | 16.99 µs | 1.06 µs | 16.56 µs | 1.62 µs | 1.50 µs | **398 ns** |
| 1024 | 5.75 µs | 248.70 µs | 5.81 µs | 256.20 µs | 7.96 µs | 6.12 µs | **1.67 µs** |

## Struct-of-arrays maps — value-size sweep (`ColumnMap` / `SortedColumnMap`)

The same map logic with keys and values in *two* parallel stores instead of one
`(K, V)` store, so a lookup touches a dense key column and skips the value payloads.
`K = u64`; `V` sweeps `u64` (8 B) → `[u64; 8]` (64 B). Median for a batch of `n`
lookups; **bold** = faster layout *for that value size* (the array-of-structs
`SortedMap`/`UnsortedMap` vs its column twin). Measured 2026-06-27, same setup.

### Sorted — `SortedColumnMap` vs `SortedMap` (binary search, `O(log n)`)

`get_hit` (reads the value):

| n | AoS 8 B | SoA 8 B | AoS 64 B | SoA 64 B |
|--:|--:|--:|--:|--:|
| 16 | **24 ns** | 33 ns | **28 ns** | 33 ns |
| 64 | **170 ns** | 179 ns | 209 ns | **166 ns** |
| 256 | 937 ns | **812 ns** | 1.11 µs | **825 ns** |
| 1024 | 5.25 µs | **4.42 µs** | 6.25 µs | **4.58 µs** |
| 4096 | 28.8 µs | **23.0 µs** | 40.1 µs | **26.2 µs** |

`get_miss` (no value load):

| n | AoS 8 B | SoA 8 B | AoS 64 B | SoA 64 B |
|--:|--:|--:|--:|--:|
| 16 | 25 ns | **22 ns** | 30 ns | **24 ns** |
| 64 | 174 ns | **131 ns** | 202 ns | **131 ns** |
| 256 | 1.02 µs | **745 ns** | 1.18 µs | **771 ns** |
| 1024 | 5.65 µs | **4.04 µs** | 6.19 µs | **4.02 µs** |
| 4096 | 28.9 µs | **21.3 µs** | 38.5 µs | **20.9 µs** |

The column split wins at scale and on misses (the search never touches the value
column) — ~1.8× for 64-byte misses at `n = 4096`. The exception is small-`n` **hits**:
the value load is a second cache line for SoA but rides the key's line in AoS, so
`SortedMap` leads at `n ≤ 16`. Net: `SortedColumnMap` pays off for large values with
lookup-heavy, key-ordered workloads; `SortedMap` is the default.

### Unsorted — `ColumnMap` vs `UnsortedMap` (linear scan, `O(n)`)

Both queries scan the dense key column as a folded reduction — `contains_key` via the
stdlib boolean `contains`, `get` via the fixed-trip `chunked_position` (a chunked
OR-reduction LLVM lowers to branchless compares) — so both pull far ahead of the strided
AoS scan, ≈ value-size-independent. For large values a cache-bandwidth effect (the scan
never touches the value column) stacks on top. Misses (whole-array scan), median batch
of `n`:

| n | `contains_key` AoS 64 B | SoA 64 B | `get` AoS 64 B | SoA 64 B |
|--:|--:|--:|--:|--:|
| 64 | 1.10 µs | **0.29 µs** | 1.14 µs | **0.50 µs** |
| 256 | 16.9 µs | **4.58 µs** | 18.6 µs | **7.75 µs** |
| 4096 | 8.30 ms | **1.19 ms** | 8.51 ms | **2.01 ms** |

`contains_key` is ~3.7–7× faster on the column layout and `get` ~2.3–4.2× — the win
holds down to small `n` on misses (the dense scan's edge is value-size-independent), and
the `get` win now covers word-sized values too, where it was previously a wash. The one
spot the column map doesn't lead is word-sized **hits** at `n ≲ 64`: the match is found
early — blunting the scan advantage — and the value sits in a second cache line, so AoS
(value beside the key) is ~par or a hair faster there. `n = 16` is omitted as
timer-noise-dominated (batches of tens of nanoseconds against ~41 ns precision). See
`benches/soa.rs`.

## Sets (growable, over `Vec`)

### `build_random` — build from keys in random order

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 19 ns | 14 ns | 16 ns | **11 ns** | 49 ns | 55 ns | 27 ns |
| 16 | 113 ns | **94 ns** | 166 ns | 98 ns | 124 ns | 193 ns | 111 ns |
| 64 | 364 ns | 325 ns | 999 ns | 760 ns | 552 ns | 656 ns | **299 ns** |
| 256 | 1.43 µs | 2.75 µs | 5.46 µs | 10.08 µs | 1.92 µs | 2.39 µs | **1.21 µs** |
| 1024 | 5.96 µs | 43.83 µs | 39.06 µs | 144.90 µs | 8.79 µs | 9.50 µs | **2.54 µs** |

### `build_sorted` — build from keys already ascending

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 12 ns | 10 ns | 12 ns | **8.8 ns** | 32 ns | 40 ns | 17 ns |
| 16 | 67 ns | 79 ns | 103 ns | 83 ns | **55 ns** | 143 ns | 79 ns |
| 64 | 173 ns | 283 ns | 506 ns | 721 ns | **164 ns** | 499 ns | 294 ns |
| 256 | 489 ns | 2.42 µs | 1.32 µs | 9.21 µs | **479 ns** | 2.80 µs | 828 ns |
| 1024 | **1.41 µs** | 37.45 µs | 4.60 µs | 139.30 µs | 1.83 µs | 7.39 µs | 2.17 µs |

### `contains_hit` — membership, all present

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 3.1 ns | 4.3 ns | **3.1 ns** | 4.0 ns | 4.2 ns | 24 ns | 3.4 ns |
| 16 | 23 ns | 17 ns | 23 ns | 49 ns | 37 ns | 97 ns | **15 ns** |
| 64 | 140 ns | 164 ns | 139 ns | 573 ns | 227 ns | 375 ns | **71 ns** |
| 256 | 791 ns | 2.37 µs | 791 ns | 8.87 µs | 1.46 µs | 1.51 µs | **255 ns** |
| 1024 | 4.21 µs | 41.14 µs | 4.25 µs | 127.40 µs | 7.08 µs | 6.00 µs | **1.05 µs** |

### `contains_miss` — membership, none present

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 3.1 ns | 4.1 ns | **3.1 ns** | 5.8 ns | 4.3 ns | 20 ns | 4.6 ns |
| 16 | 22 ns | 22 ns | 22 ns | 83 ns | 47 ns | 82 ns | **20 ns** |
| 64 | 139 ns | 289 ns | 140 ns | 1.14 µs | 252 ns | 336 ns | **87 ns** |
| 256 | 786 ns | 4.58 µs | 791 ns | 18.27 µs | 1.62 µs | 1.44 µs | **330 ns** |
| 1024 | 4.25 µs | 73.37 µs | 4.25 µs | 254.00 µs | 7.58 µs | 5.85 µs | **1.37 µs** |

### `remove` — remove every element

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 9.0 ns | **3.8 ns** | 9.8 ns | 7.6 ns | 21 ns | 32 ns | 11 ns |
| 16 | 110 ns | **27 ns** | 108 ns | 39 ns | 149 ns | 151 ns | 77 ns |
| 64 | 812 ns | 427 ns | 807 ns | 246 ns | 682 ns | 718 ns | **220 ns** |
| 256 | 4.37 µs | 12.37 µs | 4.37 µs | 2.67 µs | 3.33 µs | 2.92 µs | **802 ns** |
| 1024 | 37.08 µs | 74.89 µs | 36.87 µs | 37.99 µs | 14.16 µs | 11.58 µs | **2.71 µs** |

### `iter` — sum every element

| n | pouch Sorted | pouch Unsorted | sorted-vec | vecmap-rs | BTreeSet | HashSet | FxHashSet |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0.8 ns | **0.7 ns** | 0.8 ns | 1.1 ns | 6.5 ns | 1.6 ns | 1.2 ns |
| 16 | **0.8 ns** | 0.8 ns | **0.8 ns** | 0.8 ns | 31 ns | 6.9 ns | 6.8 ns |
| 64 | **3.2 ns** | 3.3 ns | 3.2 ns | 3.2 ns | 194 ns | 29 ns | 30 ns |
| 256 | 12 ns | 13 ns | 12 ns | **12 ns** | 791 ns | 118 ns | 136 ns |
| 1024 | 56 ns | 56 ns | **55 ns** | 56 ns | 3.19 µs | 546 ns | 656 ns |

## Construction strategy (`SortedMap` / `SortedSet` over `Vec`)

The same final contents (distinct keys) built three ways — the payoff of the bulk
constructors over a repeated-`try_insert` loop:

- `insert_loop` — `try_insert` per entry, random input: `O(n²)`, binary-search + tail
  shift each time.
- `try_from_iter` — same random input, append all then one `sort_unstable` + dedup:
  `O(n log n)`. This is what the `build_random` tables above use.
- `from_sorted_iter` — ascending input, append-only, no sort or search: `O(n)`.

### Map (`SortedMap<Vec<(u64, u64)>>`)

| n | insert_loop | try_from_iter | from_sorted_iter |
|--:|--:|--:|--:|
| 4 | 24 ns | 13 ns | **12 ns** |
| 16 | 182 ns | 104 ns | **73 ns** |
| 64 | 1.09 µs | 386 ns | **155 ns** |
| 256 | 7.56 µs | 1.35 µs | **372 ns** |
| 1024 | 55.29 µs | 5.92 µs | **979 ns** |

### Set (`SortedSet<Vec<u64>>`)

| n | insert_loop | try_from_iter | from_sorted_iter |
|--:|--:|--:|--:|
| 4 | 14 ns | 15 ns | **10 ns** |
| 16 | 157 ns | 127 ns | **76 ns** |
| 64 | 901 ns | 307 ns | **156 ns** |
| 256 | 4.75 µs | 1.17 µs | **333 ns** |
| 1024 | 35.04 µs | 5.67 µs | **885 ns** |

At `n = 1024` the bulk constructors beat the insert-per-element loop by **~6×**
(`try_from_iter`) and **~40×** (`from_sorted_iter`) for the set — ~9× / ~56× for the map.
A 1024-key sorted set builds in ~0.89 µs from already-sorted input versus ~35 µs one at
a time.

## Fixed-capacity / `no_std` (capacity matched to `n`)

Inline storage: pouch over `heapless::Vec` vs [micromap](https://crates.io/crates/micromap)
and `heapless::LinearMap`.

### Maps

**`build`**

| n | pouch Unsorted/heapless | heapless::LinearMap | micromap |
|--:|--:|--:|--:|
| 4 | **6.1 ns** | 8.3 ns | 14 ns |
| 16 | **54 ns** | 59 ns | 55 ns |
| 64 | 723 ns | 1.34 µs | **364 ns** |
| 256 | 9.35 µs | 9.00 µs | **4.96 µs** |

**`get_hit`**

| n | pouch Unsorted/heapless | heapless::LinearMap | micromap |
|--:|--:|--:|--:|
| 4 | **3.3 ns** | 4.0 ns | **3.3 ns** |
| 16 | **43 ns** | 47 ns | 46 ns |
| 64 | 671 ns | **572 ns** | 1.27 µs |
| 256 | 9.96 µs | **9.04 µs** | **9.04 µs** |

**`get_miss`**

| n | pouch Unsorted/heapless | heapless::LinearMap | micromap |
|--:|--:|--:|--:|
| 4 | 6.9 ns | 6.7 ns | **4.2 ns** |
| 16 | 77 ns | **77 ns** | 83 ns |
| 64 | **1.01 µs** | 1.05 µs | 1.12 µs |
| 256 | **16.74 µs** | 19.08 µs | 16.83 µs |

### Sets

**`build`**

| n | pouch Unsorted/heapless | pouch Sorted/heapless | micromap |
|--:|--:|--:|--:|
| 4 | 4.4 ns | 16 ns | **3.5 ns** |
| 16 | **28 ns** | 113 ns | 30 ns |
| 64 | **302 ns** | 838 ns | 445 ns |
| 256 | **2.46 µs** | 4.71 µs | 6.00 µs |

**`contains_hit`**

| n | pouch Unsorted/heapless | pouch Sorted/heapless | micromap |
|--:|--:|--:|--:|
| 4 | 4.3 ns | 3.1 ns | **2.6 ns** |
| 16 | **17 ns** | 21 ns | 47 ns |
| 64 | 164 ns | **139 ns** | 609 ns |
| 256 | 3.37 µs | **781 ns** | 8.85 µs |

**`contains_miss`**

| n | pouch Unsorted/heapless | pouch Sorted/heapless | micromap |
|--:|--:|--:|--:|
| 4 | 5.8 ns | **3.1 ns** | 3.8 ns |
| 16 | 23 ns | **21 ns** | 84 ns |
| 64 | 289 ns | **142 ns** | 1.00 µs |
| 256 | 4.62 µs | **781 ns** | 16.79 µs |

## Backend sweep — same `SortedSet` op, every backend

Big-O is identical across backends (each store is a contiguous array); only the
constant moves. `Vec` pays an allocation that inline stores skip at small `n`, and
the gap closes as `n` grows.

### `build` — sorted insert, random order

| n | Vec | SmallVec | ArrayVec | heapless::Vec |
|--:|--:|--:|--:|--:|
| 16 | 187 ns | 160 ns | 135 ns | **134 ns** |
| 64 | 1.04 µs | 1.02 µs | 1.03 µs | **994 ns** |
| 256 | **5.46 µs** | 5.71 µs | 5.71 µs | 5.58 µs |

### `contains_hit` — membership, all present

| n | Vec | SmallVec | ArrayVec | heapless::Vec |
|--:|--:|--:|--:|--:|
| 16 | **23 ns** | 27 ns | 24 ns | 23 ns |
| 64 | **145 ns** | 151 ns | 149 ns | 148 ns |
| 256 | **807 ns** | 828 ns | 838 ns | 838 ns |

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
| `ColumnMap` | 526 B | +534 B |
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

