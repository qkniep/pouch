# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Store` / `StoreMut` / `StoreNew` / `Unbounded` traits abstracting over the
  `Vec`, `SmallVec`, `TinyVec`, `ArrayVec`, and `heapless::Vec` backends.
- `Capped<S>` wrapper adding a runtime logical-capacity bound to any store.
- `SortedSet` / `SortedMap` — order kept in the store, `O(log n)` lookup.
- `UnsortedSet` / `UnsortedMap` — `O(1)` insert/delete (append + swap-remove),
  `O(n)` search, requiring only `Eq` rather than `Ord`.
- `ColumnMap` — struct-of-arrays unsorted map: keys and values in two parallel
  stores, for a dense, value-free key scan (faster lookups for large values).
- `SortedColumnMap` — the sorted struct-of-arrays map (binary search over a dense
  `[K]` column); the large-value lookup win with key-ordered iteration.
- Bulk constructors (`try_from_iter` / `try_from_sorted_iter` / `from_sorted_iter`)
  and `try_extend`, with `Unbounded`-gated `FromIterator` / `Extend`.
- `clear` on every collection, and `get_mut` on every map — an in-place value
  update without the `entry` ceremony.

### Changed

- Raised the minimum supported Rust version to **1.78** — where `std`'s
  `slice::binary_search_by` gained the cmov-based branchless bisection that the
  sorted collections' lookups ride on.
