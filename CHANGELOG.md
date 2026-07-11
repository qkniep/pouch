# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-11

Initial release. MSRV: Rust **1.87**.

### Added

Store layer:

- `Store` / `StoreMut` / `StoreNew` / `Unbounded` traits — the backend contract;
  implement them once to plug a new backing store into every collection.
- Backends: `Vec`, `SmallVec`, `TinyVec`, `ArrayVec`, and `heapless::Vec` (each
  behind the feature of the same name), plus the always-available borrowed
  stores — read-only `&[T]` / `&[T; N]`, and `ScratchVec` over a `&mut [T]`
  scratch buffer.
- `Capped<S>` wrapper adding a runtime logical-capacity bound to any store.
- `Spill<A, B>` two-tier store: fills tier `A`, then spills into tier `B`
  (e.g. inline array → borrowed buffer).

Collections:

- `SortedSet` / `SortedMap` — order kept in the store, `O(log n)` lookup.
- `UnsortedSet` / `UnsortedMap` — `O(1)` insert/delete (append + swap-remove),
  `O(n)` search, requiring only `Eq` rather than `Ord`.
- `Bag` — unconstrained `Vec`-shaped sequence (duplicates kept, insertion
  order, no `Eq`/`Ord` bound); the ergonomic facade for composed stores.
- `UnsortedColumnMap` / `SortedColumnMap` (feature `soa`) — struct-of-arrays
  maps holding keys and values in two parallel stores, so lookups scan or
  binary-search a dense key column without pulling values through cache.
- Type aliases: `Set` / `Map` (the blessed inline-then-heap defaults over
  `SmallVec`) and `SliceSet` / `SliceMap` (read-only lookup tables over
  borrowed sorted slices, e.g. `static` tables in flash).

Collection API:

- Fallible-first mutation: `try_insert` hands back the rejected element via
  `CapacityError` on bounded stores; the infallible `insert` is gated on the
  `Unbounded` marker. Duplicates and value replacements consume no capacity.
- Bulk constructors (`try_from_iter` / `try_from_sorted_iter` /
  `from_sorted_iter`) and `try_extend`, with `Unbounded`-gated `FromIterator` /
  `Extend`. Sets dedup; the map builders reject duplicate keys
  (`BuildError::DuplicateKey`) instead of picking a winner.
- `Entry` API on every map (`ColumnEntry` on the column maps) — one lookup for
  insert-or-update, with `or_try_insert` everywhere and `or_insert` gated on
  `Unbounded`, mirroring `insert` / `try_insert`.
- std-style borrowed-key lookups (`K: Borrow<Q>`) and `range` on the sorted
  flavors.
- Merge-based set algebra on `SortedSet` — `union` / `intersection` /
  `difference` / `symmetric_difference` iterators and `is_subset` /
  `is_superset` / `is_disjoint` predicates, `O(n + m)`, cross-backend;
  `UnsortedSet` gets the three predicates.
- `Hash` / `PartialOrd` / `Ord` on the sorted flavors (off their canonical
  stored order), so a set can key a `HashMap` or live in a `BTreeSet`.
- Backend introspection and preallocation: `store()` / `into_store()`
  (`stores()` / `into_stores()` on the column maps) and `reserve(additional)`.
- `clear` on every collection, and `get_mut` on every map — an in-place value
  update without the `entry` ceremony.

Serde (feature `serde`):

- `Serialize` / `Deserialize` for every collection — sets and bags as
  sequences, maps as maps. Deserialization enforces the bulk-build policy:
  sets dedup, maps reject duplicate keys, and a bounded store filling
  mid-stream is a data error, so deserializing into a fixed-capacity
  collection is input validation for free.

[0.1.0]: https://github.com/qkniep/pouch/releases/tag/v0.1.0
