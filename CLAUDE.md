# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pouch` is a `no_std`-first Rust library of small, fast, **backend-generic** sets and
maps. A collection is generic over its backing store, so the same set/map logic runs
over `Vec`, `SmallVec`, `TinyVec`, `ArrayVec`, or `heapless::Vec` — heap, inline, or
hybrid — optionally bounded by a runtime cap. Two collection flavors:
`SortedSet`/`SortedMap` (order kept in the store, `O(log n)` lookup) and
`UnsortedSet`/`UnsortedMap` (`O(1)` insert/delete, `O(n)` search, only `Eq` required).
`Bag` is the unconstrained sequence facade (duplicates, insertion order, no `Eq`/`Ord`
bound) that gives composed stores an ergonomic `Vec`-shaped API. `UnsortedColumnMap` and
`SortedColumnMap` (both behind the `soa` feature) are struct-of-arrays variants of
`UnsortedMap` / `SortedMap` (keys and values in two parallel stores) — see the layout
note below.

Module layout (modern `foo.rs` + `foo/` style, no `mod.rs` files):

```
src/lib.rs            facade: crate docs, no_std setup, `pub use` re-exports, type aliases
src/error.rs          CapacityError, BuildError, SortedBuildError
src/store.rs          Store / StoreMut / StoreNew / Unbounded traits  (pub mod store)
src/store/capped.rs   Capped<S> adapter
src/store/backend.rs  mostly cfg-gated `mod vec; mod smallvec; …` — impls only, nothing exported
src/store/backend/*   one file per backend (slice, vec, smallvec, tinyvec, arrayvec, heapless)
src/set.rs            SortedSet, UnsortedSet
src/set/algebra.rs    merge-based set algebra: Union/Intersection/… iterators, subset/disjoint predicates
src/map.rs            SortedMap, UnsortedMap  (+ src/map/entry.rs: Entry over Elem = (K, V))
src/bag.rs            Bag (unconstrained Vec-shaped sequence facade over any store)
src/serde_impls.rs    cfg(serde): Serialize/Deserialize for every collection
src/column_map.rs     cfg(soa): UnsortedColumnMap (struct-of-arrays unsorted map — two stores; + column_map/entry.rs: ColumnEntry)
src/sorted_column_map.rs  cfg(soa): SortedColumnMap (struct-of-arrays sorted map — two stores)
tests/smoke.rs        integration tests
tests/properties.rs   property-based differential tests (proptest vs BTreeMap/BTreeSet/Vec oracles)
```

`lib.rs` re-exports everything to the crate root, so the public API is flat
(`pouch::SortedSet`, `pouch::Capped`, …); `store` is the one module exposed publicly,
for backend authors. The collection layer is thin but now has bulk constructors
(`try_from_iter` / `try_from_sorted_iter` / `from_sorted_iter`, plus `try_extend` and
`Unbounded`-gated `FromIterator`/`Extend`) and an `Entry` API on every map
(`map.entry(k)`; one lookup for insert-or-update, with the infallible `or_insert`
gated on `Unbounded` and `or_try_insert` everywhere, mirroring `insert`/`try_insert`).
The single-store maps use `Entry` (`src/map/entry.rs`, over `Elem = (K, V)`); the
two-store column maps use the parallel `ColumnEntry` (`src/column_map/entry.rs`, over
two stores, `or_insert` gated on **both** columns being `Unbounded`, one combined-cap
pre-check on a vacant insert). Lookups take std-style borrowed keys (`K: Borrow<Q>`,
`Q: Ord/Eq + ?Sized` — `get("k")` on a `Map<String, _>` allocates nothing), each map
routing every key match through one private `search`/`position`. Unsized *range*
bounds need the tuple-of-`Bound`s shape (`range::<str, _>((Bound::Included("a"), …))`)
— range sugar like `"a".."m"` only bounds `&str` itself, exactly as in `BTreeMap`.
Every collection exposes its store read-only (`store()` / `into_store()`; the column
maps `stores()` / `into_stores()`) — backend introspection like `SmallVec::spilled()`
or `Spill::is_spilled` without a mutable path that could break invariants — and a
`reserve(additional)` that forwards the `StoreMut::reserve` growth hint (see below).
The **sorted** flavors derive `Hash`/`PartialOrd`/`Ord` off their canonical stored
order (structural = semantic, same argument as their `PartialEq`), so a `Set<u32>`
can key a `HashMap` or live in a `BTreeSet`; the unsorted flavors can't derive any
of these (swap-remove makes stored order incidental). This relies on the store's
own `Eq`/`Ord`/`Hash` behaving like its `as_slice()` — true of every backend, and
`Spill` implements them manually over `as_slice()` for exactly this reason (which
tier the elements live in is position, not content; a structural derive would call
a spilled-then-shrunk store unequal to a never-spilled twin). Caveat: `SortedColumnMap`'s
derived order compares column-wise (all keys, then all values), not entry-interleaved
like `SortedMap`. `SortedSet` has merge-based set algebra (`union` / `intersection` /
`difference` / `symmetric_difference` iterators plus `is_subset` / `is_superset` /
`is_disjoint`, `src/set/algebra.rs`): `O(n + m)` two-pointer walks over the sorted
slices, cross-store (`other` may use a different backend — e.g. union with a `SliceSet`
flash table), with the predicates switching to `O(n log m)` binary searches when one
side is ≥16× smaller (BTreeSet's adaptivity). `UnsortedSet` gets only the three
predicates (`O(n·m)` scans — no order, no merge). `MapIter` is a handwritten struct,
NOT an `iter::Map<_, fn(…)>` alias — naming that alias forces a function *pointer*,
which can survive to codegen as an indirect call in hot loops. Serde support lives
behind the `serde` feature (`src/serde_impls.rs`): sets/bags serialize as sequences
and maps as maps, and deserialization routes through the same `try_from_iter`
builders as everything else — sets dedup, **maps reject duplicate keys** (a
deliberate difference from std's silently-last-wins serde impls), a bounded store
filling mid-stream is a `de::Error` (bounded deserialization = input validation),
and wire-claimed lengths are capped before reaching `reserve` (serde's "cautious"
policy). `Capped`/`ScratchVec`-backed collections serialize but can't deserialize
(no `StoreNew` — they need runtime state); build via `from_store` + `try_extend`.
Comparators are the planned next step.

## Commands

`just check` mirrors the core CI (fmt, clippy, build, test, doc, deny, machete, typos);
run it before pushing. `just setup` installs the dev tools. Individual recipes:

```sh
just test                            # nextest (--all-features) + doctests, all --locked
just fmt-fix                         # apply nightly rustfmt (config requires nightly)
just clippy                          # clippy --all-targets --all-features -D warnings
just hack                            # cargo-hack feature powerset (--no-dev-deps; see gotcha)
cargo nextest run <test_name>        # run one test, e.g. try_insert_at_shifts_into_position
```

Because behavior is gated behind features, a green default run is **not** sufficient.
`just hack` (the feature powerset) is the real gate; when touching trait impls or
feature gates, also spot-check single backends:

```sh
cargo build --no-default-features                                  # core-only path — must always compile
cargo nextest run --lib --no-default-features --features alloc     # Vec + Capped, no inline backends
cargo nextest run --lib --no-default-features --features heapless  # fixed-cap, fully alloc-free
```

The `--lib` is required: it scopes the run to the in-crate unit tests, which gate
themselves per backend with `#[cfg(feature = …)]`. `tests/smoke.rs` and
`tests/properties.rs` name every backend ungated, so their `[[test]]` entries in
`Cargo.toml` carry `required-features` for the full feature set — under any
partial set (the default included, now that defaults are lean) cargo
**silently skips** the target rather than failing to build it. A green partial
run therefore says nothing about those suites; only the all-features run
(`just test`, CI's test job) executes them.

`tests/properties.rs` is the property-based layer and owns the **semantics**:
differential op sequences checked step-by-step against std oracles (`Vec` for
the store contract, `BTreeSet`/`BTreeMap` for the collections — insert/remove/
capacity behavior, duplicates-consume-no-capacity, column length-lock), plus
set-algebra, bulk-builder, and serde-policy properties. A new backend earns its
correctness argument with one line in the `store_contract!` list; do **not**
add per-backend collection instantiations — those are one per *behavior class*
({unbounded, bounded, hybrid} × {sorted, unsorted}), because the collection
layer is generic over `Store` and can't vary by backend. Example tests (the
unit modules and `smoke.rs`) deliberately cover only what the harness doesn't
drive — trait wiring (`Extend`/`FromIterator`, infallible `insert`/`or_insert`),
iterators, borrowed forms, `range`, entry variants, panic-safety, debug guards —
and each partial-feature config keeps at least one executing test module so the
spot-check commands above stay meaningful; don't add example tests for
capacity/duplicate semantics. On failure proptest writes a seed file next to
the source (`tests/properties.proptest-regressions`) — commit it; it replays
the exact case first on every future run.

## Architecture — the three orthogonal axes

The core design separates three concerns that other crates usually fuse. Keep them
separate when extending:

1. **Storage** — *where* elements live (heap / inline / hybrid). This is the `Store` /
   `StoreMut` / `StoreNew` trait family, implemented once per backend.
2. **Bound** — the max logical element count, exposed as `Store::capacity() -> Option<usize>`
   (`None` = unbounded). A runtime bound is added orthogonally via the `Capped<S>` wrapper,
   *not* per backend.
3. **Ordering** — sorted vs unsorted. This lives in the **collection** layer, never in
   the store. Stores are ordering-agnostic and only deal in indices. Sorted variants use
   `binary_search` + a shifting `try_insert_at`/`remove_at`; unsorted variants append
   (`try_insert_at(len, …)`) and `swap_remove_at`, trading `O(n)` search for `O(1)`
   structural mutation and needing only `Eq` on the element instead of `Ord`.

`swap_remove_at` is a provided `StoreMut` method (last-element swap then tail removal),
so it is `O(1)` on every backend and no backend implements it. It is the unsorted
collections' delete primitive; sorted collections must not use it (it breaks order).

**Layout — the one deliberate exception.** Every collection above holds exactly *one*
`Store` (`Elem = T` for sets, `(K, V)` for maps); `UnsortedColumnMap` (`src/column_map.rs`) is
the sole exception, holding *two* length-locked stores — `keys: SK` (`Elem = K`) and
`values: SV` (`Elem = V`). This is struct-of-arrays: a key lookup scans a dense `[K]`
slice instead of reading the key out of every `(K, V)` pair, so it never loads the
value payloads. That stacks **two** wins. First, the dense scan vectorizes: `get`/`remove`
locate the key via `chunked_position`, a fixed-trip OR-reduction LLVM folds to
branchless/SIMD compares (which the strided `(K, V)` scan can't) — a ~2× edge over a plain
`iter().position()` even on word-sized values, across all `n`, sharpest on misses. (An
earlier `contains`-based bench overstated the raw effect, but the shipping *index*-returning
`get`/`remove` keep the ~2× — see `benches/soa.rs` `locate`.) Second, the scan never pulls
value payloads through cache, a bandwidth saving ≈ proportional to `sizeof(V)/sizeof(K)`
that stacks on top for large values once the map outgrows cache (a further ~2× for 64-byte
values at `n ≥ 4k`). The cost is paid in API, not invariants: no
`as_slice() -> &[(K, V)]` (use `keys()`/`values()`), `from_store` takes two stores, and
`capacity()` is the `min` of the two columns' bounds. SoA can't be a `Store` (the
`as_slice -> &[Elem]` contract is AoS by definition), so it must live as a two-store
collection. There are **two** of these (both behind the `soa` feature): `UnsortedColumnMap`
(unsorted) and its sorted twin `SortedColumnMap` (`src/sorted_column_map.rs`). The sorted
twin earns its keep only for
*large* values — the strided `(K, V)` binary search drags value bytes through cache,
the dense `[K]` search does not (~1.2–1.3× at `sizeof(V)/sizeof(K) ≳ 4`); for word-sized
values it gains little, and at small `n` it *loses* on hits (the value now lives in a
separate cache line), so `SortedMap` stays the default. Both pre-check the combined cap
on insert so neither column half-inserts (no rollback). They differ on delete: `UnsortedColumnMap`
swap-removes the same index in both columns (`O(1)`, order-free); `SortedColumnMap` must
shift both in lockstep to keep keys sorted (`O(n)`).

### Store trait layer (the contract every backend implements)

- `try_insert_at(i, value)` is the **single universal mutation primitive**. Everything
  else (sorted insert, dedup) is built on it in the collection layer. Its `Err` arm
  returns the rejected element via `CapacityError<T>` and is reachable for fixed-cap
  backends *and* for any store wrapped in `Capped`.
- `StoreNew` is kept separate from `Default` on purpose: `Capped` needs a runtime cap and
  so must be excluded from no-argument construction (use `Capped::with_capacity` /
  `from_store`).
- `StoreMut::reserve(additional)` is a **defaulted no-op growth hint**, not a separate
  trait: its promise is "no reallocation before `len + additional`", which stores that
  never reallocate (fixed-cap, borrowed) satisfy trivially — and a default method is the
  only stable-Rust way for `append_all` (bound: `StoreMut`) to consult `size_hint`.
  Growable backends override it natively; `Capped` clamps the request to its remaining
  logical headroom; `Spill` pre-arms the *spill* tier when the projected length
  overflows the inline tier (so the migration itself allocates nothing).
- `Unbounded` is a marker trait. It is the gate that lets the collection layer expose an
  **infallible** `insert` (see `SortedSet::insert`). Implement it ONLY for genuinely
  unbounded growable backends (`Vec`, `SmallVec`, `TinyVec`). Fixed-cap backends
  (`ArrayVec`, `heapless::Vec`) and **anything wrapped in `Capped`** must NOT be `Unbounded`.

### Adding a new backend

Add `src/store/backend/<name>.rs` implementing `Store`, `StoreMut`, `StoreNew` (and
`Unbounded` only if genuinely unbounded), gate it on a feature with one line in
`src/store/backend.rs` (`#[cfg(feature = "<name>")] mod <name>;`), and add the feature
to `Cargo.toml`. The feature gate lives on the `mod` line, so the file itself needs no
per-item `#[cfg]`. Prefer a backend's **native** shifting insert/remove (one memmove)
when it has one — every current backend does, including `heapless::Vec`
(`Vec::insert`/`Vec::remove`). For a genuinely **push-only** store, synthesize
`try_insert_at`/`remove_at` with `push`/`pop` + `rotate_right(1)`/`rotate_left(1)` — but
note that rotate-by-one still monomorphizes core's general `ptr_rotate` (hundreds of
bytes of flash), so it's a fallback, not the default. `src/store/backend/heapless.rs`
documents both in its module comment.

A backend may be **read-only**: implement `Store` alone and skip `StoreMut` /
`StoreNew` / `Unbounded`. `src/store/backend/slice.rs` (`&[T]` and `&[T; N]`) is the
one shipped example — it backs lookups (`contains` / `get`) but no mutation, reports
`capacity() == Some(len)` (a borrowed slice is permanently full), needs no
dependency or `alloc`, and is therefore the **sole ungated** `mod` in
`backend.rs` (usable even under `--no-default-features`). Its headline use is
wrapping a `static` sorted table via `from_store` (`SliceSet` / `SliceMap`) for
zero-alloc lookups out of flash. This is *why* the read-only lookups (`get`,
`contains_key`, and the private `position`/`search`) live in each collection's
`impl<S: Store>` block, **not** the `impl<S: StoreMut>` block — a read-only
backend must reach them. Keep new read-only accessors in the `Store` block;
only `&mut`-returning ones (`get_mut`) belong under `StoreMut`.

## Invariants and gotchas

- **Two distinct "full" conditions** — do not conflate them:
  1. *Logical capacity* (an `ArrayVec`/`heapless` bound, or a `Capped` cap) → recoverable
     `CapacityError`. This is what the crate models.
  2. *Allocator OOM* (a growable backend can't grow) → `Vec::insert` aborts; out of scope.
     Note `Capped<Vec<_>>` is **not** abort-free — it can still OOM below its cap.
- **Duplicates / replacements consume no capacity.** A duplicate set insert or a map-value
  replacement must succeed even when the store is at its bound — it errors only on a
  genuinely new element. Preserve this when adding collection methods.
- **Bulk construction is built on the O(1)-append primitive, not a new `Store` method.**
  `try_insert_at(len, v)` is amortized `O(1)` on every backend (a native insert at `len`
  shifts nothing; a push-only fallback's `rotate_right(1)` runs over a 1-element tail, a
  no-op), so `try_from_iter` (append-all →
  `sort_unstable` → swap-compact dedup, `O(n log n)`) and `try_from_sorted_iter` /
  `from_sorted_iter` (append-only, `O(n)`) live entirely in the collection layer. Use
  `sort_unstable` (alloc-free, in `core`), never stable `sort` (alloc-gated); dedup by
  swap-compaction so no `Copy` bound is needed. Caveat: the unsorted builder appends
  *before* deduping, so a bounded store can overflow on the raw count even when the deduped
  result would fit — `try_insert` in a loop keeps the "dups consume no capacity" guarantee;
  the bulk builders trade it for speed.
- **Sets dedup, maps reject — the bulk-build duplicate policy.** A set duplicate is
  unambiguous, so set builders silently drop it. A map duplicate *key* is ambiguous (which
  value wins?), so the map builders (`try_from_iter` / `try_from_sorted_iter`) return
  `BuildError::DuplicateKey` rather than pick arbitrarily — `try_from_sorted_iter` detects
  it *before* the append, so a dup never consumes a slot. The sequential ops (`try_insert`,
  `try_extend`, `Extend`) stay **last-wins**. Maps therefore expose **no `FromIterator`**
  (it can't be fallible); sets do. `FromIterator`, `Extend`, and the infallible
  `from_sorted_iter` are `Unbounded`-gated, mirroring `insert`.
- **`no_std`-first.** `lib.rs` is `#![no_std]`; `alloc`/`std` are pulled in only behind
  features. Don't reach for `std` in core logic. `std` exists mainly so `CapacityError`
  can implement `std::error::Error`.
- **`Capped::capacity()` returns the *effective* cap** = `min(our cap, inner's own bound)`,
  so capping an already-bounded store does the expected thing.
- **Map lifetime quirk (E0311):** returning `&V` projected from `Elem = (K, V)` needs
  explicit `K: 'a, V: 'a` bounds — rustc won't infer implied bounds through the
  associated-type projection. See `SortedMap::get`; expect to repeat this as the map API grows.
- **Lints are enforced (`Cargo.toml [lints]`, CI uses `-D warnings`).** `unsafe_code` is
  `forbid`; `missing_debug_implementations` is on, so every new public type needs
  `#[derive(Debug)]`; a public `len` needs an `is_empty` and a public `new` needs a
  `Default` (clippy). New public collection structs should mirror their sorted/unsorted twin.
- **Feature powerset uses `--no-dev-deps`.** It checks the public feature surface in
  isolation, catching a missing `#[cfg(feature = …)]` gate that a dev-dependency could
  otherwise mask. Dev targets aren't the reason (smoke and every bench carry
  `required-features`, so partial sets skip rather than break them) and aren't covered:
  `just hack` and `feature-powerset.yml` check the library surface only.

## Feature flags (`Cargo.toml`)

`default = ["std", "smallvec"]` — deliberately lean (enough for the blessed `Set`/`Map`
aliases, no more); the other backends are opt-in. `std → alloc`; `smallvec`/`tinyvec`
imply `alloc`; `arrayvec`/`heapless` are alloc-free. Each optional backend dependency is
gated by the matching feature and pulled in with `default-features = false`. Two more
features carry no backend: `soa` gates the struct-of-arrays column maps
(`UnsortedColumnMap` / `SortedColumnMap`, backend-agnostic), and `serde` gates the
`Serialize`/`Deserialize` impls (`src/serde_impls.rs`, `no_std`).
