//! `pouch` — allocation-avoiding flat sets and maps for small collections:
//! the default [`Set`] / [`Map`] keep their elements **inline** until they
//! outgrow `N`, so many small collections nested in a larger structure
//! (adjacency lists, per-key buckets, `Vec<Set<_>>`) cost about one heap
//! allocation instead of one each.
//!
//! Under the hood the memory strategy is a type parameter: write against one
//! collection API and choose *where the elements live* by naming the backing
//! store: heap (`Vec`), inline (`SmallVec`, `TinyVec`, `ArrayVec`,
//! `heapless::Vec`), borrowed (`&[T]`, [`ScratchVec`]), or a composition
//! ([`Capped`] adds a runtime bound to any store, [`Spill`] chains two tiers).
//! The crate is honest about capacity — on a bounded store every insert is
//! fallible and hands the rejected element back ([`CapacityError`]), while the
//! [`Unbounded`] marker is what unlocks the infallible `insert`; the type
//! system remembers which stores can fail. And none of the core needs an
//! allocator: the crate is `#![no_std]`-first and `#![forbid(unsafe_code)]`,
//! so the fixed-cap and borrowed backends run unchanged on embedded targets.
//!
//! Three orthogonal axes are separated deliberately, one per layer:
//!   * **storage**  — where elements live (heap / inline / hybrid / borrowed): the
//!     [`store::Store`] trait family, implemented once per backend.
//!   * **bound**    — max logical element count: `Store::capacity() -> Option<usize>`,
//!     with [`Capped`] adding a runtime bound to any store.
//!   * **ordering** — sorted vs unsorted, in the collection layer: [`SortedSet`] /
//!     [`UnsortedSet`] and [`SortedMap`] / [`UnsortedMap`], NOT the store.
//!
//! # Picking a store
//!
//! [`Set`] / [`Map`] (inline up to `N`, heap after — the nested-collections
//! default) are the only blessed aliases besides the read-only [`SliceSet`] /
//! [`SliceMap`]. Every other combination is spelled, not named — spelling it
//! *is* the API:
//!
//! | I want… | Spell it |
//! |---|---|
//! | many small sets inside a big structure | `Set<T>`, tuned via `Set<T, N>` (= `SortedSet<SmallVec<[T; N]>>`) |
//! | one large heap set | `SortedSet<Vec<T>>` |
//! | a hard capacity, allocation-free | `SortedSet<ArrayVec<T, N>>` or `SortedSet<heapless::Vec<T, N>>` |
//! | a runtime cap on a growable store | `SortedSet<Capped<Vec<T>>>` |
//! | zero-alloc lookups in a `static` table | [`SliceSet`] / [`SliceMap`] (read-only, in flash) |
//! | inline, overflowing into a borrowed buffer | `SortedSet<Spill<ArrayVec<T, N>, ScratchVec<T>>>` |
//! | `Eq`-only elements, `O(1)` delete | `UnsortedSet<…>` / `UnsortedMap<…>` |
//! | a `Vec`-shaped view of a composed store | [`Bag`], e.g. `Bag<Capped<Vec<T>>>` |
//!
//! The same spellings work for maps — the element type is `(K, V)`:
//! `SortedMap<ArrayVec<(K, V), N>>`, `SortedMap<Capped<Vec<(K, V)>>>`, ….
//!
//! # Specialists
//!
//! Behind the non-default `soa` feature live the struct-of-arrays maps
//! (`UnsortedColumnMap` and its sorted twin `SortedColumnMap`): keys and values
//! in two parallel stores, so a key scan walks a dense `[K]` column and never
//! drags value payloads through cache. Worth it for large values or miss-heavy
//! scans; for everything else [`SortedMap`] is the right default.
//!
//! # NOTE on the two "full"s
//!
//!   1. *logical capacity* (an arrayvec/heapless bound, or a `Capped` cap) -> recoverable
//!      [`CapacityError`], modelled here.
//!   2. *allocator OOM* (a growable backend cannot grow) -> `Vec::insert` aborts; only
//!      `try_reserve` surfaces it. Out of scope. A `Capped<Vec<_>>` is NOT abort-free; it
//!      can still OOM below its cap.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod bag;
#[cfg(feature = "soa")]
mod column_map;
mod error;
mod map;
mod set;
#[cfg(feature = "soa")]
mod sorted_column_map;
pub mod store;

pub use bag::Bag;
#[cfg(feature = "soa")]
pub use column_map::{ColumnEntry, OccupiedColumnEntry, UnsortedColumnMap, VacantColumnEntry};
pub use error::{BuildError, CapacityError};
pub use map::{Entry, MapIter, OccupiedEntry, SortedMap, UnsortedMap, VacantEntry};
pub use set::{SortedSet, UnsortedSet};
#[cfg(feature = "smallvec")]
use smallvec::SmallVec;
#[cfg(feature = "soa")]
pub use sorted_column_map::SortedColumnMap;
// The store *adapters* appear in user-written type signatures
// (`SortedSet<Capped<Vec<T>>>`), so they live at the root alongside `Unbounded`
// (which gates the infallible APIs and shows up in user bounds). The `Store` /
// `StoreMut` / `StoreNew` contract is for backend authors and generic code and
// stays under [`store`] — one canonical path per item.
pub use store::{Capped, ScratchVec, Spill, Unbounded};

// ---------------------------------------------------------------------------
// The blessed type aliases. Deliberately few: `Set`/`Map` name the opinionated
// default (inline-then-heap), `SliceSet`/`SliceMap` name the read-only
// borrowed-table trick that is otherwise hard to discover. Every other
// store/collection combination is spelled out at the use site — see the
// "Picking a store" table in the crate docs.
// ---------------------------------------------------------------------------

/// The recommended default set: a sorted set that keeps its elements **inline**
/// (no heap allocation) until it outgrows `N`, then spills to the heap. This is the
/// pick for the case the crate is built for — many small sets nested inside a larger
/// structure (`Vec<Set<_>>`, adjacency lists, per-key buckets), where avoiding a heap
/// allocation *per inner set* is the win.
///
/// `Set<T>` just works; `Set<T, N>` tunes the inline capacity. Keep `N` small —
/// `size_of::<Set<T, N>>` grows with `N · size_of::<T>()`, and you may have millions
/// of these. Reach for `SortedSet<Vec<T>>` for a single large set, or
/// `SortedSet<ArrayVec<T, N>>` / `SortedSet<heapless::Vec<T, N>>` for a hard cap
/// with no allocator.
///
/// ```
/// use pouch::Set;
/// let mut s: Set<u32> = Set::default();
/// s.insert(2);
/// s.insert(1);
/// s.insert(2); // duplicate
/// assert_eq!(s.as_slice(), &[1, 2]); // sorted, inline, no allocation
/// ```
#[cfg(feature = "smallvec")]
pub type Set<T, const N: usize = 8> = SortedSet<SmallVec<[T; N]>>;
/// The recommended default map — the [`Set`] story for key/value pairs: entries live
/// inline until the map outgrows `N`. `Map<K, V>` just works; `Map<K, V, N>` tunes
/// the inline capacity (keep it small).
#[cfg(feature = "smallvec")]
pub type Map<K, V, const N: usize = 8> = SortedMap<SmallVec<[(K, V); N]>>;

/// A **read-only** sorted set over a borrowed slice — no dependency, no `alloc`,
/// so it works in any build. Wrap an already-sorted, duplicate-free slice (e.g. a
/// `static` table living in flash) via [`SortedSet::from_store`] for zero-alloc
/// `contains` with no copy: `SliceSet::from_store(&TABLE[..])`. It exposes only
/// the read API ([`store::Store`], not [`store::StoreMut`]).
pub type SliceSet<'a, T> = SortedSet<&'a [T]>;
/// A **read-only** sorted map over a borrowed `&[(K, V)]` slice — the [`SliceSet`]
/// story for key/value pairs. Wrap a `static` sorted-by-key table via
/// [`SortedMap::from_store`] for zero-alloc `get`/`contains_key` straight out of
/// flash: `SliceMap::from_store(&TABLE[..])`.
pub type SliceMap<'a, K, V> = SortedMap<&'a [(K, V)]>;
