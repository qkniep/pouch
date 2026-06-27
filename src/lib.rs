//! `pouch` — small, fast, backend-generic sets and maps.
//!
//! A `pouch` is a small container that holds whatever you put in it regardless of
//! the backing store: `Vec`, `SmallVec`, `TinyVec`, `ArrayVec`, or `heapless::Vec`,
//! optionally bounded by a runtime cap. `no_std`-first.
//!
//! Three orthogonal axes are separated deliberately, one per module:
//!   * **storage**   — where elements live (heap / inline / hybrid): the [`Store`] trait
//!     family, in [`store`].
//!   * **bound**     — max logical element count: `Store::capacity() -> Option<usize>`,
//!     with [`Capped`] adding a runtime bound to any store.
//!   * **ordering**  — sorted vs unsorted, in the collection layer: [`SortedSet`] /
//!     [`UnsortedSet`] and [`SortedMap`] / [`UnsortedMap`], NOT the store.
//!
//! [`ColumnMap`] is a struct-of-arrays variant of the unsorted map: keys and values
//! live in two parallel stores for a denser key scan that vectorizes (~2× over a
//! plain index scan, even for word-sized values) and skips value payloads (a further
//! win for large values once the map outgrows cache), trading the `&[(K, V)]` view.
//! [`SortedColumnMap`] is its sorted twin — the two collections backed by two stores.
//!
//! `try_insert_at` is the single universal mutation primitive. Its `Err` arm is
//! reachable for fixed-capacity backends (arrayvec / heapless) *and* for growable
//! backends wrapped in [`Capped`]. The [`Unbounded`] marker is what lets a
//! collection expose an infallible `insert`.
//!
//! NOTE on the two "full"s:
//!   1. *logical capacity* (arrayvec/heapless bound, or a `Capped` cap) -> recoverable
//!      [`CapacityError`], modelled here.
//!   2. *allocator OOM* (a growable backend cannot grow) -> `Vec::insert` aborts; only
//!      `try_reserve` surfaces it. Out of scope. A `Capped<Vec<_>>` is NOT abort-free; it
//!      can still OOM below its cap.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod column_map;
mod error;
mod map;
mod set;
mod sorted_column_map;
pub mod store;

// ---------------------------------------------------------------------------
// Ergonomic type aliases (they span both the store and collection layers)
// ---------------------------------------------------------------------------
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "arrayvec")]
use arrayvec::ArrayVec;
pub use column_map::{ColumnEntry, ColumnMap, OccupiedColumnEntry, VacantColumnEntry};
pub use error::{BuildError, CapacityError, SortedBuildError};
pub use map::{Entry, OccupiedEntry, SortedMap, UnsortedMap, VacantEntry};
pub use set::{SortedSet, UnsortedSet};
#[cfg(feature = "smallvec")]
use smallvec::SmallVec;
pub use sorted_column_map::SortedColumnMap;
pub use store::{Capped, Store, StoreMut, StoreNew, Unbounded};

#[cfg(feature = "alloc")]
pub type VecSet<T> = SortedSet<Vec<T>>;
#[cfg(feature = "alloc")]
pub type VecMap<K, V> = SortedMap<Vec<(K, V)>>;
#[cfg(feature = "alloc")]
pub type CappedVecSet<T> = SortedSet<Capped<Vec<T>>>;

/// The recommended default set: a sorted set that keeps its elements **inline**
/// (no heap allocation) until it outgrows `N`, then spills to the heap. This is the
/// pick for the case the crate is built for — many small sets nested inside a larger
/// structure (`Vec<Set<_>>`, adjacency lists, per-key buckets), where avoiding a heap
/// allocation *per inner set* is the win.
///
/// `Set<T>` just works; `Set<T, N>` tunes the inline capacity. Keep `N` small —
/// `size_of::<Set<T, N>>` grows with `N · size_of::<T>()`, and you may have millions
/// of these. Reach for [`VecSet`] for a single large set, or [`ArraySet`] /
/// [`HeaplessSet`] for a hard cap with no allocator.
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

/// A sorted set inline up to `N`, spilling to the heap beyond it. [`Set`] is this
/// with a default `N`; name `SmallSet` when you want the capacity explicit.
#[cfg(feature = "smallvec")]
pub type SmallSet<T, const N: usize> = SortedSet<SmallVec<[T; N]>>;
/// A sorted map inline up to `N`, spilling to the heap beyond it. [`Map`] is this
/// with a default `N`.
#[cfg(feature = "smallvec")]
pub type SmallMap<K, V, const N: usize> = SortedMap<SmallVec<[(K, V); N]>>;

#[cfg(feature = "arrayvec")]
pub type ArraySet<T, const N: usize> = SortedSet<ArrayVec<T, N>>;

#[cfg(feature = "heapless")]
pub type HeaplessSet<T, const N: usize> = SortedSet<heapless::Vec<T, N>>;

#[cfg(feature = "alloc")]
pub type UnsortedVecSet<T> = UnsortedSet<Vec<T>>;
#[cfg(feature = "alloc")]
pub type UnsortedVecMap<K, V> = UnsortedMap<Vec<(K, V)>>;
/// A [`ColumnMap`] with both columns heap-backed by `Vec` — the struct-of-arrays
/// unsorted map (keys and values in separate allocations).
#[cfg(feature = "alloc")]
pub type ColumnVecMap<K, V> = ColumnMap<Vec<K>, Vec<V>>;
/// A [`SortedColumnMap`] with both columns heap-backed by `Vec` — the sorted
/// struct-of-arrays map (keys and values in separate allocations).
#[cfg(feature = "alloc")]
pub type SortedColumnVecMap<K, V> = SortedColumnMap<Vec<K>, Vec<V>>;
