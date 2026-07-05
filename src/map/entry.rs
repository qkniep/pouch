//! The [`Entry`] API for [`SortedMap`](crate::SortedMap) and
//! [`UnsortedMap`](crate::UnsortedMap): **one** key lookup for a
//! read-modify-write, instead of a separate `get` then `try_insert` (each its
//! own `O(log n)` / `O(n)` search).
//!
//! `map.entry(key)` resolves the slot once and hands back an [`Entry`] borrowing
//! the map. The split mirrors the crate's fallible model: insertion through a
//! vacant entry is `try_*` and returns the rejected `(key, value)` on a bounded
//! store at capacity, while the infallible `or_insert` / `insert` are gated on an
//! [`Unbounded`] store — exactly as a map's own `insert` / `try_insert` are.
//!
//! One [`Entry`] type serves both maps; they differ only in how an occupied entry
//! is removed (order-preserving shift for sorted, `O(1)` swap-remove for
//! unsorted), selected when the entry is built.

use core::fmt;

use crate::error::CapacityError;
use crate::store::{StoreMut, Unbounded};

/// Whether the parent map keeps key order — selects the removal primitive for an
/// occupied entry (order-preserving `remove_at` vs `O(1)` `swap_remove_at`).
#[derive(Clone, Copy)]
enum Kind {
    Sorted,
    Unsorted,
}

/// A view into a single slot of a map, returned by `entry`.
///
/// Resolving the lookup once, it is either [`Occupied`](Entry::Occupied) (the key is
/// present) or [`Vacant`](Entry::Vacant) (absent, with its insertion point captured).
pub enum Entry<'a, S, K> {
    /// The key is present in the map.
    Occupied(OccupiedEntry<'a, S>),
    /// The key is absent; the entry holds it and the slot it would take.
    Vacant(VacantEntry<'a, S, K>),
}

/// An [`Entry`] for a key that is already in the map.
pub struct OccupiedEntry<'a, S> {
    store: &'a mut S,
    index: usize,
    kind: Kind,
}

/// An [`Entry`] for a key that is not yet in the map; owns the key until inserted.
pub struct VacantEntry<'a, S, K> {
    store: &'a mut S,
    index: usize,
    key: K,
}

impl<'a, S> OccupiedEntry<'a, S> {
    pub(super) fn sorted(store: &'a mut S, index: usize) -> Self {
        OccupiedEntry {
            store,
            index,
            kind: Kind::Sorted,
        }
    }
    pub(super) fn unsorted(store: &'a mut S, index: usize) -> Self {
        OccupiedEntry {
            store,
            index,
            kind: Kind::Unsorted,
        }
    }
}

impl<'a, S, K> VacantEntry<'a, S, K> {
    pub(super) fn new(store: &'a mut S, index: usize, key: K) -> Self {
        VacantEntry { store, index, key }
    }
}

// `into_mut` / `try_insert` project `&'a mut V` out of `Elem = (K, V)`, which
// needs the explicit `K/V: 'a` bounds rustc won't infer through the associated
// type (the same E0311 quirk as `SortedMap::get`).
impl<'a, K, V, S> Entry<'a, S, K>
where
    S: StoreMut<Elem = (K, V)>,
    K: 'a,
    V: 'a,
{
    /// The key this entry resolves, present or not.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => e.key(),
            Entry::Vacant(e) => &e.key,
        }
    }

    /// Runs `f` on the value if the key is present, then returns the entry — for
    /// the update half of an update-or-insert chained before `or_insert`.
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }

    /// Returns the value for the key, inserting `default` if vacant.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying the rejected `(key, default)` only when the
    /// key is absent and a bounded store is at capacity.
    pub fn or_try_insert(self, default: V) -> Result<&'a mut V, CapacityError<(K, V)>> {
        match self {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => e.try_insert(default),
        }
    }

    /// Like [`or_try_insert`](Self::or_try_insert) but computes the default lazily,
    /// so it runs `f` only when the key is absent.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying the rejected `(key, f())` only when the key
    /// is absent and a bounded store is at capacity.
    pub fn or_try_insert_with<F: FnOnce() -> V>(
        self,
        f: F,
    ) -> Result<&'a mut V, CapacityError<(K, V)>> {
        match self {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => e.try_insert(f()),
        }
    }
}

// Infallible inserts — available only when the store can never hit a logical
// cap, mirroring the map's own `insert`.
impl<'a, K, V, S> Entry<'a, S, K>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: 'a,
    V: 'a,
{
    /// Returns the value for the key, inserting `default` if vacant.
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_try_insert(default)
            .unwrap_or_else(|_| unreachable!("Unbounded store reported a capacity failure"))
    }

    /// Returns the value for the key, inserting `f()` if vacant (computed only then).
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        self.or_try_insert_with(f)
            .unwrap_or_else(|_| unreachable!("Unbounded store reported a capacity failure"))
    }

    /// Returns the value for the key, inserting `V::default()` if vacant.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
}

impl<'a, K, V, S> OccupiedEntry<'a, S>
where
    S: StoreMut<Elem = (K, V)>,
    K: 'a,
    V: 'a,
{
    /// Returns the key in this slot.
    pub fn key(&self) -> &K {
        &self.store.as_slice()[self.index].0
    }

    /// Returns a reference to the value.
    pub fn get(&self) -> &V {
        &self.store.as_slice()[self.index].1
    }

    /// Returns a mutable reference to the value, borrowing the entry.
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.store.as_mut_slice()[self.index].1
    }

    /// Returns a mutable reference to the value with the map's lifetime, consuming the
    /// entry.
    pub fn into_mut(self) -> &'a mut V {
        &mut self.store.as_mut_slice()[self.index].1
    }

    /// Replaces the value, returning the old one.
    ///
    /// Consumes no capacity, so it never fails.
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }

    /// Removes the entry, returning its value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry, returning the key and value.
    ///
    /// Order-preserving for a sorted map (`O(n)` shift), `O(1)` swap-remove for an
    /// unsorted one.
    pub fn remove_entry(self) -> (K, V) {
        match self.kind {
            Kind::Sorted => self.store.remove_at(self.index),
            Kind::Unsorted => self.store.swap_remove_at(self.index),
        }
    }
}

impl<'a, K, V, S> VacantEntry<'a, S, K>
where
    S: StoreMut<Elem = (K, V)>,
    K: 'a,
    V: 'a,
{
    /// Returns the key that would be inserted.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes back ownership of the key without inserting.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts `value` for the key and returns a mutable reference to it.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying the rejected `(key, value)` only when a
    /// bounded store is full.
    pub fn try_insert(self, value: V) -> Result<&'a mut V, CapacityError<(K, V)>> {
        let VacantEntry { store, index, key } = self;
        store.try_insert_at(index, (key, value))?;
        Ok(&mut store.as_mut_slice()[index].1)
    }
}

impl<'a, K, V, S> VacantEntry<'a, S, K>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: 'a,
    V: 'a,
{
    /// Inserts `value` for the key and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        self.try_insert(value)
            .unwrap_or_else(|_| unreachable!("Unbounded store reported a capacity failure"))
    }
}

// Lifetime-/bound-free Debug: entries are transient handles, so print the slot,
// not the whole borrowed store.
impl<S> fmt::Debug for OccupiedEntry<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<S, K> fmt::Debug for VacantEntry<'_, S, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantEntry")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<S, K> fmt::Debug for Entry<'_, S, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Occupied(e) => f.debug_tuple("Occupied").field(e).finish(),
            Entry::Vacant(e) => f.debug_tuple("Vacant").field(e).finish(),
        }
    }
}
