//! The [`ColumnEntry`] API for [`UnsortedColumnMap`](crate::UnsortedColumnMap) and
//! [`SortedColumnMap`](crate::SortedColumnMap): **one** key lookup for a
//! read-modify-write, the two-store (struct-of-arrays) counterpart of the
//! single-store [`Entry`](crate::Entry).
//!
//! `map.entry(key)` resolves the slot once and hands back a [`ColumnEntry`]
//! borrowing *both* columns. The split mirrors the crate's fallible model:
//! insertion through a vacant entry is `try_*` and returns the rejected
//! `(key, value)` when the combined cap is hit, while the infallible
//! `or_insert` / `insert` are gated on **both** columns being [`Unbounded`] —
//! exactly as a column map's own `extend` is.
//!
//! One [`ColumnEntry`] type serves both maps; they differ only in how an
//! occupied slot is vacated — an order-preserving lockstep `remove_at` for the
//! sorted map, an `O(1)` lockstep `swap_remove_at` for the unsorted one —
//! selected when the entry is built. A vacant insert is uniform: a single
//! `try_insert_at(index, …)` into each column at the slot the lookup captured
//! (`len` for the unsorted append, the sort position for the sorted shift),
//! guarded by one combined-cap pre-check so neither column half-inserts.

use core::fmt;

use super::combined_capacity;
use crate::error::CapacityError;
use crate::store::{StoreMut, Unbounded};

/// Whether the parent map keeps key order — selects the removal primitive for an
/// occupied entry (order-preserving `remove_at` vs `O(1)` `swap_remove_at`),
/// applied in lockstep to both columns.
#[derive(Clone, Copy)]
enum Kind {
    Sorted,
    Unsorted,
}

/// A view into a single slot of a column map, returned by `entry`. Resolving the
/// lookup once, it is either [`Occupied`](ColumnEntry::Occupied) (the key is
/// present) or [`Vacant`](ColumnEntry::Vacant) (absent, with its insertion point
/// captured). The two-column counterpart of [`Entry`](crate::Entry).
pub enum ColumnEntry<'a, SK, SV, K> {
    /// The key is present in the map.
    Occupied(OccupiedColumnEntry<'a, SK, SV>),
    /// The key is absent; the entry holds it and the slot it would take.
    Vacant(VacantColumnEntry<'a, SK, SV, K>),
}

/// A [`ColumnEntry`] for a key that is already in the map.
pub struct OccupiedColumnEntry<'a, SK, SV> {
    keys: &'a mut SK,
    values: &'a mut SV,
    index: usize,
    kind: Kind,
}

/// A [`ColumnEntry`] for a key that is not yet in the map; owns the key until
/// inserted.
pub struct VacantColumnEntry<'a, SK, SV, K> {
    keys: &'a mut SK,
    values: &'a mut SV,
    index: usize,
    key: K,
}

impl<'a, SK, SV> OccupiedColumnEntry<'a, SK, SV> {
    pub(crate) fn sorted(keys: &'a mut SK, values: &'a mut SV, index: usize) -> Self {
        OccupiedColumnEntry {
            keys,
            values,
            index,
            kind: Kind::Sorted,
        }
    }
    pub(crate) fn unsorted(keys: &'a mut SK, values: &'a mut SV, index: usize) -> Self {
        OccupiedColumnEntry {
            keys,
            values,
            index,
            kind: Kind::Unsorted,
        }
    }
}

impl<'a, SK, SV, K> VacantColumnEntry<'a, SK, SV, K> {
    pub(crate) fn new(keys: &'a mut SK, values: &'a mut SV, index: usize, key: K) -> Self {
        VacantColumnEntry {
            keys,
            values,
            index,
            key,
        }
    }
}

impl<'a, K, V, SK, SV> ColumnEntry<'a, SK, SV, K>
where
    SK: StoreMut<Elem = K>,
    SV: StoreMut<Elem = V>,
    K: 'a,
    V: 'a,
{
    /// The key this entry resolves, present or not.
    pub fn key(&self) -> &K {
        match self {
            ColumnEntry::Occupied(e) => e.key(),
            ColumnEntry::Vacant(e) => &e.key,
        }
    }

    /// Run `f` on the value if the key is present, then return the entry — for
    /// the update half of an update-or-insert chained before `or_insert`.
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let ColumnEntry::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }

    /// The value for the key, inserting `default` if vacant. `Err` (carrying the
    /// rejected `(key, default)`) only when the combined cap is hit.
    pub fn or_try_insert(self, default: V) -> Result<&'a mut V, CapacityError<(K, V)>> {
        match self {
            ColumnEntry::Occupied(e) => Ok(e.into_mut()),
            ColumnEntry::Vacant(e) => e.try_insert(default),
        }
    }

    /// Like [`or_try_insert`](Self::or_try_insert) but computes the default
    /// lazily, so it runs `f` only when the key is absent.
    pub fn or_try_insert_with<F: FnOnce() -> V>(
        self,
        f: F,
    ) -> Result<&'a mut V, CapacityError<(K, V)>> {
        match self {
            ColumnEntry::Occupied(e) => Ok(e.into_mut()),
            ColumnEntry::Vacant(e) => e.try_insert(f()),
        }
    }
}

// Infallible inserts — available only when **both** columns can never hit a
// logical cap, mirroring the map's own `extend`.
impl<'a, K, V, SK, SV> ColumnEntry<'a, SK, SV, K>
where
    SK: StoreMut<Elem = K> + Unbounded,
    SV: StoreMut<Elem = V> + Unbounded,
    K: 'a,
    V: 'a,
{
    /// The value for the key, inserting `default` if vacant.
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_try_insert(default)
            .unwrap_or_else(|_| unreachable!("Unbounded columns reported a capacity failure"))
    }

    /// The value for the key, inserting `f()` if vacant (computed only then).
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        self.or_try_insert_with(f)
            .unwrap_or_else(|_| unreachable!("Unbounded columns reported a capacity failure"))
    }

    /// The value for the key, inserting `V::default()` if vacant.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
}

impl<'a, K, V, SK, SV> OccupiedColumnEntry<'a, SK, SV>
where
    SK: StoreMut<Elem = K>,
    SV: StoreMut<Elem = V>,
    K: 'a,
    V: 'a,
{
    /// The key in this slot.
    pub fn key(&self) -> &K {
        &self.keys.as_slice()[self.index]
    }

    /// A reference to the value.
    pub fn get(&self) -> &V {
        &self.values.as_slice()[self.index]
    }

    /// A mutable reference to the value, borrowing the entry.
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.values.as_mut_slice()[self.index]
    }

    /// A mutable reference to the value with the map's lifetime, consuming the entry.
    pub fn into_mut(self) -> &'a mut V {
        &mut self.values.as_mut_slice()[self.index]
    }

    /// Replace the value, returning the old one. Consumes no capacity, so it
    /// never fails.
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }

    /// Remove the entry, returning its value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Remove the entry, returning the key and value. Both columns are vacated
    /// in lockstep so they stay aligned: an order-preserving `remove_at`
    /// (`O(n)` shift) for a sorted map, an `O(1)` `swap_remove_at` for an
    /// unsorted one.
    pub fn remove_entry(self) -> (K, V) {
        match self.kind {
            Kind::Sorted => (
                self.keys.remove_at(self.index),
                self.values.remove_at(self.index),
            ),
            Kind::Unsorted => (
                self.keys.swap_remove_at(self.index),
                self.values.swap_remove_at(self.index),
            ),
        }
    }
}

impl<'a, K, V, SK, SV> VacantColumnEntry<'a, SK, SV, K>
where
    SK: StoreMut<Elem = K>,
    SV: StoreMut<Elem = V>,
    K: 'a,
    V: 'a,
{
    /// The key that would be inserted.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Take back ownership of the key without inserting.
    pub fn into_key(self) -> K {
        self.key
    }

    /// Insert `value` for the key and return a mutable reference to it. `Err`
    /// (carrying the rejected `(key, value)`) only when the combined cap is hit.
    ///
    /// The columns are length-locked, so one pre-check against the combined
    /// bound guarantees both inserts below succeed — no half-insert, no
    /// rollback (the same guard as the map's own `try_insert`).
    pub fn try_insert(self, value: V) -> Result<&'a mut V, CapacityError<(K, V)>> {
        let VacantColumnEntry {
            keys,
            values,
            index,
            key,
        } = self;
        if let Some(cap) = combined_capacity(keys.capacity(), values.capacity()) {
            if keys.len() >= cap {
                return Err(CapacityError((key, value)));
            }
        }
        keys.try_insert_at(index, key)
            .expect("capacity pre-checked above");
        values
            .try_insert_at(index, value)
            .expect("capacity pre-checked above");
        Ok(&mut values.as_mut_slice()[index])
    }
}

impl<'a, K, V, SK, SV> VacantColumnEntry<'a, SK, SV, K>
where
    SK: StoreMut<Elem = K> + Unbounded,
    SV: StoreMut<Elem = V> + Unbounded,
    K: 'a,
    V: 'a,
{
    /// Insert `value` for the key and return a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        self.try_insert(value)
            .unwrap_or_else(|_| unreachable!("Unbounded columns reported a capacity failure"))
    }
}

// Lifetime-/bound-free Debug: entries are transient handles, so print the slot,
// not the whole borrowed columns.
impl<SK, SV> fmt::Debug for OccupiedColumnEntry<'_, SK, SV> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedColumnEntry")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<SK, SV, K> fmt::Debug for VacantColumnEntry<'_, SK, SV, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantColumnEntry")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl<SK, SV, K> fmt::Debug for ColumnEntry<'_, SK, SV, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnEntry::Occupied(e) => f.debug_tuple("Occupied").field(e).finish(),
            ColumnEntry::Vacant(e) => f.debug_tuple("Vacant").field(e).finish(),
        }
    }
}
