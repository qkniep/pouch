//! Sorted column-oriented (struct-of-arrays) map — the **layout** variant of
//! the sorted map.
//!
//! [`SortedMap`](crate::SortedMap) keeps `Elem = (K, V)` pairs interleaved in
//! one key-sorted store; [`SortedColumnMap`] instead keeps keys and values in
//! *two* parallel, length-locked stores — `keys: SK` (`Elem = K`, kept sorted)
//! and `values: SV` (`Elem = V`) — so a lookup binary-searches a dense `[K]`
//! slice rather than striding over `(K, V)` pairs. It is the sorted twin of
//! [`UnsortedColumnMap`](crate::UnsortedColumnMap) and the struct-of-arrays twin of
//! [`SortedMap`](crate::SortedMap).
//!
//! Its payoff is **narrow**, and worth stating plainly (measured against
//! `SortedMap`, `target-cpu=native`):
//!   * **large values win.** For `sizeof(V)/sizeof(K) ≳ 4` (e.g. 32–64-byte values) it
//!     runs ~1.2–1.3× faster on both hits and misses at every `n`: the strided `(K, V)`
//!     binary search drags value bytes through cache, the dense `[K]` search does not.
//!   * **word values are a wash — or a small-`n` *loss* on hits.** With word-sized values
//!     a hit must fetch the value from the *separate* column (a second cache line) where
//!     `SortedMap` has it co-located beside the key, so at small `n` `SortedMap` is the
//!     faster of the two on hits; the split only repays on misses (no value load) or at
//!     large `n`.
//!
//! So reach for this only when values are large **and** you need key-ordered
//! iteration **and** lookups dominate; otherwise prefer
//! [`SortedMap`](crate::SortedMap). If you do not
//! need ordering, [`UnsortedColumnMap`](crate::UnsortedColumnMap) is the unsorted SoA
//! map.
//!
//! Same two-store API trade as [`UnsortedColumnMap`](crate::UnsortedColumnMap): no
//! `as_slice() -> &[(K, V)]` (enumerate via [`keys`](SortedColumnMap::keys) /
//! [`values`](SortedColumnMap::values)),
//! [`from_store`](SortedColumnMap::from_store) takes two stores, and
//! [`capacity`](SortedColumnMap::capacity) is the `min` of the two columns'
//! bounds. Unlike `UnsortedColumnMap`'s O(1) swap-remove, the order-preserving
//! [`try_insert`](SortedColumnMap::try_insert) /
//! [`remove`](SortedColumnMap::remove) shift *both* columns in lockstep (`O(log
//! n)` search, `O(n)` shift).

use core::borrow::Borrow;

use crate::column_map::{combined_capacity, ColumnEntry, OccupiedColumnEntry, VacantColumnEntry};
use crate::error::{BuildError, CapacityError};
use crate::store::{Store, StoreMut, StoreNew, Unbounded};

/// A map kept sorted by key, stored **column-wise**: keys in `SK` (sorted),
/// values in `SV`, kept the same length so `keys[i]` pairs with `values[i]`.
/// The struct-of-arrays counterpart of [`SortedMap`](crate::SortedMap) — trades
/// the `&[(K, V)]` view for a dense key column the binary search strides
/// through without touching values (a win only for large values; see the module
/// docs). Needs `K: Ord`.
// The stored order is canonical (sorted by key, unique keys), so the structural
// derives are the semantic ones, as for `SortedMap` — this map can key another
// map or live in a `BTreeSet`. One caveat: the derived `PartialOrd`/`Ord`
// compare **column-wise** (all keys, then all values) — a valid total order
// consistent with `Eq`, but not the entry-interleaved order of the AoS
// `SortedMap`/`BTreeMap`; don't expect the two flavors to sort collections of
// maps identically.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SortedColumnMap<SK, SV> {
    keys: SK,
    values: SV,
}

impl<SK: StoreNew, SV: StoreNew> SortedColumnMap<SK, SV> {
    pub fn new() -> Self {
        SortedColumnMap {
            keys: SK::new(),
            values: SV::new(),
        }
    }
}

impl<SK: StoreNew, SV: StoreNew> Default for SortedColumnMap<SK, SV> {
    fn default() -> Self {
        Self::new()
    }
}

impl<SK: Store, SV: Store> SortedColumnMap<SK, SV> {
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Returns the effective logical capacity: the `min` of the two columns' bounds
    /// (`None` = unbounded). Capping either column caps the map.
    pub fn capacity(&self) -> Option<usize> {
        combined_capacity(self.keys.capacity(), self.values.capacity())
    }
    /// Returns the keys as a contiguous slice, in ascending order — the dense search
    /// target. `zip` with [`values`](Self::values) to iterate entries by
    /// key.
    pub fn keys(&self) -> &[SK::Elem] {
        self.keys.as_slice()
    }
    /// Returns the values as a contiguous slice, index-aligned with
    /// [`keys`](Self::keys) (so also in ascending key order).
    pub fn values(&self) -> &[SV::Elem] {
        self.values.as_slice()
    }
    /// Borrows the two backing stores, `(keys, values)` — the door to
    /// backend-specific introspection (`spilled()`, allocated capacity, …), as
    /// [`SortedSet::store`](crate::SortedSet::store) is for the single-store
    /// collections. Shared-ref only: `&mut` access could desync the columns or
    /// unsort the keys.
    pub fn stores(&self) -> (&SK, &SV) {
        (&self.keys, &self.values)
    }
    /// Consumes the map and hands back its stores, `(keys, values)`, entries
    /// intact, index-aligned, and still in ascending key order — the inverse
    /// of [`from_store`](Self::from_store).
    pub fn into_stores(self) -> (SK, SV) {
        (self.keys, self.values)
    }
}

impl<SK: StoreMut, SV: StoreMut> SortedColumnMap<SK, SV> {
    /// Removes every entry, clearing both columns and keeping their allocated
    /// capacity. Needs no `Ord` bound — it only truncates the stores.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
    /// Pre-allocates both columns so at least `additional` more entries fit
    /// without a reallocation — see
    /// [`SortedSet::reserve`](crate::SortedSet::reserve).
    pub fn reserve(&mut self, additional: usize) {
        self.keys.reserve(additional);
        self.values.reserve(additional);
    }
}

impl<K, V, SK, SV> SortedColumnMap<SK, SV>
where
    SK: Store<Elem = K>,
    SV: Store<Elem = V>,
    K: Ord,
{
    /// Wraps two stores **assumed equal-length, with keys sorted ascending and
    /// free of duplicates** — the sorted-column-map invariants. No scan,
    /// sort, or alignment is performed; a length mismatch would desync
    /// key/value pairs and an out-of-order or duplicate-keyed column yields
    /// wrong lookups. Both preconditions are `debug_assert!`-checked (zero
    /// cost in release). For a runtime-checked ascending
    /// build use [`try_from_sorted_iter`](Self::try_from_sorted_iter); to build
    /// from arbitrary input use [`try_from_iter`](Self::try_from_iter).
    pub fn from_store(keys: SK, values: SV) -> Self {
        debug_assert_eq!(
            keys.len(),
            values.len(),
            "SortedColumnMap::from_store: key and value columns must have equal length",
        );
        debug_assert!(
            keys.as_slice().windows(2).all(|w| w[0] < w[1]),
            "SortedColumnMap::from_store: keys must be sorted and free of duplicate keys",
        );
        SortedColumnMap { keys, values }
    }

    /// Binary searches the dense key column. `Ok(i)` is the index of the matching
    /// entry; `Err(i)` is the insertion point that keeps the column sorted.
    /// Every key lookup — `get`, `contains_key`, `try_insert`, `remove`,
    /// the builders — routes through this one search, so they can never
    /// disagree on which entry is "the one for this key" (and the
    /// `Borrow`-keyed match — and any future comparator — lands in exactly
    /// one place).
    fn search<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.keys
            .as_slice()
            .binary_search_by(|k| k.borrow().cmp(key))
    }

    // No E0311 lifetime dance here (unlike `SortedMap::get`): `values.as_slice()`
    // is already `&[V]`, so projecting `&V` needs no associated-type-projection
    // bound — elision ties the result to `&self`.
    //
    // `key` may be any borrowed form of `K` (a `String`-keyed column map answers
    // `get("k")` without allocating), with the usual `Borrow` contract that the
    // borrowed form's `Ord` agrees with `K`'s.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.search(key).ok().map(|i| &self.values.as_slice()[i])
    }

    /// Returns `true` if `key` is present. `O(log n)`. Unlike
    /// [`UnsortedColumnMap`](crate::UnsortedColumnMap), which scans its key column with a
    /// chunked boolean fold, the sorted layout shares the `O(log
    /// n)` binary search with [`get`](Self::get): a linear scan would
    /// forfeit the very `O(log n)` the ordering buys. `key` may be any
    /// borrowed form of `K`, like [`get`](Self::get).
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.search(key).is_ok()
    }
}

impl<K, V, SK, SV> SortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K>,
    SV: StoreMut<Elem = V>,
    K: Ord,
{
    /// Returns a mutable reference to `key`'s value, or `None` if absent — for an
    /// in-place update without the [`entry`](Self::entry) ceremony. `O(log n)`. No
    /// E0311 lifetime dance (unlike
    /// [`SortedMap::get_mut`](crate::SortedMap::get_mut)): the value column is
    /// already `&mut [V]`, so elision ties the result to `&mut self`. `key` may be
    /// any borrowed form of `K`, like [`get`](Self::get).
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let i = self.search(key).ok()?;
        Some(&mut self.values.as_mut_slice()[i])
    }

    /// Inserts a brand-new entry at index `i` in both columns, keeping them
    /// aligned, or hand it back at capacity. The columns are length-locked,
    /// so a single pre-check against the combined bound guarantees both
    /// inserts below succeed — no half-insert, no rollback. `i == len` is
    /// the O(1) tail append; `i < len` shifts.
    fn insert_entry_at(&mut self, i: usize, key: K, value: V) -> Result<(), CapacityError<(K, V)>> {
        if let Some(cap) = self.capacity() {
            if self.keys.len() >= cap {
                return Err(CapacityError((key, value)));
            }
        }
        self.keys
            .try_insert_at(i, key)
            .expect("capacity pre-checked above");
        self.values
            .try_insert_at(i, value)
            .expect("capacity pre-checked above");
        Ok(())
    }

    /// Inserts or replaces, preserving key order. Replacing an existing key
    /// touches only the value column, consumes no capacity, and so can
    /// never fail — only a genuinely new key at the bound errors. `O(log
    /// n)` search, `O(n)` shift to make room (or `O(1)` to replace in
    /// place).
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError<(K, V)>> {
        match self.search(&key) {
            Ok(i) => {
                let slot = &mut self.values.as_mut_slice()[i];
                Ok(Some(core::mem::replace(slot, value)))
            }
            Err(i) => self.insert_entry_at(i, key, value).map(|()| None),
        }
    }

    /// Removes the entry for `key`, returning its value. Order-preserving shift
    /// in *both* columns (not the swap-remove
    /// [`UnsortedColumnMap`](crate::UnsortedColumnMap) can use): `O(log n)` search,
    /// `O(n)` shift. `key` may be any borrowed form of `K`, like
    /// [`get`](Self::get).
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.search(key) {
            Ok(i) => {
                self.keys.remove_at(i);
                Some(self.values.remove_at(i))
            }
            Err(_) => None,
        }
    }

    /// Resolves `key`'s slot **once** and returns a [`ColumnEntry`] for an
    /// insert-or-update, avoiding the second binary search a separate
    /// [`get`](Self::get) + [`try_insert`](Self::try_insert) would pay. `O(log
    /// n)` to locate; a vacant entry inserts at the sort position (`O(n)`
    /// lockstep shift), an occupied one removes the same way (order preserved).
    pub fn entry(&mut self, key: K) -> ColumnEntry<'_, SK, SV, K> {
        match self.search(&key) {
            Ok(index) => ColumnEntry::Occupied(OccupiedColumnEntry::sorted(
                &mut self.keys,
                &mut self.values,
                index,
            )),
            Err(index) => ColumnEntry::Vacant(VacantColumnEntry::new(
                &mut self.keys,
                &mut self.values,
                index,
                key,
            )),
        }
    }

    /// Inserts every entry, one at a time, **last-wins** (a repeated key
    /// replaces the earlier value rather than erroring). `O(k·n)`. To
    /// reject duplicate keys instead, build a fresh map with
    /// [`try_from_iter`](Self::try_from_iter).
    ///
    /// On overflow only the one rejected entry is recoverable: the iterator is
    /// dropped along with any entries it has not yet yielded.
    pub fn try_extend<I>(&mut self, iter: I) -> Result<(), CapacityError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        for (key, value) in iter {
            self.try_insert(key, value)?;
        }
        Ok(())
    }
}

impl<K, V, SK, SV> SortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + StoreNew,
    SV: StoreMut<Elem = V> + StoreNew,
    K: Ord,
{
    /// Builds from an arbitrary (unordered) iterator of entries, **requiring
    /// every key to be unique**. `O(n²)`: each entry is binary-searched
    /// against the keys already placed and inserted in order (an
    /// order-preserving shift, like a one-at-a-time
    /// [`try_insert`](Self::try_insert)). A repeated key is rejected — a map
    /// can't drop a duplicate key without arbitrarily picking a value. For
    /// last-wins override semantics use [`try_extend`](Self::try_extend) /
    /// `extend`.
    ///
    /// Unlike [`SortedMap::try_from_iter`](crate::SortedMap::try_from_iter)
    /// (append-all then sort once, `O(n log n)`), two parallel columns
    /// cannot be co-sorted without a scratch buffer, so this stays `O(n²)`
    /// — matching
    /// [`UnsortedColumnMap::try_from_iter`](crate::UnsortedColumnMap::try_from_iter). The
    /// upside: a duplicate key is caught *before* it is inserted, so it
    /// never consumes capacity even on a bounded store.
    pub fn try_from_iter<I>(iter: I) -> Result<Self, BuildError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut map = Self::new();
        for (key, value) in iter {
            match map.search(&key) {
                Ok(_) => return Err(BuildError::DuplicateKey((key, value))),
                // CapacityError -> BuildError::Capacity via `From`.
                Err(i) => map.insert_entry_at(i, key, value)?,
            }
        }
        Ok(map)
    }

    /// Builds from an iterator whose entries are already in ascending key order,
    /// in `O(n)` — no search, no shifting, just a tail append per entry
    /// into both columns. Like [`try_from_iter`](Self::try_from_iter) it
    /// requires unique keys, and it detects a duplicate (or a misordered
    /// key) *before* the append, so either is rejected even at capacity
    /// (neither consumes a slot).
    ///
    /// Unlike [`from_store`](Self::from_store), the ascending-order promise is
    /// enforced in every build profile: a key smaller than its predecessor
    /// is returned as [`BuildError::Unsorted`] rather than silently
    /// trusted.
    pub fn try_from_sorted_iter<I>(iter: I) -> Result<Self, BuildError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut map = Self::new();
        for (key, value) in iter {
            if let Some(prev) = map.keys.as_slice().last() {
                if key < *prev {
                    return Err(BuildError::Unsorted((key, value)));
                }
                if *prev == key {
                    return Err(BuildError::DuplicateKey((key, value)));
                }
            }
            let i = map.keys.len();
            map.insert_entry_at(i, key, value)?; // tail append, O(1)
        }
        Ok(map)
    }
}

impl<K, V, SK, SV> SortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + Unbounded,
    SV: StoreMut<Elem = V> + Unbounded,
    K: Ord,
{
    /// Infallibly inserts or replaces, returning the previous value — available
    /// only when **both** columns are [`Unbounded`]. The infallible twin of
    /// [`try_insert`](Self::try_insert).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.try_insert(key, value) {
            Ok(prev) => prev,
            Err(_) => unreachable!("Unbounded columns reported a capacity failure"),
        }
    }
}

impl<K, V, SK, SV> Extend<(K, V)> for SortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + Unbounded,
    SV: StoreMut<Elem = V> + Unbounded,
    K: Ord,
{
    /// Extends the map, last-wins and infallible — available only when **both** columns
    /// are [`Unbounded`]. As with [`SortedMap`](crate::SortedMap), there is
    /// deliberately no `FromIterator`: fresh construction rejects duplicate
    /// keys, while `extend` overrides them.
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        match self.try_extend(iter) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded columns reported a capacity failure"),
        }
    }
}

// Vec is the unbounded backend, so the `Unbounded`-gated `extend` and the
// last-wins / strict-build distinction run here.
#[cfg(all(test, feature = "alloc"))]
mod alloc_tests {
    use alloc::vec::Vec;

    use crate::error::BuildError;
    use crate::{ColumnEntry, SortedColumnMap};

    #[test]
    fn insert_keeps_order_get_and_replace() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<&str>> = SortedColumnMap::new();
        assert_eq!(m.try_insert(2, "b"), Ok(None));
        assert_eq!(m.try_insert(1, "a"), Ok(None));
        // Inserts shift to keep the key column sorted (not appended like UnsortedColumnMap).
        assert_eq!(m.keys(), &[1, 2]);
        assert_eq!(m.values(), &["a", "b"]);
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&9), None);
        // Replacing a key returns the old value and adds no entry.
        assert_eq!(m.try_insert(1, "A"), Ok(Some("a")));
        assert_eq!(m.get(&1), Some(&"A"));
        assert_eq!(m.len(), 2);
        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&9));
    }

    #[test]
    fn remove_shifts_and_keeps_columns_aligned() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<&str>> = SortedColumnMap::new();
        m.try_extend([(3, "c"), (1, "a"), (2, "b")]).unwrap();
        assert_eq!(m.keys(), &[1, 2, 3]); // sorted regardless of insertion order
        assert_eq!(m.values(), &["a", "b", "c"]);
        // Order-preserving shift (not swap) in BOTH columns.
        assert_eq!(m.remove(&1), Some("a"));
        assert_eq!(m.keys(), &[2, 3]);
        assert_eq!(m.values(), &["b", "c"]);
        assert_eq!(m.get(&3), Some(&"c"));
        assert_eq!(m.get(&1), None);
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn try_from_iter_sorts_and_rejects_duplicate_key() {
        let m: SortedColumnMap<Vec<i32>, Vec<&str>> =
            SortedColumnMap::try_from_iter([(3, "c"), (1, "a"), (2, "b")]).unwrap();
        assert_eq!(m.keys(), &[1, 2, 3]);
        assert_eq!(m.get(&2), Some(&"b"));
        // A duplicate key is caught at the search, before insertion: the second
        // occurrence is handed back.
        let err =
            SortedColumnMap::<Vec<i32>, Vec<&str>>::try_from_iter([(1, "a"), (2, "b"), (1, "z")])
                .expect_err("duplicate key 1");
        match err {
            BuildError::DuplicateKey(entry) => assert_eq!(entry, (1, "z")),
            BuildError::Capacity(_) | BuildError::Unsorted(_) => {
                panic!("expected a duplicate-key error")
            }
        }
    }

    #[test]
    fn try_from_sorted_iter_rejects_unsorted_and_dup() {
        let m: SortedColumnMap<Vec<i32>, Vec<&str>> =
            SortedColumnMap::try_from_sorted_iter([(1, "a"), (2, "b"), (5, "e")]).unwrap();
        assert_eq!(m.get(&5), Some(&"e"));
        // A key smaller than its predecessor is Unsorted (checked before the dup test).
        let err = SortedColumnMap::<Vec<i32>, Vec<&str>>::try_from_sorted_iter([
            (1, "a"),
            (3, "c"),
            (2, "b"),
        ])
        .expect_err("key 2 after key 3 is descending");
        match err {
            BuildError::Unsorted(entry) => assert_eq!(entry, (2, "b")),
            BuildError::Capacity(_) | BuildError::DuplicateKey(_) => {
                panic!("expected an unsorted error")
            }
        }
        // A duplicate among sorted input is rejected before the append.
        let err2 = SortedColumnMap::<Vec<i32>, Vec<&str>>::try_from_sorted_iter([
            (1, "a"),
            (1, "z"),
            (2, "b"),
        ])
        .expect_err("duplicate key 1");
        assert_eq!(err2.into_inner(), (1, "z"));
    }

    #[test]
    fn extend_is_last_wins() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<&str>> = SortedColumnMap::new();
        m.extend([(1, "a"), (2, "b")]);
        m.extend([(2, "B"), (3, "c")]); // key 2 overridden
        assert_eq!(m.len(), 3);
        assert_eq!(m.keys(), &[1, 2, 3]);
        assert_eq!(m.get(&2), Some(&"B"));
        assert_eq!(m.get(&3), Some(&"c"));
    }

    #[test]
    fn entry_or_insert_inserts_then_updates_in_one_lookup() {
        // The headline use: tally occurrences with a single search per item.
        let mut counts: SortedColumnMap<Vec<&str>, Vec<u32>> = SortedColumnMap::new();
        for w in ["b", "a", "b", "c", "a", "b"] {
            *counts.entry(w).or_insert(0) += 1;
        }
        assert_eq!(counts.get(&"a"), Some(&2));
        assert_eq!(counts.get(&"b"), Some(&3));
        assert_eq!(counts.get(&"c"), Some(&1));
        // Vacant entries insert at the sort position, so the columns stay key-ordered.
        assert_eq!(counts.keys(), &["a", "b", "c"]);
        assert_eq!(counts.values(), &[2, 3, 1]);
    }

    #[test]
    fn entry_and_modify_then_or_insert() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<i32>> = SortedColumnMap::new();
        // Vacant: `and_modify` is a no-op, `or_insert` seeds the value.
        m.entry(1).and_modify(|v| *v += 100).or_insert(10);
        assert_eq!(m.get(&1), Some(&10));
        // Occupied: `and_modify` runs, `or_insert` is ignored.
        m.entry(1).and_modify(|v| *v += 100).or_insert(10);
        assert_eq!(m.get(&1), Some(&110));
    }

    #[test]
    fn entry_occupied_insert_and_remove_keeps_order() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<&str>> =
            SortedColumnMap::try_from_iter([(1, "a"), (2, "b"), (3, "c")]).unwrap();
        match m.entry(2) {
            ColumnEntry::Occupied(mut e) => {
                assert_eq!(e.key(), &2);
                assert_eq!(e.insert("B"), "b"); // replace returns the old value
                assert_eq!(e.remove(), "B"); // then remove it
            }
            ColumnEntry::Vacant(_) => panic!("key 2 is present"),
        }
        // Order-preserving lockstep shift, so both columns stay ascending and aligned.
        assert_eq!(m.keys(), &[1, 3]);
        assert_eq!(m.values(), &["a", "c"]);
    }

    #[test]
    fn entry_vacant_into_key_inserts_nothing() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<&str>> = SortedColumnMap::new();
        match m.entry(7) {
            ColumnEntry::Vacant(e) => {
                assert_eq!(e.key(), &7);
                assert_eq!(e.into_key(), 7); // take the key back without inserting
            }
            ColumnEntry::Occupied(_) => panic!("the map is empty"),
        }
        assert!(m.is_empty());
    }

    #[test]
    fn get_mut_and_clear() {
        let mut m: SortedColumnMap<Vec<i32>, Vec<i32>> =
            SortedColumnMap::try_from_iter([(2, 20), (1, 10)]).unwrap();
        *m.get_mut(&1).unwrap() += 5;
        assert_eq!(m.get(&1), Some(&15));
        assert_eq!(m.get_mut(&9), None);
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.keys(), &[] as &[i32]);
        assert_eq!(m.values(), &[] as &[i32]);
        m.try_insert(3, 30).unwrap(); // both columns usable again
        assert_eq!(m.keys(), &[3]);
    }

    #[test]
    fn clone_and_eq() {
        let a: SortedColumnMap<Vec<i32>, Vec<&str>> =
            SortedColumnMap::try_from_iter([(1, "a"), (2, "b")]).unwrap();
        let mut b = a.clone();
        assert_eq!(a, b); // PartialEq compares both columns
        b.try_insert(3, "c").unwrap();
        assert_ne!(a, b); // the clone is independent
                          // Different build order, same mapping -> equal (canonical key order).
        let c: SortedColumnMap<Vec<i32>, Vec<&str>> =
            SortedColumnMap::try_from_iter([(2, "b"), (1, "a")]).unwrap();
        assert_eq!(a, c);
        // A differing value breaks equality even with identical keys.
        let d: SortedColumnMap<Vec<i32>, Vec<&str>> =
            SortedColumnMap::try_from_iter([(1, "a"), (2, "B")]).unwrap();
        assert_ne!(a, d);
    }

    // The trust-contract guards fire only in debug builds, so gate these on it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "equal length")]
    fn from_store_rejects_unequal_columns() {
        let _ =
            SortedColumnMap::<Vec<i32>, Vec<&str>>::from_store(Vec::from([1, 2]), Vec::from(["a"]));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "sorted and free of duplicate keys")]
    fn from_store_rejects_unsorted_keys() {
        let _ = SortedColumnMap::<Vec<i32>, Vec<&str>>::from_store(
            Vec::from([3, 1]),
            Vec::from(["c", "a"]),
        );
    }
}

// heapless is the alloc-free fixed-cap backend: exercises the bounded paths,
// the capacity pre-check, and the order-preserving insert-at-cap.
#[cfg(all(test, feature = "heapless"))]
mod heapless_tests {
    use heapless::Vec;

    use crate::SortedColumnMap;

    #[test]
    fn capacity_reports_fixed_bound() {
        let m: SortedColumnMap<Vec<u8, 4>, Vec<u8, 4>> = SortedColumnMap::new();
        assert_eq!(m.capacity(), Some(4));
    }

    #[test]
    fn try_insert_overflow_hands_back_the_pair() {
        let mut m: SortedColumnMap<Vec<u8, 2>, Vec<u8, 2>> = SortedColumnMap::new();
        m.try_insert(2, 20).unwrap();
        m.try_insert(3, 30).unwrap();
        // A genuinely new key at the bound errors, returning the whole pair; nothing is
        // half-inserted (both columns stay length 2) — even when it would sort in the
        // middle (the cap is pre-checked before any shift).
        let err = m.try_insert(1, 10).expect_err("at cap 2");
        assert_eq!(err.into_inner(), (1, 10));
        assert_eq!(m.len(), 2);
        assert_eq!(m.keys(), &[2, 3]);
        // A replacement at the bound still succeeds (consumes no capacity).
        assert_eq!(m.try_insert(2, 99), Ok(Some(20)));
    }

    #[test]
    fn entry_or_try_insert_respects_capacity() {
        // Cap 2, full. A bounded column has no infallible `or_insert`;
        // `or_try_insert` updates an occupied slot (no capacity used) but rejects
        // a new key against the combined cap — even one that would sort in the
        // middle (the cap is pre-checked before any shift).
        let mut m: SortedColumnMap<Vec<u8, 2>, Vec<u8, 2>> = SortedColumnMap::new();
        m.try_extend([(2, 20), (3, 30)]).unwrap();

        // Occupied update in place succeeds even at capacity.
        *m.entry(2)
            .or_try_insert(0)
            .expect("update consumes no capacity") = 21;
        assert_eq!(m.get(&2), Some(&21));

        // A genuinely new key at the bound is rejected, handing back `(key, value)`;
        // nothing is half-inserted (both columns stay length 2).
        let err = m.entry(1).or_try_insert(10).expect_err("columns are full");
        assert_eq!(err.into_inner(), (1, 10));
        assert_eq!(m.len(), 2);
        assert_eq!(m.keys(), &[2, 3]);
    }

    #[test]
    fn from_sorted_iter_dup_beats_capacity() {
        // Cap 2 with only one slot used: the dup is rejected as a duplicate, not as a
        // capacity failure — a duplicate key consumes no capacity.
        let err = SortedColumnMap::<Vec<u8, 2>, Vec<u8, 2>>::try_from_sorted_iter([(1, 1), (1, 2)])
            .expect_err("duplicate key 1");
        assert_eq!(err.into_inner(), (1, 2));
    }
}

// Mixed backends: the effective cap is the tighter column's bound. Needs both
// `alloc` (the unbounded value column) and `heapless` (the bounded key column).
#[cfg(all(test, feature = "alloc", feature = "heapless"))]
mod hetero_tests {
    use alloc::vec::Vec;

    use heapless::Vec as HVec;

    use crate::SortedColumnMap;

    #[test]
    fn capacity_is_min_of_the_two_columns() {
        // Bounded keys (cap 2), unbounded values: the map is bounded at 2.
        let mut m: SortedColumnMap<HVec<u8, 2>, Vec<u16>> = SortedColumnMap::new();
        assert_eq!(m.capacity(), Some(2));
        m.try_insert(1, 10).unwrap();
        m.try_insert(2, 20).unwrap();
        let err = m.try_insert(3, 30).expect_err("key column is full at 2");
        assert_eq!(err.into_inner(), (3, 30));
        assert_eq!(m.len(), 2);
    }
}
