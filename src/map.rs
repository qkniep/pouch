//! Map collections — the **ordering** axis for `Elem = (K, V)`.
//!
//! [`SortedMap`] keeps its store ordered by key (`O(log n)` lookup);
//! [`UnsortedMap`] appends and swap-removes (`O(1)` mutation, `O(n)` search) and
//! needs only `K: Eq` rather than `K: Ord`.

use crate::error::{BuildError, CapacityError};
use crate::store::{append_all, push, Store, StoreMut, StoreNew, Unbounded};

mod entry;

pub use entry::{Entry, OccupiedEntry, VacantEntry};

/// A map kept sorted by key in its backing store (`Elem = (K, V)`).
#[derive(Debug)]
pub struct SortedMap<S> {
    store: S,
}

impl<S: StoreNew> SortedMap<S> {
    pub fn new() -> Self {
        SortedMap { store: S::new() }
    }
}

impl<S: StoreNew> Default for SortedMap<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Store> SortedMap<S> {
    pub fn len(&self) -> usize {
        self.store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    pub fn capacity(&self) -> Option<usize> {
        self.store.capacity()
    }
    /// The entries as a contiguous `(K, V)` slice, in ascending key order. The
    /// sole iteration accessor: enumerate keys/values via `.iter()`.
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
}

impl<S: StoreMut> SortedMap<S> {
    /// Remove every entry, keeping the backing store's allocated capacity. Needs
    /// no `Ord` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

impl<K, V, S> SortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: Ord,
{
    /// Wrap a store **assumed already sorted by key and free of duplicate keys** —
    /// the invariant `binary_search` (and thus [`get`](Self::get) /
    /// [`try_insert`](Self::try_insert) / [`remove`](Self::remove)) relies on. No
    /// sort is performed; an out-of-order or duplicate-keyed store yields wrong
    /// lookups. The precondition is only `debug_assert!`-checked (zero cost in
    /// release). For a runtime-checked ascending build use
    /// [`try_from_sorted_iter`](Self::try_from_sorted_iter); to build from arbitrary
    /// input use [`try_from_iter`](Self::try_from_iter).
    pub fn from_store(store: S) -> Self {
        debug_assert!(
            store.as_slice().windows(2).all(|w| w[0].0 < w[1].0),
            "SortedMap::from_store: store must be sorted by key and free of duplicate keys",
        );
        SortedMap { store }
    }
}

impl<K, V, S> SortedMap<S>
where
    S: StoreMut<Elem = (K, V)>,
    K: Ord,
{
    // NOTE: returning `&V` derived from the projected `Elem = (K, V)` slice needs
    // an explicit `K/V: 'a` bound — rustc does not infer implied bounds through the
    // associated-type projection (E0311). Worth knowing when you build out the API.
    pub fn get<'a>(&'a self, key: &K) -> Option<&'a V>
    where
        K: 'a,
        V: 'a,
    {
        let s = self.store.as_slice();
        s.binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|i| &s[i].1)
    }

    /// A mutable reference to `key`'s value, or `None` if absent — for an in-place
    /// update without the [`entry`](Self::entry) ceremony. Carries the same
    /// explicit `K/V: 'a` bounds as [`get`](Self::get) (the E0311 quirk).
    pub fn get_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>
    where
        K: 'a,
        V: 'a,
    {
        let i = self
            .store
            .as_slice()
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()?;
        Some(&mut self.store.as_mut_slice()[i].1)
    }

    /// Whether `key` is present. `O(log n)` — like [`get`](Self::get) but yields a
    /// yes/no answer, so it needs no value lifetime.
    pub fn contains_key(&self, key: &K) -> bool {
        self.store
            .as_slice()
            .binary_search_by(|(k, _)| k.cmp(key))
            .is_ok()
    }

    /// Insert or replace. Replacing an existing key consumes no capacity and so
    /// can never fail — only a genuinely new key at the bound errors.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError<(K, V)>> {
        match self.store.as_slice().binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(i) => {
                let slot = &mut self.store.as_mut_slice()[i].1;
                Ok(Some(core::mem::replace(slot, value)))
            }
            Err(i) => self.store.try_insert_at(i, (key, value)).map(|()| None),
        }
    }

    /// Remove the entry for `key`, returning its value. Order-preserving shift:
    /// `O(log n)` search, `O(n)` shift.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.store.as_slice().binary_search_by(|(k, _)| k.cmp(key)) {
            Ok(i) => Some(self.store.remove_at(i).1),
            Err(_) => None,
        }
    }

    /// Resolve `key`'s slot **once** and return an [`Entry`] for an
    /// insert-or-update, avoiding the second search a separate
    /// [`get`](Self::get) + [`try_insert`](Self::try_insert) would pay. `O(log n)`.
    ///
    /// ```
    /// use pouch::Map;
    ///
    /// let mut counts: Map<&str, u32> = Map::default();
    /// for w in ["a", "b", "a", "a"] {
    ///     *counts.entry(w).or_insert(0) += 1; // one lookup per word, not two
    /// }
    /// assert_eq!(counts.get(&"a"), Some(&3));
    /// assert_eq!(counts.get(&"b"), Some(&1));
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<'_, S, K> {
        match self.store.as_slice().binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(index) => Entry::Occupied(OccupiedEntry::sorted(&mut self.store, index)),
            Err(index) => Entry::Vacant(VacantEntry::new(&mut self.store, index, key)),
        }
    }

    /// Insert every entry, one at a time, **last-wins**: a repeated key replaces
    /// the earlier value rather than erroring (so this returns only a
    /// [`CapacityError`], never a duplicate-key error). `O(k·n)`. To instead
    /// reject duplicate keys, build a fresh map with
    /// [`try_from_iter`](Self::try_from_iter).
    ///
    /// On overflow only the one rejected entry is recoverable: the iterator is
    /// dropped along with any entries it has not yet yielded. Drive
    /// [`try_insert`](Self::try_insert) yourself over an iterator you keep if the
    /// unconsumed tail must survive.
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

impl<K, V, S> SortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + StoreNew,
    K: Ord,
{
    /// Build from an arbitrary (unordered) iterator of entries in `O(n log n)`,
    /// **requiring every key to be unique**: append all, sort by key, then reject
    /// any duplicate. Unlike a set's silent dedup, a map can't drop a duplicate
    /// key without arbitrarily choosing which value to keep, so it errors instead
    /// (the second entry of the clashing pair is handed back). For last-wins
    /// override semantics use [`try_extend`](Self::try_extend) / `extend`.
    ///
    /// Errors with [`BuildError::Capacity`] if a bounded store fills. As with the
    /// set builder, entries are appended *before* the dedup/sort pass, so on a
    /// bounded backend a capacity overflow surfaces during the append even if the
    /// final unique map would fit.
    pub fn try_from_iter<I>(iter: I) -> Result<Self, BuildError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut store = S::new();
        append_all(&mut store, iter)?;
        store.as_mut_slice().sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // After sorting, equal keys are adjacent; the first such pair is a dup.
        let dup = store
            .as_slice()
            .windows(2)
            .position(|w| w[0].0 == w[1].0)
            .map(|i| i + 1);
        if let Some(i) = dup {
            return Err(BuildError::DuplicateKey(store.remove_at(i)));
        }
        Ok(Self::from_store(store))
    }

    /// Build from an iterator whose entries are already in ascending key order, in
    /// `O(n)` — no sort. Like [`try_from_iter`](Self::try_from_iter) it requires
    /// unique keys, but it detects a duplicate (and a misordered key) *before* the
    /// append, so either is rejected even at capacity (neither consumes a slot).
    ///
    /// Unlike [`from_store`](Self::from_store), the ascending-order promise is
    /// enforced in every build profile: a key smaller than its predecessor is
    /// returned as [`BuildError::Unsorted`] rather than silently trusted. The check
    /// is one comparison per entry — the same one the dedup already needs.
    pub fn try_from_sorted_iter<I>(iter: I) -> Result<Self, BuildError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut store = S::new();
        for entry in iter {
            if let Some((prev_key, _)) = store.as_slice().last() {
                if entry.0 < *prev_key {
                    return Err(BuildError::Unsorted(entry));
                }
                if *prev_key == entry.0 {
                    return Err(BuildError::DuplicateKey(entry));
                }
            }
            push(&mut store, entry)?;
        }
        Ok(Self::from_store(store))
    }
}

impl<K, V, S> Extend<(K, V)> for SortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: Ord,
{
    /// Last-wins, infallible — available only for an [`Unbounded`] store.
    /// Deliberately no `FromIterator`: fresh construction is strict about
    /// duplicate keys (see [`try_from_iter`](SortedMap::try_from_iter)), whereas
    /// `extend` matches the standard-library override semantics.
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        match self.try_extend(iter) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

/// A map with no key ordering (`Elem = (K, V)`): lookup is a linear scan, insert
/// appends, delete swap-removes. The unsorted counterpart of [`SortedMap`];
/// needs only `K: Eq`, not `K: Ord`.
#[derive(Debug)]
pub struct UnsortedMap<S> {
    store: S,
}

impl<S: StoreNew> UnsortedMap<S> {
    pub fn new() -> Self {
        UnsortedMap { store: S::new() }
    }
}

impl<S: StoreNew> Default for UnsortedMap<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Store> UnsortedMap<S> {
    pub fn len(&self) -> usize {
        self.store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    pub fn capacity(&self) -> Option<usize> {
        self.store.capacity()
    }
    /// The entries as a contiguous `(K, V)` slice, in insertion order modulo the
    /// swaps that [`remove`](Self::remove) performs. The sole iteration accessor:
    /// enumerate keys/values via `.iter()`.
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
}

impl<S: StoreMut> UnsortedMap<S> {
    /// Remove every entry, keeping the backing store's allocated capacity. Needs
    /// no `Eq` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

impl<K, V, S> UnsortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: Eq,
{
    /// Wrap a store **assumed free of duplicate keys** — the map invariant. No scan
    /// is performed; a duplicate key would shadow itself and let the same entry be
    /// removed twice. The precondition is `debug_assert!`-checked (zero cost in
    /// release). To build from arbitrary input, use
    /// [`try_from_iter`](Self::try_from_iter).
    pub fn from_store(store: S) -> Self {
        debug_assert!(
            {
                let s = store.as_slice();
                !(1..s.len()).any(|i| s[..i].iter().any(|(k, _)| *k == s[i].0))
            },
            "UnsortedMap::from_store: store must be free of duplicate keys",
        );
        UnsortedMap { store }
    }
}

impl<K, V, S> UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)>,
    K: Eq,
{
    /// Index of the entry whose key equals `key`, or `None`. Every key lookup —
    /// `get`, `try_insert`, `remove`, `try_from_iter` — routes through this single
    /// scan, so they can never disagree on which entry is "the one for this key"
    /// (and a future `Borrow`/comparator match lands in exactly one place).
    fn position(&self, key: &K) -> Option<usize> {
        self.store.as_slice().iter().position(|(k, _)| k == key)
    }

    pub fn get<'a>(&'a self, key: &K) -> Option<&'a V>
    where
        K: 'a,
        V: 'a,
    {
        self.position(key).map(|i| &self.store.as_slice()[i].1)
    }

    /// A mutable reference to `key`'s value, or `None` if absent — for an in-place
    /// update without the [`entry`](Self::entry) ceremony. Routes through the same
    /// internal linear scan as [`get`](Self::get); carries the same explicit
    /// `K/V: 'a` bounds (the E0311 quirk).
    pub fn get_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>
    where
        K: 'a,
        V: 'a,
    {
        let i = self.position(key)?;
        Some(&mut self.store.as_mut_slice()[i].1)
    }

    /// Whether `key` is present. `O(n)` — routes through the same internal linear
    /// scan as the other lookups, so it stays consistent with [`get`](Self::get).
    pub fn contains_key(&self, key: &K) -> bool {
        self.position(key).is_some()
    }

    /// Insert or replace. Replacing an existing key consumes no capacity and so
    /// can never fail — only a genuinely new key at the bound errors. O(n) lookup,
    /// O(1) to append or replace in place.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError<(K, V)>> {
        if let Some(i) = self.position(&key) {
            let slot = &mut self.store.as_mut_slice()[i].1;
            return Ok(Some(core::mem::replace(slot, value)));
        }
        push(&mut self.store, (key, value)).map(|()| None)
    }

    /// Remove the entry for `key`, returning its value. Swap-remove: O(1), order
    /// not preserved.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let i = self.position(key)?;
        Some(self.store.swap_remove_at(i).1)
    }

    /// Resolve `key`'s slot **once** and return an [`Entry`] for an
    /// insert-or-update, avoiding the second scan a separate [`get`](Self::get) +
    /// [`try_insert`](Self::try_insert) would pay. `O(n)` to locate; an occupied
    /// entry removes via `O(1)` swap (order not preserved).
    pub fn entry(&mut self, key: K) -> Entry<'_, S, K> {
        match self.position(&key) {
            Some(index) => Entry::Occupied(OccupiedEntry::unsorted(&mut self.store, index)),
            None => {
                let index = self.store.len();
                Entry::Vacant(VacantEntry::new(&mut self.store, index, key))
            }
        }
    }

    /// Insert every entry, one at a time, **last-wins** (a repeated key replaces
    /// the earlier value rather than erroring). `O(k·n)`. To reject duplicate keys
    /// instead, build a fresh map with [`try_from_iter`](Self::try_from_iter).
    ///
    /// On overflow only the one rejected entry is recoverable: the iterator is
    /// dropped along with any entries it has not yet yielded. Drive
    /// [`try_insert`](Self::try_insert) yourself over an iterator you keep if the
    /// unconsumed tail must survive.
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

impl<K, V, S> UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + StoreNew,
    K: Eq,
{
    /// Build from an iterator of entries, **requiring every key to be unique**.
    /// `O(n²)`: each entry is scanned against those already kept (an unsorted map
    /// has no faster dedup without `Ord`), and a repeated key is rejected — a map
    /// can't drop a duplicate key without arbitrarily picking a value. For
    /// last-wins override semantics use [`try_extend`](Self::try_extend) / `extend`.
    pub fn try_from_iter<I>(iter: I) -> Result<Self, BuildError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut map = Self::new();
        for (key, value) in iter {
            if map.position(&key).is_some() {
                return Err(BuildError::DuplicateKey((key, value)));
            }
            push(&mut map.store, (key, value))?;
        }
        Ok(map)
    }
}

impl<K, V, S> Extend<(K, V)> for UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: Eq,
{
    /// Last-wins, infallible — available only for an [`Unbounded`] store. As with
    /// [`SortedMap`], there is deliberately no `FromIterator`: fresh construction
    /// rejects duplicate keys, while `extend` overrides them.
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        match self.try_extend(iter) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

// Vec is the unbounded backend, so the `Unbounded`-gated `extend` and the
// last-wins / strict-build distinction both run here.
#[cfg(all(test, feature = "alloc"))]
mod alloc_tests {
    use alloc::vec::Vec;

    use crate::error::BuildError;
    use crate::{Entry, SortedMap, UnsortedMap};

    #[test]
    fn sorted_try_from_iter_unique_keys() {
        let m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(3, "c"), (1, "a"), (2, "b")]).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&2), Some(&"b"));
        assert_eq!(m.get(&3), Some(&"c"));
    }

    #[test]
    fn sorted_try_from_iter_rejects_duplicate_key() {
        let err = SortedMap::<Vec<(i32, &str)>>::try_from_iter([(1, "a"), (2, "b"), (1, "z")])
            .expect_err("duplicate key 1");
        // Unstable sort means *which* of the two value strings is handed back is
        // not promised — only that it's the clashing key. That ambiguity is
        // exactly why construction rejects rather than picks.
        match err {
            BuildError::DuplicateKey(entry) => assert_eq!(entry.0, 1),
            BuildError::Capacity(_) | BuildError::Unsorted(_) => {
                panic!("expected a duplicate-key error")
            }
        }
    }

    #[test]
    fn sorted_from_sorted_iter_rejects_dup_before_capacity() {
        let m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_sorted_iter([(1, "a"), (2, "b"), (5, "e")]).unwrap();
        assert_eq!(m.get(&5), Some(&"e"));

        // Sorted input detects the dup before appending, so the *second* entry is
        // handed back deterministically.
        let err =
            SortedMap::<Vec<(i32, &str)>>::try_from_sorted_iter([(1, "a"), (1, "z"), (2, "b")])
                .expect_err("duplicate key 1");
        assert_eq!(err.into_inner(), (1, "z"));
    }

    #[test]
    fn maps_contains_key() {
        let sm: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(2, "b"), (1, "a")]).unwrap();
        assert!(sm.contains_key(&1));
        assert!(!sm.contains_key(&3));

        let mut um: UnsortedMap<Vec<(i32, &str)>> = UnsortedMap::new();
        um.try_insert(5, "e").unwrap();
        assert!(um.contains_key(&5));
        assert!(!um.contains_key(&6));
    }

    #[test]
    fn sorted_remove_returns_value_and_keeps_order() {
        let mut m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(3, "c"), (1, "a"), (2, "b")]).unwrap();
        assert_eq!(m.remove(&2), Some("b"));
        assert_eq!(m.remove(&2), None); // already gone
        assert_eq!(m.len(), 2);
        // Order is preserved (shift, not swap), so the slice stays ascending.
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&3), Some(&"c"));
    }

    #[test]
    fn sorted_as_slice_is_key_ordered() {
        let m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(3, "c"), (1, "a"), (2, "b")]).unwrap();
        // as_slice yields the entries sorted by key — the only iteration accessor.
        assert_eq!(m.as_slice(), &[(1, "a"), (2, "b"), (3, "c")]);
        let keys: Vec<i32> = m.as_slice().iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, &[1, 2, 3]);
    }

    #[test]
    fn unsorted_as_slice_enumerates_entries() {
        let mut m: UnsortedMap<Vec<(i32, &str)>> = UnsortedMap::new();
        m.try_insert(1, "a").unwrap();
        m.try_insert(2, "b").unwrap();
        m.try_insert(3, "c").unwrap();
        // Insertion order is preserved until a swap-remove reshuffles it.
        assert_eq!(m.as_slice(), &[(1, "a"), (2, "b"), (3, "c")]);
        assert_eq!(m.remove(&1), Some("a")); // last entry swaps into slot 0
        assert_eq!(m.as_slice(), &[(3, "c"), (2, "b")]);
    }

    #[test]
    fn extend_is_last_wins() {
        let mut m: SortedMap<Vec<(i32, &str)>> = SortedMap::new();
        m.extend([(1, "a"), (2, "b")]);
        m.extend([(2, "B"), (3, "c")]); // key 2 overridden
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(&2), Some(&"B"));
        assert_eq!(m.get(&3), Some(&"c"));
    }

    #[test]
    fn unsorted_try_from_iter_rejects_duplicate_key() {
        let err = UnsortedMap::<Vec<(i32, &str)>>::try_from_iter([(1, "a"), (2, "b"), (1, "z")])
            .expect_err("duplicate key 1");
        // Detected at append, so the third entry is handed back deterministically.
        assert_eq!(err.into_inner(), (1, "z"));
    }

    // Key order is enforced in *every* build profile, returned as an error (not a
    // debug-only panic). The check runs before the dup check, so a smaller key is
    // Unsorted, not DuplicateKey.
    #[test]
    fn sorted_try_from_sorted_iter_rejects_unsorted_keys() {
        let err =
            SortedMap::<Vec<(i32, &str)>>::try_from_sorted_iter([(1, "a"), (3, "c"), (2, "b")])
                .expect_err("key 2 after key 3 is descending");
        match err {
            BuildError::Unsorted(entry) => assert_eq!(entry, (2, "b")),
            BuildError::Capacity(_) | BuildError::DuplicateKey(_) => {
                panic!("expected an unsorted error")
            }
        }
    }

    #[test]
    fn entry_or_insert_inserts_then_updates_in_one_lookup() {
        // The headline use: tally occurrences with a single search per item.
        let mut counts: SortedMap<Vec<(&str, u32)>> = SortedMap::new();
        for w in ["a", "b", "a", "c", "a", "b"] {
            *counts.entry(w).or_insert(0) += 1;
        }
        assert_eq!(counts.get(&"a"), Some(&3));
        assert_eq!(counts.get(&"b"), Some(&2));
        assert_eq!(counts.get(&"c"), Some(&1));
        // A vacant entry inserts at the binary-search slot, so order is kept.
        assert_eq!(counts.as_slice(), &[("a", 3), ("b", 2), ("c", 1)]);
    }

    #[test]
    fn entry_and_modify_then_or_insert() {
        let mut m: SortedMap<Vec<(i32, i32)>> = SortedMap::new();
        // Vacant: `and_modify` is a no-op, `or_insert` seeds the value.
        m.entry(1).and_modify(|v| *v += 100).or_insert(10);
        assert_eq!(m.get(&1), Some(&10));
        // Occupied: `and_modify` runs, `or_insert` is ignored.
        m.entry(1).and_modify(|v| *v += 100).or_insert(10);
        assert_eq!(m.get(&1), Some(&110));
    }

    #[test]
    fn entry_occupied_insert_and_remove_keeps_order() {
        let mut m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(1, "a"), (2, "b"), (3, "c")]).unwrap();
        match m.entry(2) {
            Entry::Occupied(mut e) => {
                assert_eq!(e.key(), &2);
                assert_eq!(e.insert("B"), "b"); // replace returns the old value
                assert_eq!(e.remove(), "B"); // then remove it
            }
            Entry::Vacant(_) => panic!("key 2 is present"),
        }
        // Sorted remove shifts, so the remaining entries stay ascending.
        assert_eq!(m.as_slice(), &[(1, "a"), (3, "c")]);
    }

    #[test]
    fn entry_vacant_into_key_inserts_nothing() {
        let mut m: SortedMap<Vec<(i32, &str)>> = SortedMap::new();
        match m.entry(7) {
            Entry::Vacant(e) => {
                assert_eq!(e.key(), &7);
                assert_eq!(e.into_key(), 7); // take the key back without inserting
            }
            Entry::Occupied(_) => panic!("the map is empty"),
        }
        assert!(m.is_empty());
    }

    #[test]
    fn unsorted_entry_occupied_removes_by_swap() {
        let mut m: UnsortedMap<Vec<(i32, &str)>> = UnsortedMap::new();
        for (k, v) in [(1, "a"), (2, "b"), (3, "c")] {
            m.try_insert(k, v).unwrap();
        }
        // An occupied entry on an unsorted map swap-removes: the last entry fills
        // the hole, so order is not preserved.
        match m.entry(1) {
            Entry::Occupied(e) => assert_eq!(e.remove(), "a"),
            Entry::Vacant(_) => panic!("key 1 is present"),
        }
        assert_eq!(m.as_slice(), &[(3, "c"), (2, "b")]);
        // A vacant entry appends.
        *m.entry(9).or_insert("z") = "Z";
        assert_eq!(m.get(&9), Some(&"Z"));
    }

    #[test]
    fn get_mut_updates_in_place() {
        let mut sm: SortedMap<Vec<(i32, i32)>> =
            SortedMap::try_from_iter([(1, 10), (2, 20)]).unwrap();
        *sm.get_mut(&1).unwrap() += 5;
        assert_eq!(sm.get(&1), Some(&15));
        assert_eq!(sm.get_mut(&9), None);

        let mut um: UnsortedMap<Vec<(i32, i32)>> = UnsortedMap::new();
        um.try_insert(1, 10).unwrap();
        *um.get_mut(&1).unwrap() = 99;
        assert_eq!(um.get(&1), Some(&99));
        assert_eq!(um.get_mut(&9), None);
    }

    #[test]
    fn clear_empties_both_map_flavors() {
        let mut sm: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(1, "a"), (2, "b")]).unwrap();
        sm.clear();
        assert!(sm.is_empty());
        assert_eq!(sm.get(&1), None);
        assert_eq!(sm.try_insert(3, "c"), Ok(None)); // usable again
        assert_eq!(sm.as_slice(), &[(3, "c")]);

        let mut um: UnsortedMap<Vec<(i32, &str)>> = UnsortedMap::new();
        um.try_insert(1, "a").unwrap();
        um.clear();
        assert!(um.is_empty());
        assert_eq!(um.get(&1), None);
    }

    // The trust-contract guards fire only in debug builds, so gate these on it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "sorted by key and free of duplicate keys")]
    fn sorted_from_store_rejects_unsorted_keys() {
        let _ = SortedMap::from_store(alloc::vec![(3, "c"), (1, "a")]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "free of duplicate keys")]
    fn unsorted_from_store_rejects_duplicate_keys() {
        let _ = UnsortedMap::from_store(alloc::vec![(1, "a"), (1, "z")]);
    }
}

// heapless is the alloc-free fixed-cap backend: exercises the bounded paths.
#[cfg(all(test, feature = "heapless"))]
mod heapless_tests {
    use heapless::Vec;

    use crate::error::BuildError;
    use crate::SortedMap;

    #[test]
    fn try_from_iter_capacity_overflow() {
        let err = SortedMap::<Vec<(u8, u8), 2>>::try_from_iter([(1, 1), (2, 2), (3, 3)])
            .expect_err("third entry overflows cap 2");
        match err {
            BuildError::Capacity(entry) => assert_eq!(entry, (3, 3)),
            BuildError::DuplicateKey(_) | BuildError::Unsorted(_) => {
                panic!("expected a capacity error")
            }
        }
    }

    #[test]
    fn capacity_reports_fixed_bound() {
        // A bounded sorted map reads its own cap without reaching into the store.
        let m: SortedMap<Vec<(u8, u8), 4>> = SortedMap::new();
        assert_eq!(m.capacity(), Some(4));
    }

    #[test]
    fn entry_or_try_insert_respects_capacity() {
        // Cap 2, full. A bounded store has no infallible `or_insert`; `or_try_insert`
        // updates an occupied slot (no capacity used) but rejects a new key.
        let mut m: SortedMap<Vec<(u8, u8), 2>> =
            SortedMap::try_from_sorted_iter([(1, 10), (2, 20)]).unwrap();

        // Occupied update in place succeeds even at capacity.
        *m.entry(1)
            .or_try_insert(0)
            .expect("update consumes no capacity") = 11;
        assert_eq!(m.get(&1), Some(&11));

        // A genuinely new key at the bound is rejected, handing back `(key, value)`.
        let err = m.entry(3).or_try_insert(30).expect_err("store is full");
        assert_eq!(err.into_inner(), (3, 30));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn from_sorted_iter_dup_beats_capacity() {
        // Cap 2 with only one slot used: the dup is rejected as a duplicate, not
        // as a capacity failure — a duplicate key consumes no capacity.
        let err = SortedMap::<Vec<(u8, u8), 2>>::try_from_sorted_iter([(1, 1), (1, 2)])
            .expect_err("duplicate key 1");
        assert_eq!(err.into_inner(), (1, 2));
    }
}
