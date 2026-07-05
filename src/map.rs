//! Map collections — the **ordering** axis for `Elem = (K, V)`.
//!
//! [`SortedMap`] keeps its store ordered by key (`O(log n)` lookup);
//! [`UnsortedMap`] appends and swap-removes (`O(1)` mutation, `O(n)` search) and
//! needs only `K: Eq` rather than `K: Ord`.

use core::borrow::Borrow;
use core::ops::RangeBounds;

use crate::error::{BuildError, CapacityError};
use crate::set::subrange;
use crate::store::{append_all, push, retain_in, Store, StoreMut, StoreNew, Unbounded};

mod entry;

pub use entry::{Entry, OccupiedEntry, VacantEntry};

/// Iterator over a map's entries as `(&K, &V)` pairs — what [`SortedMap::iter`] and
/// [`UnsortedMap::iter`] return and `&map` iterates as.
///
/// A thin wrapper over the underlying `&[(K, V)]` slice iterator (double-ended,
/// exact-size, fused).
// A named struct rather than an `iter::Map<_, fn(…)>` type alias: naming the
// alias forces the projection into a function *pointer*, which can survive to
// codegen as an indirect call — and a map iterated in a hot loop is exactly
// where that shows. The struct keeps the projection a direct, inlinable call
// and leaves room to change the representation later.
#[derive(Clone, Debug)]
pub struct MapIter<'a, K, V> {
    inner: core::slice::Iter<'a, (K, V)>,
}

impl<'a, K, V> MapIter<'a, K, V> {
    fn new(entries: &'a [(K, V)]) -> Self {
        MapIter {
            inner: entries.iter(),
        }
    }
}

impl<'a, K, V> Iterator for MapIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner.next().map(entry_refs)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn nth(&mut self, n: usize) -> Option<(&'a K, &'a V)> {
        self.inner.nth(n).map(entry_refs)
    }
    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }
    #[inline]
    fn last(self) -> Option<(&'a K, &'a V)> {
        self.inner.last().map(entry_refs)
    }
    // Forward internal iteration so `for_each`/`sum`-style consumers get the
    // slice iterator's unrolled loop, not the `next()` default.
    #[inline]
    fn fold<B, F: FnMut(B, (&'a K, &'a V)) -> B>(self, init: B, mut f: F) -> B {
        self.inner.fold(init, |acc, e| f(acc, entry_refs(e)))
    }
}

impl<'a, K, V> DoubleEndedIterator for MapIter<'a, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner.next_back().map(entry_refs)
    }
    #[inline]
    fn rfold<B, F: FnMut(B, (&'a K, &'a V)) -> B>(self, init: B, mut f: F) -> B {
        self.inner.rfold(init, |acc, e| f(acc, entry_refs(e)))
    }
}

impl<K, V> ExactSizeIterator for MapIter<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> core::iter::FusedIterator for MapIter<'_, K, V> {}

/// Splits a borrowed entry into borrowed key/value halves — the projection under
/// [`MapIter`].
fn entry_refs<K, V>(entry: &(K, V)) -> (&K, &V) {
    (&entry.0, &entry.1)
}

/// A map kept sorted by key in its backing store (`Elem = (K, V)`).
// The stored order is canonical (sorted by key, unique keys), so the structural
// derives are the semantic ones: the derived `Hash`/`PartialOrd`/`Ord`
// (lexicographic over `(K, V)` entries in ascending key order, exactly
// `BTreeMap`'s) are consistent with the derived `PartialEq`, letting a
// `SortedMap` key another map or live in a `BTreeSet`. The unsorted twin can
// derive none of these (swap-remove makes its stored order incidental).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SortedMap<S> {
    store: S,
}

impl<S: StoreNew> SortedMap<S> {
    /// Creates an empty `SortedMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pouch::Map;
    /// let m: Map<&str, u32> = Map::new();
    /// assert!(m.is_empty());
    /// ```
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
    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.store.len()
    }
    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    /// Returns the logical capacity, or `None` if unbounded.
    pub fn capacity(&self) -> Option<usize> {
        self.store.capacity()
    }
    /// Returns the entries as a contiguous `(K, V)` slice, in ascending key order.
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
    /// Borrows the backing store, for backend-specific introspection (`spilled()`,
    /// allocated capacity, …) — see [`SortedSet::store`](crate::SortedSet::store).
    ///
    /// Shared-ref only: `&mut` access could break the sorted-by-key invariant that
    /// [`from_store`](Self::from_store) trusts.
    pub fn store(&self) -> &S {
        &self.store
    }
    /// Consumes the map and hands back its store, entries intact and still in
    /// ascending key order — the inverse of [`from_store`](Self::from_store).
    pub fn into_store(self) -> S {
        self.store
    }
}

impl<S: StoreMut> SortedMap<S> {
    /// Removes every entry, keeping the backing store's allocated capacity.
    ///
    /// Needs no `Ord` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
    /// Pre-allocates so at least `additional` more entries fit without a
    /// reallocation — see [`SortedSet::reserve`](crate::SortedSet::reserve).
    pub fn reserve(&mut self, additional: usize) {
        self.store.reserve(additional);
    }
}

// The iteration accessors need no `K: Ord` — they only walk the store. (The
// explicit `K/V: 'a` bounds throughout are the E0311 projection quirk; see
// `get`.)
impl<K, V, S> SortedMap<S>
where
    S: Store<Elem = (K, V)>,
{
    /// Returns an iterator over the entries as `(&K, &V)` pairs, in ascending key order.
    pub fn iter<'a>(&'a self) -> MapIter<'a, K, V>
    where
        K: 'a,
        V: 'a,
    {
        MapIter::new(self.store.as_slice())
    }

    /// Returns an iterator over the keys in ascending order.
    pub fn keys<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a K> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_slice().iter().map(|(k, _)| k)
    }

    /// Returns an iterator over the values, in ascending order of their keys.
    pub fn values<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a V> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_slice().iter().map(|(_, v)| v)
    }

    /// Returns the entry with the smallest key, or `None` if empty. `O(1)`.
    pub fn first_key_value<'a>(&'a self) -> Option<(&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_slice().first().map(entry_refs)
    }

    /// Returns the entry with the largest key, or `None` if empty. `O(1)`.
    pub fn last_key_value<'a>(&'a self) -> Option<(&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_slice().last().map(entry_refs)
    }
}

// The mutating iteration accessors need no `K: Ord` either: handing out `&mut V`
// can't unsort the keys (only `&mut K` could, so there is no `keys_mut`).
impl<K, V, S> SortedMap<S>
where
    S: StoreMut<Elem = (K, V)>,
{
    /// Returns an iterator over the entries as `(&K, &mut V)` pairs, in ascending key
    /// order — bulk in-place value updates without touching the keys.
    pub fn iter_mut<'a>(
        &'a mut self,
    ) -> impl DoubleEndedIterator<Item = (&'a K, &'a mut V)> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_mut_slice().iter_mut().map(|(k, v)| (&*k, v))
    }

    /// Returns a mutable iterator over the values, in ascending order of their keys.
    pub fn values_mut<'a>(
        &'a mut self,
    ) -> impl DoubleEndedIterator<Item = &'a mut V> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_mut_slice().iter_mut().map(|(_, v)| v)
    }

    /// Retains only the entries for which `f` returns `true`, preserving key order.
    /// `O(n)`.
    ///
    /// The predicate gets `&mut V`, so it can update the entries it keeps.
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        retain_in(&mut self.store, |(k, v)| f(k, v));
    }
}

impl<K, V, S> SortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: Ord,
{
    /// Wraps a store **assumed already sorted by key and free of duplicate keys** — the
    /// invariant `binary_search` (and thus [`get`](Self::get) /
    /// [`try_insert`](Self::try_insert) / [`remove`](Self::remove)) relies on.
    ///
    /// No sort is performed; an out-of-order or duplicate-keyed store yields wrong
    /// lookups. The precondition is only `debug_assert!`-checked (zero cost in release).
    /// For a runtime-checked ascending build use
    /// [`try_from_sorted_iter`](Self::try_from_sorted_iter); to build from arbitrary
    /// input use [`try_from_iter`](Self::try_from_iter).
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `store` is not sorted by key or contains duplicate
    /// keys; release builds trust the precondition unchecked.
    pub fn from_store(store: S) -> Self {
        debug_assert!(
            store.as_slice().windows(2).all(|w| w[0].0 < w[1].0),
            "SortedMap::from_store: store must be sorted by key and free of duplicate keys",
        );
        SortedMap { store }
    }

    /// Binary searches the store by key.
    ///
    /// `Ok(i)` is the index of the matching entry; `Err(i)` the insertion point that
    /// keeps the keys sorted. Every key lookup — `get`, `contains_key`, `try_insert`,
    /// `remove`, `entry` — routes through this one search, so they can never disagree on
    /// which entry is "the one for this key" (and the `Borrow`-keyed match lands in
    /// exactly one place).
    fn search<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.store
            .as_slice()
            .binary_search_by(|(k, _)| k.borrow().cmp(key))
    }

    // NOTE: returning `&V` derived from the projected `Elem = (K, V)` slice needs
    // an explicit `K/V: 'a` bound — rustc does not infer implied bounds through the
    // associated-type projection (E0311). Worth knowing when you build out the API.
    /// Returns a reference to the value corresponding to `key`, or `None` if
    /// absent. `O(log n)`.
    ///
    /// `key` may be any borrowed form of `K` — a `SortedMap<Vec<(String, V)>>`
    /// answers `get("k")` without allocating a `String` to ask — with the usual
    /// [`Borrow`] contract that the borrowed form's `Ord` agrees with `K`'s.
    ///
    /// # Examples
    ///
    /// ```
    /// use pouch::Map;
    /// let mut m: Map<&str, u32> = Map::default();
    /// m.insert("a", 1);
    /// assert_eq!(m.get("a"), Some(&1));
    /// assert_eq!(m.get("z"), None);
    /// ```
    pub fn get<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        V: 'a,
    {
        let i = self.search(key).ok()?;
        Some(&self.store.as_slice()[i].1)
    }

    /// Returns `true` if `key` is present.
    ///
    /// `O(log n)` — like [`get`](Self::get) but yields a yes/no answer, so it needs no
    /// value lifetime. `key` may be any borrowed form of `K`.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.search(key).is_ok()
    }

    /// Returns the entries whose keys fall within `range`, as a subslice of the sorted
    /// store — the sorted layout's native range query.
    ///
    /// Two `O(log n)` bound searches, zero copies. The bounds may be any borrowed form of
    /// `K`, like [`get`](Self::get); as with `BTreeMap::range`, an **unsized** form
    /// (`str`, `[u8]`) needs the explicit tuple-of-`Bound`s shape: `map.range::<str,
    /// _>((Bound::Included("a"), Bound::Excluded("m")))`.
    ///
    /// # Panics
    ///
    /// Panics if the range's start is greater than its end — which includes an exclusive
    /// range whose bounds are equal *and* present in the map (e.g. `(Bound::Excluded(k),
    /// Bound::Excluded(k))` when `k` is a key).
    pub fn range<Q, R>(&self, range: R) -> &[(K, V)]
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        subrange(self.store.as_slice(), range, |(k, _)| k.borrow())
    }
}

impl<K, V, S> SortedMap<S>
where
    S: StoreMut<Elem = (K, V)>,
    K: Ord,
{
    /// Returns a mutable reference to `key`'s value, or `None` if absent — for an
    /// in-place update without the [`entry`](Self::entry) ceremony.
    ///
    /// Carries the same explicit `K/V: 'a` bounds as [`get`](Self::get) (the E0311
    /// quirk), and takes any borrowed form of `K` the same way.
    pub fn get_mut<'a, Q>(&'a mut self, key: &Q) -> Option<&'a mut V>
    where
        K: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        V: 'a,
    {
        let i = self.search(key).ok()?;
        Some(&mut self.store.as_mut_slice()[i].1)
    }

    /// Inserts or replaces.
    ///
    /// Replacing an existing key consumes no capacity and so can never fail — only a
    /// genuinely new key at the bound errors.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying `(key, value)` if `key` is new and the map
    /// is at its logical [`capacity`](Self::capacity); replacing an existing key
    /// never errors.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError<(K, V)>> {
        // Append-mostly fast path — see `SortedSet::try_insert`. Strict `>` (not
        // `>=`) is load-bearing: an equal key must fall through to `search` and
        // replace the value (a replacement consumes no capacity), never append.
        if self.store.as_slice().last().is_none_or(|(k, _)| key > *k) {
            return push(&mut self.store, (key, value)).map(|()| None);
        }
        match self.search(&key) {
            Ok(i) => {
                let slot = &mut self.store.as_mut_slice()[i].1;
                Ok(Some(core::mem::replace(slot, value)))
            }
            Err(i) => self.store.try_insert_at(i, (key, value)).map(|()| None),
        }
    }

    /// Removes the entry for `key`, returning its value.
    ///
    /// Order-preserving shift: `O(log n)` search, `O(n)` shift. `key` may be any borrowed
    /// form of `K`, like [`get`](Self::get).
    ///
    /// # Examples
    ///
    /// ```
    /// use pouch::Map;
    /// let mut m: Map<&str, u32> = Map::default();
    /// m.insert("a", 1);
    /// assert_eq!(m.remove("a"), Some(1)); // returns the removed value
    /// assert_eq!(m.remove("a"), None); // already gone
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.search(key) {
            Ok(i) => Some(self.store.remove_at(i).1),
            Err(_) => None,
        }
    }

    /// Resolves `key`'s slot **once** and returns an [`Entry`] for an
    /// insert-or-update, avoiding the second search a separate
    /// [`get`](Self::get) + [`try_insert`](Self::try_insert) would pay. `O(log n)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pouch::Map;
    ///
    /// let mut counts: Map<&str, u32> = Map::default();
    /// for w in ["a", "b", "a", "a"] {
    ///     *counts.entry(w).or_insert(0) += 1; // one lookup per word, not two
    /// }
    /// assert_eq!(counts.get("a"), Some(&3));
    /// assert_eq!(counts.get("b"), Some(&1));
    /// ```
    pub fn entry(&mut self, key: K) -> Entry<'_, S, K> {
        match self.search(&key) {
            Ok(index) => Entry::Occupied(OccupiedEntry::sorted(&mut self.store, index)),
            Err(index) => Entry::Vacant(VacantEntry::new(&mut self.store, index, key)),
        }
    }

    /// Inserts every entry, one at a time, **last-wins**: a repeated key replaces the
    /// earlier value rather than erroring (so this returns only a [`CapacityError`],
    /// never a duplicate-key error). `O(k·n)`.
    ///
    /// To instead reject duplicate keys, build a fresh map with
    /// [`try_from_iter`](Self::try_from_iter).
    ///
    /// On overflow only the one rejected entry is recoverable: the iterator is
    /// dropped along with any entries it has not yet yielded. Drive
    /// [`try_insert`](Self::try_insert) yourself over an iterator you keep if the
    /// unconsumed tail must survive.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] with the first entry that doesn't fit when a
    /// bounded store fills; earlier entries are kept.
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
    /// Builds from an arbitrary (unordered) iterator of entries in `O(n log n)`,
    /// **requiring every key to be unique**: append all, sort by key, then reject any
    /// duplicate.
    ///
    /// Unlike a set's silent dedup, a map can't drop a duplicate key without arbitrarily
    /// choosing which value to keep, so it errors instead (one of the clashing entries is
    /// handed back — the sort is unstable, so which of the two is unspecified). For
    /// last-wins override semantics use [`try_extend`](Self::try_extend) / `extend`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::DuplicateKey`] carrying one of the colliding entries of a
    /// repeated key, or [`BuildError::Capacity`] if a bounded store fills. As with the
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

    /// Builds from an iterator whose entries are already in ascending key order, in
    /// `O(n)` — no sort.
    ///
    /// Like [`try_from_iter`](Self::try_from_iter) it requires unique keys, but it
    /// detects a duplicate (and a misordered key) *before* the append, so either is
    /// rejected even at capacity (neither consumes a slot).
    ///
    /// Unlike [`from_store`](Self::from_store), the ascending-order promise is
    /// enforced in every build profile: a key smaller than its predecessor is
    /// returned as [`BuildError::Unsorted`] rather than silently trusted. The check
    /// is one comparison per entry — the same one the dedup already needs.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Unsorted`] if a key is smaller than its predecessor,
    /// [`BuildError::DuplicateKey`] if a key repeats, or [`BuildError::Capacity`] if
    /// a bounded store fills — each carrying the offending entry.
    pub fn try_from_sorted_iter<I>(iter: I) -> Result<Self, BuildError<(K, V)>>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let mut store = S::new();
        let iter = iter.into_iter();
        store.reserve(iter.size_hint().0);
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

impl<K, V, S> SortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: Ord,
{
    /// Infallibly inserts or replaces, returning the previous value — available only when
    /// the backing store is [`Unbounded`].
    ///
    /// The infallible twin of [`try_insert`](Self::try_insert).
    ///
    /// # Examples
    ///
    /// ```
    /// use pouch::Map;
    /// let mut m: Map<&str, u32> = Map::default();
    /// assert_eq!(m.insert("a", 1), None); // new key
    /// assert_eq!(m.insert("a", 2), Some(1)); // replaced; previous value returned
    /// assert_eq!(m.get("a"), Some(&2));
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.try_insert(key, value) {
            Ok(prev) => prev,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<K, V, S> SortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + StoreNew + Unbounded,
    K: Ord,
{
    /// Builds from an ascending iterator — the infallible
    /// [`try_from_sorted_iter`](Self::try_from_sorted_iter), available only for an
    /// [`Unbounded`] store. `O(n)`.
    ///
    /// # Panics
    ///
    /// Panics if the keys are not in ascending order or a key repeats — an
    /// infallible builder has no error channel, and a map cannot silently pick
    /// which duplicate value wins.
    pub fn from_sorted_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
    {
        match Self::try_from_sorted_iter(iter) {
            Ok(map) => map,
            Err(BuildError::Capacity(_)) => {
                unreachable!("Unbounded store reported a capacity failure")
            }
            Err(BuildError::DuplicateKey(_)) => {
                panic!("from_sorted_iter: duplicate key")
            }
            Err(BuildError::Unsorted(_)) => {
                panic!("from_sorted_iter: keys were not in ascending order")
            }
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a SortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: 'a,
    V: 'a,
{
    type Item = (&'a K, &'a V);
    type IntoIter = MapIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Consumes the map, yielding owned `(K, V)` entries in ascending key order.
///
/// Available when the backing store is itself consumable into its elements.
impl<S> IntoIterator for SortedMap<S>
where
    S: Store + IntoIterator<Item = <S as Store>::Elem>,
{
    type Item = S::Elem;
    type IntoIter = <S as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
    }
}

impl<K, V, S> Extend<(K, V)> for SortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: Ord,
{
    /// Extends the map, last-wins and infallible — available only for an [`Unbounded`]
    /// store.
    ///
    /// Deliberately no `FromIterator`: fresh construction is strict about duplicate keys
    /// (see [`try_from_iter`](SortedMap::try_from_iter)), whereas `extend` matches the
    /// standard-library override semantics.
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        match self.try_extend(iter) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

/// A map with no key ordering (`Elem = (K, V)`): lookup is a linear scan, insert appends,
/// delete swap-removes.
///
/// The unsorted counterpart of [`SortedMap`]; needs only `K: Eq`, not `K: Ord`.
// Derives `Clone` but not `PartialEq`/`Eq` (nor `Hash`/`Ord`): correct map
// equality is key-order-independent, yet swap-remove lets two equal maps store
// their entries in different orders, so a structural derive would wrongly call
// them unequal. The sorted twin derives all of these because its stored order
// is canonical.
#[derive(Clone, Debug)]
pub struct UnsortedMap<S> {
    store: S,
}

impl<S: StoreNew> UnsortedMap<S> {
    /// Creates an empty `UnsortedMap`.
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
    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.store.len()
    }
    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    /// Returns the logical capacity, or `None` if unbounded.
    pub fn capacity(&self) -> Option<usize> {
        self.store.capacity()
    }
    /// Returns the entries as a contiguous `(K, V)` slice, in insertion order modulo the
    /// swaps that [`remove`](Self::remove) performs.
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
    /// Borrows the backing store, for backend-specific introspection (`spilled()`,
    /// allocated capacity, …) — see [`SortedSet::store`](crate::SortedSet::store).
    ///
    /// Shared-ref only: `&mut` access could smuggle in a duplicate key, breaking the map
    /// invariant.
    pub fn store(&self) -> &S {
        &self.store
    }
    /// Consumes the map and hands back its store, entries intact (in no
    /// particular order) — the inverse of [`from_store`](Self::from_store).
    pub fn into_store(self) -> S {
        self.store
    }
}

impl<S: StoreMut> UnsortedMap<S> {
    /// Removes every entry, keeping the backing store's allocated capacity.
    ///
    /// Needs no `Eq` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
    /// Pre-allocates so at least `additional` more entries fit without a
    /// reallocation — see [`SortedSet::reserve`](crate::SortedSet::reserve).
    pub fn reserve(&mut self, additional: usize) {
        self.store.reserve(additional);
    }
}

// Iteration accessors, `K: Eq`-free — they only walk the store. (The explicit
// `K/V: 'a` bounds are the E0311 projection quirk; see `get`.)
impl<K, V, S> UnsortedMap<S>
where
    S: Store<Elem = (K, V)>,
{
    /// Returns an iterator over the entries as `(&K, &V)` pairs, in no particular order.
    pub fn iter<'a>(&'a self) -> MapIter<'a, K, V>
    where
        K: 'a,
        V: 'a,
    {
        MapIter::new(self.store.as_slice())
    }

    /// Returns an iterator over the keys, in no particular order.
    pub fn keys<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a K> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_slice().iter().map(|(k, _)| k)
    }

    /// Returns an iterator over the values, in no particular order.
    pub fn values<'a>(&'a self) -> impl DoubleEndedIterator<Item = &'a V> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_slice().iter().map(|(_, v)| v)
    }
}

impl<K, V, S> UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)>,
{
    /// Returns an iterator over the entries as `(&K, &mut V)` pairs — bulk in-place value
    /// updates.
    pub fn iter_mut<'a>(
        &'a mut self,
    ) -> impl DoubleEndedIterator<Item = (&'a K, &'a mut V)> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_mut_slice().iter_mut().map(|(k, v)| (&*k, v))
    }

    /// Returns a mutable iterator over the values, in no particular order.
    pub fn values_mut<'a>(
        &'a mut self,
    ) -> impl DoubleEndedIterator<Item = &'a mut V> + ExactSizeIterator
    where
        K: 'a,
        V: 'a,
    {
        self.store.as_mut_slice().iter_mut().map(|(_, v)| v)
    }

    /// Retains only the entries for which `f` returns `true`. `O(n)`.
    ///
    /// The predicate gets `&mut V`, so it can update the entries it keeps.
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        retain_in(&mut self.store, |(k, v)| f(k, v));
    }
}

impl<K, V, S> UnsortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: Eq,
{
    /// Wraps a store **assumed free of duplicate keys** — the map invariant.
    ///
    /// No scan is performed; a duplicate key would shadow itself and let the same entry
    /// be removed twice. The precondition is `debug_assert!`-checked (zero cost in
    /// release). To build from arbitrary input, use
    /// [`try_from_iter`](Self::try_from_iter).
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `store` contains duplicate keys; release builds
    /// trust the precondition unchecked.
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

    /// Returns the index of the entry whose key equals `key`, or `None`.
    ///
    /// Every key lookup — `get`, `try_insert`, `remove`, `try_from_iter` — routes through
    /// this single scan, so they can never disagree on which entry is "the one for this
    /// key" (and the `Borrow`-keyed match — and any future comparator — lands in exactly
    /// one place).
    fn position<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.store
            .as_slice()
            .iter()
            .position(|(k, _)| k.borrow() == key)
    }

    /// Returns a reference to the value corresponding to `key`, or `None` if
    /// absent. `O(n)` linear scan.
    ///
    /// `key` may be any borrowed form of `K` — an
    /// `UnsortedMap<Vec<(String, V)>>` answers `get("k")` without allocating a
    /// `String` to ask — with the usual [`Borrow`] contract that the borrowed
    /// form's `Eq` agrees with `K`'s.
    pub fn get<'a, Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q> + 'a,
        Q: Eq + ?Sized,
        V: 'a,
    {
        self.position(key).map(|i| &self.store.as_slice()[i].1)
    }

    /// Returns `true` if `key` is present.
    ///
    /// `O(n)` — routes through the same internal linear scan as the other lookups, so it
    /// stays consistent with [`get`](Self::get), and takes any borrowed form of `K` the
    /// same way.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.position(key).is_some()
    }
}

impl<K, V, S> UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)>,
    K: Eq,
{
    /// Returns a mutable reference to `key`'s value, or `None` if absent — for an
    /// in-place update without the [`entry`](Self::entry) ceremony.
    ///
    /// Routes through the same internal linear scan as [`get`](Self::get) and takes any
    /// borrowed form of `K` the same way; carries the same explicit `K/V: 'a` bounds (the
    /// E0311 quirk).
    pub fn get_mut<'a, Q>(&'a mut self, key: &Q) -> Option<&'a mut V>
    where
        K: Borrow<Q> + 'a,
        Q: Eq + ?Sized,
        V: 'a,
    {
        let i = self.position(key)?;
        Some(&mut self.store.as_mut_slice()[i].1)
    }

    /// Inserts or replaces.
    ///
    /// Replacing an existing key consumes no capacity and so can never fail — only a
    /// genuinely new key at the bound errors. O(n) lookup, O(1) to append or replace in
    /// place.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying `(key, value)` if `key` is new and the map
    /// is at its logical [`capacity`](Self::capacity); replacing an existing key
    /// never errors.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError<(K, V)>> {
        if let Some(i) = self.position(&key) {
            let slot = &mut self.store.as_mut_slice()[i].1;
            return Ok(Some(core::mem::replace(slot, value)));
        }
        push(&mut self.store, (key, value)).map(|()| None)
    }

    /// Removes the entry for `key`, returning its value.
    ///
    /// Swap-remove: O(1), order not preserved. `key` may be any borrowed form of `K`,
    /// like [`get`](Self::get).
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let i = self.position(key)?;
        Some(self.store.swap_remove_at(i).1)
    }

    /// Resolves `key`'s slot **once** and returns an [`Entry`] for an insert-or-update,
    /// avoiding the second scan a separate [`get`](Self::get) +
    /// [`try_insert`](Self::try_insert) would pay.
    ///
    /// `O(n)` to locate; an occupied entry removes via `O(1)` swap (order not preserved).
    pub fn entry(&mut self, key: K) -> Entry<'_, S, K> {
        match self.position(&key) {
            Some(index) => Entry::Occupied(OccupiedEntry::unsorted(&mut self.store, index)),
            None => {
                let index = self.store.len();
                Entry::Vacant(VacantEntry::new(&mut self.store, index, key))
            }
        }
    }

    /// Inserts every entry, one at a time, **last-wins** (a repeated key replaces the
    /// earlier value rather than erroring). `O(k·n)`.
    ///
    /// To reject duplicate keys instead, build a fresh map with
    /// [`try_from_iter`](Self::try_from_iter).
    ///
    /// On overflow only the one rejected entry is recoverable: the iterator is
    /// dropped along with any entries it has not yet yielded. Drive
    /// [`try_insert`](Self::try_insert) yourself over an iterator you keep if the
    /// unconsumed tail must survive.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] with the first entry that doesn't fit when a
    /// bounded store fills; earlier entries are kept.
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
    /// Builds from an iterator of entries, **requiring every key to be unique**.
    ///
    /// `O(n²)`: each entry is scanned against those already kept (an unsorted map has no
    /// faster dedup without `Ord`), and a repeated key is rejected — a map can't drop a
    /// duplicate key without arbitrarily picking a value. For last-wins override
    /// semantics use [`try_extend`](Self::try_extend) / `extend`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::DuplicateKey`] with the second entry of a repeated key,
    /// or [`BuildError::Capacity`] if a bounded store fills.
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

impl<K, V, S> UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: Eq,
{
    /// Infallibly inserts or replaces, returning the previous value — available only when
    /// the backing store is [`Unbounded`].
    ///
    /// The infallible twin of [`try_insert`](Self::try_insert).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.try_insert(key, value) {
            Ok(prev) => prev,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a UnsortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: 'a,
    V: 'a,
{
    type Item = (&'a K, &'a V);
    type IntoIter = MapIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Consumes the map, yielding owned `(K, V)` entries in no particular order.
///
/// Available when the backing store is itself consumable into its elements.
impl<S> IntoIterator for UnsortedMap<S>
where
    S: Store + IntoIterator<Item = <S as Store>::Elem>,
{
    type Item = S::Elem;
    type IntoIter = <S as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
    }
}

impl<K, V, S> Extend<(K, V)> for UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + Unbounded,
    K: Eq,
{
    /// Extends the map, last-wins and infallible — available only for an [`Unbounded`]
    /// store.
    ///
    /// As with [`SortedMap`], there is deliberately no `FromIterator`: fresh construction
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

    // `MapIter` is a real struct wrapping the slice iterator: double-ended,
    // exact-size, fused, with forwarded internal iteration (`fold`).
    #[test]
    fn map_iter_is_double_ended_exact_and_fused() {
        use alloc::string::String;

        let m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(1, "a"), (2, "b"), (3, "c")]).unwrap();
        let mut it = m.iter();
        assert_eq!(it.len(), 3);
        assert_eq!(it.next(), Some((&1, &"a")));
        assert_eq!(it.next_back(), Some((&3, &"c")));
        assert_eq!(it.len(), 1);
        assert_eq!(it.next(), Some((&2, &"b")));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None); // fused
        let joined = m.iter().fold(String::new(), |mut s, (k, v)| {
            use core::fmt::Write;
            write!(s, "{k}{v}").expect("writing to a String cannot fail");
            s
        });
        assert_eq!(joined, "1a2b3c");
    }

    // The on-mission `Borrow` payoff: `String` keys, `&str` queries — no
    // allocation to ask, in either flavor.
    #[test]
    fn lookups_take_borrowed_forms() {
        use alloc::string::{String, ToString};
        use core::ops::Bound;

        let mut m: SortedMap<Vec<(String, u32)>> =
            SortedMap::try_from_iter([("a".to_string(), 1), ("b".to_string(), 2)]).unwrap();
        assert_eq!(m.get("a"), Some(&1));
        assert!(m.contains_key("b"));
        assert!(!m.contains_key("z"));
        *m.get_mut("a").unwrap() += 10;
        assert_eq!(m.get("a"), Some(&11));
        // Unsized bounds (`str`) need the tuple-of-`Bound`s shape (range sugar
        // like `"a".."c"` is a `Range<&str>`, which can only bound `&str`).
        assert_eq!(
            m.range::<str, _>((Bound::Included("b"), Bound::Unbounded)),
            &[("b".to_string(), 2)]
        );
        assert_eq!(m.remove("a"), Some(11));
        assert_eq!(m.get("a"), None);

        let mut u: UnsortedMap<Vec<(String, u32)>> =
            UnsortedMap::try_from_iter([("x".to_string(), 9)]).unwrap();
        assert_eq!(u.get("x"), Some(&9));
        assert!(u.contains_key("x"));
        *u.get_mut("x").unwrap() = 10;
        assert_eq!(u.remove("x"), Some(10));
        assert_eq!(u.get("x"), None);
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

    // The monotonic-append fast path in `try_insert` must be observably identical
    // to the binary-search path: ascending inserts stay sorted, an equal key
    // replaces (never appends a duplicate), and out-of-order inserts still land
    // in place.
    #[test]
    fn sorted_try_insert_monotonic_fast_path() {
        let mut m: SortedMap<Vec<(u32, u32)>> = SortedMap::new();
        for k in 0..100u32 {
            assert_eq!(m.insert(k, k), None); // every insert hits the tail append
        }
        assert!(m.keys().copied().eq(0..100)); // sorted, unique
        assert_eq!(m.insert(99, 999), Some(99)); // equal-to-max key → replace, not append
        assert_eq!(m.len(), 100);
        assert_eq!(m.get(&99), Some(&999));
        m.insert(200, 200); // strictly-greater → fast-path append
        m.insert(150, 150); // mid-range → binary-search path
        assert!(m.keys().copied().eq((0..100).chain([150, 200])));
        assert_eq!(m.get(&150), Some(&150));
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
    fn insert_is_infallible_on_unbounded_stores() {
        let mut m: SortedMap<Vec<(i32, &str)>> = SortedMap::new();
        assert_eq!(m.insert(2, "b"), None);
        assert_eq!(m.insert(1, "a"), None);
        assert_eq!(m.insert(2, "B"), Some("b")); // replace hands back the old value
        assert_eq!(m.as_slice(), &[(1, "a"), (2, "B")]);

        let mut um: UnsortedMap<Vec<(i32, &str)>> = UnsortedMap::new();
        assert_eq!(um.insert(1, "a"), None);
        assert_eq!(um.insert(1, "A"), Some("a"));
        assert_eq!(um.get(&1), Some(&"A"));
    }

    #[test]
    fn from_sorted_iter_builds_in_order() {
        let m = SortedMap::<Vec<(i32, &str)>>::from_sorted_iter([(1, "a"), (2, "b")]);
        assert_eq!(m.as_slice(), &[(1, "a"), (2, "b")]);
    }

    #[test]
    #[should_panic(expected = "duplicate key")]
    fn from_sorted_iter_panics_on_duplicate_key() {
        let _ = SortedMap::<Vec<(i32, &str)>>::from_sorted_iter([(1, "a"), (1, "z")]);
    }

    #[test]
    fn iteration_accessors_walk_entries() {
        let m: SortedMap<Vec<(i32, i32)>> =
            SortedMap::try_from_iter([(2, 20), (1, 10), (3, 30)]).unwrap();
        // iter / &m yield (&K, &V) in ascending key order.
        assert!(m.iter().eq([(&1, &10), (&2, &20), (&3, &30)]));
        assert!((&m).into_iter().next_back() == Some((&3, &30))); // double-ended
        assert!(m.keys().eq(&[1, 2, 3]));
        assert!(m.values().eq(&[10, 20, 30]));
        assert_eq!(m.first_key_value(), Some((&1, &10)));
        assert_eq!(m.last_key_value(), Some((&3, &30)));
        // by-value consumption yields owned entries.
        let owned: Vec<(i32, i32)> = m.into_iter().collect();
        assert_eq!(owned, &[(1, 10), (2, 20), (3, 30)]);

        let um: UnsortedMap<Vec<(i32, i32)>> =
            UnsortedMap::try_from_iter([(1, 10), (2, 20)]).unwrap();
        assert_eq!(um.iter().count(), 2);
        assert_eq!(um.keys().count(), 2);
    }

    #[test]
    fn values_mut_and_iter_mut_update_in_place() {
        let mut m: SortedMap<Vec<(i32, i32)>> =
            SortedMap::try_from_iter([(1, 10), (2, 20)]).unwrap();
        for v in m.values_mut() {
            *v += 1;
        }
        assert_eq!(m.as_slice(), &[(1, 11), (2, 21)]);
        for (k, v) in m.iter_mut() {
            *v += *k;
        }
        assert_eq!(m.as_slice(), &[(1, 12), (2, 23)]);
    }

    #[test]
    fn retain_filters_and_can_mutate_kept_values() {
        let mut m: SortedMap<Vec<(i32, i32)>> =
            SortedMap::try_from_iter([(1, 10), (2, 20), (3, 30), (4, 40)]).unwrap();
        m.retain(|k, v| {
            *v += 1; // mutate every visited value, keep even keys only
            k % 2 == 0
        });
        assert_eq!(m.as_slice(), &[(2, 21), (4, 41)]); // key order preserved

        let mut um: UnsortedMap<Vec<(i32, i32)>> =
            UnsortedMap::try_from_iter([(1, 10), (2, 20), (3, 30)]).unwrap();
        um.retain(|k, _| *k != 2);
        assert_eq!(um.len(), 2);
        assert!(!um.contains_key(&2));
    }

    #[test]
    fn range_returns_key_bounded_subslice() {
        let m: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(1, "a"), (3, "c"), (5, "e"), (7, "g")]).unwrap();
        assert_eq!(m.range(3..7), &[(3, "c"), (5, "e")]);
        assert_eq!(m.range(..=3), &[(1, "a"), (3, "c")]);
        assert_eq!(m.range(4..), &[(5, "e"), (7, "g")]);
        // A full range can't infer the borrowed key type (every `Q` fits
        // `RangeFull`), so it takes a turbofish — same as `BTreeMap::range`.
        assert_eq!(m.range::<i32, _>(..), m.as_slice());
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

    #[test]
    fn clone_and_eq_for_sorted_map() {
        let a: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(1, "a"), (2, "b")]).unwrap();
        let mut b = a.clone();
        assert_eq!(a, b); // PartialEq compares keys *and* values
        b.try_insert(3, "c").unwrap();
        assert_ne!(a, b); // the clone is independent
        assert_eq!(a.len(), 2);
        // Different build order, same mapping -> equal (canonical key order).
        let c: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(2, "b"), (1, "a")]).unwrap();
        assert_eq!(a, c);
        // A differing value breaks equality even with identical keys.
        let d: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(1, "a"), (2, "B")]).unwrap();
        assert_ne!(a, d);
    }

    #[test]
    fn clone_for_unsorted_map_is_independent() {
        // UnsortedMap derives Clone but not PartialEq (order-sensitive).
        let mut a: UnsortedMap<Vec<(i32, &str)>> = UnsortedMap::new();
        a.try_insert(1, "a").unwrap();
        let b = a.clone();
        a.try_insert(2, "b").unwrap();
        assert_eq!(b.len(), 1); // clone unaffected
        assert_eq!(b.get(&1), Some(&"a"));
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
