//! Column-oriented (struct-of-arrays) map — the **layout** variant of the
//! unsorted map.
//!
//! [`UnsortedMap`](crate::UnsortedMap) stores `Elem = (K, V)` pairs interleaved
//! in one store (array-of-structs). [`UnsortedColumnMap`] instead keeps keys and values
//! in *two* parallel stores — `keys: SK` (`Elem = K`) and `values: SV` (`Elem =
//! V`), length-locked so `keys[i]` pairs with `values[i]`. A key lookup then
//! scans a dense `[K]` slice instead of reading the key out of every `(K, V)`
//! pair, which stacks two wins over the interleaved scan. First, the dense scan
//! is vectorization-friendly: `get`/`remove` locate the key with a fixed-trip
//! reduction (`chunked_position`) that LLVM folds to branchless compares —
//! which the strided `(K, V)` scan can't — for a ~2× edge even on word-sized
//! values, across all `n` and sharpest on misses (which scan the whole column).
//! Second, the scan never pulls value payloads through cache, a bandwidth
//! saving ≈ proportional to `sizeof(V)/sizeof(K)` that stacks on top for large
//! values once the map outgrows cache (a further ~2× for 64-byte values at `n ≥
//! 4k`). See `benches/soa.rs`.
//!
//! The trade is deliberate and is why this is a separate type, not a tweak to
//! `UnsortedMap`:
//!   * no `as_slice() -> &[(K, V)]`, since the pairs don't exist contiguously — enumerate
//!     via [`keys`](UnsortedColumnMap::keys) / [`values`](UnsortedColumnMap::values),
//!     which `zip` back together;
//!   * [`from_store`](UnsortedColumnMap::from_store) takes two stores (no zero-copy wrap
//!     of an existing `Vec<(K, V)>`);
//!   * two backends to name, and the effective [`capacity`](UnsortedColumnMap::capacity)
//!     is the `min` of the two columns' bounds.
//!
//! [`SortedColumnMap`](crate::SortedColumnMap) is the sorted twin (the same
//! two-store layout, keys kept ordered for an `O(log n)` binary search). It
//! exists only because the SoA win is real for *large* values; for word-sized
//! values the split gains little — or loses on small-`n` hits, which fetch the
//! value from a separate cache line — so [`SortedMap`](crate::SortedMap) stays
//! the default. See its module docs.

use core::borrow::Borrow;

use crate::error::{BuildError, CapacityError};
use crate::set::chunked_contains;
use crate::store::{push, Store, StoreMut, StoreNew, Unbounded};

mod entry;

pub use entry::{ColumnEntry, OccupiedColumnEntry, VacantColumnEntry};

/// Returns the effective bound of a two-column map: the tighter of the two columns' caps
/// (`None` = unbounded), mirroring `Capped`'s min-of-bounds rule.
///
/// Shared with [`SortedColumnMap`](crate::SortedColumnMap), the sorted two-column map.
pub(crate) fn combined_capacity(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// Returns the index of the first slot in `keys` equal to `needle`, or `None` — the dense
/// key-column scan behind every [`UnsortedColumnMap`] lookup.
///
/// Structured as a fixed-trip OR-reduction rather than a plain
/// `iter().position()`: each `LANES`-wide chunk is scanned in full — no early
/// exit *within* a chunk — and the per-element equalities are folded into one
/// `hit` flag, so the only data-dependent branch fires once per chunk instead
/// of once per element. For primitive keys LLVM lowers that reduction to
/// branchless conditional-compares (or vector compares on targets with wide
/// SIMD); the short scalar locate then runs only inside the one chunk that
/// matched. The result is ~2× over `iter().position()` on a miss or a long
/// scan, and neutral for key types that don't fold (same total comparisons; at
/// most one extra chunk's worth on a hit). `LANES = 8` keeps the chunk-level
/// early exit fine enough that small-`n` hits don't regress against the
/// early-exit baseline. See `benches/soa.rs` (`locate`).
fn chunked_position<K, Q>(keys: &[K], needle: &Q) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    const LANES: usize = 8;
    let mut offset = 0;
    let mut chunks = keys.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        let mut hit = false;
        for k in chunk {
            hit |= k.borrow() == needle;
        }
        if hit {
            let i = chunk
                .iter()
                .position(|k| k.borrow() == needle)
                .expect("the chunk reduction reported a match");
            return Some(offset + i);
        }
        offset += LANES;
    }
    chunks
        .remainder()
        .iter()
        .position(|k| k.borrow() == needle)
        .map(|i| offset + i)
}

/// A map with no key ordering, stored **column-wise**: keys in `SK`, values in `SV`, kept
/// the same length.
///
/// The struct-of-arrays counterpart of [`UnsortedMap`](crate::UnsortedMap) — trades the
/// `&[(K, V)]` view for a dense, value-free key scan (faster for large values; see the
/// module docs). Needs only `K: Eq`.
// Derives `Clone` but not `PartialEq`/`Eq` (nor `Hash`/`Ord`): correct map
// equality is key-order-independent, yet swap-remove lets two equal maps store
// their columns in different orders, so a structural derive would wrongly call
// them unequal. The sorted twin derives all of these because its stored order
// is canonical.
#[derive(Clone, Debug)]
pub struct UnsortedColumnMap<SK, SV> {
    keys: SK,
    values: SV,
}

impl<SK: StoreNew, SV: StoreNew> UnsortedColumnMap<SK, SV> {
    /// Creates an empty `UnsortedColumnMap`.
    pub fn new() -> Self {
        UnsortedColumnMap {
            keys: SK::new(),
            values: SV::new(),
        }
    }
}

impl<SK: StoreNew, SV: StoreNew> Default for UnsortedColumnMap<SK, SV> {
    fn default() -> Self {
        Self::new()
    }
}

impl<SK: Store, SV: Store> UnsortedColumnMap<SK, SV> {
    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Returns the effective logical capacity: the `min` of the two columns' bounds
    /// (`None` = unbounded).
    ///
    /// Capping either column caps the map.
    pub fn capacity(&self) -> Option<usize> {
        combined_capacity(self.keys.capacity(), self.values.capacity())
    }
    /// Returns the keys as a contiguous slice — the dense scan target.
    ///
    /// `zip` with [`values`](Self::values) to iterate entries.
    pub fn keys(&self) -> &[SK::Elem] {
        self.keys.as_slice()
    }
    /// Returns the values as a contiguous slice, index-aligned with
    /// [`keys`](Self::keys).
    pub fn values(&self) -> &[SV::Elem] {
        self.values.as_slice()
    }
}

impl<SK: Store, SV: Store> UnsortedColumnMap<SK, SV> {
    /// Borrows the two backing stores, `(keys, values)` — the door to backend-specific
    /// introspection (`spilled()`, allocated capacity, …), as
    /// [`SortedSet::store`](crate::SortedSet::store) is for the single-store collections.
    ///
    /// Shared-ref only: `&mut` access could desync the columns or smuggle in a duplicate
    /// key.
    pub fn stores(&self) -> (&SK, &SV) {
        (&self.keys, &self.values)
    }
    /// Consumes the map and hands back its stores, `(keys, values)`, entries
    /// intact and index-aligned — the inverse of
    /// [`from_store`](Self::from_store).
    pub fn into_stores(self) -> (SK, SV) {
        (self.keys, self.values)
    }
}

impl<SK: StoreMut, SV: StoreMut> UnsortedColumnMap<SK, SV> {
    /// Removes every entry, clearing both columns and keeping their allocated capacity.
    ///
    /// Needs no `Eq` bound — it only truncates the stores.
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

impl<K, V, SK, SV> UnsortedColumnMap<SK, SV>
where
    SK: Store<Elem = K>,
    SV: Store<Elem = V>,
    K: Eq,
{
    /// Wraps two stores **assumed equal-length and free of duplicate keys** — the
    /// column-map invariants.
    ///
    /// No scan or alignment is performed; a length mismatch would desync key/value pairs
    /// and a duplicate key would shadow itself. Both preconditions are
    /// `debug_assert!`-checked (zero cost in release). To build from an arbitrary
    /// iterator, use [`try_from_iter`](Self::try_from_iter).
    ///
    /// # Panics
    ///
    /// In debug builds, panics if the columns' lengths differ or the keys contain
    /// duplicates; release builds trust the preconditions unchecked.
    pub fn from_store(keys: SK, values: SV) -> Self {
        debug_assert_eq!(
            keys.len(),
            values.len(),
            "UnsortedColumnMap::from_store: key and value columns must have equal length",
        );
        debug_assert!(
            {
                let ks = keys.as_slice();
                !(1..ks.len()).any(|i| ks[..i].contains(&ks[i]))
            },
            "UnsortedColumnMap::from_store: keys must be free of duplicates",
        );
        UnsortedColumnMap { keys, values }
    }

    /// Returns the index of the entry whose key equals `key`, or `None`.
    ///
    /// Every key lookup — `get`, `try_insert`, `remove`, `try_from_iter` — routes through
    /// this single dense `[K]` scan, so they can never disagree on which entry is "the
    /// one for this key". This contiguous, value-free scan is the layout's whole point;
    /// [`chunked_position`] gives it the branchless, vectorization-friendly shape a
    /// short-circuiting `iter().position()` can't take.
    fn position<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        chunked_position(self.keys.as_slice(), key)
    }

    // No E0311 lifetime dance here (unlike `UnsortedMap::get`): `values.as_slice()`
    // is already `&[V]`, so projecting `&V` needs no associated-type-projection
    // bound — elision ties the result to `&self`.
    /// Returns a reference to the value corresponding to `key`, or `None` if
    /// absent. `O(n)` scan over the dense key column.
    ///
    /// `key` may be any borrowed form of `K` (a `String`-keyed column map answers
    /// `get("k")` without allocating), with the usual [`Borrow`] contract that the
    /// borrowed form's `Eq` agrees with `K`'s.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.position(key).map(|i| &self.values.as_slice()[i])
    }

    /// Returns `true` if `key` is present.
    ///
    /// `O(n)`, but unlike [`get`](Self::get) it needs only a yes/no answer — so it uses
    /// the boolean chunked fold (`chunked_contains`, the crate's mirror of the standard
    /// library's specialized `slice::contains`, whose `&K` needle borrowed-form lookups
    /// can't supply), skipping `chunked_position`'s index recovery. The index-returning
    /// lookups (`get`/`remove`) keep the comparable branchless scan via
    /// `chunked_position`, so the broad-`n`, value-independent edge over
    /// [`UnsortedMap`](crate::UnsortedMap) — whose strided `(K, V)` scan can't fold the
    /// same way — holds across the board.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        chunked_contains(self.keys.as_slice(), key)
    }
}

impl<K, V, SK, SV> UnsortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K>,
    SV: StoreMut<Elem = V>,
    K: Eq,
{
    /// Returns a mutable reference to `key`'s value, or `None` if absent — for an
    /// in-place update without the [`entry`](Self::entry) ceremony.
    ///
    /// No E0311 lifetime dance (unlike
    /// [`UnsortedMap::get_mut`](crate::UnsortedMap::get_mut)): the value column is
    /// already `&mut [V]`, so elision ties the result to `&mut self`. `key` may be any
    /// borrowed form of `K`, like [`get`](Self::get).
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let i = self.position(key)?;
        Some(&mut self.values.as_mut_slice()[i])
    }

    /// Appends a brand-new entry to both columns, or hands it back at capacity.
    ///
    /// The columns are length-locked, so a single pre-check against the combined bound
    /// guarantees both pushes below succeed — no half-insert, no rollback.
    fn push_entry(&mut self, key: K, value: V) -> Result<(), CapacityError<(K, V)>> {
        if let Some(cap) = self.capacity() {
            if self.keys.len() >= cap {
                return Err(CapacityError((key, value)));
            }
        }
        push(&mut self.keys, key).expect("capacity pre-checked above");
        push(&mut self.values, value).expect("capacity pre-checked above");
        Ok(())
    }

    /// Inserts or replaces.
    ///
    /// Replacing an existing key touches only the value column, consumes no capacity, and
    /// so can never fail — only a genuinely new key at the bound errors. O(n) lookup,
    /// O(1) to append or replace in place.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying `(key, value)` if `key` is new and the
    /// columns' combined cap is hit; replacing an existing key never errors.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, CapacityError<(K, V)>> {
        if let Some(i) = self.position(&key) {
            let slot = &mut self.values.as_mut_slice()[i];
            return Ok(Some(core::mem::replace(slot, value)));
        }
        self.push_entry(key, value).map(|()| None)
    }

    /// Removes the entry for `key`, returning its value.
    ///
    /// Swap-removes at the same index in both columns, keeping them aligned: O(1), order
    /// not preserved. `key` may be any borrowed form of `K`, like [`get`](Self::get).
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let i = self.position(key)?;
        self.keys.swap_remove_at(i);
        Some(self.values.swap_remove_at(i))
    }

    /// Resolves `key`'s slot **once** and returns a [`ColumnEntry`] for an
    /// insert-or-update, avoiding the second scan a separate [`get`](Self::get) +
    /// [`try_insert`](Self::try_insert) would pay.
    ///
    /// `O(n)` to locate; a vacant entry appends to both columns, an occupied one removes
    /// via `O(1)` lockstep swap (order not preserved).
    pub fn entry(&mut self, key: K) -> ColumnEntry<'_, SK, SV, K> {
        match self.position(&key) {
            Some(index) => ColumnEntry::Occupied(OccupiedColumnEntry::unsorted(
                &mut self.keys,
                &mut self.values,
                index,
            )),
            None => {
                let index = self.keys.len();
                ColumnEntry::Vacant(VacantColumnEntry::new(
                    &mut self.keys,
                    &mut self.values,
                    index,
                    key,
                ))
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
    /// dropped along with any entries it has not yet yielded.
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

impl<K, V, SK, SV> UnsortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + StoreNew,
    SV: StoreMut<Elem = V> + StoreNew,
    K: Eq,
{
    /// Builds from an iterator of entries, **requiring every key to be unique**.
    ///
    /// `O(n²)`: each entry's key is scanned against those already kept (no faster dedup
    /// without `Ord`), and a repeated key is rejected — a map can't drop a duplicate key
    /// without arbitrarily picking a value. For last-wins override semantics use
    /// [`try_extend`](Self::try_extend) / `extend`.
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
            map.push_entry(key, value)?; // CapacityError ->
                                         // BuildError::Capacity
        }
        Ok(map)
    }
}

impl<K, V, SK, SV> UnsortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + Unbounded,
    SV: StoreMut<Elem = V> + Unbounded,
    K: Eq,
{
    /// Infallibly inserts or replaces, returning the previous value — available only when
    /// **both** columns are [`Unbounded`].
    ///
    /// The infallible twin of [`try_insert`](Self::try_insert).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.try_insert(key, value) {
            Ok(prev) => prev,
            Err(_) => unreachable!("Unbounded columns reported a capacity failure"),
        }
    }
}

impl<K, V, SK, SV> Extend<(K, V)> for UnsortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + Unbounded,
    SV: StoreMut<Elem = V> + Unbounded,
    K: Eq,
{
    /// Extends the map, last-wins and infallible — available only when **both** columns
    /// are [`Unbounded`].
    ///
    /// As with [`UnsortedMap`](crate::UnsortedMap), there is deliberately no
    /// `FromIterator`: fresh construction rejects duplicate keys, while `extend`
    /// overrides them.
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
    use crate::{ColumnEntry, UnsortedColumnMap};

    #[test]
    fn insert_get_and_replace() {
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<&str>> = UnsortedColumnMap::new();
        assert_eq!(m.try_insert(1, "a"), Ok(None));
        assert_eq!(m.try_insert(2, "b"), Ok(None));
        assert_eq!(m.get(&1), Some(&"a"));
        assert_eq!(m.get(&9), None);
        // Replacing a key returns the old value and adds no entry.
        assert_eq!(m.try_insert(1, "A"), Ok(Some("a")));
        assert_eq!(m.get(&1), Some(&"A"));
        assert_eq!(m.len(), 2);
        // contains_key agrees with get (the vectorizable membership path).
        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&9));
    }

    #[test]
    fn remove_swaps_and_keeps_columns_aligned() {
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<&str>> = UnsortedColumnMap::new();
        m.try_extend([(1, "a"), (2, "b"), (3, "c")]).unwrap();
        // Swap-remove pulls the last entry (3, "c") into slot 0 — in *both* columns.
        assert_eq!(m.remove(&1), Some("a"));
        assert_eq!(m.keys(), &[3, 2]);
        assert_eq!(m.values(), &["c", "b"]);
        // The surviving keys still resolve to their own values (columns aligned).
        assert_eq!(m.get(&3), Some(&"c"));
        assert_eq!(m.get(&2), Some(&"b"));
        assert_eq!(m.get(&1), None);
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn try_from_iter_rejects_duplicate_key() {
        let err =
            UnsortedColumnMap::<Vec<i32>, Vec<&str>>::try_from_iter([(1, "a"), (2, "b"), (1, "z")])
                .expect_err("duplicate key 1");
        // Detected at the scan before any push, so the third entry is handed back.
        match err {
            BuildError::DuplicateKey(entry) => assert_eq!(entry, (1, "z")),
            BuildError::Capacity(_) | BuildError::Unsorted(_) => {
                panic!("expected a duplicate-key error")
            }
        }
    }

    #[test]
    fn extend_is_last_wins() {
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<&str>> = UnsortedColumnMap::new();
        m.extend([(1, "a"), (2, "b")]);
        m.extend([(2, "B"), (3, "c")]); // key 2 overridden
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(&2), Some(&"B"));
        assert_eq!(m.get(&3), Some(&"c"));
    }

    #[test]
    fn entry_or_insert_inserts_then_updates_in_one_lookup() {
        // The headline use: tally occurrences with a single scan per item.
        let mut counts: UnsortedColumnMap<Vec<&str>, Vec<u32>> = UnsortedColumnMap::new();
        for w in ["a", "b", "a", "c", "a", "b"] {
            *counts.entry(w).or_insert(0) += 1;
        }
        assert_eq!(counts.get(&"a"), Some(&3));
        assert_eq!(counts.get(&"b"), Some(&2));
        assert_eq!(counts.get(&"c"), Some(&1));
        // Vacant entries append, so the columns stay insertion-ordered and aligned.
        assert_eq!(counts.keys(), &["a", "b", "c"]);
        assert_eq!(counts.values(), &[3, 2, 1]);
    }

    #[test]
    fn entry_and_modify_then_or_insert() {
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<i32>> = UnsortedColumnMap::new();
        // Vacant: `and_modify` is a no-op, `or_insert` seeds the value.
        m.entry(1).and_modify(|v| *v += 100).or_insert(10);
        assert_eq!(m.get(&1), Some(&10));
        // Occupied: `and_modify` runs, `or_insert` is ignored.
        m.entry(1).and_modify(|v| *v += 100).or_insert(10);
        assert_eq!(m.get(&1), Some(&110));
    }

    #[test]
    fn entry_occupied_insert_and_remove_swaps_columns() {
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<&str>> = UnsortedColumnMap::new();
        m.try_extend([(1, "a"), (2, "b"), (3, "c")]).unwrap();
        match m.entry(1) {
            ColumnEntry::Occupied(mut e) => {
                assert_eq!(e.key(), &1);
                assert_eq!(e.insert("A"), "a"); // replace returns the old value
                assert_eq!(e.remove(), "A"); // then swap-remove it
            }
            ColumnEntry::Vacant(_) => panic!("key 1 is present"),
        }
        // Swap-remove pulls the last entry (3, "c") into slot 0 — in *both* columns.
        assert_eq!(m.keys(), &[3, 2]);
        assert_eq!(m.values(), &["c", "b"]);
    }

    #[test]
    fn entry_vacant_into_key_inserts_nothing() {
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<&str>> = UnsortedColumnMap::new();
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
        let mut m: UnsortedColumnMap<Vec<i32>, Vec<i32>> = UnsortedColumnMap::new();
        m.try_extend([(1, 10), (2, 20)]).unwrap();
        *m.get_mut(&2).unwrap() += 5;
        assert_eq!(m.get(&2), Some(&25));
        assert_eq!(m.get_mut(&9), None);
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.keys(), &[] as &[i32]);
        assert_eq!(m.values(), &[] as &[i32]);
        assert_eq!(m.try_insert(7, 70), Ok(None)); // both columns usable again
        assert_eq!(m.keys(), &[7]);
    }

    #[test]
    fn clone_is_independent() {
        // UnsortedColumnMap derives Clone but not PartialEq (order-sensitive).
        let mut a: UnsortedColumnMap<Vec<i32>, Vec<&str>> = UnsortedColumnMap::new();
        a.try_extend([(1, "a"), (2, "b")]).unwrap();
        let b = a.clone();
        a.try_insert(3, "c").unwrap();
        assert_eq!(b.len(), 2); // clone unaffected by the later insert
        assert_eq!(b.get(&1), Some(&"a"));
        assert_eq!(b.get(&2), Some(&"b"));
        assert_eq!(b.get(&3), None);
    }

    // The trust-contract guards fire only in debug builds, so gate these on it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "equal length")]
    fn from_store_rejects_unequal_columns() {
        let _ = UnsortedColumnMap::<Vec<i32>, Vec<&str>>::from_store(
            Vec::from([1, 2]),
            Vec::from(["a"]),
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "free of duplicates")]
    fn from_store_rejects_duplicate_keys() {
        let _ = UnsortedColumnMap::<Vec<i32>, Vec<&str>>::from_store(
            Vec::from([1, 1]),
            Vec::from(["a", "z"]),
        );
    }
}

// heapless is the alloc-free fixed-cap backend: exercises the bounded paths and
// the capacity pre-check.
#[cfg(all(test, feature = "heapless"))]
mod heapless_tests {
    use heapless::Vec;

    use crate::UnsortedColumnMap;

    #[test]
    fn capacity_reports_fixed_bound() {
        let m: UnsortedColumnMap<Vec<u8, 4>, Vec<u8, 4>> = UnsortedColumnMap::new();
        assert_eq!(m.capacity(), Some(4));
    }

    #[test]
    fn entry_or_try_insert_respects_capacity() {
        // Cap 2, full. A bounded column has no infallible `or_insert`;
        // `or_try_insert` updates an occupied slot (no capacity used) but rejects
        // a new key against the combined cap.
        let mut m: UnsortedColumnMap<Vec<u8, 2>, Vec<u8, 2>> = UnsortedColumnMap::new();
        m.try_extend([(1, 10), (2, 20)]).unwrap();

        // Occupied update in place succeeds even at capacity.
        *m.entry(1)
            .or_try_insert(0)
            .expect("update consumes no capacity") = 11;
        assert_eq!(m.get(&1), Some(&11));

        // A genuinely new key at the bound is rejected, handing back `(key, value)`;
        // nothing is half-inserted (both columns stay length 2).
        let err = m.entry(3).or_try_insert(30).expect_err("columns are full");
        assert_eq!(err.into_inner(), (3, 30));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn try_insert_overflow_hands_back_the_pair() {
        let mut m: UnsortedColumnMap<Vec<u8, 2>, Vec<u8, 2>> = UnsortedColumnMap::new();
        m.try_insert(1, 10).unwrap();
        m.try_insert(2, 20).unwrap();
        // A genuinely new key at the bound errors, returning the whole pair; nothing
        // is half-inserted (both columns stay length 2).
        let err = m.try_insert(3, 30).expect_err("at cap 2");
        assert_eq!(err.into_inner(), (3, 30));
        assert_eq!(m.len(), 2);
        // A replacement at the bound still succeeds (consumes no capacity).
        assert_eq!(m.try_insert(2, 99), Ok(Some(20)));
    }
}

// Mixed backends: the effective cap is the tighter column's bound. Needs both
// `alloc` (the unbounded value column) and `heapless` (the bounded key column).
#[cfg(all(test, feature = "alloc", feature = "heapless"))]
mod hetero_tests {
    use alloc::vec::Vec;

    use heapless::Vec as HVec;

    use crate::UnsortedColumnMap;

    #[test]
    fn capacity_is_min_of_the_two_columns() {
        // Bounded keys (cap 2), unbounded values: the map is bounded at 2.
        let mut m: UnsortedColumnMap<HVec<u8, 2>, Vec<u16>> = UnsortedColumnMap::new();
        assert_eq!(m.capacity(), Some(2));
        m.try_insert(1, 10).unwrap();
        m.try_insert(2, 20).unwrap();
        let err = m.try_insert(3, 30).expect_err("key column is full at 2");
        assert_eq!(err.into_inner(), (3, 30));
        assert_eq!(m.len(), 2);
    }
}
