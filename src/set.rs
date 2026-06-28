//! Set collections — the **ordering** axis for `Elem = T`.
//!
//! [`SortedSet`] keeps its store ordered (`O(log n)` lookup via `binary_search`);
//! [`UnsortedSet`] appends and swap-removes (`O(1)` mutation, `O(n)` search) and
//! needs only `Eq` rather than `Ord`.

use crate::error::{CapacityError, SortedBuildError};
use crate::store::{append_all, push, Store, StoreMut, StoreNew, Unbounded};

/// Re-establish the sorted-set invariant over a store filled in arbitrary order:
/// `sort_unstable` (allocation-free, so `core`-only) then drop adjacent
/// duplicates. `O(n log n)` sort + `O(n)` dedup. The dedup keeps the first of
/// each equal run, compacting distinct values to the front with `swap` (no `Copy`
/// bound) before popping the duplicate tail — each pop is `remove_at(len - 1)`,
/// which is `O(1)` on every backend.
fn sort_dedup<S>(store: &mut S)
where
    S: StoreMut,
    S::Elem: Ord,
{
    store.as_mut_slice().sort_unstable();
    let s = store.as_mut_slice();
    if s.is_empty() {
        return;
    }
    let mut write = 0;
    for read in 1..s.len() {
        if s[read] != s[write] {
            write += 1;
            if write != read {
                s.swap(write, read);
            }
        }
    }
    let keep = write + 1;
    while store.len() > keep {
        store.remove_at(store.len() - 1);
    }
}

/// A set kept sorted in its backing store; lookup is `binary_search`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortedSet<S> {
    store: S,
}

impl<S: StoreNew> SortedSet<S> {
    pub fn new() -> Self {
        SortedSet { store: S::new() }
    }
}

impl<S: StoreNew> Default for SortedSet<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Store> SortedSet<S> {
    /// Wrap a store **assumed already sorted and free of duplicates** — the
    /// invariant `binary_search` (and thus [`contains`](Self::contains) /
    /// [`try_insert`](Self::try_insert) / [`remove`](Self::remove)) relies on. No
    /// sort is performed; an out-of-order or duplicate-bearing store yields wrong
    /// lookups. The precondition is only `debug_assert!`-checked (zero cost in
    /// release). For a runtime-checked ascending build use
    /// [`try_from_sorted_iter`](Self::try_from_sorted_iter); to build from arbitrary
    /// input use [`try_from_iter`](Self::try_from_iter).
    pub fn from_store(store: S) -> Self
    where
        S::Elem: Ord,
    {
        debug_assert!(
            store.as_slice().windows(2).all(|w| w[0] < w[1]),
            "SortedSet::from_store: store must be sorted and free of duplicates",
        );
        SortedSet { store }
    }
    pub fn len(&self) -> usize {
        self.store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    pub fn capacity(&self) -> Option<usize> {
        self.store.capacity()
    }
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
    pub fn contains(&self, value: &S::Elem) -> bool
    where
        S::Elem: Ord,
    {
        self.store.as_slice().binary_search(value).is_ok()
    }
}

impl<S: StoreMut> SortedSet<S> {
    /// Remove every element, keeping the backing store's allocated capacity. Needs
    /// no `Ord` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

impl<S: StoreMut> SortedSet<S>
where
    S::Elem: Ord,
{
    /// Insert preserving order. `Ok(true)` if newly added, `Ok(false)` if already
    /// present (a duplicate consumes no capacity and never errors).
    pub fn try_insert(&mut self, value: S::Elem) -> Result<bool, CapacityError<S::Elem>> {
        match self.store.as_slice().binary_search(&value) {
            Ok(_) => Ok(false),
            Err(i) => self.store.try_insert_at(i, value).map(|()| true),
        }
    }

    pub fn remove(&mut self, value: &S::Elem) -> bool {
        match self.store.as_slice().binary_search(value) {
            Ok(i) => {
                self.store.remove_at(i);
                true
            }
            Err(_) => false,
        }
    }

    /// Insert every item from `iter`, one at a time. `O(k·n)` for `k` items —
    /// each is a [`try_insert`](Self::try_insert), so duplicates never consume
    /// capacity and the set stays valid even if a bounded store fills partway: on
    /// error the items inserted so far are kept and the rejected element returned.
    /// For bulk loads into an [`Unbounded`] store, `.extend()` / `.collect()` use
    /// a faster `O(n log n)` sort-once path instead.
    ///
    /// Only that one rejected element is recoverable: the iterator is dropped on
    /// error along with any items it has not yet yielded. Drive
    /// [`try_insert`](Self::try_insert) yourself over an iterator you keep if the
    /// unconsumed tail must survive an overflow.
    pub fn try_extend<I>(&mut self, iter: I) -> Result<(), CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        for value in iter {
            self.try_insert(value)?;
        }
        Ok(())
    }
}

impl<S: StoreMut + StoreNew> SortedSet<S>
where
    S::Elem: Ord,
{
    /// Build from an arbitrary (unordered) iterator in `O(n log n)`: append every
    /// item, then sort and drop duplicates once. Beats repeated
    /// [`try_insert`](Self::try_insert) — which is `O(n²)`, a shift per element —
    /// for bulk construction.
    ///
    /// Errors with the rejected element if the store fills. Note for bounded
    /// backends: items are appended *before* the dedup pass, so a duplicate-heavy
    /// iterator can overflow the bound even when the deduplicated result would
    /// fit. Use [`try_insert`](Self::try_insert) in a loop if duplicates must
    /// never consume capacity.
    pub fn try_from_iter<I>(iter: I) -> Result<Self, CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        let mut store = S::new();
        append_all(&mut store, iter)?;
        sort_dedup(&mut store);
        Ok(Self::from_store(store))
    }

    /// Build from an iterator whose items are already in ascending order, in
    /// `O(n)` — no sort, no shifting, just an append per distinct element. Equal
    /// neighbours are dropped, so duplicate runs in the input are fine.
    ///
    /// Unlike [`from_store`](Self::from_store), the ascending-order promise is
    /// enforced in every build profile: an item smaller than its predecessor is
    /// returned as [`SortedBuildError::Unsorted`] rather than silently trusted. The
    /// check is one comparison per item — the same one the dedup already needs. A
    /// bounded store that fills yields [`SortedBuildError::Capacity`].
    pub fn try_from_sorted_iter<I>(iter: I) -> Result<Self, SortedBuildError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        let mut store = S::new();
        for value in iter {
            if let Some(prev) = store.as_slice().last() {
                if value < *prev {
                    return Err(SortedBuildError::Unsorted(value));
                }
                if *prev == value {
                    continue; // adjacent duplicate — sets dedup silently
                }
            }
            push(&mut store, value)?;
        }
        Ok(Self::from_store(store))
    }
}

impl<S: StoreMut + Unbounded> SortedSet<S>
where
    S::Elem: Ord,
{
    /// Infallible insert — available only when the backing store is [`Unbounded`].
    pub fn insert(&mut self, value: S::Elem) -> bool {
        match self.try_insert(value) {
            Ok(b) => b,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<S: StoreMut + StoreNew + Unbounded> SortedSet<S>
where
    S::Elem: Ord,
{
    /// Infallible [`try_from_sorted_iter`](Self::try_from_sorted_iter) — available
    /// only for an [`Unbounded`] store. `O(n)`.
    pub fn from_sorted_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = S::Elem>,
    {
        match Self::try_from_sorted_iter(iter) {
            Ok(set) => set,
            // Capacity can't happen on an Unbounded store; misordered input can —
            // and an infallible builder has no error channel, so it must panic.
            Err(SortedBuildError::Capacity(_)) => {
                unreachable!("Unbounded store reported a capacity failure")
            }
            Err(SortedBuildError::Unsorted(_)) => {
                panic!("from_sorted_iter: input was not in ascending order")
            }
        }
    }
}

impl<S> FromIterator<S::Elem> for SortedSet<S>
where
    S: StoreMut + StoreNew + Unbounded,
    S::Elem: Ord,
{
    /// `.collect()` into a sorted set. `O(n log n)`; see
    /// [`try_from_iter`](Self::try_from_iter) for the bounded-store counterpart.
    fn from_iter<I: IntoIterator<Item = S::Elem>>(iter: I) -> Self {
        match SortedSet::try_from_iter(iter) {
            Ok(set) => set,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<S> Extend<S::Elem> for SortedSet<S>
where
    S: StoreMut + Unbounded,
    S::Elem: Ord,
{
    /// Append every item, then sort and dedup once — `O((n + k) log(n + k))`,
    /// faster than [`try_extend`](Self::try_extend)'s one-at-a-time shift for a
    /// large `k`. Only for [`Unbounded`] stores; bounded ones use `try_extend`.
    fn extend<I: IntoIterator<Item = S::Elem>>(&mut self, iter: I) {
        if append_all(&mut self.store, iter).is_err() {
            unreachable!("Unbounded store reported a capacity failure");
        }
        sort_dedup(&mut self.store);
    }
}

/// A set with no ordering: membership is a linear scan, insert appends, delete
/// swap-removes. The unsorted counterpart of [`SortedSet`] — prefer it when
/// `Elem` isn't `Ord`, or when n is small enough that skipping the sorted
/// insert's shift wins.
// Derives `Clone` but not `PartialEq`/`Eq`: correct set equality is
// order-independent, yet swap-remove lets two equal sets store their elements in
// different orders, so a structural derive would wrongly call them unequal. The
// sorted twin derives equality because its stored order is canonical.
#[derive(Clone, Debug)]
pub struct UnsortedSet<S> {
    store: S,
}

impl<S: StoreNew> UnsortedSet<S> {
    pub fn new() -> Self {
        UnsortedSet { store: S::new() }
    }
}

impl<S: StoreNew> Default for UnsortedSet<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Store> UnsortedSet<S> {
    /// Wrap a store **assumed free of duplicates** — the set invariant. No scan is
    /// performed; duplicates would inflate `len` and let the same value be removed
    /// twice. The precondition is `debug_assert!`-checked (zero cost in release).
    /// To build from arbitrary input, use [`try_from_iter`](Self::try_from_iter).
    pub fn from_store(store: S) -> Self
    where
        S::Elem: Eq,
    {
        debug_assert!(
            {
                let s = store.as_slice();
                !(1..s.len()).any(|i| s[..i].contains(&s[i]))
            },
            "UnsortedSet::from_store: store must be free of duplicates",
        );
        UnsortedSet { store }
    }
    pub fn len(&self) -> usize {
        self.store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    pub fn capacity(&self) -> Option<usize> {
        self.store.capacity()
    }
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
    pub fn contains(&self, value: &S::Elem) -> bool
    where
        S::Elem: Eq,
    {
        self.store.as_slice().contains(value)
    }
}

impl<S: StoreMut> UnsortedSet<S> {
    /// Remove every element, keeping the backing store's allocated capacity. Needs
    /// no `Eq` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

impl<S: StoreMut> UnsortedSet<S>
where
    S::Elem: Eq,
{
    /// Append at the tail. `Ok(true)` if newly added, `Ok(false)` if already
    /// present (a duplicate consumes no capacity and never errors). O(n) to
    /// reject a duplicate, O(1) to append.
    pub fn try_insert(&mut self, value: S::Elem) -> Result<bool, CapacityError<S::Elem>> {
        if self.store.as_slice().contains(&value) {
            return Ok(false);
        }
        push(&mut self.store, value).map(|()| true)
    }

    /// Remove by swapping in the last element (O(1)); does not preserve order.
    pub fn remove(&mut self, value: &S::Elem) -> bool {
        match self.store.as_slice().iter().position(|e| e == value) {
            Some(i) => {
                self.store.swap_remove_at(i);
                true
            }
            None => false,
        }
    }

    /// Insert every item from `iter`, skipping duplicates. `O(k·n)` for `k` items;
    /// on error the items inserted so far are kept and the rejected element
    /// returned. Without `Ord` there is no faster dedup — for bulk loads prefer a
    /// [`SortedSet`].
    ///
    /// Only that one rejected element is recoverable: the iterator is dropped on
    /// error along with any items it has not yet yielded. Drive
    /// [`try_insert`](Self::try_insert) yourself over an iterator you keep if the
    /// unconsumed tail must survive an overflow.
    pub fn try_extend<I>(&mut self, iter: I) -> Result<(), CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        for value in iter {
            self.try_insert(value)?;
        }
        Ok(())
    }
}

impl<S: StoreMut + StoreNew> UnsortedSet<S>
where
    S::Elem: Eq,
{
    /// Build from an iterator, skipping duplicates. `O(n²)`: each item is scanned
    /// against those already kept (an unsorted set has no faster dedup without
    /// `Ord`). For large inputs prefer a [`SortedSet`].
    pub fn try_from_iter<I>(iter: I) -> Result<Self, CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        let mut set = Self::new();
        set.try_extend(iter)?;
        Ok(set)
    }
}

impl<S: StoreMut + Unbounded> UnsortedSet<S>
where
    S::Elem: Eq,
{
    /// Infallible insert — available only when the backing store is [`Unbounded`].
    pub fn insert(&mut self, value: S::Elem) -> bool {
        match self.try_insert(value) {
            Ok(b) => b,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<S> FromIterator<S::Elem> for UnsortedSet<S>
where
    S: StoreMut + StoreNew + Unbounded,
    S::Elem: Eq,
{
    /// `.collect()` into an unsorted set, skipping duplicates. `O(n²)`.
    fn from_iter<I: IntoIterator<Item = S::Elem>>(iter: I) -> Self {
        match UnsortedSet::try_from_iter(iter) {
            Ok(set) => set,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<S> Extend<S::Elem> for UnsortedSet<S>
where
    S: StoreMut + Unbounded,
    S::Elem: Eq,
{
    fn extend<I: IntoIterator<Item = S::Elem>>(&mut self, iter: I) {
        match self.try_extend(iter) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

// Vec is the unbounded backend, so the `Unbounded`-gated paths (`collect`,
// `extend`, `from_sorted_iter`) and the fallible builders run here.
#[cfg(all(test, feature = "alloc"))]
mod alloc_tests {
    use alloc::vec::Vec;

    use crate::{SortedBuildError, SortedSet, UnsortedSet};

    #[test]
    fn try_from_iter_sorts_and_dedups() {
        let set: SortedSet<Vec<i32>> = SortedSet::try_from_iter([3, 1, 2, 3, 1, 4]).unwrap();
        assert_eq!(set.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn collect_into_sorted_set() {
        let set: SortedSet<Vec<i32>> = [5, 5, 2, 8, 2].into_iter().collect();
        assert_eq!(set.as_slice(), &[2, 5, 8]);
    }

    #[test]
    fn from_sorted_iter_drops_adjacent_runs() {
        let set: SortedSet<Vec<i32>> =
            SortedSet::try_from_sorted_iter([1, 1, 2, 3, 3, 3, 5]).unwrap();
        assert_eq!(set.as_slice(), &[1, 2, 3, 5]);
        // infallible twin, available because Vec is Unbounded.
        let set2 = SortedSet::<Vec<i32>>::from_sorted_iter([1, 2, 2, 4]);
        assert_eq!(set2.as_slice(), &[1, 2, 4]);
    }

    #[test]
    fn extend_merges_and_stays_sorted() {
        let mut set: SortedSet<Vec<i32>> = SortedSet::from_sorted_iter([2, 4, 6]);
        set.extend([5, 1, 4]); // 4 is already present
        assert_eq!(set.as_slice(), &[1, 2, 4, 5, 6]);
    }

    #[test]
    fn try_extend_keeps_order_across_calls() {
        let mut set: SortedSet<Vec<i32>> = SortedSet::new();
        set.try_extend([3, 1, 2]).unwrap();
        set.try_extend([0, 2, 5]).unwrap(); // 2 is a duplicate
        assert_eq!(set.as_slice(), &[0, 1, 2, 3, 5]);
    }

    #[test]
    fn unsorted_collect_dedups() {
        let set: UnsortedSet<Vec<i32>> = [1, 2, 1, 3, 2].into_iter().collect();
        assert_eq!(set.len(), 3);
        for x in [1, 2, 3] {
            assert!(set.contains(&x));
        }
    }

    // Order is enforced in *every* build profile (not just debug), so this is a
    // plain returned error, not a debug-only panic.
    #[test]
    fn try_from_sorted_iter_rejects_unsorted() {
        let err = SortedSet::<Vec<i32>>::try_from_sorted_iter([1, 3, 2])
            .expect_err("3 then 2 is descending");
        match err {
            SortedBuildError::Unsorted(x) => assert_eq!(x, 2),
            SortedBuildError::Capacity(_) => panic!("expected an unsorted error"),
        }
    }

    #[test]
    #[should_panic(expected = "not in ascending order")]
    fn from_sorted_iter_panics_on_unsorted() {
        let _ = SortedSet::<Vec<i32>>::from_sorted_iter([1, 3, 2]);
    }

    #[test]
    fn clear_empties_both_set_flavors() {
        let mut sorted: SortedSet<Vec<i32>> = SortedSet::from_sorted_iter([1, 2, 3]);
        sorted.clear();
        assert!(sorted.is_empty());
        assert_eq!(sorted.as_slice(), &[] as &[i32]);
        assert!(sorted.insert(5)); // usable again after clear
        assert_eq!(sorted.as_slice(), &[5]);

        let mut unsorted: UnsortedSet<Vec<i32>> = [1, 2, 3].into_iter().collect();
        unsorted.clear();
        assert!(unsorted.is_empty());
        assert!(unsorted.insert(9));
        assert_eq!(unsorted.len(), 1);
    }

    #[test]
    fn clone_and_eq_for_sorted_set() {
        let a: SortedSet<Vec<i32>> = SortedSet::from_sorted_iter([1, 2, 3]);
        let mut b = a.clone();
        assert_eq!(a, b); // PartialEq: same contents
        b.insert(4);
        assert_ne!(a, b); // the clone is independent of the original
        assert_eq!(a.as_slice(), &[1, 2, 3]);
        // Built in a different order, still equal — the stored order is canonical.
        let c: SortedSet<Vec<i32>> = [3, 1, 2, 1].into_iter().collect();
        assert_eq!(a, c);
    }

    #[test]
    fn clone_for_unsorted_set_is_independent() {
        // UnsortedSet derives Clone but not PartialEq (order-sensitive).
        let mut a: UnsortedSet<Vec<i32>> = [1, 2, 3].into_iter().collect();
        let b = a.clone();
        a.insert(4);
        assert_eq!(b.len(), 3); // clone unaffected by the later insert
        assert!(b.contains(&1) && b.contains(&2) && b.contains(&3));
    }

    // The trust-contract guards fire only in debug builds, so gate these on it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "sorted and free of duplicates")]
    fn sorted_from_store_rejects_unsorted() {
        let _ = SortedSet::from_store(alloc::vec![3, 1, 2]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "free of duplicates")]
    fn unsorted_from_store_rejects_duplicates() {
        let _ = UnsortedSet::from_store(alloc::vec![1, 2, 1]);
    }
}

// heapless is the alloc-free fixed-cap backend, so these run under
// `--no-default-features --features heapless` and exercise the bounded paths
// (bounded append, capacity overflow).
#[cfg(all(test, feature = "heapless"))]
mod heapless_tests {
    use heapless::Vec;

    use crate::SortedSet;

    #[test]
    fn try_from_iter_into_fixed_cap() {
        // Four raw items (within cap) dedup down to three after the sort pass.
        let set: SortedSet<Vec<u8, 4>> =
            SortedSet::try_from_iter([3, 1, 2, 3]).expect("raw count fits the cap");
        assert_eq!(set.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn try_from_iter_overflows_before_dedup() {
        // Items are appended raw, so the run of 3s overflows cap 3 even though the
        // deduplicated result {1, 2, 3} would fit exactly — the documented caveat.
        let err = SortedSet::<Vec<u8, 3>>::try_from_iter([1, 2, 3, 3, 3])
            .expect_err("raw append exceeds the bound");
        assert_eq!(err.into_inner(), 3);
    }

    #[test]
    fn try_from_sorted_iter_dedups_within_cap() {
        let set: SortedSet<Vec<u8, 5>> =
            SortedSet::try_from_sorted_iter([1, 1, 2, 4, 4]).expect("fits");
        assert_eq!(set.as_slice(), &[1, 2, 4]);
    }
}
