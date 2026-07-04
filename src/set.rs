//! Set collections — the **ordering** axis for `Elem = T`.
//!
//! [`SortedSet`] keeps its store ordered (`O(log n)` lookup via `binary_search`);
//! [`UnsortedSet`] appends and swap-removes (`O(1)` mutation, `O(n)` search) and
//! needs only `Eq` rather than `Ord`.

use core::borrow::Borrow;
use core::ops::{Bound, RangeBounds};

use crate::error::{BuildError, CapacityError};
use crate::store::{append_all, push, retain_in, Store, StoreMut, StoreNew, Unbounded};

mod algebra;

pub use algebra::{Difference, Intersection, SymmetricDifference, Union};

/// Resolve a `RangeBounds` over a sorted slice into index bounds via
/// `partition_point`, projecting each element to its search key with `key`
/// (identity for sets, `.0` for maps). Shared by the sorted collections'
/// `range` accessors.
pub(crate) fn subrange<T, Q: Ord + ?Sized, R: RangeBounds<Q>>(
    slice: &[T],
    range: R,
    key: impl Fn(&T) -> &Q,
) -> &[T] {
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(q) => slice.partition_point(|e| key(e) < q),
        Bound::Excluded(q) => slice.partition_point(|e| key(e) <= q),
    };
    let end = match range.end_bound() {
        Bound::Unbounded => slice.len(),
        Bound::Included(q) => slice.partition_point(|e| key(e) <= q),
        Bound::Excluded(q) => slice.partition_point(|e| key(e) < q),
    };
    // An inverted range (start > end) falls through to the slice indexing
    // panic, mirroring `BTreeMap::range`.
    &slice[start..end]
}

/// Whether any element of `haystack` compares equal to `needle` through its
/// [`Borrow`]ed form — the membership scan under the unsorted collections'
/// `contains` (shared with [`Bag`](crate::Bag) and the `soa` column map).
///
/// Mirrors the standard library's specialized `slice::contains` shape — a
/// fixed-trip OR-fold per chunk, no early exit *within* a chunk — which LLVM
/// lowers to branchless/vector compares for primitive elements. We can't just
/// call `slice::contains`: its needle must be `&T`, and borrowed-form lookups
/// only have a `&Q`. A plain early-exit `iter().any()` measured 2.5–4.5×
/// slower on `u64` hits at `n ≥ 16` (the data-dependent branch per element
/// defeats vectorization); the sub-chunk tail (`n < 8`) keeps the early-exit
/// scan, which wins at tiny `n`.
pub(crate) fn chunked_contains<T, Q>(haystack: &[T], needle: &Q) -> bool
where
    T: Borrow<Q>,
    Q: Eq + ?Sized,
{
    const LANES: usize = 8;
    let mut chunks = haystack.chunks_exact(LANES);
    for chunk in chunks.by_ref() {
        if chunk
            .iter()
            .fold(false, |hit, x| hit | (x.borrow() == needle))
        {
            return true;
        }
    }
    chunks.remainder().iter().any(|x| x.borrow() == needle)
}

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
// The stored order is canonical (sorted, deduplicated), so the structural
// derives are the semantic ones: equal sets are byte-identical stores, and the
// derived `Hash`/`PartialOrd`/`Ord` (lexicographic over elements in ascending
// order, exactly `BTreeSet`'s) are consistent with the derived `PartialEq`.
// That's what lets a `SortedSet` key another map or live in a `BTreeSet` — the
// nested-collections niche the crate is built for. The unsorted twin can
// derive none of these (swap-remove makes its stored order incidental).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Borrow the backing store — the door to backend-specific introspection
    /// the collection API doesn't abstract: `spilled()` on a `SmallVec`,
    /// [`is_spilled`](crate::Spill::is_spilled) on a [`Spill`](crate::Spill),
    /// a backend's inherent `capacity()` for *allocated* (not logical)
    /// capacity. Shared-ref only: `&mut` access could break the
    /// sorted-and-deduplicated invariant that
    /// [`from_store`](Self::from_store) trusts.
    ///
    /// ```
    /// use pouch::Set;
    /// let mut s: Set<u32, 2> = Set::default();
    /// s.insert(1);
    /// s.insert(2);
    /// assert!(!s.store().spilled()); // still inline
    /// s.insert(3);
    /// assert!(s.store().spilled()); // outgrew N = 2 — now on the heap
    /// ```
    pub fn store(&self) -> &S {
        &self.store
    }
    /// Consume the set and hand back its store, elements intact and still in
    /// ascending order — the inverse of [`from_store`](Self::from_store), for
    /// reusing the buffer or handing a sorted `Vec` to an API that wants one.
    pub fn into_store(self) -> S {
        self.store
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
    /// Iterate the elements in ascending order.
    pub fn iter(&self) -> core::slice::Iter<'_, S::Elem> {
        self.store.as_slice().iter()
    }
    /// The smallest element, or `None` if empty. `O(1)`.
    pub fn first(&self) -> Option<&S::Elem> {
        self.store.as_slice().first()
    }
    /// The largest element, or `None` if empty. `O(1)`.
    pub fn last(&self) -> Option<&S::Elem> {
        self.store.as_slice().last()
    }
    /// Whether `value` is in the set. `O(log n)`. `value` may be any borrowed
    /// form of the element type — a `SortedSet<Vec<String>>` answers
    /// `contains("x")` without allocating a `String` to ask — as long as the
    /// borrowed form's `Ord` agrees with the element type's (the [`Borrow`]
    /// contract, as in the std collections).
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        S::Elem: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.store
            .as_slice()
            .binary_search_by(|e| e.borrow().cmp(value))
            .is_ok()
    }
    /// The elements within `range`, as a subslice of the sorted store — the
    /// sorted layout's native range query. Two `O(log n)` bound searches, zero
    /// copies; iterate, index, or re-slice the result freely. The bounds may be
    /// any borrowed form of the element type, like [`contains`](Self::contains);
    /// as with `BTreeSet::range`, an **unsized** form (`str`, `[u8]`) needs the
    /// explicit tuple-of-`Bound`s shape — range sugar like `"a".."m"` is a
    /// `Range<&str>`, which can only bound `&str` itself:
    /// `set.range::<str, _>((Bound::Included("a"), Bound::Excluded("m")))`.
    ///
    /// ```
    /// use pouch::Set;
    /// let s: Set<u32> = (0..10).collect();
    /// assert_eq!(s.range(3..6), &[3, 4, 5]);
    /// assert_eq!(s.range(7..), &[7, 8, 9]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the range's start is greater than its end.
    pub fn range<Q, R>(&self, range: R) -> &[S::Elem]
    where
        S::Elem: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        subrange(self.store.as_slice(), range, |e| e.borrow())
    }

    // --- Set algebra ------------------------------------------------------
    // All merge walks over the two already-sorted slices: `O(n + m)`, no
    // allocation, results in ascending order. `other` may use a *different*
    // store (`S2`) — a heap set can union with a `static` `SliceSet` table.

    /// Whether every element of `self` is in `other`. `O(n + m)` merge walk,
    /// switching to `O(n log m)` binary searches when `self` is ≥16× smaller.
    pub fn is_subset<S2>(&self, other: &SortedSet<S2>) -> bool
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        algebra::is_subset(self.as_slice(), other.as_slice())
    }

    /// Whether every element of `other` is in `self` —
    /// [`is_subset`](Self::is_subset) with the arguments flipped.
    pub fn is_superset<S2>(&self, other: &SortedSet<S2>) -> bool
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        other.is_subset(self)
    }

    /// Whether `self` and `other` share no element. `O(n + m)` merge walk,
    /// switching to binary searches when one side is ≥16× smaller.
    pub fn is_disjoint<S2>(&self, other: &SortedSet<S2>) -> bool
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        algebra::is_disjoint(self.as_slice(), other.as_slice())
    }

    /// Iterate the elements in `self`, `other`, or both, ascending — each
    /// shared element once. Collect into an [`Unbounded`] set with
    /// `.cloned().collect()`, or `try_extend` a bounded one.
    ///
    /// ```
    /// use pouch::Set;
    /// let a: Set<u32> = [1, 2, 3].into_iter().collect();
    /// let b: Set<u32> = [2, 3, 4].into_iter().collect();
    /// assert!(a.union(&b).eq(&[1, 2, 3, 4]));
    /// assert!(a.intersection(&b).eq(&[2, 3]));
    /// assert!(a.difference(&b).eq(&[1]));
    /// assert!(a.symmetric_difference(&b).eq(&[1, 4]));
    /// ```
    pub fn union<'a, S2>(&'a self, other: &'a SortedSet<S2>) -> Union<'a, S::Elem>
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        Union::new(self.as_slice(), other.as_slice())
    }

    /// Iterate the elements in both `self` and `other`, ascending. See
    /// [`union`](Self::union).
    pub fn intersection<'a, S2>(&'a self, other: &'a SortedSet<S2>) -> Intersection<'a, S::Elem>
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        Intersection::new(self.as_slice(), other.as_slice())
    }

    /// Iterate the elements in `self` but not `other`, ascending. See
    /// [`union`](Self::union).
    pub fn difference<'a, S2>(&'a self, other: &'a SortedSet<S2>) -> Difference<'a, S::Elem>
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        Difference::new(self.as_slice(), other.as_slice())
    }

    /// Iterate the elements in exactly one of `self`, `other`, ascending. See
    /// [`union`](Self::union).
    pub fn symmetric_difference<'a, S2>(
        &'a self,
        other: &'a SortedSet<S2>,
    ) -> SymmetricDifference<'a, S::Elem>
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Ord,
    {
        SymmetricDifference::new(self.as_slice(), other.as_slice())
    }
}

impl<S: StoreMut> SortedSet<S> {
    /// Remove every element, keeping the backing store's allocated capacity. Needs
    /// no `Ord` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
    /// Pre-allocate so at least `additional` more elements fit without a
    /// reallocation — pay the growth once up front instead of as spikes
    /// mid-burst ([`StoreMut::reserve`]).
    /// Stores that never reallocate (fixed-capacity, borrowed) have nothing to
    /// do; a [`Spill`](crate::Spill) pre-arms its spill tier so even the
    /// migration allocates nothing. To *start* with capacity, wrap a pre-sized
    /// store instead: `SortedSet::from_store(Vec::with_capacity(n))`.
    ///
    /// ```
    /// use pouch::SortedSet;
    /// let mut s: SortedSet<Vec<u64>> = SortedSet::new();
    /// s.reserve(1_000); // one allocation now, none during the burst
    /// for x in 0..1_000 {
    ///     s.insert(x);
    /// }
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.store.reserve(additional);
    }
    /// Keep only the elements for which `f` returns `true`, preserving order.
    /// `O(n)`, and needs no `Ord` bound — dropping elements can't unsort the rest.
    pub fn retain<F: FnMut(&S::Elem) -> bool>(&mut self, mut f: F) {
        retain_in(&mut self.store, |e| f(e));
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

    /// Remove `value`, returning whether it was present. Order-preserving
    /// shift: `O(log n)` search, `O(n)` shift. `value` may be any borrowed form
    /// of the element type, like [`contains`](Self::contains).
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        S::Elem: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self
            .store
            .as_slice()
            .binary_search_by(|e| e.borrow().cmp(value))
        {
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
    /// returned as [`BuildError::Unsorted`] rather than silently trusted. The
    /// check is one comparison per item — the same one the dedup already needs. A
    /// bounded store that fills yields [`BuildError::Capacity`]. (A set build
    /// never returns [`BuildError::DuplicateKey`] — duplicates dedup silently.)
    pub fn try_from_sorted_iter<I>(iter: I) -> Result<Self, BuildError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        let mut store = S::new();
        let iter = iter.into_iter();
        store.reserve(iter.size_hint().0);
        for value in iter {
            if let Some(prev) = store.as_slice().last() {
                if value < *prev {
                    return Err(BuildError::Unsorted(value));
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
            // Capacity can't happen on an Unbounded store and a set build dedups
            // rather than erroring; misordered input can happen — and an
            // infallible builder has no error channel, so it must panic.
            Err(BuildError::Capacity(_)) => {
                unreachable!("Unbounded store reported a capacity failure")
            }
            Err(BuildError::DuplicateKey(_)) => {
                unreachable!("set builds dedup duplicates silently")
            }
            Err(BuildError::Unsorted(_)) => {
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

impl<'a, S: Store> IntoIterator for &'a SortedSet<S> {
    type Item = &'a S::Elem;
    type IntoIter = core::slice::Iter<'a, S::Elem>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Consume the set, yielding its elements in ascending order. Available when
/// the backing store is itself consumable into its elements (every owning
/// backend is; a borrowed `&[T]` store is not — it can't give up owned values).
impl<S> IntoIterator for SortedSet<S>
where
    S: Store + IntoIterator<Item = <S as Store>::Elem>,
{
    type Item = S::Elem;
    type IntoIter = <S as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
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
// Derives `Clone` but not `PartialEq`/`Eq` (nor `Hash`/`Ord`): correct set
// equality is order-independent, yet swap-remove lets two equal sets store their
// elements in different orders, so a structural derive would wrongly call them
// unequal. The sorted twin derives all of these because its stored order is
// canonical.
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
    /// Borrow the backing store, for backend-specific introspection
    /// (`spilled()`, allocated capacity, …) — see
    /// [`SortedSet::store`](crate::SortedSet::store). Shared-ref only: `&mut`
    /// access could smuggle in a duplicate, breaking the set invariant.
    pub fn store(&self) -> &S {
        &self.store
    }
    /// Consume the set and hand back its store, elements intact (in no
    /// particular order) — the inverse of [`from_store`](Self::from_store).
    pub fn into_store(self) -> S {
        self.store
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
    /// Iterate the elements, in no particular order.
    pub fn iter(&self) -> core::slice::Iter<'_, S::Elem> {
        self.store.as_slice().iter()
    }
    /// Whether `value` is in the set. `O(n)` linear scan. `value` may be any
    /// borrowed form of the element type — a `String` set answers
    /// `contains("x")` without allocating — with the usual [`Borrow`] contract
    /// that the borrowed form's `Eq` agrees with the element type's.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        S::Elem: Borrow<Q> + Eq,
        Q: Eq + ?Sized,
    {
        chunked_contains(self.store.as_slice(), value)
    }

    // Without an order there is no merge walk, so the predicates are scans:
    // `O(n·m)`, honest about what `Eq`-only membership costs. For the
    // element-yielding algebra (union & co.) use the sorted flavor. As there,
    // `other` may use a different store.

    /// Whether every element of `self` is in `other`. `O(n·m)` — a
    /// [`contains`](Self::contains) scan per element.
    pub fn is_subset<S2>(&self, other: &UnsortedSet<S2>) -> bool
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Eq,
    {
        self.len() <= other.len() && self.iter().all(|x| other.contains(x))
    }

    /// Whether every element of `other` is in `self` —
    /// [`is_subset`](Self::is_subset) with the arguments flipped.
    pub fn is_superset<S2>(&self, other: &UnsortedSet<S2>) -> bool
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Eq,
    {
        other.is_subset(self)
    }

    /// Whether `self` and `other` share no element. `O(n·m)`, scanning with
    /// the smaller set on the outside.
    pub fn is_disjoint<S2>(&self, other: &UnsortedSet<S2>) -> bool
    where
        S2: Store<Elem = S::Elem>,
        S::Elem: Eq,
    {
        if self.len() <= other.len() {
            self.iter().all(|x| !other.contains(x))
        } else {
            other.iter().all(|x| !self.contains(x))
        }
    }
}

impl<S: StoreMut> UnsortedSet<S> {
    /// Remove every element, keeping the backing store's allocated capacity. Needs
    /// no `Eq` bound — it only truncates the store.
    pub fn clear(&mut self) {
        self.store.clear();
    }
    /// Pre-allocate so at least `additional` more elements fit without a
    /// reallocation — see [`SortedSet::reserve`](crate::SortedSet::reserve).
    pub fn reserve(&mut self, additional: usize) {
        self.store.reserve(additional);
    }
    /// Keep only the elements for which `f` returns `true`. `O(n)`; needs no
    /// `Eq` bound.
    pub fn retain<F: FnMut(&S::Elem) -> bool>(&mut self, mut f: F) {
        retain_in(&mut self.store, |e| f(e));
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
    /// `value` may be any borrowed form of the element type, like
    /// [`contains`](Self::contains).
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        S::Elem: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        match self
            .store
            .as_slice()
            .iter()
            .position(|e| e.borrow() == value)
        {
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

impl<'a, S: Store> IntoIterator for &'a UnsortedSet<S> {
    type Item = &'a S::Elem;
    type IntoIter = core::slice::Iter<'a, S::Elem>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Consume the set, yielding its elements in no particular order. Available
/// when the backing store is itself consumable into its elements.
impl<S> IntoIterator for UnsortedSet<S>
where
    S: Store + IntoIterator<Item = <S as Store>::Elem>,
{
    type Item = S::Elem;
    type IntoIter = <S as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
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

    use crate::{BuildError, SortedSet, UnsortedSet};

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
            BuildError::Unsorted(x) => assert_eq!(x, 2),
            BuildError::Capacity(_) | BuildError::DuplicateKey(_) => {
                panic!("expected an unsorted error")
            }
        }
    }

    #[test]
    fn iter_and_into_iter_yield_sorted_elements() {
        let set: SortedSet<Vec<i32>> = [3, 1, 2].into_iter().collect();
        // by-ref: `iter()` and `&set` are the same slice iterator.
        assert!(set.iter().eq(&[1, 2, 3]));
        let doubled: Vec<i32> = (&set).into_iter().map(|x| x * 2).collect();
        assert_eq!(doubled, &[2, 4, 6]);
        // by-value: consuming the set yields owned elements, still ascending.
        let owned: Vec<i32> = set.into_iter().collect();
        assert_eq!(owned, &[1, 2, 3]);

        let unsorted: UnsortedSet<Vec<i32>> = [3, 1, 2].into_iter().collect();
        assert_eq!(unsorted.iter().count(), 3);
        let mut owned: Vec<i32> = unsorted.into_iter().collect();
        owned.sort_unstable();
        assert_eq!(owned, &[1, 2, 3]);
    }

    #[test]
    fn first_last_and_range() {
        let set: SortedSet<Vec<i32>> = SortedSet::from_sorted_iter([1, 3, 5, 7, 9]);
        assert_eq!(set.first(), Some(&1));
        assert_eq!(set.last(), Some(&9));
        // range is a subslice: half-open, inclusive, and open-ended bounds, with
        // bounds that fall between elements.
        assert_eq!(set.range(3..7), &[3, 5]);
        assert_eq!(set.range(3..=7), &[3, 5, 7]);
        assert_eq!(set.range(..4), &[1, 3]);
        assert_eq!(set.range(4..), &[5, 7, 9]);
        // A full range can't infer the borrowed key type (every `Q` fits
        // `RangeFull`), so it takes a turbofish — same as `BTreeSet::range`.
        assert_eq!(set.range::<i32, _>(..), &[1, 3, 5, 7, 9]);
        assert_eq!(set.range(4..4), &[] as &[i32]);

        let empty: SortedSet<Vec<i32>> = SortedSet::new();
        assert_eq!(empty.first(), None);
        assert_eq!(empty.range::<i32, _>(..), &[] as &[i32]);
    }

    #[test]
    fn set_algebra_merges_ascending() {
        let a: SortedSet<Vec<u32>> = [1, 2, 3, 5].into_iter().collect();
        let b: SortedSet<Vec<u32>> = [2, 3, 4].into_iter().collect();
        assert!(a.union(&b).eq(&[1, 2, 3, 4, 5]));
        assert!(a.intersection(&b).eq(&[2, 3]));
        assert!(a.difference(&b).eq(&[1, 5]));
        assert!(b.difference(&a).eq(&[4])); // difference is directional
        assert!(a.symmetric_difference(&b).eq(&[1, 4, 5]));

        // The iterators collect straight back into a set.
        let u: SortedSet<Vec<u32>> = a.union(&b).copied().collect();
        assert_eq!(u.as_slice(), &[1, 2, 3, 4, 5]);

        // Cross-store: the other set only contributes a sorted slice, so a
        // heap set can union with a read-only static-table SliceSet.
        static TABLE: [u32; 2] = [4, 6];
        let table = crate::SliceSet::from_store(&TABLE[..]);
        assert!(a.union(&table).eq(&[1, 2, 3, 4, 5, 6]));

        // Empty edges.
        let empty: SortedSet<Vec<u32>> = SortedSet::new();
        assert!(empty.union(&a).eq(a.iter()));
        assert_eq!(empty.intersection(&a).count(), 0);
        assert!(empty.difference(&a).next().is_none());
        assert!(a.symmetric_difference(&empty).eq(a.iter()));
    }

    #[test]
    fn subset_superset_disjoint() {
        let small: SortedSet<Vec<u32>> = [2, 40].into_iter().collect();
        // 64 elements: the ≥16× size gap exercises the binary-search path.
        let big: SortedSet<Vec<u32>> = (0..64).map(|x| x * 2).collect();
        assert!(small.is_subset(&big));
        assert!(big.is_superset(&small));
        assert!(!big.is_subset(&small));
        assert!(!small.is_disjoint(&big));
        let odd: SortedSet<Vec<u32>> = [1, 3, 41].into_iter().collect();
        assert!(small.is_disjoint(&odd) && odd.is_disjoint(&big));

        // Similar sizes take the linear merge path.
        let cover: SortedSet<Vec<u32>> = [2, 4, 40].into_iter().collect();
        assert!(small.is_subset(&cover));
        assert!(!cover.is_subset(&small)); // longer than `small`
        let empty: SortedSet<Vec<u32>> = SortedSet::new();
        assert!(empty.is_subset(&small) && small.is_superset(&empty));
        assert!(empty.is_disjoint(&empty));

        // The unsorted flavor gets the Eq-only scan predicates.
        let ua: UnsortedSet<Vec<u32>> = [3, 1].into_iter().collect();
        let ub: UnsortedSet<Vec<u32>> = [1, 2, 3].into_iter().collect();
        assert!(ua.is_subset(&ub));
        assert!(ub.is_superset(&ua));
        assert!(!ua.is_disjoint(&ub));
        let uc: UnsortedSet<Vec<u32>> = [9].into_iter().collect();
        assert!(ua.is_disjoint(&uc));
    }

    // The sorted flavor derives `PartialOrd`/`Ord`/`Hash` off its canonical
    // stored order, so it nests: element-lexicographic ordering (BTreeSet's),
    // membership in an outer BTreeSet, dedup of equal inner sets.
    #[test]
    fn sorted_set_orders_and_nests() {
        use alloc::collections::BTreeSet;

        let a: SortedSet<Vec<i32>> = [1, 2].into_iter().collect();
        let b: SortedSet<Vec<i32>> = [1, 3].into_iter().collect();
        let prefix: SortedSet<Vec<i32>> = [1].into_iter().collect();
        assert!(a < b); // element-lexicographic, like BTreeSet
        assert!(prefix < a); // a strict prefix sorts first

        let mut nested: BTreeSet<SortedSet<Vec<i32>>> = BTreeSet::new();
        nested.insert(a.clone());
        nested.insert(b);
        nested.insert(a.clone()); // equal inner set — deduped by the outer set
        assert_eq!(nested.len(), 2);
        assert!(nested.contains(&a));
        assert_eq!(nested.iter().next(), Some(&a)); // smallest first
    }

    // `store`/`into_store` round-trip with `from_store`, and `reserve` is
    // observable through `store()` via `Vec`'s *inherent* (allocated) capacity.
    #[test]
    fn store_access_and_reserve() {
        let mut set: SortedSet<Vec<i32>> = SortedSet::new();
        assert_eq!(set.store().capacity(), 0); // Vec's own capacity: allocated
        set.reserve(100);
        assert!(set.store().capacity() >= 100); // one allocation, up front
        let before = set.store().capacity();
        set.extend([3, 1, 2]);
        assert_eq!(set.store().capacity(), before); // burst caused no growth

        // into_store hands back the sorted, deduplicated Vec; from_store
        // round-trips it.
        let v: Vec<i32> = set.into_store();
        assert_eq!(v, &[1, 2, 3]);
        let set2: SortedSet<Vec<i32>> = SortedSet::from_store(v);
        assert!(set2.contains(&2));
    }

    // The on-mission `Borrow` payoff: `String` elements, `&str` queries — no
    // allocation to ask, in either flavor.
    #[test]
    fn lookups_take_borrowed_forms() {
        use alloc::string::{String, ToString};
        use core::ops::Bound;

        let mut set: SortedSet<Vec<String>> =
            ["b", "a", "c"].iter().map(ToString::to_string).collect();
        assert!(set.contains("b"));
        assert!(!set.contains("z"));
        // Unsized bounds (`str`) need the tuple-of-`Bound`s shape (range sugar
        // like `"a".."c"` is a `Range<&str>`, which can only bound `&str`).
        assert_eq!(
            set.range::<str, _>((Bound::Included("a"), Bound::Excluded("c"))),
            &["a".to_string(), "b".to_string()]
        );
        assert!(set.remove("a"));
        assert!(!set.contains("a"));

        let mut unsorted: UnsortedSet<Vec<String>> =
            ["x", "y"].iter().map(ToString::to_string).collect();
        assert!(unsorted.contains("x"));
        assert!(!unsorted.contains("z"));
        assert!(unsorted.remove("x"));
        assert!(!unsorted.contains("x"));
    }

    #[test]
    fn retain_keeps_matching_elements_in_order() {
        let mut set: SortedSet<Vec<i32>> = (1..=6).collect();
        set.retain(|x| x % 2 == 0);
        assert_eq!(set.as_slice(), &[2, 4, 6]); // still sorted
        set.retain(|_| false);
        assert!(set.is_empty());

        let mut unsorted: UnsortedSet<Vec<i32>> = (1..=6).collect();
        unsorted.retain(|x| x % 2 == 1);
        assert_eq!(unsorted.len(), 3);
        assert!(unsorted.contains(&1) && unsorted.contains(&3) && unsorted.contains(&5));
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
