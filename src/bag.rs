//! Bag — the `Vec`-shaped facade over any [`Store`].
//!
//! A [`Bag`] holds values with duplicates allowed, in insertion order, and no
//! invariant of any kind: `try_push` appends in `O(1)`, `pop` and `swap_remove`
//! delete in `O(1)`, `remove` deletes in `O(n)` keeping order. **Its job is to
//! give the crate's *composed* stores an ergonomic sequence API.** A raw
//! [`Capped`](crate::Capped)-, [`Spill`](crate::Spill)- or
//! [`ScratchVec`](crate::ScratchVec)-built store only speaks the index-based
//! [`StoreMut`] contract; wrap it in a `Bag` and it speaks `Vec`:
//! `Bag<Capped<Vec<T>>>` is a capped vector, `Bag<Spill<ArrayVec<…>,
//! ScratchVec<…>>>` a two-tier allocation-free one. (Over a plain `Vec` or
//! `SmallVec` a `Bag` adds nothing — use the backend directly.)
//!
//! It is also the cheapest collection in the crate: it needs **no bound on the
//! element type** (no `Eq` / `Ord` / `Hash`), so bulk construction is a bare
//! append — no dedup, no sort, no duplicate-key check, and a fully
//! unconstrained `FromIterator`. The `Eq`-gated [`contains`](Bag::contains) /
//! [`count`](Bag::count) add multiset queries without constraining the core.

use core::borrow::Borrow;

use crate::error::CapacityError;
use crate::set::chunked_contains;
use crate::store::{append_all, push, retain_in, Store, StoreMut, StoreNew, Unbounded};

/// A sequence of values with duplicates allowed and no ordering or uniqueness invariant.
///
/// The crate's lightest collection — `Vec`-like over any backend, with no bound on the
/// element type.
// Derives order-sensitive `PartialEq`/`Eq`, like the `Vec` it faces: a bag is a
// sequence, so equal means the same elements in the same order — two bags a
// `swap_remove` left in different orders are different sequences, exactly as for
// `Vec` (which pairs `swap_remove` with a structural `Eq` and nobody calls its
// order incidental). This is *unlike* the unsorted set/map twins, which withhold
// these: a set's identity is order-free, so its stored order is incidental and a
// structural derive would wrongly split equal sets. A bag has no order-free
// identity — its order is observable (`get`, `as_slice`) and therefore semantic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bag<S> {
    store: S,
}

impl<S: StoreNew> Bag<S> {
    /// Creates an empty `Bag`.
    pub fn new() -> Self {
        Bag { store: S::new() }
    }
}

impl<S: StoreNew> Default for Bag<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Store> Bag<S> {
    /// Wraps a store as a bag.
    ///
    /// Every store is a valid bag (no invariant to uphold), so this is infallible and
    /// needs no element bound — the cheapest of the crate's `from_store` constructors.
    pub fn from_store(store: S) -> Self {
        Bag { store }
    }
    /// Borrows the backing store, for backend-specific introspection
    /// (`spilled()`, allocated capacity, …) — see
    /// [`SortedSet::store`](crate::SortedSet::store).
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }
    /// Consumes the bag and hands back its store, elements intact and in
    /// insertion order — the inverse of [`from_store`](Self::from_store).
    #[must_use]
    pub fn into_store(self) -> S {
        self.store
    }
    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }
    /// Returns `true` if the bag contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    /// Returns the logical capacity, or `None` if unbounded.
    #[must_use]
    pub fn max_capacity(&self) -> Option<usize> {
        self.store.max_capacity()
    }
    /// Returns the elements as a contiguous slice, in insertion order.
    #[must_use]
    pub fn as_slice(&self) -> &[S::Elem] {
        self.store.as_slice()
    }
    /// Returns an iterator over the elements in insertion order.
    pub fn iter(&self) -> core::slice::Iter<'_, S::Elem> {
        self.store.as_slice().iter()
    }
    /// Returns a reference to the element at `i` in insertion order, or `None` if out of
    /// bounds.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&S::Elem> {
        self.store.as_slice().get(i)
    }
    /// Returns `true` if any element equals `value`.
    ///
    /// `O(n)` linear scan; gated on `Eq` so the rest of the bag stays bound-free. `value`
    /// may be any borrowed form of the element type (a `String` bag answers
    /// `contains("x")` without allocating), with the usual [`Borrow`] contract that the
    /// borrowed form's `Eq` agrees with the element type's.
    #[must_use]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        S::Elem: Borrow<Q> + Eq,
        Q: Eq + ?Sized,
    {
        chunked_contains(self.store.as_slice(), value)
    }
}

impl<S: StoreMut> Bag<S> {
    /// Returns a mutable slice of the elements, for in-place edits.
    ///
    /// A bag has no invariant, so arbitrary mutation (reorder, overwrite) is always
    /// valid.
    pub fn as_mut_slice(&mut self) -> &mut [S::Elem] {
        self.store.as_mut_slice()
    }
    /// Returns a mutable iterator over the elements, in insertion order.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, S::Elem> {
        self.store.as_mut_slice().iter_mut()
    }
    /// Retains only the elements for which `f` returns `true`, preserving order. `O(n)`.
    ///
    /// The predicate gets `&mut`, so it can edit the elements it keeps — a bag has no
    /// invariant an edit could break.
    pub fn retain<F: FnMut(&mut S::Elem) -> bool>(&mut self, f: F) {
        retain_in(&mut self.store, f);
    }
    /// Returns a mutable reference to the element at `i`, or `None` if out of bounds.
    #[must_use]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut S::Elem> {
        self.store.as_mut_slice().get_mut(i)
    }
    /// Removes every element, keeping the backing store's allocated capacity.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Pre-allocates so at least `additional` more elements fit without a
    /// reallocation — see [`SortedSet::reserve`](crate::SortedSet::reserve).
    pub fn reserve(&mut self, additional: usize) {
        self.store.reserve(additional);
    }

    /// Appends `value` at the tail. `O(1)`.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] carrying `value` if a bounded store is at capacity;
    /// a bag never rejects for any other reason.
    pub fn try_push(&mut self, value: S::Elem) -> Result<(), CapacityError<S::Elem>> {
        push(&mut self.store, value)
    }

    /// Removes and returns the last element, or `None` if empty. `O(1)`.
    pub fn pop(&mut self) -> Option<S::Elem> {
        let len = self.store.len();
        (len > 0).then(|| self.store.remove_at(len - 1))
    }

    /// Removes and returns the element at `i` by swapping the last element into its place
    /// — `O(1)`, but **does not preserve order**.
    ///
    /// Prefer this over [`remove`](Self::remove) when order doesn't matter.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    pub fn swap_remove(&mut self, i: usize) -> S::Elem {
        self.store.swap_remove_at(i)
    }

    /// Removes and returns the element at `i`, shifting the tail down to preserve
    /// order — `O(n)`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    pub fn remove(&mut self, i: usize) -> S::Elem {
        self.store.remove_at(i)
    }

    /// Appends every item from `iter` at the tail.
    ///
    /// `O(k)` for `k` items — a bare append, no dedup.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] with the rejected element if a bounded store fills;
    /// the items pushed so far are kept and the iterator's unconsumed tail is
    /// dropped.
    pub fn try_extend<I>(&mut self, iter: I) -> Result<(), CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        append_all(&mut self.store, iter)
    }
}

impl<S: StoreMut + StoreNew> Bag<S> {
    /// Builds from an iterator by appending every item, in `O(n)`.
    ///
    /// No dedup and no element bound — the simplest bulk build in the crate.
    ///
    /// # Errors
    ///
    /// Returns [`CapacityError`] with the rejected element if a bounded store fills.
    pub fn try_from_iter<I>(iter: I) -> Result<Self, CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        let mut store = S::new();
        append_all(&mut store, iter)?;
        Ok(Self::from_store(store))
    }
}

impl<S: StoreMut + Unbounded> Bag<S> {
    /// Infallibly appends `value` — available only when the backing store is
    /// [`Unbounded`].
    pub fn push(&mut self, value: S::Elem) {
        match self.try_push(value) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<'a, S: Store> IntoIterator for &'a Bag<S> {
    type Item = &'a S::Elem;
    type IntoIter = core::slice::Iter<'a, S::Elem>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, S: StoreMut> IntoIterator for &'a mut Bag<S> {
    type Item = &'a mut S::Elem;
    type IntoIter = core::slice::IterMut<'a, S::Elem>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Consumes the bag, yielding its elements in insertion order.
///
/// Available when the backing store is itself consumable into its elements.
impl<S> IntoIterator for Bag<S>
where
    S: Store + IntoIterator<Item = <S as Store>::Elem>,
{
    type Item = S::Elem;
    type IntoIter = <S as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.store.into_iter()
    }
}

impl<S> FromIterator<S::Elem> for Bag<S>
where
    S: StoreMut + StoreNew + Unbounded,
{
    /// Collects an iterator into a bag — `O(n)`, no dedup, no element bound.
    ///
    /// Unlike the maps (whose duplicate-key policy makes a fallible build), a bag's
    /// `FromIterator` can't fail on an [`Unbounded`] store.
    fn from_iter<I: IntoIterator<Item = S::Elem>>(iter: I) -> Self {
        match Bag::try_from_iter(iter) {
            Ok(bag) => bag,
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<S> Extend<S::Elem> for Bag<S>
where
    S: StoreMut + Unbounded,
{
    fn extend<I: IntoIterator<Item = S::Elem>>(&mut self, iter: I) {
        match self.try_extend(iter) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

// Vec is the unbounded backend, so the `Unbounded`-gated paths (`push`, `collect`,
// `extend`) and the order-preserving / swap deletes run here.
#[cfg(all(test, feature = "alloc"))]
mod alloc_tests {
    use alloc::vec::Vec;

    use crate::Bag;

    #[test]
    fn push_keeps_insertion_order_and_duplicates() {
        let mut bag: Bag<Vec<i32>> = Bag::new();
        bag.push(2);
        bag.push(1);
        bag.push(2); // duplicates are kept — no membership discipline
        assert_eq!(bag.as_slice(), &[2, 1, 2]);
        assert_eq!(bag.len(), 3);
        assert!(bag.contains(&1));
        assert!(!bag.contains(&9));
    }

    #[test]
    fn collect_and_extend_append_everything() {
        let mut bag: Bag<Vec<i32>> = [1, 1, 2].into_iter().collect();
        bag.extend([3, 1]);
        assert_eq!(bag.as_slice(), &[1, 1, 2, 3, 1]);
    }

    #[test]
    fn pop_and_removes() {
        let mut bag: Bag<Vec<i32>> = Bag::try_from_iter([10, 20, 30, 40]).unwrap();
        assert_eq!(bag.pop(), Some(40));
        // swap_remove pulls the last element into the hole (order not preserved).
        assert_eq!(bag.swap_remove(0), 10);
        assert_eq!(bag.as_slice(), &[30, 20]);
        // remove shifts the tail down (order preserved).
        assert_eq!(bag.remove(0), 30);
        assert_eq!(bag.as_slice(), &[20]);
        assert_eq!(bag.pop(), Some(20));
        assert_eq!(bag.pop(), None);
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut bag: Bag<Vec<i32>> = Bag::try_from_iter([1, 2, 3]).unwrap();
        *bag.get_mut(1).unwrap() = 99;
        assert_eq!(bag.as_slice(), &[1, 99, 3]);
        assert!(bag.get_mut(3).is_none());
    }

    #[test]
    fn clear_then_reuse() {
        let mut bag: Bag<Vec<i32>> = [1, 2, 3].into_iter().collect();
        bag.clear();
        assert!(bag.is_empty());
        bag.push(7);
        assert_eq!(bag.as_slice(), &[7]);
    }

    #[test]
    fn iteration_and_retain() {
        let mut bag: Bag<Vec<i32>> = [1, 2, 2, 3].into_iter().collect();
        assert!(bag.iter().eq(&[1, 2, 2, 3]));
        assert!((&bag).into_iter().count() == 4);
        for x in &mut bag {
            *x *= 10;
        }
        assert_eq!(bag.as_slice(), &[10, 20, 20, 30]);
        // retain's `&mut` predicate can edit the kept elements as it filters.
        bag.retain(|x| {
            *x += 1;
            *x > 15
        });
        assert_eq!(bag.as_slice(), &[21, 21, 31]);
        let owned: Vec<i32> = bag.into_iter().collect();
        assert_eq!(owned, &[21, 21, 31]);
    }

    #[test]
    fn clone_is_independent() {
        let mut a: Bag<Vec<i32>> = [1, 2, 3].into_iter().collect();
        let b = a.clone();
        a.push(4);
        assert_eq!(b.as_slice(), &[1, 2, 3]); // clone unaffected by the later push
    }

    #[test]
    fn eq_is_order_sensitive_like_vec() {
        let a: Bag<Vec<i32>> = [1, 2, 3].into_iter().collect();
        let b: Bag<Vec<i32>> = [1, 2, 3].into_iter().collect();
        assert_eq!(a, b); // same elements, same order
                          // A `swap_remove` leaves a different order, so a different sequence...
        let mut c: Bag<Vec<i32>> = [1, 2, 3].into_iter().collect();
        c.swap_remove(0); // -> [3, 2]
        assert_eq!(c.as_slice(), &[3, 2]);
        // ...equal to a bag built in that order, unequal to the same multiset
        // in another order (a bag is a sequence, not an order-free multiset).
        assert_eq!(c, [3, 2].into_iter().collect());
        assert_ne!(c, [2, 3].into_iter().collect::<Bag<Vec<i32>>>());
    }
}

// heapless is the alloc-free fixed-cap backend, so these run under
// `--no-default-features --features heapless` and exercise the bounded paths.
#[cfg(all(test, feature = "heapless"))]
mod heapless_tests {
    use heapless::Vec;

    use crate::Bag;

    #[test]
    fn try_push_overflows_at_capacity() {
        let mut bag: Bag<Vec<u8, 2>> = Bag::new();
        bag.try_push(1).expect("within cap");
        bag.try_push(2).expect("within cap");
        // A bag keeps duplicates, so the only rejection is a genuine capacity hit.
        let err = bag.try_push(3).expect_err("third push exceeds cap 2");
        assert_eq!(err.into_inner(), 3);
        assert_eq!(bag.as_slice(), &[1, 2]);
    }

    #[test]
    fn try_from_iter_overflows_on_raw_count() {
        // No dedup, so even all-equal input overflows on its raw length.
        let err =
            Bag::<Vec<u8, 3>>::try_from_iter([5, 5, 5, 5]).expect_err("four items exceed cap 3");
        assert_eq!(err.into_inner(), 5);
    }
}
