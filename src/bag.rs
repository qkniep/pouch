//! Bag — a multiset/sequence with **no membership discipline**.
//!
//! A [`Bag`] holds values with duplicates allowed, in insertion order — a `Vec`
//! lifted over any [`Store`]: `try_push` appends in `O(1)`, `pop` and `swap_remove`
//! delete in `O(1)`, `remove` deletes in `O(n)` keeping order. It is the cheapest
//! collection in the crate: it needs **no bound on the element type** (no `Eq` / `Ord` /
//! `Hash`), so bulk construction is a bare append — no dedup, no sort, no duplicate-key
//! check, and a fully unconstrained `FromIterator`. Reach for it for the inside-a-map
//! case where you accumulate values per key and never need uniqueness (multimap values,
//! group-by, per-key event logs); the `Eq`-gated [`contains`](Bag::contains) /
//! [`count`](Bag::count) add multiset queries without constraining the core.

use crate::error::CapacityError;
use crate::store::{append_all, push, Store, StoreMut, StoreNew, Unbounded};

/// A sequence of values with duplicates allowed and no ordering or uniqueness
/// invariant. The crate's lightest collection — `Vec`-like over any backend, with
/// no bound on the element type.
// Derives `Clone` but not `PartialEq`/`Eq`: a bag's multiset equality is
// order-independent, yet `swap_remove` lets two equal bags store their elements in
// different orders, so a structural derive would wrongly call them unequal — the
// same reason the unsorted set/map twins withhold it.
#[derive(Clone, Debug)]
pub struct Bag<S> {
    store: S,
}

impl<S: StoreNew> Bag<S> {
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
    /// Wrap a store as a bag. Every store is a valid bag (no invariant to uphold),
    /// so this is infallible and needs no element bound — the cheapest of the
    /// crate's `from_store` constructors.
    pub fn from_store(store: S) -> Self {
        Bag { store }
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
    /// The element at `i` in insertion order, or `None` if out of bounds.
    pub fn get(&self, i: usize) -> Option<&S::Elem> {
        self.store.as_slice().get(i)
    }
    /// Whether any element equals `value`. `O(n)` linear scan; gated on `Eq` so the
    /// rest of the bag stays bound-free.
    pub fn contains(&self, value: &S::Elem) -> bool
    where
        S::Elem: Eq,
    {
        self.store.as_slice().contains(value)
    }
    /// How many elements equal `value` — the multiset multiplicity. `O(n)`.
    pub fn count(&self, value: &S::Elem) -> usize
    where
        S::Elem: Eq,
    {
        self.store.as_slice().iter().filter(|e| *e == value).count()
    }
}

impl<S: StoreMut> Bag<S> {
    /// Mutable view of the elements, for in-place edits. A bag has no invariant, so
    /// arbitrary mutation (reorder, overwrite) is always valid.
    pub fn as_mut_slice(&mut self) -> &mut [S::Elem] {
        self.store.as_mut_slice()
    }
    /// Mutable reference to the element at `i`, or `None` if out of bounds.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut S::Elem> {
        self.store.as_mut_slice().get_mut(i)
    }
    /// Remove every element, keeping the backing store's allocated capacity.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Append at the tail. `O(1)`. Errors with the rejected element iff a bounded
    /// store is at capacity; a bag never rejects for any other reason.
    pub fn try_push(&mut self, value: S::Elem) -> Result<(), CapacityError<S::Elem>> {
        push(&mut self.store, value)
    }

    /// Remove and return the last element, or `None` if empty. `O(1)`.
    pub fn pop(&mut self) -> Option<S::Elem> {
        let len = self.store.len();
        (len > 0).then(|| self.store.remove_at(len - 1))
    }

    /// Remove and return the element at `i` by swapping the last element into its
    /// place — `O(1)`, but **does not preserve order**. Panics if `i` is out of
    /// bounds. Prefer this over [`remove`](Self::remove) when order doesn't matter.
    pub fn swap_remove(&mut self, i: usize) -> S::Elem {
        self.store.swap_remove_at(i)
    }

    /// Remove and return the element at `i`, shifting the tail down to preserve
    /// order — `O(n)`. Panics if `i` is out of bounds.
    pub fn remove(&mut self, i: usize) -> S::Elem {
        self.store.remove_at(i)
    }

    /// Append every item from `iter` at the tail. `O(k)` for `k` items — a bare
    /// append, no dedup. On a capacity failure the items pushed so far are kept and
    /// the rejected element returned (the iterator's unconsumed tail is dropped).
    pub fn try_extend<I>(&mut self, iter: I) -> Result<(), CapacityError<S::Elem>>
    where
        I: IntoIterator<Item = S::Elem>,
    {
        append_all(&mut self.store, iter)
    }
}

impl<S: StoreMut + StoreNew> Bag<S> {
    /// Build from an iterator by appending every item, in `O(n)`. No dedup and no
    /// element bound — the simplest bulk build in the crate. Errors with the
    /// rejected element if a bounded store fills.
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
    /// Infallible append — available only when the backing store is [`Unbounded`].
    pub fn push(&mut self, value: S::Elem) {
        match self.try_push(value) {
            Ok(()) => {}
            Err(_) => unreachable!("Unbounded store reported a capacity failure"),
        }
    }
}

impl<S> FromIterator<S::Elem> for Bag<S>
where
    S: StoreMut + StoreNew + Unbounded,
{
    /// `.collect()` into a bag — `O(n)`, no dedup, no element bound. Unlike the
    /// maps (whose duplicate-key policy makes a fallible build), a bag's
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
        assert_eq!(bag.count(&2), 2);
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
    fn clone_is_independent() {
        // Bag derives Clone but not PartialEq (order-sensitive multiset).
        let mut a: Bag<Vec<i32>> = [1, 2, 3].into_iter().collect();
        let b = a.clone();
        a.push(4);
        assert_eq!(b.as_slice(), &[1, 2, 3]); // clone unaffected by the later push
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
