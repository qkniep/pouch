//! The store contract — the **storage** and **bound** axes of the crate.
//!
//! `Store` / `StoreMut` / `StoreNew` abstract over where elements live and how
//! they are mutated; `Unbounded` marks the backends that never hit a logical
//! cap; [`Capped`] adds a runtime bound to any store. Concrete backends (`Vec`,
//! `SmallVec`, `TinyVec`, `ArrayVec`, `heapless::Vec`) implement these traits in
//! the private `backend` submodule — nothing there is named, it exists only for
//! its trait impls.

use crate::error::CapacityError;

mod backend;
mod capped;
mod scratch;
mod spill;

pub use capped::Capped;
pub use scratch::ScratchVec;
pub use spill::Spill;

/// Read access to a contiguous backing store of `Elem`.
///
/// `Elem` is an associated type (not a generic parameter): a given container
/// type has exactly one element type, and this keeps constructors free of an
/// unconstrained `T`. A set stores `Elem = T`; a map stores `Elem = (K, V)`.
pub trait Store {
    type Elem;

    fn as_slice(&self) -> &[Self::Elem];

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Logical capacity. `None` == unbounded (limited only by allocator OOM).
    fn capacity(&self) -> Option<usize>;
}

/// Mutation primitives. The collection layer builds sorted/unsorted semantics
/// on top of index-based insert/remove; the store itself is ordering-agnostic.
pub trait StoreMut: Store {
    /// Insert `value` at index `i` (shifting the tail right). `i <= len`.
    /// Returns `Err` iff the store is at logical capacity.
    fn try_insert_at(
        &mut self,
        i: usize,
        value: Self::Elem,
    ) -> Result<(), CapacityError<Self::Elem>>;

    /// Remove and return the element at index `i`. `i < len`.
    fn remove_at(&mut self, i: usize) -> Self::Elem;

    /// Remove the element at index `i` in O(1) by swapping the last element into
    /// its place; order is **not** preserved. `i < len`. This is the unsorted
    /// collections' delete primitive — sorted ones can't use it without breaking
    /// their ordering invariant. Provided in terms of `remove_at(len - 1)`, which
    /// drops the tail and so is O(1) on every backend.
    fn swap_remove_at(&mut self, i: usize) -> Self::Elem {
        debug_assert!(
            i < self.len(),
            "swap_remove_at: index out of bounds (empty store has no element to remove)",
        );
        let last = self.len() - 1;
        self.as_mut_slice().swap(i, last);
        self.remove_at(last)
    }

    /// Mutable slice, for in-place value updates (e.g. replacing a map value,
    /// which consumes no capacity).
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];

    fn clear(&mut self);
}

/// Construct an empty store. Kept separate from `Default` so that [`Capped`]
/// (which needs a runtime cap) is excluded; use `Capped::with_capacity` /
/// `from_store` for bounded wrappers.
pub trait StoreNew: Store + Sized {
    fn new() -> Self;
}

/// Marker: this store never reports a *logical*-capacity failure, so the
/// collection layer may expose an infallible `insert`. (Allocator OOM still
/// aborts; that is a separate concern — see crate docs.)
///
/// Implemented only for genuinely unbounded growable backends. Wrapping any
/// store in [`Capped`] removes this guarantee by construction.
pub trait Unbounded {}

/// Append `value` at the tail. `try_insert_at(len, …)` is the universal O(1)
/// append on every backend (a native shifting insert at `len` shifts nothing; a
/// push-only fallback's `rotate_right(1)` runs over a 1-element tail — also a
/// no-op), so it is the single primitive every bulk builder is built on. Errors
/// with the rejected element iff the store is at capacity.
pub(crate) fn push<S: StoreMut>(
    store: &mut S,
    value: S::Elem,
) -> Result<(), CapacityError<S::Elem>> {
    let i = store.len();
    store.try_insert_at(i, value)
}

/// Keep only the elements for which `f` returns `true`, preserving the order of
/// the kept ones — the shared engine under every collection's `retain`. `O(n)`
/// with no `Copy`/`Clone` bound: each kept element is swapped down to its final
/// slot (the slots in between hold only doomed elements, whose relative order
/// doesn't matter), then the doomed tail is popped off — each pop is
/// `remove_at(len - 1)`, `O(1)` on every backend. The predicate gets `&mut` so
/// map `retain` can offer in-place value mutation; set `retain` narrows it to
/// `&` (an element edit could break the set invariant).
pub(crate) fn retain_in<S: StoreMut>(store: &mut S, mut f: impl FnMut(&mut S::Elem) -> bool) {
    let s = store.as_mut_slice();
    let mut write = 0;
    for read in 0..s.len() {
        if f(&mut s[read]) {
            if write != read {
                s.swap(write, read);
            }
            write += 1;
        }
    }
    while store.len() > write {
        store.remove_at(store.len() - 1);
    }
}

/// Append every item from `iter` at the tail via [`push`] — the shared loop
/// under the bulk collection builders (`try_from_iter`, `extend`, …). Stops at
/// the first capacity failure, returning the rejected element; the items
/// appended so far are kept.
pub(crate) fn append_all<S, I>(store: &mut S, iter: I) -> Result<(), CapacityError<S::Elem>>
where
    S: StoreMut,
    I: IntoIterator<Item = S::Elem>,
{
    for value in iter {
        push(store, value)?;
    }
    Ok(())
}

// `swap_remove_at` is a provided method, so it's exercised through a concrete
// backend. heapless is the alloc-free one, so this runs even on the minimal
// `--no-default-features --features heapless` config.
#[cfg(all(test, feature = "heapless"))]
mod tests {
    use heapless::Vec;

    use super::{Store, StoreMut, StoreNew};

    #[test]
    fn swap_remove_at_swaps_in_the_last_element() {
        let mut v: Vec<u8, 4> = StoreNew::new();
        for x in [1, 2, 3, 4] {
            v.try_insert_at(v.len(), x).expect("within capacity");
        }
        // Removing a non-last index swaps the *last* element into the hole, so
        // order is not preserved — only membership is.
        assert_eq!(v.swap_remove_at(1), 2);
        assert_eq!(v.as_slice(), &[1, 4, 3]);
        // Removing the last index degenerates to a plain tail pop.
        assert_eq!(v.swap_remove_at(2), 3);
        assert_eq!(v.as_slice(), &[1, 4]);
    }

    // The precondition guard fires only in debug builds, so gate this on it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn swap_remove_at_on_empty_store_panics() {
        let mut v: Vec<u8, 4> = StoreNew::new();
        v.swap_remove_at(0);
    }
}
