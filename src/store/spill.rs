//! [`Spill<A, B>`]: a two-tier store — live in `A` until it fills, then migrate
//! every element to `B` and continue there. The **storage**-axis analogue of
//! [`Capped`](crate::Capped): a composable adapter, not a new container.
//!
//! Contiguity (the `as_slice() -> &[Elem]` contract) holds because the elements
//! live in exactly **one** tier at a time — the migration moves them all at once,
//! they are never split across `A` and `B`. So the logical capacity is `B`'s (you
//! always end up there); size `B` to hold at least `A`'s capacity, or the
//! migration has nowhere to land.
//!
//! `Spill<ArrayVec<[T; N]>, Vec<T>>` reproduces a `SmallVec` by composition (no
//! heap until spill, then unbounded), and being [`Unbounded`] through `B` it gets
//! the collection layer's infallible `insert`. The real payoff is exotic spill
//! tiers no crate ships — e.g. `Spill<ArrayVec<T, N>, ScratchVec<T>>` for a
//! two-level, fully allocation-free store.

use super::{push, Store, StoreMut, StoreNew, Unbounded};
use crate::error::CapacityError;

/// Two-tier store: `inline` until it fills, then everything migrates to `spill`
/// and stays there. See the module docs.
#[derive(Debug)]
pub struct Spill<A, B> {
    inline: A,
    spill: B,
    spilled: bool,
}

impl<A: Store, B: Store> Spill<A, B> {
    /// Build from two **empty** tiers (debug-checked). The general constructor —
    /// use it when the spill tier needs a runtime resource it can't `new()`, like
    /// a [`ScratchVec`](crate::ScratchVec) borrowing a buffer. When both tiers are
    /// `StoreNew` and empty, [`StoreNew::new`] is the no-argument shortcut.
    pub fn from_tiers(inline: A, spill: B) -> Self {
        debug_assert!(
            inline.is_empty() && spill.is_empty(),
            "Spill::from_tiers: both tiers must start empty",
        );
        Spill {
            inline,
            spill,
            spilled: false,
        }
    }
}

impl<A: StoreNew, B: Store> Spill<A, B> {
    /// Build with a fresh empty inline tier and the given (empty) spill tier — the
    /// ergonomic constructor for `Spill<SomeStoreNew, ScratchVec<_>>` and friends.
    pub fn with_spill(spill: B) -> Self {
        Self::from_tiers(A::new(), spill)
    }
}

impl<A, B> Spill<A, B> {
    /// Whether the store has spilled into its second tier. Once spilled it stays
    /// spilled until [`clear`](StoreMut::clear) (no migrate-back), so this also
    /// reports "has ever overflowed the inline tier".
    pub fn is_spilled(&self) -> bool {
        self.spilled
    }
}

impl<A, B> Spill<A, B>
where
    A: StoreMut,
    B: StoreMut<Elem = A::Elem>,
{
    /// Move every element from `inline` to `spill`, order preserved, in `O(n)`
    /// using only the universal primitives: pop the tail (`remove_at(len - 1)` is
    /// `O(1)` on every backend — no shift) into `spill`, which reverses order, then
    /// reverse `spill` once to restore it.
    fn migrate(&mut self) {
        while !self.inline.is_empty() {
            let last = self.inline.len() - 1;
            let value = self.inline.remove_at(last);
            // The spill tier must hold at least the inline capacity (see module
            // docs); it is empty here, so this cannot fail for a well-sized `B`.
            push(&mut self.spill, value)
                .expect("Spill: spill tier must hold at least the inline capacity");
        }
        self.spill.as_mut_slice().reverse();
        self.spilled = true;
    }
}

impl<A, B> Store for Spill<A, B>
where
    A: Store,
    B: Store<Elem = A::Elem>,
{
    type Elem = A::Elem;

    fn as_slice(&self) -> &[Self::Elem] {
        if self.spilled {
            self.spill.as_slice()
        } else {
            self.inline.as_slice()
        }
    }

    fn capacity(&self) -> Option<usize> {
        // Everything ends up in `spill`, so its bound is the logical capacity.
        self.spill.capacity()
    }
}

impl<A, B> StoreMut for Spill<A, B>
where
    A: StoreMut,
    B: StoreMut<Elem = A::Elem>,
{
    fn try_insert_at(
        &mut self,
        i: usize,
        value: Self::Elem,
    ) -> Result<(), CapacityError<Self::Elem>> {
        if self.spilled {
            return self.spill.try_insert_at(i, value);
        }
        match self.inline.try_insert_at(i, value) {
            Ok(()) => Ok(()),
            // Inline is full: migrate everything to `spill`, then retry there. A
            // failure now is a genuine capacity error (the spill tier is full too).
            Err(CapacityError(value)) => {
                self.migrate();
                self.spill.try_insert_at(i, value)
            }
        }
    }

    fn remove_at(&mut self, i: usize) -> Self::Elem {
        if self.spilled {
            self.spill.remove_at(i)
        } else {
            self.inline.remove_at(i)
        }
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        if self.spilled {
            self.spill.as_mut_slice()
        } else {
            self.inline.as_mut_slice()
        }
    }

    fn clear(&mut self) {
        self.inline.clear();
        self.spill.clear();
        self.spilled = false;
    }
}

impl<A: StoreNew, B: StoreNew<Elem = A::Elem>> StoreNew for Spill<A, B> {
    fn new() -> Self {
        Spill {
            inline: A::new(),
            spill: B::new(),
            spilled: false,
        }
    }
}

// Unbounded iff the spill tier is — that's where every element ends up, so its
// boundlessness is what lets the collection layer expose an infallible `insert`.
impl<A, B: Unbounded> Unbounded for Spill<A, B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ScratchVec;

    // Core-only: both tiers are `ScratchVec`, so this whole test runs under
    // `--no-default-features` with no dependency and no `alloc`.
    #[test]
    fn spills_from_inline_to_spill_preserving_order() {
        let mut small = [0u8; 2]; // inline tier: holds 2
        let mut big = [0u8; 8]; // spill tier: holds 8
        let mut s = Spill::from_tiers(ScratchVec::new(&mut small), ScratchVec::new(&mut big));

        // Effective capacity is the spill tier's bound, before and after spilling.
        assert_eq!(s.capacity(), Some(8));

        // Fill the inline tier.
        push(&mut s, 1).expect("room");
        push(&mut s, 2).expect("room");
        assert!(!s.is_spilled());
        assert_eq!(s.as_slice(), &[1, 2]);

        // One more triggers migration; order is preserved across the move.
        push(&mut s, 3).expect("room");
        assert!(s.is_spilled());
        assert_eq!(s.as_slice(), &[1, 2, 3]);

        // Keep growing in the spill tier, including a non-tail insert.
        s.try_insert_at(0, 0).expect("room"); // [0, 1, 2, 3]
        push(&mut s, 4).expect("room"); // [0, 1, 2, 3, 4]
        assert_eq!(s.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn errors_only_when_the_spill_tier_is_full() {
        let mut small = [0u8; 1];
        let mut big = [0u8; 2];
        let mut s = Spill::from_tiers(ScratchVec::new(&mut small), ScratchVec::new(&mut big));

        push(&mut s, 1).expect("room"); // inline
        push(&mut s, 2).expect("room"); // spills, now in the 2-slot tier
        let err = push(&mut s, 3).expect_err("spill tier is full");
        assert_eq!(err.into_inner(), 3);
        assert_eq!(s.as_slice(), &[1, 2]); // unchanged on overflow
    }

    #[test]
    fn remove_and_clear_work_across_the_boundary() {
        let mut small = [0u8; 2];
        let mut big = [0u8; 8];
        let mut s = Spill::from_tiers(ScratchVec::new(&mut small), ScratchVec::new(&mut big));
        for x in [1, 2, 3, 4] {
            push(&mut s, x).expect("room");
        }
        assert!(s.is_spilled());
        assert_eq!(s.remove_at(1), 2); // remove from the spill tier
        assert_eq!(s.as_slice(), &[1, 3, 4]);

        s.clear();
        assert!(s.is_empty());
        assert!(!s.is_spilled()); // clear resets to the inline tier
        push(&mut s, 9).expect("room");
        assert!(!s.is_spilled()); // and the inline tier is live again
        assert_eq!(s.as_slice(), &[9]);
    }

    // The smallvec-by-composition shape: inline ArrayVec spilling to an unbounded
    // Vec. Being `Unbounded` through `B`, it earns the collection layer's
    // infallible `insert`.
    #[cfg(all(feature = "arrayvec", feature = "alloc"))]
    #[test]
    fn arrayvec_spilling_to_vec_backs_a_sorted_set() {
        use alloc::vec::Vec;

        use arrayvec::ArrayVec;

        use crate::SortedSet;

        let mut set: SortedSet<Spill<ArrayVec<u32, 4>, Vec<u32>>> = SortedSet::new();
        // Insert out of order and past the inline bound; the set stays sorted and
        // spills transparently.
        for x in [5, 3, 8, 1, 9, 2, 7] {
            set.insert(x); // infallible — store is Unbounded via Vec
        }
        assert_eq!(set.as_slice(), &[1, 2, 3, 5, 7, 8, 9]);
        assert!(set.contains(&7));
        assert!(!set.contains(&4));
        assert_eq!(set.capacity(), None); // unbounded, via the Vec spill tier
    }

    // Fully allocation-free collection: an inline ArrayVec spilling into a borrowed
    // scratch buffer, never touching the heap.
    #[cfg(feature = "arrayvec")]
    #[test]
    fn arrayvec_spilling_to_scratch_backs_a_sorted_set() {
        use arrayvec::ArrayVec;

        use crate::SortedSet;

        let mut scratch = [0u32; 16];
        // `set`'s type annotation pins the inline tier to `ArrayVec<u32, 4>`, so
        // `with_spill` builds it empty and borrows the scratch buffer for the rest.
        let mut set: SortedSet<Spill<ArrayVec<u32, 4>, ScratchVec<u32>>> =
            SortedSet::from_store(Spill::with_spill(ScratchVec::new(&mut scratch)));
        for x in [5, 3, 8, 1, 9, 2, 7] {
            set.try_insert(x).expect("within scratch capacity");
        }
        assert_eq!(set.as_slice(), &[1, 2, 3, 5, 7, 8, 9]);
        assert_eq!(set.capacity(), Some(16));
    }
}
