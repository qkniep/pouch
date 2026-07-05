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
//! `Spill<ArrayVec<T, N>, Vec<T>>` reproduces a `SmallVec` by composition (no
//! heap until spill, then unbounded), and being [`Unbounded`] through `B` it gets
//! the collection layer's infallible `insert`. The real payoff is exotic spill
//! tiers no crate ships — e.g. `Spill<ArrayVec<T, N>, ScratchVec<T>>` for a
//! two-level, fully allocation-free store.

use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

use super::{push, Store, StoreMut, StoreNew, Unbounded};
use crate::error::CapacityError;

/// Two-tier store: `inline` until it fills, then everything migrates to `spill` and stays
/// there.
///
/// The elements live in exactly one tier at a time — the migration moves them all at
/// once — so the `as_slice()` contiguity contract holds and the logical capacity is
/// `spill`'s. Size `B` to hold at least `A`'s capacity, or the migration has nowhere to
/// land. When `B` is [`Unbounded`], so is the `Spill`, which unlocks the collection
/// layer's infallible `insert`.
#[derive(Clone, Debug)]
pub struct Spill<A, B> {
    inline: A,
    spill: B,
    spilled: bool,
}

// Comparisons and hashing are manual, over `as_slice()` — the *logical*
// contents — not the struct fields: which tier the elements currently live in
// is position, not content. A structural derive would compare `spilled` and
// both tiers, calling a spilled-then-shrunk store unequal to a never-spilled
// one holding the same elements. The slice-based impls keep the collection
// derives (`SortedSet<Spill<…>>: Eq/Ord/Hash`) semantic, matching every real
// backend (whose `Eq`/`Ord`/`Hash` behave like its `as_slice()`).
impl<A, B> PartialEq for Spill<A, B>
where
    A: Store,
    B: Store<Elem = A::Elem>,
    A::Elem: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<A, B> Eq for Spill<A, B>
where
    A: Store,
    B: Store<Elem = A::Elem>,
    A::Elem: Eq,
{
}

// `PartialOrd` can't defer to `cmp` — its element bound is only `PartialOrd`
// (matching the slice impls it delegates to), so the two impls are separate.
impl<A, B> PartialOrd for Spill<A, B>
where
    A: Store,
    B: Store<Elem = A::Elem>,
    A::Elem: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<A, B> Ord for Spill<A, B>
where
    A: Store,
    B: Store<Elem = A::Elem>,
    A::Elem: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<A, B> Hash for Spill<A, B>
where
    A: Store,
    B: Store<Elem = A::Elem>,
    A::Elem: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Slice hashing (length prefix + elements) — the same shape `Vec` and
        // the inline backends hash with.
        self.as_slice().hash(state);
    }
}

impl<A: Store, B: Store> Spill<A, B> {
    /// Builds from two **empty** tiers (debug-checked).
    ///
    /// The general constructor — use it when the spill tier needs a runtime resource it
    /// can't `new()`, like a [`ScratchVec`](crate::ScratchVec) borrowing a buffer. When
    /// both tiers are `StoreNew` and empty, [`StoreNew::new`] is the no-argument
    /// shortcut.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if either tier is non-empty, or if the spill tier's
    /// bound is smaller than the inline tier's capacity, which would otherwise report a
    /// `capacity()` below the inline tier's live `len()` and only panic later, when the
    /// migration has nowhere to land. Release builds trust the precondition unchecked.
    pub fn from_tiers(inline: A, spill: B) -> Self {
        debug_assert!(
            inline.is_empty() && spill.is_empty(),
            "Spill::from_tiers: both tiers must start empty",
        );
        // The spill tier must hold at least the inline capacity (module docs): an
        // unbounded spill always does; a bounded one must be `>=` the inline bound.
        // An unbounded inline tier never fills, so it imposes no constraint.
        debug_assert!(
            spill
                .capacity()
                .is_none_or(|sc| inline.capacity().is_none_or(|ic| sc >= ic)),
            "Spill::from_tiers: spill tier capacity must be at least the inline tier's",
        );
        Spill {
            inline,
            spill,
            spilled: false,
        }
    }
}

impl<A: StoreNew, B: Store> Spill<A, B> {
    /// Builds with a fresh empty inline tier and the given (empty) spill tier — the
    /// ergonomic constructor for `Spill<SomeStoreNew, ScratchVec<_>>` and friends.
    pub fn with_spill(spill: B) -> Self {
        Self::from_tiers(A::new(), spill)
    }
}

impl<A, B> Spill<A, B> {
    /// Returns `true` if the store has spilled into its second tier.
    ///
    /// Once spilled it stays spilled until [`clear`](StoreMut::clear) (no migrate-back),
    /// so this also reports "has ever overflowed the inline tier".
    pub fn is_spilled(&self) -> bool {
        self.spilled
    }
}

impl<A, B> Spill<A, B>
where
    A: StoreMut,
    B: StoreMut<Elem = A::Elem>,
{
    /// Moves every element from `inline` to `spill`, order preserved, in `O(n)`
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

    /// Pre-arms the tier the elements will live in.
    ///
    /// Once spilled, forwards to the spill tier. Before that, if the projected length
    /// (`len + additional`) still fits the inline tier there is nothing to do; if it
    /// doesn't, migration is coming — so reserve the *spill* tier for the whole projected
    /// population now, and the spill boundary itself allocates nothing when it hits.
    fn reserve(&mut self, additional: usize) {
        if self.spilled {
            self.spill.reserve(additional);
        } else if let Some(cap) = self.inline.capacity() {
            let projected = self.inline.len() + additional;
            if projected > cap {
                // Pre-spill the spill tier is empty (elements live in exactly
                // one tier), so `projected` more elements is the full need.
                self.spill.reserve(projected);
            }
        } else {
            // Inline tier is unbounded (e.g. `Spill<Vec, _>`): it can never
            // overflow, so it holds the whole population and never spills.
            // Forward the hint so a bulk build reserves once up front instead
            // of reallocating per push.
            self.inline.reserve(additional);
        }
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

/// Consuming iteration: migrates any still-inline elements into the spill tier
/// (order-preserving, `O(n)`, cannot fail — the tier is empty and must hold at least the
/// inline capacity), then yield the spill tier's iterator.
///
/// One iterator type instead of an either-tier enum; the one-time move is paid only by a
/// bag/set/map consumed before it ever spilled.
impl<A, B> IntoIterator for Spill<A, B>
where
    A: StoreMut,
    B: StoreMut<Elem = A::Elem> + IntoIterator<Item = A::Elem>,
{
    type Item = A::Elem;
    type IntoIter = B::IntoIter;

    fn into_iter(mut self) -> Self::IntoIter {
        if !self.spilled {
            self.migrate();
        }
        self.spill.into_iter()
    }
}

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

    // Equality and ordering are content-based (`as_slice`), not structural: a
    // spilled-then-shrunk store equals a never-spilled one holding the same
    // elements, even though their tiers differ.
    #[cfg(all(feature = "arrayvec", feature = "alloc"))]
    #[test]
    fn eq_and_ord_ignore_spill_state() {
        use alloc::vec::Vec;

        use arrayvec::ArrayVec;

        let mut spilled: Spill<ArrayVec<u32, 2>, Vec<u32>> = StoreNew::new();
        for x in [1, 2, 3] {
            push(&mut spilled, x).expect("unbounded via Vec");
        }
        // Shrink back under the inline bound; the store stays spilled.
        spilled.remove_at(2);
        assert!(spilled.is_spilled());
        assert_eq!(spilled.as_slice(), &[1, 2]);

        let mut inline: Spill<ArrayVec<u32, 2>, Vec<u32>> = StoreNew::new();
        push(&mut inline, 1).expect("room");
        push(&mut inline, 2).expect("room");
        assert!(!inline.is_spilled());

        assert_eq!(spilled, inline); // same contents, different tiers
        assert_eq!(spilled.cmp(&inline), core::cmp::Ordering::Equal);
        push(&mut inline, 3).expect("room"); // now [1, 2, 3]
        assert!(spilled < inline); // slice-lexicographic, tier-blind
    }

    // `reserve` pre-arms the tier the elements will live in: a projected
    // length that overflows the inline tier reserves the (empty) spill tier
    // for the whole population, so the later migration allocates nothing.
    #[cfg(all(feature = "arrayvec", feature = "alloc"))]
    #[test]
    fn reserve_pre_arms_the_spill_tier() {
        use alloc::vec::Vec;

        use arrayvec::ArrayVec;

        let mut s: Spill<ArrayVec<u32, 4>, Vec<u32>> = StoreNew::new();
        push(&mut s, 1).expect("room");

        // Projected length fits inline: nothing reserved anywhere.
        s.reserve(2);
        assert_eq!(s.spill.capacity(), 0);

        // Projected length (1 + 10) overflows the inline 4: the spill tier is
        // reserved for the whole projected population up front...
        s.reserve(10);
        assert!(!s.is_spilled()); // ...without spilling yet
        assert!(s.spill.capacity() >= 11);
        let armed = s.spill.capacity();

        // ...so the migration and the growth burst reallocate nothing.
        for x in 2..=11 {
            push(&mut s, x).expect("unbounded via Vec");
        }
        assert!(s.is_spilled());
        assert_eq!(s.spill.capacity(), armed);

        // Once spilled, reserve forwards to the spill tier directly.
        s.reserve(1000);
        assert!(s.spill.capacity() >= 1011);
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

    // A spill tier smaller than the inline tier would report a `capacity()` below the
    // inline tier's live `len()`; the debug guard rejects it at construction. Gated on
    // `debug_assertions`, since the check compiles out in release.
    #[cfg(all(debug_assertions, feature = "arrayvec"))]
    #[test]
    #[should_panic(expected = "spill tier capacity must be at least the inline tier's")]
    fn from_tiers_rejects_spill_smaller_than_inline() {
        use arrayvec::ArrayVec;

        let mut scratch = [0u32; 2]; // spill holds 2 < inline's 4
        let _: Spill<ArrayVec<u32, 4>, ScratchVec<u32>> =
            Spill::with_spill(ScratchVec::new(&mut scratch));
    }
}
