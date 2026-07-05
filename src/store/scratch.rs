//! [`ScratchVec`]: a fixed-capacity store over a **borrowed** `&mut [T]` — fully
//! allocation-free, capacity = the buffer's length.
//!
//! Unlike `ArrayVec` / `heapless::Vec` the storage is borrowed, not owned: the
//! caller hands over a slice (a stack array, a `static mut` region, an arena
//! chunk) and `ScratchVec` uses it as backing for up to `buf.len()` elements.
//! Its headline use is as the *spill* tier of a [`Spill`](crate::Spill): small
//! common case inline, rare overflow into a big shared buffer, never touching the
//! heap — a two-level store no existing crate ships (`SmallVec`/`TinyVec` spill to
//! the global heap; `ArrayVec`/`heapless::Vec` are single-tier).
//!
//! Mutation requires `T: Default`, the same trade `TinyVec` makes: every buffer
//! slot must always hold a valid `T` because `pouch` forbids `unsafe`, so there is
//! no `MaybeUninit` hole to leave behind. A vacated slot is refilled with
//! `T::default()`; `clear` drops the live elements and defaults their slots.
//! Insert needs no `Default` (it only permutes existing values), but the bound
//! sits on the whole `StoreMut` impl for one uniform contract; the read-only
//! [`Store`] impl is unconstrained.

use core::{fmt, mem};

use crate::error::CapacityError;
use crate::store::{Store, StoreMut};

/// A fixed-capacity store backed by a borrowed `&'a mut [T]`.
///
/// Capacity is fixed at the buffer length; it never allocates. Mutation requires
/// `T: Default` (the trade `TinyVec` makes): `pouch` forbids `unsafe`, so a vacated slot
/// can't be left uninitialized and is refilled with `T::default()` instead. The
/// read-only [`Store`] impl carries no such bound.
pub struct ScratchVec<'a, T> {
    buf: &'a mut [T],
    len: usize,
}

impl<'a, T> ScratchVec<'a, T> {
    /// Wraps `buf` as empty scratch storage (logical length 0).
    ///
    /// The values already in `buf` are treated as junk and overwritten as elements are
    /// inserted.
    pub fn new(buf: &'a mut [T]) -> Self {
        ScratchVec { buf, len: 0 }
    }
}

impl<T> Store for ScratchVec<'_, T> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        &self.buf[..self.len]
    }
    fn capacity(&self) -> Option<usize> {
        Some(self.buf.len())
    }
}

impl<T: Default> StoreMut for ScratchVec<'_, T> {
    fn try_insert_at(&mut self, i: usize, value: T) -> Result<(), CapacityError<T>> {
        debug_assert!(
            i <= self.len,
            "try_insert_at: index out of bounds (can insert at most one past the end)",
        );
        if self.len >= self.buf.len() {
            return Err(CapacityError(value));
        }
        // Rotate the (junk) slot at `len` round to `i`, which shifts buf[i..len]
        // up by one, then drop that junk as `value` lands at `i`. Every slot still
        // holds exactly one live `T` — no hole, no `Default` needed for insert.
        // (rotate_right(1) pulls in core's `ptr_rotate`; a backend minimising flash
        // would hand-roll the one-slot shift — see the heapless backend's note.)
        self.buf[i..=self.len].rotate_right(1);
        drop(mem::replace(&mut self.buf[i], value));
        self.len += 1;
        Ok(())
    }

    fn remove_at(&mut self, i: usize) -> T {
        debug_assert!(
            i < self.len,
            "remove_at: index out of bounds (empty store has no element to remove)",
        );
        // Rotate the element at `i` to the end of the live region, drop `len`, then
        // take it out — leaving a `default()` behind so the buffer stays fully
        // initialised.
        self.buf[i..self.len].rotate_left(1);
        self.len -= 1;
        mem::take(&mut self.buf[self.len])
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf[..self.len]
    }

    fn clear(&mut self) {
        // Drop the live elements (run their destructors), defaulting their slots so
        // the borrowed buffer stays valid for its owner.
        for slot in &mut self.buf[..self.len] {
            drop(mem::take(slot));
        }
        self.len = 0;
    }
}

// Deliberately NOT `StoreNew` (construction needs the borrowed buffer, like
// `Capped` needs a cap) and NOT `Unbounded` (capacity is fixed at the buffer len).

// Show only the live elements and the bound — never the junk past `len`.
impl<T: fmt::Debug> fmt::Debug for ScratchVec<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScratchVec")
            .field("len", &self.len)
            .field("capacity", &self.buf.len())
            .field("elements", &self.as_slice())
            .finish()
    }
}

// Core-only (no dependency, no `alloc`), so these run in every config, including
// the `--no-default-features` path that has no other mutable backend.
#[cfg(test)]
mod tests {
    use super::*;

    fn from_slice<'a>(buf: &'a mut [u8], items: &[u8]) -> ScratchVec<'a, u8> {
        let mut v = ScratchVec::new(buf);
        for &x in items {
            v.try_insert_at(v.len(), x).expect("within capacity");
        }
        v
    }

    #[test]
    fn capacity_is_the_buffer_length() {
        let mut buf = [0u8; 4];
        let v = ScratchVec::new(&mut buf);
        assert_eq!(v.capacity(), Some(4));
        assert!(v.is_empty());
    }

    #[test]
    fn try_insert_at_shifts_into_position() {
        let mut buf = [0u8; 5];
        let mut v = ScratchVec::new(&mut buf);
        v.try_insert_at(0, 20).expect("room"); // [20]
        v.try_insert_at(0, 10).expect("room"); // front:  [10, 20]
        v.try_insert_at(2, 30).expect("room"); // end:    [10, 20, 30]
        v.try_insert_at(1, 15).expect("room"); // middle: [10, 15, 20, 30]
        assert_eq!(v.as_slice(), &[10, 15, 20, 30]);
    }

    #[test]
    fn remove_at_shifts_tail_down() {
        let mut buf = [0u8; 5];
        let mut v = from_slice(&mut buf, &[10, 15, 20, 30]);
        assert_eq!(v.remove_at(1), 15); // middle
        assert_eq!(v.as_slice(), &[10, 20, 30]);
        assert_eq!(v.remove_at(0), 10); // front
        assert_eq!(v.as_slice(), &[20, 30]);
        assert_eq!(v.remove_at(1), 30); // last
        assert_eq!(v.as_slice(), &[20]);
    }

    #[test]
    fn try_insert_at_overflow_hands_back_the_value() {
        let mut buf = [0u8; 2];
        let mut v = from_slice(&mut buf, &[1, 2]);
        let err = v.try_insert_at(0, 9).expect_err("buffer is full");
        assert_eq!(err.into_inner(), 9);
        assert_eq!(v.as_slice(), &[1, 2]); // unchanged on overflow
    }

    #[test]
    fn clear_drops_live_elements_and_resets_len() {
        use core::cell::Cell;

        // A `Default` type that bumps a counter on drop *only* when it was tagged —
        // so the default-filled junk slots don't count, isolating the elements
        // `clear` actually dropped.
        #[derive(Default)]
        struct Noisy<'c>(Option<&'c Cell<u32>>);
        impl Drop for Noisy<'_> {
            fn drop(&mut self) {
                if let Some(c) = self.0 {
                    c.set(c.get() + 1);
                }
            }
        }

        let drops = Cell::new(0);
        let mut buf: [Noisy; 4] = Default::default();
        let mut v = ScratchVec::new(&mut buf);
        v.try_insert_at(0, Noisy(Some(&drops))).expect("room");
        v.try_insert_at(1, Noisy(Some(&drops))).expect("room");
        v.clear();
        assert_eq!(drops.get(), 2); // the two live elements were dropped
        assert!(v.is_empty());
    }
}
