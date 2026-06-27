//! `heapless::Vec`: fixed capacity `N`, no alloc. Not `Unbounded`.
//!
//! heapless::Vec has native shifting `insert`/`remove` (each a single memmove),
//! so we use them directly. The push-only template — `push` + `rotate_right(1)`
//! for insert, `rotate_left(1)` + `pop` for remove — is the fallback for a store
//! that truly only exposes `push`/`pop`; avoid it when a native shift exists, as
//! rotating by one still pulls in core's general `ptr_rotate` (hundreds of bytes
//! of flash).

use heapless::Vec;

use crate::error::CapacityError;
use crate::store::{Store, StoreMut, StoreNew};

impl<T, const N: usize> Store for Vec<T, N> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        &self[..]
    }
    fn capacity(&self) -> Option<usize> {
        Some(N)
    }
}
impl<T, const N: usize> StoreMut for Vec<T, N> {
    fn try_insert_at(&mut self, i: usize, value: T) -> Result<(), CapacityError<T>> {
        // heapless::Vec has a native shifting insert (one memmove) that hands the
        // value back on overflow — exactly our `CapacityError` contract. We use it
        // over the push-only `push` + `self[i..].rotate_right(1)` synthesis: rotate
        // by one still monomorphizes core's general `ptr_rotate` (hundreds of bytes
        // of flash), which matters for embedded. See the module note for the
        // push-only fallback.
        Vec::insert(self, i, value).map_err(CapacityError)
    }
    fn remove_at(&mut self, i: usize) -> T {
        // native shifting remove (one memmove); same rationale as try_insert_at.
        Vec::remove(self, i)
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self[..]
    }
    fn clear(&mut self) {
        Vec::clear(self)
    }
}
impl<T, const N: usize> StoreNew for Vec<T, N> {
    fn new() -> Self {
        Vec::new()
    }
}

// White-box tests for the native shifting insert/remove. These live
// here (gated for free by the `mod heapless` feature line) so they run under the
// alloc-free `--no-default-features --features heapless` config.
#[cfg(test)]
mod tests {
    use super::*;

    // Append through the public primitive, the way the collection layer drives it.
    fn from_slice<const N: usize>(items: &[u8]) -> Vec<u8, N> {
        let mut v: Vec<u8, N> = StoreNew::new();
        for &x in items {
            v.try_insert_at(v.len(), x).expect("within capacity");
        }
        v
    }

    #[test]
    fn try_insert_at_shifts_into_position() {
        let mut v: Vec<u8, 5> = StoreNew::new();
        v.try_insert_at(0, 20).expect("room"); // [20]
        v.try_insert_at(0, 10).expect("room"); // front:  [10, 20]
        v.try_insert_at(2, 30).expect("room"); // end:    [10, 20, 30]
        v.try_insert_at(1, 15).expect("room"); // middle: [10, 15, 20, 30]
        assert_eq!(v.as_slice(), &[10, 15, 20, 30]);
    }

    #[test]
    fn remove_at_shifts_tail_down() {
        let mut v: Vec<u8, 5> = from_slice(&[10, 15, 20, 30]);
        assert_eq!(v.remove_at(1), 15); // middle
        assert_eq!(v.as_slice(), &[10, 20, 30]);
        assert_eq!(v.remove_at(0), 10); // front
        assert_eq!(v.as_slice(), &[20, 30]);
        assert_eq!(v.remove_at(1), 30); // last
        assert_eq!(v.as_slice(), &[20]);
    }

    #[test]
    fn try_insert_at_overflow_hands_back_the_value() {
        let mut v: Vec<u8, 2> = from_slice(&[1, 2]);
        let err = v.try_insert_at(0, 9).expect_err("store is full");
        assert_eq!(err.into_inner(), 9);
        assert_eq!(v.as_slice(), &[1, 2]); // unchanged on overflow
    }
}
