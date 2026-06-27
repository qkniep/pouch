//! [`Capped`]: a runtime logical-capacity bound layered over any growable store.

use super::{Store, StoreMut, StoreNew};
use crate::error::CapacityError;

/// Adds a **runtime** logical-capacity bound to any backing store, turning its
/// (otherwise infallible) inserts into recoverable [`CapacityError`]s. This is
/// the factoring that makes `max_capacity` orthogonal to storage: cap logic is
/// written once here rather than per backend.
#[derive(Debug)]
pub struct Capped<S> {
    inner: S,
    cap: usize,
}

impl<S> Capped<S> {
    /// Wrap `inner` with a runtime cap, **assuming its current length does not
    /// already exceed `cap`** — the `len() <= capacity()` invariant the rest of
    /// the crate relies on (e.g. any `capacity() - len()` remaining math, which
    /// would otherwise underflow). The precondition is only `debug_assert!`-checked
    /// (zero cost in release), mirroring the collection-layer `from_store`. To
    /// start from an empty store instead, use [`with_capacity`](Self::with_capacity).
    pub fn from_store(inner: S, cap: usize) -> Self
    where
        S: Store,
    {
        debug_assert!(
            inner.len() <= cap,
            "Capped::from_store: store length must not exceed cap",
        );
        Capped { inner, cap }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: StoreNew> Capped<S> {
    pub fn with_capacity(cap: usize) -> Self {
        Capped {
            inner: S::new(),
            cap,
        }
    }
}

impl<S: Store> Store for Capped<S> {
    type Elem = S::Elem;

    fn as_slice(&self) -> &[S::Elem] {
        self.inner.as_slice()
    }

    fn capacity(&self) -> Option<usize> {
        // Effective cap = min(our cap, inner's own bound if any).
        Some(self.inner.capacity().map_or(self.cap, |c| c.min(self.cap)))
    }
}

impl<S: StoreMut> StoreMut for Capped<S> {
    fn try_insert_at(&mut self, i: usize, value: S::Elem) -> Result<(), CapacityError<S::Elem>> {
        if self.inner.len() >= self.cap {
            return Err(CapacityError(value));
        }
        self.inner.try_insert_at(i, value)
    }

    fn remove_at(&mut self, i: usize) -> S::Elem {
        self.inner.remove_at(i)
    }

    fn as_mut_slice(&mut self) -> &mut [S::Elem] {
        self.inner.as_mut_slice()
    }

    fn clear(&mut self) {
        self.inner.clear()
    }
}

// Capped is deliberately NOT `Unbounded`.

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn capacity_over_unbounded_inner_is_our_cap() {
        use alloc::vec::Vec;
        let c: Capped<Vec<u8>> = Capped::with_capacity(3);
        assert_eq!(c.capacity(), Some(3));
    }

    #[cfg(feature = "heapless")]
    #[test]
    fn capacity_is_min_of_our_cap_and_inner_bound() {
        use heapless::Vec;
        // our cap is the tighter bound
        let tight: Capped<Vec<u8, 5>> = Capped::with_capacity(3);
        assert_eq!(tight.capacity(), Some(3));
        // the inner backend's own bound is the tighter one
        let loose: Capped<Vec<u8, 2>> = Capped::with_capacity(5);
        assert_eq!(loose.capacity(), Some(2));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn try_insert_at_errors_at_cap_and_preserves_value() {
        use alloc::vec::Vec;
        let mut c: Capped<Vec<u8>> = Capped::with_capacity(2);
        c.try_insert_at(0, 1).expect("room");
        c.try_insert_at(1, 2).expect("room");
        let err = c.try_insert_at(2, 9).expect_err("at cap");
        assert_eq!(err.into_inner(), 9);
        assert_eq!(c.len(), 2); // rejected element did not land
    }

    // The trust-contract guard fires only in debug builds, so gate this on it.
    #[cfg(all(debug_assertions, feature = "alloc"))]
    #[test]
    #[should_panic(expected = "length must not exceed cap")]
    fn from_store_rejects_len_over_cap() {
        use alloc::vec::Vec;
        let _: Capped<Vec<u8>> = Capped::from_store(alloc::vec![1, 2, 3, 4, 5], 2);
    }
}
