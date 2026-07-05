//! The recoverable error returned when a bounded store is at capacity.

use core::fmt;

/// Returned when a bounded store is at logical capacity.
///
/// Hands the rejected element back (mirrors `arrayvec::CapacityError<T>` / heapless
/// `Result<(), T>`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CapacityError<T>(pub(crate) T);

impl<T> CapacityError<T> {
    /// Wraps a rejected element.
    ///
    /// For third-party [`StoreMut`](crate::store::StoreMut) implementations, whose
    /// `try_insert_at` must hand the value back on overflow; everything in-crate
    /// constructs the error directly.
    pub fn new(value: T) -> Self {
        CapacityError(value)
    }

    /// Recovers the element that could not be inserted.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> fmt::Debug for CapacityError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CapacityError")
    }
}

impl<T> fmt::Display for CapacityError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("store is at logical capacity")
    }
}

#[cfg(feature = "std")]
impl<T> std::error::Error for CapacityError<T> {}

/// Error from a fallible *bulk* build (`try_from_iter` / `try_from_sorted_iter` on any
/// set or map).
///
/// Every arm hands the rejected element back, like [`CapacityError`]. One error type
/// covers all the builders; not every arm is reachable from every builder:
///
/// * [`Capacity`](BuildError::Capacity) — any builder over a bounded store.
/// * [`DuplicateKey`](BuildError::DuplicateKey) — **map** builders only. A duplicate
///   *key* is ambiguous (which value wins?), so map construction rejects it; set builders
///   dedup silently and never return this. The sequential ops (`try_insert`,
///   `try_extend`, `Extend`) stay last-wins and never raise it either.
/// * [`Unsorted`](BuildError::Unsorted) — `try_from_sorted_iter` only, which enforces its
///   ascending-order promise in every build profile (unlike `from_store`, the sorted
///   builder does not trust its input).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BuildError<T> {
    /// The store reached its logical capacity before every element was inserted.
    Capacity(T),
    /// Two entries shared a key; the second one is handed back.
    DuplicateKey(T),
    /// `try_from_sorted_iter` was given input that was not in ascending order;
    /// the offending element (smaller than its predecessor) is handed back.
    Unsorted(T),
}

impl<T> BuildError<T> {
    /// Recovers the element that could not be inserted.
    pub fn into_inner(self) -> T {
        match self {
            BuildError::Capacity(t) | BuildError::DuplicateKey(t) | BuildError::Unsorted(t) => t,
        }
    }
}

impl<T> From<CapacityError<T>> for BuildError<T> {
    fn from(err: CapacityError<T>) -> Self {
        BuildError::Capacity(err.0)
    }
}

impl<T> fmt::Debug for BuildError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BuildError::Capacity(_) => "BuildError::Capacity",
            BuildError::DuplicateKey(_) => "BuildError::DuplicateKey",
            BuildError::Unsorted(_) => "BuildError::Unsorted",
        })
    }
}

impl<T> fmt::Display for BuildError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BuildError::Capacity(_) => "store is at logical capacity",
            BuildError::DuplicateKey(_) => "duplicate key in bulk build",
            BuildError::Unsorted(_) => "input was not in ascending order",
        })
    }
}

#[cfg(feature = "std")]
impl<T> std::error::Error for BuildError<T> {}
