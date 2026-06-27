//! The recoverable error returned when a bounded store is at capacity.

use core::fmt;

/// Returned when a bounded store is at logical capacity. Hands the rejected
/// element back (mirrors `arrayvec::CapacityError<T>` / heapless `Result<(), T>`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CapacityError<T>(pub T);

impl<T> CapacityError<T> {
    /// Recover the element that could not be inserted.
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

/// Error from a fallible *bulk* map build (`SortedMap::try_from_iter` and its
/// siblings), which require every key to be unique. Every arm hands the rejected
/// entry back, like [`CapacityError`]. (One-at-a-time inserts and `extend` are
/// last-wins, so they never raise `DuplicateKey` — only `CapacityError`.)
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BuildError<T> {
    /// The store reached its logical capacity before every entry was inserted.
    Capacity(T),
    /// Two entries shared a key; the second one is handed back.
    DuplicateKey(T),
    /// `try_from_sorted_iter` was given keys that were not in ascending order; the
    /// offending entry (smaller than its predecessor) is handed back. Enforced in
    /// every build profile — unlike `from_store`, the sorted builder does not trust
    /// its input.
    Unsorted(T),
}

impl<T> BuildError<T> {
    /// Recover the entry that could not be inserted.
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
            BuildError::Unsorted(_) => "input keys were not in ascending order",
        })
    }
}

#[cfg(feature = "std")]
impl<T> std::error::Error for BuildError<T> {}

/// Error from a fallible *sorted* set build ([`SortedSet::try_from_sorted_iter`]),
/// which can fail two ways: the store fills, or the input is not ascending. A set
/// silently dedups adjacent equal values, so — unlike [`BuildError`] for maps —
/// there is no duplicate arm. Both arms hand the rejected element back.
///
/// [`SortedSet::try_from_sorted_iter`]: crate::SortedSet::try_from_sorted_iter
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortedBuildError<T> {
    /// The store reached its logical capacity before every element was inserted.
    Capacity(T),
    /// The input was not in ascending order; the offending element (smaller than
    /// its predecessor) is handed back. Enforced in every build profile — unlike
    /// `from_store`, the sorted builder does not trust its input.
    Unsorted(T),
}

impl<T> SortedBuildError<T> {
    /// Recover the element that could not be inserted.
    pub fn into_inner(self) -> T {
        match self {
            SortedBuildError::Capacity(t) | SortedBuildError::Unsorted(t) => t,
        }
    }
}

impl<T> From<CapacityError<T>> for SortedBuildError<T> {
    fn from(err: CapacityError<T>) -> Self {
        SortedBuildError::Capacity(err.0)
    }
}

impl<T> fmt::Debug for SortedBuildError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SortedBuildError::Capacity(_) => "SortedBuildError::Capacity",
            SortedBuildError::Unsorted(_) => "SortedBuildError::Unsorted",
        })
    }
}

impl<T> fmt::Display for SortedBuildError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SortedBuildError::Capacity(_) => "store is at logical capacity",
            SortedBuildError::Unsorted(_) => "input was not in ascending order",
        })
    }
}

#[cfg(feature = "std")]
impl<T> std::error::Error for SortedBuildError<T> {}
