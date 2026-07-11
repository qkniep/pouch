//! Merge-based set algebra — the iterators behind [`SortedSet::union`] & co.
//!
//! The base walk is a two-pointer merge over the two sets' already sorted
//! slices: `O(n + m)`, no allocation, no hashing, yielding references in
//! ascending order. Because the walk only needs the slices, the other set may
//! live in a **different store** — a heap set can union with a `static`
//! [`SliceSet`](crate::SliceSet) table.
//!
//! [`Intersection`], [`Difference`], and the subset/disjoint predicates are
//! **size-adaptive**: when one side is ≥16× smaller they drop the merge and
//! binary-search each element of the small side into the large one
//! (`O(n log m)`), so intersecting a 3-element set with a 100k-entry table
//! probes 3 times instead of walking 100k — the same tipping point `BTreeSet`
//! uses. [`Union`] and [`SymmetricDifference`] stay pure merges: their output
//! is `Ω(n + m)` (every distinct element can appear), so there is nothing to
//! skip. Every walk yields ascending, so [`min`](Iterator::min) is one `next`,
//! not a drain.
//!
//! [`SortedSet::union`]: crate::SortedSet::union

use core::cmp::Ordering;
use core::iter::FusedIterator;

/// Returns `true` if sorted `a` is a subset of sorted `b`.
///
/// Length check, then either per-element binary search (small `a`) or a linear merge
/// walk.
pub(crate) fn is_subset<T: Ord>(a: &[T], mut b: &[T]) -> bool {
    if a.len() > b.len() {
        return false;
    }
    if a.len().saturating_mul(16) <= b.len() {
        // `a` is much smaller: n·log m searches beat the n+m walk. Each hit
        // shrinks `b`, so later searches scan less.
        for x in a {
            match b.binary_search(x) {
                Ok(i) => b = &b[i + 1..],
                Err(_) => return false,
            }
        }
        return true;
    }
    let mut j = 0;
    'outer: for x in a {
        while j < b.len() {
            match b[j].cmp(x) {
                Ordering::Less => j += 1,
                Ordering::Equal => {
                    j += 1;
                    continue 'outer;
                }
                Ordering::Greater => return false,
            }
        }
        return false; // `b` exhausted with `x` unmatched
    }
    true
}

/// Returns `true` if sorted `a` and sorted `b` share no element.
///
/// Iterates the smaller side; binary-search path when it is much smaller. (The single
/// lifetime is what lets the two slices swap roles.)
pub(crate) fn is_disjoint<'s, T: Ord>(mut a: &'s [T], mut b: &'s [T]) -> bool {
    if a.len() > b.len() {
        core::mem::swap(&mut a, &mut b);
    }
    if a.len().saturating_mul(16) <= b.len() {
        for x in a {
            match b.binary_search(x) {
                Ok(_) => return false,
                Err(i) => b = &b[i..],
            }
        }
        return true;
    }
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => return false,
        }
    }
    true
}

/// Ascending iterator over the elements in `a`, `b`, or both — see
/// [`SortedSet::union`](crate::SortedSet::union).
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Union<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

/// Ascending iterator over the elements in both `a` and `b` — see
/// [`SortedSet::intersection`](crate::SortedSet::intersection).
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Intersection<'a, T> {
    a: &'a [T],
    b: &'a [T],
    /// When set, `a` is the (≥16×) smaller side: probe each of its elements into
    /// `b` rather than merging — `O(n log m)` instead of `O(n + m)`.
    search: bool,
}

/// Ascending iterator over the elements in `a` but not `b` — see
/// [`SortedSet::difference`](crate::SortedSet::difference).
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct Difference<'a, T> {
    a: &'a [T],
    b: &'a [T],
    /// When set, the kept side `a` is ≥16× smaller than the subtracted side `b`:
    /// probe each of its elements into `b` rather than merging.
    search: bool,
}

/// Ascending iterator over the elements in exactly one of `a`, `b` — see
/// [`SortedSet::symmetric_difference`](crate::SortedSet::symmetric_difference).
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SymmetricDifference<'a, T> {
    a: &'a [T],
    b: &'a [T],
}

impl<'a, T> Union<'a, T> {
    pub(crate) fn new(a: &'a [T], b: &'a [T]) -> Self {
        Union { a, b }
    }
}
impl<'a, T> Intersection<'a, T> {
    pub(crate) fn new(a: &'a [T], b: &'a [T]) -> Self {
        // Intersection is symmetric, so orient `a` to the smaller side whenever a
        // side is ≥16× smaller; the search walk then always probes `a` into `b`.
        if b.len().saturating_mul(16) <= a.len() {
            Intersection {
                a: b,
                b: a,
                search: true,
            }
        } else {
            let search = a.len().saturating_mul(16) <= b.len();
            Intersection { a, b, search }
        }
    }
}
impl<'a, T> Difference<'a, T> {
    pub(crate) fn new(a: &'a [T], b: &'a [T]) -> Self {
        // Difference is asymmetric: only the kept side `a` being ≥16× smaller than
        // the subtracted side `b` lets per-element probing beat the merge.
        let search = a.len().saturating_mul(16) <= b.len();
        Difference { a, b, search }
    }
}
impl<'a, T> SymmetricDifference<'a, T> {
    pub(crate) fn new(a: &'a [T], b: &'a [T]) -> Self {
        SymmetricDifference { a, b }
    }
}

// Manual `Clone` impls: a derive would bound every type parameter (`T: Clone`),
// but these iterators hold only slice borrows and clone for *any* `T` — matching
// `BTreeSet`'s algebra iterators, whose `Clone` is likewise unconditional.
impl<T> Clone for Union<'_, T> {
    fn clone(&self) -> Self {
        Union {
            a: self.a,
            b: self.b,
        }
    }
}
impl<T> Clone for Intersection<'_, T> {
    fn clone(&self) -> Self {
        Intersection {
            a: self.a,
            b: self.b,
            search: self.search,
        }
    }
}
impl<T> Clone for Difference<'_, T> {
    fn clone(&self) -> Self {
        Difference {
            a: self.a,
            b: self.b,
            search: self.search,
        }
    }
}
impl<T> Clone for SymmetricDifference<'_, T> {
    fn clone(&self) -> Self {
        SymmetricDifference {
            a: self.a,
            b: self.b,
        }
    }
}

impl<'a, T: Ord> Iterator for Union<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match (self.a.split_first(), self.b.split_first()) {
            (Some((x, a_rest)), Some((y, b_rest))) => match x.cmp(y) {
                Ordering::Less => {
                    self.a = a_rest;
                    Some(x)
                }
                Ordering::Greater => {
                    self.b = b_rest;
                    Some(y)
                }
                Ordering::Equal => {
                    // In both sets: yield once, advance both.
                    self.a = a_rest;
                    self.b = b_rest;
                    Some(x)
                }
            },
            (Some((x, a_rest)), None) => {
                self.a = a_rest;
                Some(x)
            }
            (None, Some((y, b_rest))) => {
                self.b = b_rest;
                Some(y)
            }
            (None, None) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // At least the larger set (its elements all appear); at most both.
        (
            self.a.len().max(self.b.len()),
            self.a.len().checked_add(self.b.len()),
        )
    }

    fn min(mut self) -> Option<&'a T> {
        // Output is ascending, so the first element is the minimum — one step, not
        // the draining default.
        self.next()
    }
}

impl<'a, T: Ord> Iterator for Intersection<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.search {
            // `a` is the smaller side: probe each of its elements into `b`, shrinking
            // `b` past every probe (both ascending, so the suffix always suffices).
            loop {
                let (x, rest) = self.a.split_first()?;
                self.a = rest;
                match self.b.binary_search(x) {
                    Ok(i) => {
                        self.b = &self.b[i + 1..];
                        return Some(x);
                    }
                    Err(i) => self.b = &self.b[i..],
                }
            }
        }
        loop {
            let (x, a_rest) = self.a.split_first()?;
            let (y, b_rest) = self.b.split_first()?;
            match x.cmp(y) {
                Ordering::Less => self.a = a_rest,
                Ordering::Greater => self.b = b_rest,
                Ordering::Equal => {
                    self.a = a_rest;
                    self.b = b_rest;
                    return Some(x);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.a.len().min(self.b.len())))
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<'a, T: Ord> Iterator for Difference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.search {
            // `a` (kept) is the smaller side: probe each element into `b` (subtracted)
            // and yield the misses, shrinking `b` past every probe.
            loop {
                let (x, rest) = self.a.split_first()?;
                self.a = rest;
                match self.b.binary_search(x) {
                    Ok(i) => self.b = &self.b[i + 1..],
                    Err(i) => {
                        self.b = &self.b[i..];
                        return Some(x);
                    }
                }
            }
        }
        loop {
            let (x, a_rest) = self.a.split_first()?;
            let Some((y, b_rest)) = self.b.split_first() else {
                self.a = a_rest;
                return Some(x);
            };
            match x.cmp(y) {
                Ordering::Less => {
                    self.a = a_rest;
                    return Some(x);
                }
                Ordering::Greater => self.b = b_rest,
                Ordering::Equal => {
                    // In both sets: not part of the difference.
                    self.a = a_rest;
                    self.b = b_rest;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // At most `b.len()` elements of `a` can be cancelled.
        (
            self.a.len().saturating_sub(self.b.len()),
            Some(self.a.len()),
        )
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

impl<'a, T: Ord> Iterator for SymmetricDifference<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            match (self.a.split_first(), self.b.split_first()) {
                (Some((x, a_rest)), Some((y, b_rest))) => match x.cmp(y) {
                    Ordering::Less => {
                        self.a = a_rest;
                        return Some(x);
                    }
                    Ordering::Greater => {
                        self.b = b_rest;
                        return Some(y);
                    }
                    Ordering::Equal => {
                        // In both sets: in neither side of the symmetric diff.
                        self.a = a_rest;
                        self.b = b_rest;
                    }
                },
                (Some((x, a_rest)), None) => {
                    self.a = a_rest;
                    return Some(x);
                }
                (None, Some((y, b_rest))) => {
                    self.b = b_rest;
                    return Some(y);
                }
                (None, None) => return None,
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // The shared elements cancel pairwise, so at least the length gap.
        (
            self.a.len().abs_diff(self.b.len()),
            self.a.len().checked_add(self.b.len()),
        )
    }

    fn min(mut self) -> Option<&'a T> {
        self.next()
    }
}

// Once both slices are empty every `next` is `None` — trivially fused.
impl<T: Ord> FusedIterator for Union<'_, T> {}
impl<T: Ord> FusedIterator for Intersection<'_, T> {}
impl<T: Ord> FusedIterator for Difference<'_, T> {}
impl<T: Ord> FusedIterator for SymmetricDifference<'_, T> {}
