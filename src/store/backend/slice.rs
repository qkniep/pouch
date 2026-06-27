//! `&[T]` / `&[T; N]`: a borrowed, **read-only** backend. Implements `Store`
//! only — never `StoreMut` / `StoreNew` / `Unbounded` — so it backs lookups
//! (`contains` / `get`) but no mutation. `capacity()` is `Some(len)`: a borrowed
//! slice can never grow, so it is permanently at capacity, holding exactly its
//! elements. The `&[T; N]` impl is the same backend over an array reference, so a
//! `static [_; N]` table wraps directly as `&TABLE` (no `[..]`).
//!
//! Unlike the other backends this needs no dependency and no `alloc`, so it is
//! ungated — usable even under `--no-default-features`. The headline use is a
//! `static` sorted table wrapped via `from_store` for zero-alloc, zero-RAM
//! lookups straight out of flash / `.rodata`:
//!
//! ```
//! use pouch::SortedMap;
//! static STATUS: [(u16, &str); 3] = [(200, "OK"), (404, "Not Found"), (500, "Error")];
//! let codes = SortedMap::from_store(&STATUS); // `&ARR` or `&ARR[..]` both work
//! assert_eq!(codes.get(&404), Some(&"Not Found"));
//! ```

use crate::store::Store;

impl<T> Store for &[T] {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        &self[..]
    }
    fn capacity(&self) -> Option<usize> {
        // A borrowed slice can never grow: it is permanently full, holding
        // exactly its current elements, so cap == len.
        Some(self.len())
    }
}

// Array references reuse the same read-only backend, so `&[T; N]` works wherever
// `&[T]` does — letting a `static [_; N]` table be wrapped as `&TABLE` directly.
impl<T, const N: usize> Store for &[T; N] {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        &self[..]
    }
    fn capacity(&self) -> Option<usize> {
        Some(N) // statically full at the array's length
    }
}

// Ungated, so these run in every config — including the alloc-free
// `--no-default-features` core path, the one build with no other backend.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SortedMap, SortedSet};

    #[test]
    fn read_only_slice_reports_len_and_full_capacity() {
        let xs: &[u32] = &[10, 20, 30];
        // Fully qualified: `<[T]>::as_slice` is an unstable inherent that would
        // shadow our trait method and trip `unstable_name_collisions`.
        assert_eq!(Store::as_slice(&xs), &[10, 20, 30]);
        assert_eq!(xs.len(), 3);
        assert!(!xs.is_empty());
        // A borrowed slice is permanently full: cap == len.
        assert_eq!(Store::capacity(&xs), Some(3));

        let empty: &[u32] = &[];
        assert!(empty.is_empty());
        assert_eq!(Store::capacity(&empty), Some(0));
    }

    #[test]
    fn read_only_array_ref_reports_const_capacity() {
        let xs: &[u32; 3] = &[10, 20, 30];
        assert_eq!(Store::as_slice(&xs), &[10, 20, 30]);
        assert_eq!(Store::capacity(&xs), Some(3));
    }

    #[test]
    fn backs_a_read_only_sorted_set() {
        // `&ARR` (array ref) and `&ARR[..]` (slice) both back the set.
        let from_array = SortedSet::from_store(&[1u32, 2, 3]);
        assert!(from_array.contains(&2));
        assert!(!from_array.contains(&4));
        assert_eq!(from_array.capacity(), Some(3)); // permanently full

        let from_slice = SortedSet::from_store(&[1u32, 2, 3][..]);
        assert!(from_slice.contains(&2));
    }

    #[test]
    fn backs_a_read_only_sorted_map() {
        let map = SortedMap::from_store(&[(1u8, "a"), (2, "b")]);
        assert_eq!(map.get(&2), Some(&"b"));
        assert_eq!(map.get(&3), None);
        assert!(map.contains_key(&1));
    }
}
