//! `Vec`: growable, heap-backed, unbounded.

use alloc::vec::Vec;

use crate::error::CapacityError;
use crate::store::{Store, StoreMut, StoreNew, Unbounded};

impl<T> Store for Vec<T> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        &self[..]
    }
    fn capacity(&self) -> Option<usize> {
        None // logical capacity; distinct from Vec::capacity() (allocation)
    }
}
impl<T> StoreMut for Vec<T> {
    fn try_insert_at(&mut self, i: usize, value: T) -> Result<(), CapacityError<T>> {
        self.insert(i, value);
        Ok(())
    }
    fn remove_at(&mut self, i: usize) -> T {
        self.remove(i)
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self[..]
    }
    fn clear(&mut self) {
        Vec::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }
}
impl<T> StoreNew for Vec<T> {
    fn new() -> Self {
        Vec::new()
    }
}
impl<T> Unbounded for Vec<T> {}
