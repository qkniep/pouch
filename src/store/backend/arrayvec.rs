//! `ArrayVec`: fixed capacity `N`, fallible; native shifting insert/remove. Not
//! `Unbounded`.

use arrayvec::ArrayVec;

use crate::error::CapacityError;
use crate::store::{Store, StoreMut, StoreNew};

impl<T, const N: usize> Store for ArrayVec<T, N> {
    type Elem = T;
    fn as_slice(&self) -> &[T] {
        &self[..]
    }
    fn max_capacity(&self) -> Option<usize> {
        Some(N)
    }
}
impl<T, const N: usize> StoreMut for ArrayVec<T, N> {
    fn try_insert_at(&mut self, i: usize, value: T) -> Result<(), CapacityError<T>> {
        if self.len() >= N {
            return Err(CapacityError(value));
        }
        // Safe: not full and i <= len, so `insert` cannot panic.
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
        ArrayVec::clear(self)
    }
}
impl<T, const N: usize> StoreNew for ArrayVec<T, N> {
    fn new() -> Self {
        ArrayVec::new()
    }
}
