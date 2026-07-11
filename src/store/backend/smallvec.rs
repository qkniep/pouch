//! `SmallVec`: inline `N`, spills to heap; unbounded.

use smallvec::{Array, SmallVec};

use crate::error::CapacityError;
use crate::store::{Store, StoreMut, StoreNew, Unbounded};

impl<A: Array> Store for SmallVec<A> {
    type Elem = A::Item;
    fn as_slice(&self) -> &[A::Item] {
        &self[..]
    }
    fn max_capacity(&self) -> Option<usize> {
        None
    }
}
impl<A: Array> StoreMut for SmallVec<A> {
    fn try_insert_at(&mut self, i: usize, value: A::Item) -> Result<(), CapacityError<A::Item>> {
        self.insert(i, value);
        Ok(())
    }
    fn remove_at(&mut self, i: usize) -> A::Item {
        self.remove(i)
    }
    fn as_mut_slice(&mut self) -> &mut [A::Item] {
        &mut self[..]
    }
    fn clear(&mut self) {
        SmallVec::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        SmallVec::reserve(self, additional);
    }
}
impl<A: Array> StoreNew for SmallVec<A> {
    fn new() -> Self {
        SmallVec::new()
    }
}
impl<A: Array> Unbounded for SmallVec<A> {}
