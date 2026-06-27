//! `TinyVec`: inline `N`, spills to heap; 100% safe; unbounded. Requires
//! `Item: Default`.

use tinyvec::{Array, TinyVec};

use crate::error::CapacityError;
use crate::store::{Store, StoreMut, StoreNew, Unbounded};

impl<A: Array> Store for TinyVec<A>
where
    A::Item: Default,
{
    type Elem = A::Item;
    fn as_slice(&self) -> &[A::Item] {
        &self[..]
    }
    fn capacity(&self) -> Option<usize> {
        None
    }
}
impl<A: Array> StoreMut for TinyVec<A>
where
    A::Item: Default,
{
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
        TinyVec::clear(self)
    }
}
impl<A: Array> StoreNew for TinyVec<A>
where
    A::Item: Default,
{
    fn new() -> Self {
        TinyVec::default()
    }
}
impl<A: Array> Unbounded for TinyVec<A> where A::Item: Default {}
