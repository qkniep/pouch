//! `serde` support (feature `serde`): sets and bags serialize as sequences,
//! maps as maps, all by content (`iter()` order — ascending for the sorted
//! flavors, so round-trips feed the deserializer pre-sorted input).
//!
//! Deserialization routes through the **same fallible builders as the rest of
//! the crate** (`try_from_iter`), so it enforces the bulk-build policy rather
//! than inventing a serde-specific one:
//!
//! * **sets dedup silently, maps reject duplicate keys** as a data error — unlike the
//!   standard library's silently-last-wins map impls, a duplicate key here is ambiguous
//!   input, so it fails loudly;
//! * **a bounded store that fills mid-stream is a data error** too ([`de::Error`]), not a
//!   panic — deserializing into an `ArrayVec`/`heapless` collection is input validation
//!   for free;
//! * the length claimed on the wire is treated with serde's usual caution (capped before
//!   it reaches [`reserve`](crate::store::StoreMut::reserve)), so hostile input can't
//!   force a giant allocation up front.
//!
//! **Caveat — the bulk-build tradeoff surfaces here.** The `try_from_iter` builders
//! append every entry *before* the dedup/sort pass, so on a bounded backend the
//! deserializer can reject input whose *deduplicated* result would have fit: `[1, 1, 2]`
//! into a `SortedSet<ArrayVec<u8, 2>>` is a capacity error even though `{1, 2}` fits, and
//! a duplicate map key can surface as [`BuildError::Capacity`] (the append overflowed)
//! rather than the more precise [`BuildError::DuplicateKey`] when the raw count exceeds
//! the bound first. This is the documented crate-wide bulk-build tradeoff — one
//! `O(n log n)` pass instead of a per-element `try_insert` loop — but it is most
//! surprising precisely at this untrusted-input boundary, where the reported error can
//! misidentify *why* valid-looking input was refused. Deserializing into an unbounded
//! store, or feeding pre-deduplicated input, sidesteps it; for exact-fit bounded input
//! validation, build via `from_store` + a `try_insert` loop instead.
//!
//! Deserialize needs to build a store, so it is bounded on `StoreMut +
//! StoreNew`: stores that need runtime state to construct ([`Capped`], a cap;
//! [`ScratchVec`], a buffer) serialize but don't deserialize — construct them
//! via `from_store` and fill with `try_extend` instead.
//!
//! [`de::Error`]: serde::de::Error
//! [`Capped`]: crate::Capped
//! [`ScratchVec`]: crate::ScratchVec

use core::fmt;
use core::marker::PhantomData;

use serde::de::{Deserialize, Deserializer, Error as _, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, Serializer};

use crate::store::{Store, StoreMut, StoreNew};
#[cfg(feature = "soa")]
use crate::{
    Bag, SortedColumnMap, SortedMap, SortedSet, UnsortedColumnMap, UnsortedMap, UnsortedSet,
};
#[cfg(not(feature = "soa"))]
use crate::{Bag, SortedMap, SortedSet, UnsortedMap, UnsortedSet};

/// Serde's "cautious" length policy: trust the wire's claimed length only up
/// to a small bound, so a hostile `len` can't force a giant pre-allocation.
/// (The builders still grow beyond this fine — it only shapes `reserve`.)
const CAUTIOUS_LEN_CAP: usize = 4096;

/// Drive a [`SeqAccess`] as a plain `Iterator`, so the crate's own builders
/// (`try_from_iter`) can consume it unchanged.
///
/// A deserializer error has no lane through the builder's error type, so it is stashed
/// aside and checked by the caller *before* the builder's result — an error mid-stream
/// just cuts the iterator short, and the truncated build result must not be trusted.
struct SeqIter<'de, 'a, A, T>
where
    A: SeqAccess<'de>,
{
    access: &'a mut A,
    error: &'a mut Option<A::Error>,
    marker: PhantomData<(&'de (), T)>,
}

impl<'de, A, T> Iterator for SeqIter<'de, '_, A, T>
where
    A: SeqAccess<'de>,
    T: Deserialize<'de>,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.error.is_some() {
            return None;
        }
        match self.access.next_element::<T>() {
            Ok(item) => item,
            Err(e) => {
                *self.error = Some(e);
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.access.size_hint().unwrap_or(0).min(CAUTIOUS_LEN_CAP),
            None,
        )
    }
}

/// The [`MapAccess`] twin of [`SeqIter`], yielding owned `(K, V)` entries.
struct MapEntryIter<'de, 'a, A, K, V>
where
    A: MapAccess<'de>,
{
    access: &'a mut A,
    error: &'a mut Option<A::Error>,
    marker: PhantomData<(&'de (), K, V)>,
}

impl<'de, A, K, V> Iterator for MapEntryIter<'de, '_, A, K, V>
where
    A: MapAccess<'de>,
    K: Deserialize<'de>,
    V: Deserialize<'de>,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if self.error.is_some() {
            return None;
        }
        match self.access.next_entry::<K, V>() {
            Ok(entry) => entry,
            Err(e) => {
                *self.error = Some(e);
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.access.size_hint().unwrap_or(0).min(CAUTIOUS_LEN_CAP),
            None,
        )
    }
}

// ---------------------------------------------------------------------------
// Sequence-shaped collections: SortedSet, UnsortedSet, Bag.
// ---------------------------------------------------------------------------

impl<S> Serialize for SortedSet<S>
where
    S: Store,
    S::Elem: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de, S> Deserialize<'de> for SortedSet<S>
where
    S: StoreMut + StoreNew,
    S::Elem: Deserialize<'de> + Ord,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V<S>(PhantomData<S>);
        impl<'de, S> Visitor<'de> for V<S>
        where
            S: StoreMut + StoreNew,
            S::Elem: Deserialize<'de> + Ord,
        {
            type Value = SortedSet<S>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a sequence of set elements")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = SortedSet::try_from_iter(SeqIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_seq(V(PhantomData))
    }
}

impl<S> Serialize for UnsortedSet<S>
where
    S: Store,
    S::Elem: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de, S> Deserialize<'de> for UnsortedSet<S>
where
    S: StoreMut + StoreNew,
    S::Elem: Deserialize<'de> + Eq,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V<S>(PhantomData<S>);
        impl<'de, S> Visitor<'de> for V<S>
        where
            S: StoreMut + StoreNew,
            S::Elem: Deserialize<'de> + Eq,
        {
            type Value = UnsortedSet<S>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a sequence of set elements")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = UnsortedSet::try_from_iter(SeqIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_seq(V(PhantomData))
    }
}

impl<S> Serialize for Bag<S>
where
    S: Store,
    S::Elem: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de, S> Deserialize<'de> for Bag<S>
where
    S: StoreMut + StoreNew,
    S::Elem: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V<S>(PhantomData<S>);
        impl<'de, S> Visitor<'de> for V<S>
        where
            S: StoreMut + StoreNew,
            S::Elem: Deserialize<'de>,
        {
            type Value = Bag<S>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a sequence")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = Bag::try_from_iter(SeqIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_seq(V(PhantomData))
    }
}

// ---------------------------------------------------------------------------
// Map-shaped collections: SortedMap, UnsortedMap (+ the soa column maps).
// ---------------------------------------------------------------------------

impl<K, V, S> Serialize for SortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: Serialize,
    V: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_map(self.iter())
    }
}

impl<'de, K, V, S> Deserialize<'de> for SortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + StoreNew,
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V2<S>(PhantomData<S>);
        impl<'de, K, V, S> Visitor<'de> for V2<S>
        where
            S: StoreMut<Elem = (K, V)> + StoreNew,
            K: Deserialize<'de> + Ord,
            V: Deserialize<'de>,
        {
            type Value = SortedMap<S>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a map with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = SortedMap::try_from_iter(MapEntryIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_map(V2(PhantomData))
    }
}

impl<K, V, S> Serialize for UnsortedMap<S>
where
    S: Store<Elem = (K, V)>,
    K: Serialize,
    V: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_map(self.iter())
    }
}

impl<'de, K, V, S> Deserialize<'de> for UnsortedMap<S>
where
    S: StoreMut<Elem = (K, V)> + StoreNew,
    K: Deserialize<'de> + Eq,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V2<S>(PhantomData<S>);
        impl<'de, K, V, S> Visitor<'de> for V2<S>
        where
            S: StoreMut<Elem = (K, V)> + StoreNew,
            K: Deserialize<'de> + Eq,
            V: Deserialize<'de>,
        {
            type Value = UnsortedMap<S>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a map with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = UnsortedMap::try_from_iter(MapEntryIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_map(V2(PhantomData))
    }
}

#[cfg(feature = "soa")]
impl<K, V, SK, SV> Serialize for UnsortedColumnMap<SK, SV>
where
    SK: Store<Elem = K>,
    SV: Store<Elem = V>,
    K: Serialize,
    V: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_map(self.keys().iter().zip(self.values().iter()))
    }
}

#[cfg(feature = "soa")]
impl<'de, K, V, SK, SV> Deserialize<'de> for UnsortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + StoreNew,
    SV: StoreMut<Elem = V> + StoreNew,
    K: Deserialize<'de> + Eq,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V2<SK, SV>(PhantomData<(SK, SV)>);
        impl<'de, K, V, SK, SV> Visitor<'de> for V2<SK, SV>
        where
            SK: StoreMut<Elem = K> + StoreNew,
            SV: StoreMut<Elem = V> + StoreNew,
            K: Deserialize<'de> + Eq,
            V: Deserialize<'de>,
        {
            type Value = UnsortedColumnMap<SK, SV>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a map with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = UnsortedColumnMap::try_from_iter(MapEntryIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_map(V2(PhantomData))
    }
}

#[cfg(feature = "soa")]
impl<K, V, SK, SV> Serialize for SortedColumnMap<SK, SV>
where
    SK: Store<Elem = K>,
    SV: Store<Elem = V>,
    K: Serialize,
    V: Serialize,
{
    fn serialize<Ser: Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        serializer.collect_map(self.keys().iter().zip(self.values().iter()))
    }
}

#[cfg(feature = "soa")]
impl<'de, K, V, SK, SV> Deserialize<'de> for SortedColumnMap<SK, SV>
where
    SK: StoreMut<Elem = K> + StoreNew,
    SV: StoreMut<Elem = V> + StoreNew,
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V2<SK, SV>(PhantomData<(SK, SV)>);
        impl<'de, K, V, SK, SV> Visitor<'de> for V2<SK, SV>
        where
            SK: StoreMut<Elem = K> + StoreNew,
            SV: StoreMut<Elem = V> + StoreNew,
            K: Deserialize<'de> + Ord,
            V: Deserialize<'de>,
        {
            type Value = SortedColumnMap<SK, SV>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a map with unique keys")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut error = None;
                let result = SortedColumnMap::try_from_iter(MapEntryIter {
                    access: &mut access,
                    error: &mut error,
                    marker: PhantomData,
                });
                match error {
                    Some(e) => Err(e),
                    None => result.map_err(A::Error::custom),
                }
            }
        }
        deserializer.deserialize_map(V2(PhantomData))
    }
}

// Vec exercises the unbounded paths; serde_test drives exact token streams.
// The sorted types have `PartialEq`, so they get full `assert_tokens`
// round-trips; the unsorted flavors (no `PartialEq` — order-sensitive derive
// would be wrong) serialize-check with tokens and deserialize through serde's
// value deserializers, asserting contents through the collection API.
#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use serde_test::{assert_de_tokens, assert_ser_tokens, assert_tokens, Token};

    use crate::{Bag, SortedMap, SortedSet, UnsortedMap, UnsortedSet};

    #[test]
    fn sorted_set_round_trips_as_seq() {
        let set: SortedSet<Vec<i32>> = [2, 1, 3].into_iter().collect();
        assert_tokens(
            &set,
            &[
                Token::Seq { len: Some(3) },
                Token::I32(1),
                Token::I32(2),
                Token::I32(3),
                Token::SeqEnd,
            ],
        );
    }

    #[test]
    fn sorted_set_deserializes_unsorted_input() {
        // Foreign input needn't be sorted or unique: the builder sorts and
        // dedups exactly as `try_from_iter` documents.
        let expected: SortedSet<Vec<i32>> = [1, 3].into_iter().collect();
        assert_de_tokens(
            &expected,
            &[
                Token::Seq { len: Some(3) },
                Token::I32(3),
                Token::I32(1),
                Token::I32(3),
                Token::SeqEnd,
            ],
        );
    }

    #[test]
    fn sorted_map_round_trips_as_map() {
        let map: SortedMap<Vec<(i32, &str)>> =
            SortedMap::try_from_iter([(2, "b"), (1, "a")]).unwrap();
        assert_tokens(
            &map,
            &[
                Token::Map { len: Some(2) },
                Token::I32(1),
                Token::BorrowedStr("a"),
                Token::I32(2),
                Token::BorrowedStr("b"),
                Token::MapEnd,
            ],
        );
    }

    #[test]
    fn map_rejects_duplicate_keys() {
        serde_test::assert_de_tokens_error::<SortedMap<Vec<(i32, i32)>>>(
            &[
                Token::Map { len: Some(2) },
                Token::I32(1),
                Token::I32(10),
                Token::I32(1),
                Token::I32(20),
                Token::MapEnd,
            ],
            "duplicate key in bulk build",
        );
    }

    #[test]
    fn unsorted_flavors_serialize_and_rebuild() {
        use serde::de::value::{Error as ValueError, MapDeserializer, SeqDeserializer};
        use serde::Deserialize;

        let mut set: UnsortedSet<Vec<i32>> = UnsortedSet::new();
        set.insert(1);
        set.insert(2);
        assert_ser_tokens(
            &set,
            &[
                Token::Seq { len: Some(2) },
                Token::I32(1),
                Token::I32(2),
                Token::SeqEnd,
            ],
        );

        // Deserialize via serde's value deserializers (no `PartialEq` on the
        // unsorted flavors, so contents are asserted through the API).
        let de = SeqDeserializer::<_, ValueError>::new([1, 2, 1].into_iter());
        let set = UnsortedSet::<Vec<i32>>::deserialize(de).expect("dups dedup silently");
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1) && set.contains(&2));

        // (Integer values: serde's value deserializers feed strings as
        // transient, so a borrowed `&str` value type wouldn't deserialize.)
        let de = MapDeserializer::<_, ValueError>::new([(1, 10), (2, 20)].into_iter());
        let map = UnsortedMap::<Vec<(i32, i32)>>::deserialize(de).expect("unique keys");
        assert_eq!(map.get(&2), Some(&20));

        // Unsorted maps reject duplicate keys too — same builder policy.
        let de = MapDeserializer::<_, ValueError>::new([(1, 10), (1, 99)].into_iter());
        assert!(UnsortedMap::<Vec<(i32, i32)>>::deserialize(de).is_err());

        // Bags keep duplicates: the loosest of the sequence builds.
        let de = SeqDeserializer::<_, ValueError>::new([5, 5].into_iter());
        let bag = Bag::<Vec<i32>>::deserialize(de).expect("dups allowed");
        assert_eq!(bag.as_slice(), &[5, 5]);
    }

    // Bounded deserialization is input validation: overflow is a data error
    // with the capacity message, not a panic.
    #[cfg(feature = "arrayvec")]
    #[test]
    fn bounded_deserialize_errors_at_capacity() {
        use arrayvec::ArrayVec;

        serde_test::assert_de_tokens_error::<SortedSet<ArrayVec<u8, 2>>>(
            &[
                Token::Seq { len: Some(3) },
                Token::U8(1),
                Token::U8(2),
                Token::U8(3),
                Token::SeqEnd,
            ],
            "store is at logical capacity",
        );
    }

    #[cfg(feature = "soa")]
    #[test]
    fn column_maps_round_trip_as_maps() {
        use crate::{SortedColumnMap, UnsortedColumnMap};

        let map: SortedColumnMap<Vec<i32>, Vec<i32>> =
            SortedColumnMap::try_from_iter([(2, 20), (1, 10)]).unwrap();
        assert_tokens(
            &map,
            &[
                Token::Map { len: Some(2) },
                Token::I32(1),
                Token::I32(10),
                Token::I32(2),
                Token::I32(20),
                Token::MapEnd,
            ],
        );

        use serde::de::value::{Error as ValueError, MapDeserializer};
        use serde::Deserialize;
        let de = MapDeserializer::<_, ValueError>::new([(7, 70)].into_iter());
        let map = UnsortedColumnMap::<Vec<i32>, Vec<i32>>::deserialize(de).expect("unique keys");
        assert_eq!(map.get(&7), Some(&70));
    }
}
