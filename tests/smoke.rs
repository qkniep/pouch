use pouch::*;

#[test]
fn vec_set_unbounded() {
    // Vec is Unbounded -> infallible `insert` is available.
    let mut s: SortedSet<Vec<u64>> = SortedSet::new();
    assert!(s.insert(5));
    assert!(s.insert(1));
    assert!(s.insert(3));
    assert!(!s.insert(3)); // duplicate
    assert_eq!(s.as_slice(), &[1, 3, 5]);
    assert!(s.contains(&3));
    assert!(s.remove(&3));
    assert_eq!(s.as_slice(), &[1, 5]);
    assert_eq!(s.capacity(), None);
}

#[test]
fn arrayvec_fixed_capacity() {
    let mut s: SortedSet<arrayvec::ArrayVec<u64, 3>> = SortedSet::new();
    assert_eq!(s.capacity(), Some(3));
    assert_eq!(s.try_insert(5), Ok(true));
    assert_eq!(s.try_insert(1), Ok(true));
    assert_eq!(s.try_insert(9), Ok(true));
    // full: a new element is rejected and handed back.
    match s.try_insert(2) {
        Err(e) => assert_eq!(e.into_inner(), 2),
        Ok(_) => panic!("expected capacity error"),
    }
    // a duplicate consumes no capacity even when full.
    assert_eq!(s.try_insert(5), Ok(false));
    assert_eq!(s.as_slice(), &[1, 5, 9]);
}

#[test]
fn heapless_sorted_via_rotate() {
    let mut s: SortedSet<heapless::Vec<u64, 4>> = SortedSet::new();
    for x in [3u64, 1, 4, 2] {
        assert_eq!(s.try_insert(x), Ok(true));
    }
    assert_eq!(s.as_slice(), &[1, 2, 3, 4]); // rotate-based insert keeps order
    assert!(s.remove(&3));
    assert_eq!(s.as_slice(), &[1, 2, 4]);
}

#[test]
fn smallvec_spills_and_stays_sorted() {
    // inline capacity 2; third insert spills to heap.
    let mut s: Set<u64, 2> = Set::new();
    assert!(s.insert(30));
    assert!(s.insert(10));
    assert!(s.insert(20));
    assert_eq!(s.as_slice(), &[10, 20, 30]);
    assert_eq!(s.capacity(), None); // still unbounded after spilling
}

#[test]
fn tinyvec_set() {
    // TinyVec requires Elem: Default — u64 qualifies.
    let mut s: SortedSet<tinyvec::TinyVec<[u64; 4]>> = SortedSet::new();
    assert!(s.insert(2));
    assert!(s.insert(1));
    assert_eq!(s.as_slice(), &[1, 2]);
}

#[test]
fn capped_runtime_bound_over_vec() {
    let mut s: SortedSet<Capped<Vec<u64>>> = SortedSet::from_store(Capped::with_capacity(3));
    assert_eq!(s.capacity(), Some(3));
    assert_eq!(s.try_insert(5), Ok(true));
    assert_eq!(s.try_insert(1), Ok(true));
    assert_eq!(s.try_insert(3), Ok(true));
    assert!(s.try_insert(2).is_err()); // cap reached -> Err on growable backend
    assert_eq!(s.try_insert(5), Ok(false)); // duplicate at cap is fine
    assert_eq!(s.as_slice(), &[1, 3, 5]);
}

#[test]
fn sorted_map_replace_no_capacity() {
    let mut m: SortedMap<Vec<(u64, &str)>> = SortedMap::new();
    assert_eq!(m.try_insert(2, "two"), Ok(None));
    assert_eq!(m.try_insert(1, "one"), Ok(None));
    assert_eq!(m.get(&1), Some(&"one"));
    // replacing an existing key returns the old value.
    assert_eq!(m.try_insert(2, "TWO"), Ok(Some("two")));
    assert_eq!(m.get(&2), Some(&"TWO"));
    assert_eq!(m.len(), 2);
}

#[test]
fn capped_map_replace_at_cap() {
    let mut m: SortedMap<Capped<Vec<(u64, u64)>>> = SortedMap::from_store(Capped::with_capacity(2));
    m.try_insert(1, 10).unwrap();
    m.try_insert(2, 20).unwrap();
    // full: a NEW key errors...
    assert!(m.try_insert(3, 30).is_err());
    // ...but replacing an EXISTING key consumes no capacity, so it succeeds.
    assert_eq!(m.try_insert(2, 22), Ok(Some(20)));
}

#[test]
fn unsorted_set_basic() {
    // Vec is Unbounded -> infallible `insert`.
    let mut s: UnsortedSet<Vec<u64>> = UnsortedSet::new();
    assert!(s.insert(5));
    assert!(s.insert(1));
    assert!(s.insert(3));
    assert!(!s.insert(3)); // duplicate, no capacity consumed
    assert_eq!(s.len(), 3);
    assert!(s.contains(&1) && s.contains(&5) && !s.contains(&2));
    assert_eq!(s.capacity(), None);
    // swap-remove: order is not preserved, but membership stays correct.
    assert!(s.remove(&5));
    assert!(!s.remove(&5));
    assert_eq!(s.len(), 2);
    assert!(s.contains(&1) && s.contains(&3) && !s.contains(&5));
}

#[test]
fn unsorted_set_dup_at_cap() {
    let mut s: UnsortedSet<Capped<Vec<u64>>> = UnsortedSet::from_store(Capped::with_capacity(2));
    assert_eq!(s.capacity(), Some(2));
    assert_eq!(s.try_insert(7), Ok(true));
    assert_eq!(s.try_insert(8), Ok(true));
    // full: a new element is rejected and handed back...
    match s.try_insert(9) {
        Err(e) => assert_eq!(e.into_inner(), 9),
        Ok(_) => panic!("expected capacity error"),
    }
    // ...a duplicate consumes no capacity even when full.
    assert_eq!(s.try_insert(7), Ok(false));
}

#[test]
fn unsorted_set_needs_only_eq_not_ord() {
    // A type that is Eq but deliberately not Ord still works in an unsorted set.
    #[derive(PartialEq, Eq)]
    struct NoOrd(u8);
    let mut s: UnsortedSet<Vec<NoOrd>> = UnsortedSet::new();
    assert!(s.insert(NoOrd(3)));
    assert!(!s.insert(NoOrd(3)));
    assert!(s.contains(&NoOrd(3)));
    assert_eq!(s.len(), 1);
}

#[test]
fn unsorted_map_insert_get_remove() {
    let mut m: UnsortedMap<Vec<(u64, &str)>> = UnsortedMap::new();
    assert_eq!(m.try_insert(2, "two"), Ok(None));
    assert_eq!(m.try_insert(1, "one"), Ok(None));
    assert_eq!(m.get(&1), Some(&"one"));
    // replacing an existing key returns the old value, consumes no capacity.
    assert_eq!(m.try_insert(2, "TWO"), Ok(Some("two")));
    assert_eq!(m.get(&2), Some(&"TWO"));
    assert_eq!(m.len(), 2);
    // swap-remove a key, others remain reachable.
    assert_eq!(m.remove(&1), Some("one"));
    assert_eq!(m.remove(&1), None);
    assert_eq!(m.get(&1), None);
    assert_eq!(m.get(&2), Some(&"TWO"));
    assert_eq!(m.len(), 1);
}

#[test]
fn unsorted_map_replace_at_cap() {
    let mut m: UnsortedMap<Capped<Vec<(u64, u64)>>> =
        UnsortedMap::from_store(Capped::with_capacity(2));
    m.try_insert(1, 10).unwrap();
    m.try_insert(2, 20).unwrap();
    assert!(m.try_insert(3, 30).is_err()); // new key at cap
    assert_eq!(m.try_insert(2, 22), Ok(Some(20))); // replace existing
}

#[test]
fn column_map_two_backends_insert_get_remove() {
    // The struct-of-arrays map over *different* backends per column: inline keys
    // (SmallVec), heap values (Vec). Both Unbounded, so `extend` is available.
    let mut m: UnsortedColumnMap<smallvec::SmallVec<[u64; 4]>, Vec<&str>> =
        UnsortedColumnMap::new();
    m.extend([(2, "two"), (1, "one")]);
    assert_eq!(m.get(&1), Some(&"one"));
    assert_eq!(m.try_insert(2, "TWO"), Ok(Some("two"))); // replace, no capacity
    assert_eq!(m.keys(), &[2, 1]);
    assert_eq!(m.values(), &["TWO", "one"]);
    // swap-remove keeps the columns aligned.
    assert_eq!(m.remove(&2), Some("TWO"));
    assert_eq!(m.get(&1), Some(&"one"));
    assert_eq!(m.get(&2), None);
    assert_eq!(m.capacity(), None); // both columns unbounded
}

#[test]
fn column_map_combined_cap_is_min() {
    // Cap only the value column at 2; the map is bounded at 2 (min of the columns).
    let mut m: UnsortedColumnMap<Vec<u64>, Capped<Vec<u64>>> =
        UnsortedColumnMap::from_store(Vec::new(), Capped::with_capacity(2));
    assert_eq!(m.capacity(), Some(2));
    m.try_insert(1, 10).unwrap();
    m.try_insert(2, 20).unwrap();
    assert_eq!(m.try_insert(3, 30).unwrap_err().into_inner(), (3, 30)); // new key at cap
    assert_eq!(m.try_insert(2, 22), Ok(Some(20))); // replace still succeeds
    assert_eq!(m.len(), 2);
}

#[test]
fn sorted_column_map_two_backends_keep_order() {
    // The sorted struct-of-arrays map over *different* backends per column: inline
    // keys (SmallVec), heap values (Vec). Both Unbounded, so `extend` is
    // available.
    let mut m: SortedColumnMap<smallvec::SmallVec<[u64; 4]>, Vec<&str>> = SortedColumnMap::new();
    m.extend([(3, "three"), (1, "one"), (2, "two")]);
    // Keys stay sorted across the column split; values track them by index.
    assert_eq!(m.keys(), &[1, 2, 3]);
    assert_eq!(m.values(), &["one", "two", "three"]);
    assert_eq!(m.get(&2), Some(&"two"));
    assert!(m.contains_key(&3) && !m.contains_key(&9));
    assert_eq!(m.try_insert(2, "TWO"), Ok(Some("two"))); // replace, no capacity
                                                         // Order-preserving remove (shift, not swap) keeps both columns aligned and
                                                         // sorted.
    assert_eq!(m.remove(&1), Some("one"));
    assert_eq!(m.keys(), &[2, 3]);
    assert_eq!(m.values(), &["TWO", "three"]);
    assert_eq!(m.capacity(), None); // both columns unbounded
}

#[test]
fn sorted_column_map_combined_cap_is_min() {
    // Cap only the value column at 2; the map is bounded at 2 (min of the columns).
    // A new key sorting into the middle is still rejected at the bound before
    // any shift.
    let mut m: SortedColumnMap<Vec<u64>, Capped<Vec<&str>>> =
        SortedColumnMap::from_store(Vec::new(), Capped::with_capacity(2));
    assert_eq!(m.capacity(), Some(2));
    m.try_insert(3, "c").unwrap();
    m.try_insert(1, "a").unwrap(); // sorts in front of 3 (shift), still within cap
    assert_eq!(m.keys(), &[1, 3]);
    assert_eq!(m.try_insert(2, "b").unwrap_err().into_inner(), (2, "b")); // new key at cap
    assert_eq!(m.try_insert(1, "A"), Ok(Some("a"))); // replace still succeeds
    assert_eq!(m.len(), 2);
}
