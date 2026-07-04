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

#[test]
fn borrowed_key_lookups_across_the_surface() {
    // The on-mission `Borrow` payoff, end to end on the blessed aliases:
    // `String` keys, `&str` queries — no allocation to ask.
    let mut m: Map<String, u32> = Map::default();
    m.insert("alpha".to_string(), 1);
    m.insert("beta".to_string(), 2);
    assert_eq!(m.get("alpha"), Some(&1));
    assert!(m.contains_key("beta") && !m.contains_key("gamma"));
    *m.get_mut("alpha").unwrap() += 10;
    assert_eq!(m.remove("beta"), Some(2));

    let mut s: Set<String> = Set::default();
    s.insert("gamma".to_string());
    assert!(s.contains("gamma"));
    assert!(s.remove("gamma"));

    // The column maps route the same borrowed keys through their dense scans.
    let mut cm: UnsortedColumnMap<Vec<String>, Vec<u32>> = UnsortedColumnMap::new();
    cm.try_insert("k".to_string(), 7).unwrap();
    assert_eq!(cm.get("k"), Some(&7));
    assert!(cm.contains_key("k"));
    assert_eq!(cm.remove("k"), Some(7));

    let mut scm: SortedColumnMap<Vec<String>, Vec<u32>> = SortedColumnMap::new();
    scm.try_insert("k".to_string(), 7).unwrap();
    assert_eq!(scm.get("k"), Some(&7));
    *scm.get_mut("k").unwrap() += 1;
    assert_eq!(scm.remove("k"), Some(8));
}

#[test]
fn store_access_and_reserve_across_the_surface() {
    // `store()` reaches backend introspection the collection API doesn't
    // abstract: SmallVec's `spilled()` through the blessed alias...
    let mut s: Set<u32, 2> = Set::new();
    s.insert(1);
    s.insert(2);
    assert!(!s.store().spilled());
    s.insert(3);
    assert!(s.store().spilled());

    // ...and `Spill::is_spilled` through a SortedSet.
    let mut sp: SortedSet<Spill<arrayvec::ArrayVec<u32, 2>, Vec<u32>>> = SortedSet::new();
    sp.insert(1);
    assert!(!sp.store().is_spilled());
    sp.insert(2);
    sp.insert(3);
    assert!(sp.store().is_spilled());

    // `reserve` pays growth up front; observable via the store's inherent
    // (allocated) capacity. `into_store` recovers the buffer, still sorted.
    let mut m: Map<u32, u32> = Map::default();
    m.reserve(100);
    let vec_cap = m.store().capacity();
    assert!(vec_cap >= 100);
    for x in 0..50 {
        m.insert(x, x);
    }
    assert_eq!(m.store().capacity(), vec_cap); // no growth during the burst

    // Column maps: two length-locked stores, borrowed and recovered as a pair.
    let mut cm: UnsortedColumnMap<Vec<u32>, Vec<u32>> = UnsortedColumnMap::new();
    cm.reserve(10);
    cm.try_insert(1, 10).unwrap();
    let (ks, vs) = cm.stores();
    assert!(ks.capacity() >= 10 && vs.capacity() >= 10);
    let (ks, vs) = cm.into_stores();
    assert_eq!((ks.as_slice(), vs.as_slice()), (&[1][..], &[10][..]));
}

#[test]
fn sorted_collections_are_good_citizens_in_other_structures() {
    use std::collections::hash_map::DefaultHasher;
    use std::collections::{BTreeSet, HashMap};
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(t: &T) -> u64 {
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    // Build order doesn't matter: the stored order is canonical, so equal sets
    // hash equal — the property that makes them valid HashMap keys.
    let a: Set<u32> = [3, 1, 2].into_iter().collect();
    let b: Set<u32> = [2, 3, 1, 1].into_iter().collect();
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));

    let mut by_set: HashMap<Set<u32>, &str> = HashMap::new();
    by_set.insert(a, "quorum-a");
    assert_eq!(by_set.get(&b), Some(&"quorum-a")); // looked up via the twin

    // Maps too, including as BTreeSet members (needs Ord).
    let m1: Map<u32, &str> = Map::try_from_iter([(1, "a"), (2, "b")]).unwrap();
    let m2: Map<u32, &str> = Map::try_from_iter([(2, "b"), (1, "a")]).unwrap();
    assert_eq!(hash_of(&m1), hash_of(&m2));
    let nested: BTreeSet<Map<u32, &str>> = [m1, m2].into_iter().collect();
    assert_eq!(nested.len(), 1); // equal maps collapse

    // SortedColumnMap orders column-wise — a valid total order for nesting,
    // though not the entry-interleaved order of the AoS SortedMap.
    let c1: SortedColumnMap<Vec<u32>, Vec<u32>> =
        SortedColumnMap::try_from_iter([(1, 10)]).unwrap();
    let c2: SortedColumnMap<Vec<u32>, Vec<u32>> =
        SortedColumnMap::try_from_iter([(2, 20)]).unwrap();
    assert!(c1 < c2);
    assert_eq!(hash_of(&c1), hash_of(&c1.clone()));
}

#[test]
fn spill_compares_and_hashes_by_contents() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(t: &T) -> u64 {
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    // One set spills and shrinks back; the other never spills. Spill's
    // slice-based impls make the SortedSet derives tier-blind.
    let mut a: SortedSet<Spill<arrayvec::ArrayVec<u32, 2>, Vec<u32>>> = SortedSet::new();
    for x in [1, 2, 3] {
        a.insert(x);
    }
    a.remove(&3); // back under the inline bound, but stays spilled
    assert!(a.store().is_spilled());

    let mut b: SortedSet<Spill<arrayvec::ArrayVec<u32, 2>, Vec<u32>>> = SortedSet::new();
    b.insert(1);
    b.insert(2);
    assert!(!b.store().is_spilled());

    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}
