//! Property-based tests: differential (model-based) checks of every collection
//! flavor, with the standard library's `BTreeMap` / `BTreeSet` / `Vec` as
//! oracles — the same random operation sequence runs on both sides and every
//! observable must agree after every step.
//!
//! Three layers, mirroring the crate's axes:
//!
//!   * **store contract** — random `try_insert_at` / `remove_at` / `swap_remove_at` /
//!     `reserve` sequences on every backend, checked step-by-step against a plain `Vec`
//!     model. This is the reusable correctness argument for a backend: a new one earns
//!     its keep by adding one line to the `store_contract!` list.
//!   * **collection semantics** — random op sequences on each set/map flavor, checked
//!     against `BTreeSet`/`BTreeMap`, including the bound-sensitive invariants: an insert
//!     fails **iff** the element/key is new and the store is at capacity (duplicates and
//!     replacements consume none), the rejected element is handed back, and the column
//!     maps stay length-locked. One instantiation per *behavior class* ({unbounded,
//!     bounded, hybrid} × {sorted, unsorted}), NOT per backend — the store contract
//!     already proves the backends interchangeable, and the collection layer is generic
//!     over `Store`, so re-running it per backend re-tests code that cannot vary.
//!   * **derived surfaces** — set algebra vs `BTreeSet`'s, the bulk builders'
//!     duplicate/capacity policies, and serde round-trips.
//!
//! Keys draw from a tiny domain (`0..16`) so duplicates, replacements, and
//! full-store cases dominate. Like `smoke.rs`, this target carries
//! `required-features` in Cargo.toml: under a partial feature set cargo
//! silently skips it, and only an all-features run (`just test`, CI's test
//! job) executes it.

use std::collections::{BTreeMap, BTreeSet};

use arrayvec::ArrayVec;
use pouch::store::{StoreMut, StoreNew};
use pouch::*;
use proptest::prelude::*;
use smallvec::SmallVec;
use tinyvec::TinyVec;

/// The bound shared by every bounded instantiation below. Half the key domain,
/// so op sequences regularly run the stores full.
const CAP: usize = 8;

/// Deliberately tiny key domain: collisions, duplicates, and at-capacity
/// inserts must be the common case, not the rare one.
fn key() -> impl Strategy<Value = u8> {
    0u8..16
}

fn val() -> impl Strategy<Value = u16> {
    0u16..1000
}

// ---------------------------------------------------------------------------
// Store contract: every backend behaves like a (possibly bounded) Vec.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum StoreOp {
    /// Insert at `raw % (len + 1)` — always a valid index.
    InsertAt(usize, u8),
    /// Remove at `raw % len`; skipped on an empty store.
    RemoveAt(usize),
    SwapRemoveAt(usize),
    /// A growth hint: must never change contents, length, or the bound.
    Reserve(usize),
    Clear,
}

fn store_ops() -> impl Strategy<Value = Vec<StoreOp>> {
    prop::collection::vec(
        prop_oneof![
            5 => (any::<usize>(), any::<u8>()).prop_map(|(i, v)| StoreOp::InsertAt(i, v)),
            2 => any::<usize>().prop_map(StoreOp::RemoveAt),
            2 => any::<usize>().prop_map(StoreOp::SwapRemoveAt),
            1 => (0usize..16).prop_map(StoreOp::Reserve),
            1 => Just(StoreOp::Clear),
        ],
        0..48,
    )
}

fn check_store_contract<S: StoreMut<Elem = u8>>(mut store: S, ops: &[StoreOp]) {
    let cap = store.capacity();
    let mut model: Vec<u8> = Vec::new();
    for op in ops {
        match *op {
            StoreOp::InsertAt(raw, v) => {
                let i = raw % (model.len() + 1);
                let full = cap.is_some_and(|c| model.len() >= c);
                match store.try_insert_at(i, v) {
                    Ok(()) => {
                        assert!(!full, "insert succeeded on a full store");
                        model.insert(i, v);
                    }
                    Err(e) => {
                        assert!(full, "insert failed below capacity");
                        assert_eq!(e.into_inner(), v, "rejected element must be handed back");
                    }
                }
            }
            StoreOp::RemoveAt(raw) => {
                if !model.is_empty() {
                    let i = raw % model.len();
                    assert_eq!(store.remove_at(i), model.remove(i));
                }
            }
            StoreOp::SwapRemoveAt(raw) => {
                if !model.is_empty() {
                    let i = raw % model.len();
                    assert_eq!(store.swap_remove_at(i), model.swap_remove(i));
                }
            }
            StoreOp::Reserve(n) => store.reserve(n),
            StoreOp::Clear => {
                store.clear();
                model.clear();
            }
        }
        assert_eq!(store.as_slice(), model.as_slice());
        assert_eq!(store.len(), model.len());
        assert_eq!(store.is_empty(), model.is_empty());
        // The logical bound is a property of the store, not of its fill level.
        assert_eq!(store.capacity(), cap);
    }
}

macro_rules! store_contract {
    ($($name:ident: $ctor:expr;)*) => {$(
        proptest! {
            #[test]
            fn $name(ops in store_ops()) {
                check_store_contract($ctor, &ops);
            }
        }
    )*};
}

// The per-backend layer: every backend and adapter composition, one line each.
store_contract! {
    store_contract_vec: Vec::<u8>::new();
    store_contract_smallvec: SmallVec::<[u8; 4]>::new();
    store_contract_tinyvec: <TinyVec<[u8; 4]> as StoreNew>::new();
    store_contract_arrayvec: ArrayVec::<u8, CAP>::new();
    store_contract_heapless: <heapless::Vec<u8, CAP> as StoreNew>::new();
    store_contract_capped_vec: Capped::<Vec<u8>>::with_capacity(CAP);
    // The effective bound is min(cap, inner bound) — exercised from both sides.
    store_contract_capped_arrayvec_cap_wins: Capped::from_store(ArrayVec::<u8, CAP>::new(), 5);
    store_contract_capped_arrayvec_inner_bound_wins:
        Capped::from_store(ArrayVec::<u8, CAP>::new(), CAP + 4);
    store_contract_spill_arrayvec_to_vec:
        Spill::from_tiers(ArrayVec::<u8, 4>::new(), Vec::new());
}

proptest! {
    #[test]
    fn store_contract_scratchvec(ops in store_ops()) {
        let mut buf = [0u8; CAP];
        check_store_contract(ScratchVec::new(&mut buf), &ops);
    }

    // Bounded spill tier: the whole store's capacity is the spill tier's, and
    // the inline→spill migration must be invisible to the contract.
    #[test]
    fn store_contract_spill_arrayvec_to_scratchvec(ops in store_ops()) {
        let mut buf = [0u8; CAP];
        check_store_contract(
            Spill::from_tiers(ArrayVec::<u8, 4>::new(), ScratchVec::new(&mut buf)),
            &ops,
        );
    }
}

// ---------------------------------------------------------------------------
// Sets: differential op sequences against BTreeSet.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum SetOp {
    Insert(u8),
    Remove(u8),
    Contains(u8),
    /// Keep elements with `x % (d + 1) == r % (d + 1)`.
    Retain(u8, u8),
    Clear,
}

fn set_ops() -> impl Strategy<Value = Vec<SetOp>> {
    prop::collection::vec(
        prop_oneof![
            5 => key().prop_map(SetOp::Insert),
            2 => key().prop_map(SetOp::Remove),
            2 => key().prop_map(SetOp::Contains),
            1 => (0u8..4, any::<u8>()).prop_map(|(d, r)| SetOp::Retain(d, r)),
            1 => Just(SetOp::Clear),
        ],
        0..64,
    )
}

macro_rules! set_matches_btreeset {
    ($($name:ident: $ctor:expr, $norm:expr;)*) => {$(
        proptest! {
            #[test]
            fn $name(ops in set_ops()) {
                let mut set = $ctor;
                let cap = set.capacity();
                let mut oracle: BTreeSet<u8> = BTreeSet::new();
                let normalize = $norm;
                for op in ops {
                    let full = cap.is_some_and(|c| oracle.len() >= c);
                    match op {
                        SetOp::Insert(x) => {
                            let dup = oracle.contains(&x);
                            match set.try_insert(x) {
                                Ok(newly) => {
                                    assert_eq!(newly, !dup);
                                    // A duplicate consumes no capacity, so only a
                                    // NEW element needs headroom.
                                    assert!(dup || !full, "insert succeeded on a full store");
                                    oracle.insert(x);
                                }
                                Err(e) => {
                                    assert_eq!(e.into_inner(), x);
                                    assert!(
                                        !dup && full,
                                        "only a new element at the bound may be rejected"
                                    );
                                }
                            }
                        }
                        SetOp::Remove(x) => assert_eq!(set.remove(&x), oracle.remove(&x)),
                        SetOp::Contains(x) => assert_eq!(set.contains(&x), oracle.contains(&x)),
                        SetOp::Retain(d, r) => {
                            let modulus = d + 1;
                            set.retain(|x| *x % modulus == r % modulus);
                            oracle.retain(|x| *x % modulus == r % modulus);
                        }
                        SetOp::Clear => {
                            set.clear();
                            oracle.clear();
                        }
                    }
                    assert_eq!(set.len(), oracle.len());
                    assert_eq!(set.is_empty(), oracle.is_empty());
                    let mut elems: Vec<u8> = set.iter().copied().collect();
                    normalize(&mut elems);
                    let expected: Vec<u8> = oracle.iter().copied().collect();
                    assert_eq!(elems, expected);
                }
            }
        }
    )*};
}

/// Sorted flavors: the stored order itself is checked against the oracle.
fn keep_order(_: &mut [u8]) {}
/// Unsorted flavors: stored order is incidental; compare sorted.
fn sort_first(elems: &mut [u8]) {
    elems.sort_unstable();
}

// One representative per behavior class (see module docs) — don't grow this
// list when adding a backend; add a `store_contract!` line instead. The
// smallvec instance is the suite's one collection-driven crossing of a
// hybrid's spill boundary; the unsorted flavors don't need their own.
set_matches_btreeset! {
    sorted_set_vec_matches_btreeset: SortedSet::<Vec<u8>>::new(), keep_order;
    sorted_set_smallvec_matches_btreeset: SortedSet::<SmallVec<[u8; 4]>>::new(), keep_order;
    sorted_set_arrayvec_matches_btreeset: SortedSet::<ArrayVec<u8, CAP>>::new(), keep_order;
    unsorted_set_vec_matches_btreeset: UnsortedSet::<Vec<u8>>::new(), sort_first;
    unsorted_set_arrayvec_matches_btreeset: UnsortedSet::<ArrayVec<u8, CAP>>::new(), sort_first;
}

// ---------------------------------------------------------------------------
// Maps: differential op sequences against BTreeMap.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum MapOp {
    Insert(u8, u16),
    Remove(u8),
    Get(u8),
    /// `entry(k).and_modify(+1).or_try_insert(v)` — the one-lookup upsert.
    Entry(u8, u16),
    /// Keep entries with `k % (d + 1) == r % (d + 1)`.
    Retain(u8, u8),
    Clear,
}

fn map_ops() -> impl Strategy<Value = Vec<MapOp>> {
    prop::collection::vec(
        prop_oneof![
            5 => (key(), val()).prop_map(|(k, v)| MapOp::Insert(k, v)),
            2 => key().prop_map(MapOp::Remove),
            2 => key().prop_map(MapOp::Get),
            3 => (key(), val()).prop_map(|(k, v)| MapOp::Entry(k, v)),
            1 => (0u8..4, any::<u8>()).prop_map(|(d, r)| MapOp::Retain(d, r)),
            1 => Just(MapOp::Clear),
        ],
        0..64,
    )
}

/// Sorted flavors keep entries in key order; check it verbatim.
fn keep_entry_order(_: &mut [(u8, u16)]) {}
/// Unsorted flavors: order is incidental; compare sorted.
fn sort_entries_first(entries: &mut [(u8, u16)]) {
    entries.sort_unstable();
}

// The optional trailing closure is an extra per-step invariant check (the
// column maps assert their two stores stay length-locked — a half-inserted
// entry would break the zip).
macro_rules! map_matches_btreemap {
    ($($name:ident: $ctor:expr, $norm:expr $(, $extra:expr)?;)*) => {$(
        proptest! {
            #[test]
            fn $name(ops in map_ops()) {
                let mut map = $ctor;
                let cap = map.capacity();
                let mut oracle: BTreeMap<u8, u16> = BTreeMap::new();
                let normalize = $norm;
                for op in ops {
                    let full = cap.is_some_and(|c| oracle.len() >= c);
                    match op {
                        MapOp::Insert(k, v) => {
                            let prev = oracle.get(&k).copied();
                            match map.try_insert(k, v) {
                                Ok(old) => {
                                    assert_eq!(old, prev);
                                    // A replacement consumes no capacity; only a
                                    // NEW key needs headroom.
                                    assert!(
                                        prev.is_some() || !full,
                                        "insert succeeded on a full store"
                                    );
                                    oracle.insert(k, v);
                                }
                                Err(e) => {
                                    assert_eq!(e.into_inner(), (k, v));
                                    assert!(
                                        prev.is_none() && full,
                                        "only a new key at the bound may be rejected"
                                    );
                                }
                            }
                        }
                        MapOp::Remove(k) => assert_eq!(map.remove(&k), oracle.remove(&k)),
                        MapOp::Get(k) => {
                            assert_eq!(map.get(&k), oracle.get(&k));
                            assert_eq!(map.contains_key(&k), oracle.contains_key(&k));
                        }
                        MapOp::Entry(k, v) => {
                            let prev = oracle.get(&k).copied();
                            let res = map
                                .entry(k)
                                .and_modify(|slot| *slot = slot.wrapping_add(1))
                                .or_try_insert(v);
                            match res {
                                Ok(slot) => match prev {
                                    Some(old) => {
                                        assert_eq!(*slot, old.wrapping_add(1));
                                        oracle.insert(k, old.wrapping_add(1));
                                    }
                                    None => {
                                        assert!(!full, "vacant insert succeeded on a full store");
                                        assert_eq!(*slot, v);
                                        oracle.insert(k, v);
                                    }
                                },
                                Err(e) => {
                                    assert_eq!(e.into_inner(), (k, v));
                                    assert!(
                                        prev.is_none() && full,
                                        "only a vacant entry at the bound may be rejected"
                                    );
                                }
                            }
                        }
                        MapOp::Retain(d, r) => {
                            let modulus = d + 1;
                            map.retain(|k, _| *k % modulus == r % modulus);
                            oracle.retain(|k, _| *k % modulus == r % modulus);
                        }
                        MapOp::Clear => {
                            map.clear();
                            oracle.clear();
                        }
                    }
                    assert_eq!(map.len(), oracle.len());
                    assert_eq!(map.is_empty(), oracle.is_empty());
                    let mut entries: Vec<(u8, u16)> = map.iter().map(|(k, v)| (*k, *v)).collect();
                    normalize(&mut entries);
                    let expected: Vec<(u8, u16)> = oracle.iter().map(|(k, v)| (*k, *v)).collect();
                    assert_eq!(entries, expected);
                    $(
                        let extra = $extra;
                        extra(&map);
                    )?
                }
            }
        }
    )*};
}

// Same representative policy as the sets. The column maps are not backend
// redundancy — two-store logic and combined caps are collection code of their
// own — so each column flavor appears in an unbounded and a bounded/mixed
// configuration.
map_matches_btreemap! {
    sorted_map_vec_matches_btreemap: SortedMap::<Vec<(u8, u16)>>::new(), keep_entry_order;
    sorted_map_arrayvec_matches_btreemap:
        SortedMap::<ArrayVec<(u8, u16), CAP>>::new(), keep_entry_order;
    unsorted_map_vec_matches_btreemap: UnsortedMap::<Vec<(u8, u16)>>::new(), sort_entries_first;
    unsorted_map_arrayvec_matches_btreemap:
        UnsortedMap::<ArrayVec<(u8, u16), CAP>>::new(), sort_entries_first;
    unsorted_column_map_vec_matches_btreemap:
        UnsortedColumnMap::<Vec<u8>, Vec<u16>>::new(), sort_entries_first,
        |m: &UnsortedColumnMap<Vec<u8>, Vec<u16>>| assert_eq!(m.keys().len(), m.values().len());
    // Mixed columns: unbounded inline keys, runtime-capped heap values — the
    // combined cap is the min, and an at-cap insert must not half-insert.
    unsorted_column_map_capped_values_matches_btreemap:
        UnsortedColumnMap::from_store(SmallVec::<[u8; 4]>::new(), Capped::with_capacity(CAP)),
        sort_entries_first,
        |m: &UnsortedColumnMap<SmallVec<[u8; 4]>, Capped<Vec<u16>>>| {
            assert_eq!(m.keys().len(), m.values().len());
        };
    sorted_column_map_vec_matches_btreemap:
        SortedColumnMap::<Vec<u8>, Vec<u16>>::new(), keep_entry_order,
        |m: &SortedColumnMap<Vec<u8>, Vec<u16>>| assert_eq!(m.keys().len(), m.values().len());
    sorted_column_map_arrayvec_matches_btreemap:
        SortedColumnMap::<ArrayVec<u8, CAP>, ArrayVec<u16, CAP>>::new(), keep_entry_order,
        |m: &SortedColumnMap<ArrayVec<u8, CAP>, ArrayVec<u16, CAP>>| {
            assert_eq!(m.keys().len(), m.values().len());
        };
}

// ---------------------------------------------------------------------------
// Set algebra: the merge iterators and predicates against BTreeSet's.
// ---------------------------------------------------------------------------

fn algebra_case(a: &[u8], b: &[u8]) {
    // Cross-store on purpose: the two sides use different backends.
    let sa = SortedSet::<Vec<u8>>::try_from_iter(a.iter().copied()).expect("unbounded");
    let sb = SortedSet::<SmallVec<[u8; 4]>>::try_from_iter(b.iter().copied()).expect("unbounded");
    let oa: BTreeSet<u8> = a.iter().copied().collect();
    let ob: BTreeSet<u8> = b.iter().copied().collect();

    let ours: Vec<u8> = sa.union(&sb).copied().collect();
    let stds: Vec<u8> = oa.union(&ob).copied().collect();
    assert_eq!(ours, stds, "union");
    let ours: Vec<u8> = sa.intersection(&sb).copied().collect();
    let stds: Vec<u8> = oa.intersection(&ob).copied().collect();
    assert_eq!(ours, stds, "intersection");
    let ours: Vec<u8> = sa.difference(&sb).copied().collect();
    let stds: Vec<u8> = oa.difference(&ob).copied().collect();
    assert_eq!(ours, stds, "difference");
    let ours: Vec<u8> = sa.symmetric_difference(&sb).copied().collect();
    let stds: Vec<u8> = oa.symmetric_difference(&ob).copied().collect();
    assert_eq!(ours, stds, "symmetric_difference");

    assert_eq!(sa.is_subset(&sb), oa.is_subset(&ob));
    assert_eq!(sb.is_subset(&sa), ob.is_subset(&oa));
    assert_eq!(sa.is_superset(&sb), oa.is_superset(&ob));
    assert_eq!(sb.is_superset(&sa), ob.is_superset(&oa));
    assert_eq!(sa.is_disjoint(&sb), oa.is_disjoint(&ob));
    assert!(sa.is_subset(&sa), "every set is a subset of itself");
    assert_eq!(sa.first(), oa.first());
    assert_eq!(sa.last(), oa.last());

    // The unsorted predicates are O(n·m) scans over the same contract.
    let ua = UnsortedSet::<Vec<u8>>::try_from_iter(a.iter().copied()).expect("unbounded");
    let ub = UnsortedSet::<SmallVec<[u8; 4]>>::try_from_iter(b.iter().copied()).expect("unbounded");
    assert_eq!(ua.is_subset(&ub), oa.is_subset(&ob));
    assert_eq!(ua.is_superset(&ub), oa.is_superset(&ob));
    assert_eq!(ua.is_disjoint(&ub), oa.is_disjoint(&ob));
}

proptest! {
    #[test]
    fn set_algebra_matches_btreeset(
        a in prop::collection::vec(key(), 0..48),
        b in prop::collection::vec(key(), 0..48),
    ) {
        algebra_case(&a, &b);
    }

    // The predicates switch to binary probing when one side is ≥16× smaller;
    // lopsided sizes exercise both sides of that threshold. The small side
    // samples the big one so overlap (and genuine subsets) are common.
    #[test]
    fn set_algebra_lopsided_sizes(
        a in prop::collection::vec(any::<u8>(), 64..256),
        picks in prop::collection::vec(any::<usize>(), 0..6),
        extra in prop::collection::vec(any::<u8>(), 0..3),
    ) {
        let b: Vec<u8> = picks.iter().map(|&i| a[i % a.len()]).chain(extra).collect();
        algebra_case(&a, &b);
    }
}

// ---------------------------------------------------------------------------
// Bulk builders: the documented duplicate and capacity policies.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn set_bulk_build_matches_oracle(raw in prop::collection::vec(key(), 0..40)) {
        let expected: Vec<u8> = raw.iter().copied().collect::<BTreeSet<u8>>().into_iter().collect();

        let sorted = SortedSet::<Vec<u8>>::try_from_iter(raw.iter().copied()).expect("unbounded");
        assert_eq!(sorted.as_slice(), expected.as_slice());

        let unsorted = UnsortedSet::<Vec<u8>>::try_from_iter(raw.iter().copied()).expect("unbounded");
        let mut elems: Vec<u8> = unsorted.iter().copied().collect();
        elems.sort_unstable();
        assert_eq!(elems, expected);
    }

    // Documented caveat: the bulk builders append every item BEFORE the dedup
    // pass, so the RAW count — not the deduplicated one — is what a bounded
    // store's cap polices.
    #[test]
    fn bounded_bulk_build_overflows_on_raw_count(raw in prop::collection::vec(key(), 0..=12)) {
        match SortedSet::<ArrayVec<u8, CAP>>::try_from_iter(raw.iter().copied()) {
            Ok(s) => {
                assert!(raw.len() <= CAP);
                let expected: Vec<u8> =
                    raw.iter().copied().collect::<BTreeSet<u8>>().into_iter().collect();
                assert_eq!(s.as_slice(), expected.as_slice());
            }
            Err(e) => {
                assert!(raw.len() > CAP);
                assert_eq!(e.into_inner(), raw[CAP], "first element that did not fit");
            }
        }
    }

    #[test]
    fn set_try_from_sorted_iter_enforces_ascending(raw in prop::collection::vec(key(), 0..24)) {
        // The builder's promise: accept ascending input (equal neighbours dedup
        // silently), reject the first item that undercuts its predecessor.
        let mut max_so_far: Option<u8> = None;
        let mut first_violation = None;
        for &x in &raw {
            match max_so_far {
                Some(m) if x < m => {
                    first_violation = Some(x);
                    break;
                }
                Some(m) if x == m => {}
                _ => max_so_far = Some(x),
            }
        }
        match SortedSet::<Vec<u8>>::try_from_sorted_iter(raw.iter().copied()) {
            Ok(s) => {
                assert_eq!(first_violation, None);
                let mut expected = raw.clone();
                expected.dedup();
                assert_eq!(s.as_slice(), expected.as_slice());
            }
            Err(BuildError::Unsorted(x)) => assert_eq!(Some(x), first_violation),
            Err(e) => panic!("unexpected error: {e:?}"),
        }

        // A pre-sorted copy always builds, and the O(n) path must agree with
        // the O(n log n) one.
        let mut ascending = raw.clone();
        ascending.sort_unstable();
        let fast = SortedSet::<Vec<u8>>::from_sorted_iter(ascending);
        let slow = SortedSet::<Vec<u8>>::try_from_iter(raw.iter().copied()).expect("unbounded");
        assert_eq!(fast, slow);
    }

    // Sets dedup, maps reject: a duplicate key is ambiguous input for a bulk
    // map build, while the sequential ops stay last-wins.
    #[test]
    fn map_bulk_build_rejects_duplicate_keys(raw in prop::collection::vec((key(), val()), 0..24)) {
        let mut keys: Vec<u8> = raw.iter().map(|(k, _)| *k).collect();
        keys.sort_unstable();
        let has_dup = keys.windows(2).any(|w| w[0] == w[1]);

        match SortedMap::<Vec<(u8, u16)>>::try_from_iter(raw.iter().copied()) {
            Ok(m) => {
                assert!(!has_dup);
                let oracle: BTreeMap<u8, u16> = raw.iter().copied().collect();
                let entries: Vec<(u8, u16)> = m.iter().map(|(k, v)| (*k, *v)).collect();
                assert_eq!(entries, oracle.into_iter().collect::<Vec<_>>());
            }
            Err(BuildError::DuplicateKey((k, _))) => {
                assert!(has_dup);
                assert!(raw.iter().filter(|(k2, _)| *k2 == k).count() >= 2);
            }
            Err(e) => panic!("unexpected error: {e:?}"),
        }

        // Last-wins sequential extend accepts the same input; BTreeMap's insert
        // is last-wins too, so it is the exact oracle.
        let mut m = UnsortedMap::<Vec<(u8, u16)>>::new();
        m.try_extend(raw.iter().copied()).expect("unbounded");
        let oracle: BTreeMap<u8, u16> = raw.iter().copied().collect();
        assert_eq!(m.len(), oracle.len());
        for (k, v) in &oracle {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn map_try_from_sorted_iter_polices_order_and_dups(
        raw in prop::collection::vec((key(), val()), 0..24),
    ) {
        #[derive(Debug, PartialEq)]
        enum Expect {
            Ok,
            Unsorted(u8),
            Dup(u8),
        }
        // Unlike the set builder, every accepted entry is appended (no dedup
        // skip), so the reference predecessor is simply the previous key.
        let mut prev: Option<u8> = None;
        let mut expect = Expect::Ok;
        for &(k, _) in &raw {
            if let Some(p) = prev {
                if k < p {
                    expect = Expect::Unsorted(k);
                    break;
                }
                if k == p {
                    expect = Expect::Dup(k);
                    break;
                }
            }
            prev = Some(k);
        }
        match (SortedMap::<Vec<(u8, u16)>>::try_from_sorted_iter(raw.iter().copied()), expect) {
            (Ok(m), Expect::Ok) => assert_eq!(m.len(), raw.len()),
            (Err(BuildError::Unsorted((k, _))), Expect::Unsorted(k2)) => assert_eq!(k, k2),
            (Err(BuildError::DuplicateKey((k, _))), Expect::Dup(k2)) => assert_eq!(k, k2),
            (res, exp) => panic!("result {res:?} does not match expectation {exp:?}"),
        }
    }

    // The sorted builder detects a duplicate BEFORE the append, so at an exactly
    // full bounded store it reports DuplicateKey — not a misleading Capacity.
    #[test]
    fn sorted_map_builder_detects_dup_before_append(keys in prop::collection::btree_set(key(), 4)) {
        let keys: Vec<u8> = keys.into_iter().collect();
        let last = *keys.last().expect("exactly 4 keys");
        let input = keys
            .iter()
            .map(|&k| (k, u16::from(k)))
            .chain([(last, 999u16)]);
        match SortedMap::<ArrayVec<(u8, u16), 4>>::try_from_sorted_iter(input) {
            Err(BuildError::DuplicateKey(entry)) => assert_eq!(entry, (last, 999)),
            other => panic!("expected DuplicateKey, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Serde: round-trips and the deserializer's builder policies over a real wire
// format (JSON), complementing the exact token tests in `serde_impls.rs`.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn serde_round_trips_preserve_contents(
        raw in prop::collection::vec(key(), 0..24),
        raw_entries in prop::collection::vec((key(), val()), 0..24),
    ) {
        let set = SortedSet::<Vec<u8>>::try_from_iter(raw.iter().copied()).expect("unbounded");
        let json = serde_json::to_string(&set).expect("serialize");
        let back: SortedSet<Vec<u8>> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, set);

        // Maps round-trip through last-wins extend so duplicate keys in the
        // random input don't abort construction.
        let mut map = SortedMap::<Vec<(u8, u16)>>::new();
        map.try_extend(raw_entries.iter().copied()).expect("unbounded");
        let json = serde_json::to_string(&map).expect("serialize");
        let back: SortedMap<Vec<(u8, u16)>> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, map);

        let mut cmap = SortedColumnMap::<Vec<u8>, Vec<u16>>::new();
        cmap.try_extend(raw_entries.iter().copied()).expect("unbounded");
        let json = serde_json::to_string(&cmap).expect("serialize");
        let back: SortedColumnMap<Vec<u8>, Vec<u16>> =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cmap);

        // Bags keep duplicates and order.
        let bag = Bag::<Vec<u8>>::try_from_iter(raw.iter().copied()).expect("unbounded");
        let json = serde_json::to_string(&bag).expect("serialize");
        let back: Bag<Vec<u8>> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, bag);

        // Unsorted flavors have no PartialEq (stored order is incidental):
        // compare contents.
        let uset = UnsortedSet::<Vec<u8>>::try_from_iter(raw.iter().copied()).expect("unbounded");
        let json = serde_json::to_string(&uset).expect("serialize");
        let back: UnsortedSet<Vec<u8>> = serde_json::from_str(&json).expect("deserialize");
        let mut ours: Vec<u8> = uset.iter().copied().collect();
        let mut theirs: Vec<u8> = back.iter().copied().collect();
        ours.sort_unstable();
        theirs.sort_unstable();
        assert_eq!(ours, theirs);
    }

    // A duplicate key on the wire is a data error for pouch maps — where std's
    // map impls silently keep the last value.
    #[test]
    fn serde_map_rejects_duplicate_wire_keys(
        entries in prop::collection::vec((key(), val()), 1..12),
        dup_at in any::<usize>(),
    ) {
        let dup = entries[dup_at % entries.len()];
        let body: Vec<String> = entries
            .iter()
            .chain([&dup])
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        let json = format!("{{{}}}", body.join(","));

        assert!(serde_json::from_str::<SortedMap<Vec<(u8, u16)>>>(&json).is_err());
        assert!(serde_json::from_str::<UnsortedMap<Vec<(u8, u16)>>>(&json).is_err());
        assert!(serde_json::from_str::<SortedColumnMap<Vec<u8>, Vec<u16>>>(&json).is_err());
        // The contrast the docs draw: std accepts the same wire bytes silently.
        assert!(serde_json::from_str::<BTreeMap<u8, u16>>(&json).is_ok());
    }

    // Bounded deserialization is input validation, and it polices the RAW
    // element count (the deserializer routes through `try_from_iter`, which
    // appends before deduping — the documented caveat).
    #[test]
    fn serde_bounded_deserialize_validates_raw_count(raw in prop::collection::vec(key(), 0..8)) {
        let json = serde_json::to_string(&raw).expect("serialize");
        match serde_json::from_str::<SortedSet<ArrayVec<u8, 4>>>(&json) {
            Ok(s) => {
                assert!(raw.len() <= 4);
                let expected: Vec<u8> =
                    raw.iter().copied().collect::<BTreeSet<u8>>().into_iter().collect();
                assert_eq!(s.as_slice(), expected.as_slice());
            }
            Err(_) => assert!(raw.len() > 4),
        }
    }
}
