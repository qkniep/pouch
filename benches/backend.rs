//! Backend sweep — the *same* `SortedSet` operation across every backing store,
//! to show that big-O is backend-independent (every store is a contiguous
//! array) and only the constant factor moves: `Vec` (heap memmove) vs inline
//! `SmallVec` / `ArrayVec` / `heapless::Vec` (all native memmove shifts).
//!
//! Each backend is instantiated at a capacity matching the element count via
//! `consts = SIZES`, so inline backends never spill and fixed backends never
//! overflow. `try_insert` is used throughout (the one insert available on every
//! backend, including the fixed-capacity ones).
//!
//! Run: `cargo bench --bench backend`.

use divan::counter::ItemsCount;
use divan::{black_box, Bencher};
use pouch::SortedSet;

mod common;
use common::splitmix64;

fn main() {
    divan::main();
}

const SIZES: [usize; 3] = [16, 64, 256];

fn keys(n: usize) -> Vec<u64> {
    let mut state = 0u64;
    (0..n).map(|_| splitmix64(&mut state)).collect()
}

macro_rules! backend_bench {
    ($modname:ident, $store:ty) => {
        mod $modname {
            use super::*;

            #[divan::bench(consts = SIZES)]
            fn build<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut s = SortedSet::<$store>::new();
                    for &x in &k {
                        let _ = s.try_insert(x);
                    }
                    s
                });
            }

            #[divan::bench(consts = SIZES)]
            fn contains_hit<const N: usize>(bencher: Bencher) {
                let k = keys(N);
                let mut s = SortedSet::<$store>::new();
                for &x in &k {
                    let _ = s.try_insert(x);
                }
                bencher.counter(ItemsCount::new(N)).bench(|| {
                    let mut hits = 0u64;
                    for &x in &k {
                        hits += s.contains(black_box(&x)) as u64;
                    }
                    hits
                });
            }
        }
    };
}

// `TinyVec` is omitted: its `Array` trait only covers arrays up to length 32,
// so it can't take a generic `N` up to 256 (it matches `SmallVec` inline until
// spill).
backend_bench!(vec, Vec<u64>);
backend_bench!(smallvec, ::smallvec::SmallVec<[u64; N]>);
backend_bench!(arrayvec, ::arrayvec::ArrayVec<u64, N>);
backend_bench!(heapless, ::heapless::Vec<u64, N>);
