//! Binary-size probe for `pouch` — see `size/measure.sh` (`just size`).
//!
//! Each `#[no_mangle] extern "C"` fn is a root LTO must keep, so the emitted code
//! is exactly the monomorphized op for `K = V = u32` at cap `N = 64`. measure.sh
//! builds one family at a time (`--features <name>`) and subtracts the no-feature
//! baseline (panic handler only) to get the marginal `.text` of that type.
#![no_std]
#![allow(improper_ctypes_definitions)] // we pass &mut Coll across `extern "C"` on purpose

#[allow(unused_imports)] // unused only in the no-feature baseline build
use core::hint::black_box;

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

const N: usize = 64;

#[cfg(feature = "sorted_set")]
mod sorted_set {
    use super::*;
    type C = pouch::SortedSet<heapless::Vec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn ss_insert(c: &mut C, v: u32) -> u8 {
        black_box(c.try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn ss_contains(c: &C, v: u32) -> u8 {
        black_box(c.contains(&black_box(v)) as u8)
    }
    #[no_mangle]
    pub extern "C" fn ss_remove(c: &mut C, v: u32) -> u8 {
        black_box(c.remove(&black_box(v)) as u8)
    }
}

#[cfg(feature = "unsorted_set")]
mod unsorted_set {
    use super::*;
    type C = pouch::UnsortedSet<heapless::Vec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn us_insert(c: &mut C, v: u32) -> u8 {
        black_box(c.try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn us_contains(c: &C, v: u32) -> u8 {
        black_box(c.contains(&black_box(v)) as u8)
    }
    #[no_mangle]
    pub extern "C" fn us_remove(c: &mut C, v: u32) -> u8 {
        black_box(c.remove(&black_box(v)) as u8)
    }
}

#[cfg(feature = "sorted_map")]
mod sorted_map {
    use super::*;
    type C = pouch::SortedMap<heapless::Vec<(u32, u32), N>>;
    #[no_mangle]
    pub extern "C" fn sm_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.try_insert(black_box(k), black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn sm_get(c: &C, k: u32) -> u32 {
        black_box(c.get(&black_box(k)).copied().unwrap_or(0))
    }
    #[no_mangle]
    pub extern "C" fn sm_remove(c: &mut C, k: u32) -> u32 {
        black_box(c.remove(&black_box(k)).unwrap_or(0))
    }
}

#[cfg(feature = "unsorted_map")]
mod unsorted_map {
    use super::*;
    type C = pouch::UnsortedMap<heapless::Vec<(u32, u32), N>>;
    #[no_mangle]
    pub extern "C" fn um_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.try_insert(black_box(k), black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn um_get(c: &C, k: u32) -> u32 {
        black_box(c.get(&black_box(k)).copied().unwrap_or(0))
    }
    #[no_mangle]
    pub extern "C" fn um_remove(c: &mut C, k: u32) -> u32 {
        black_box(c.remove(&black_box(k)).unwrap_or(0))
    }
}

#[cfg(feature = "column_map")]
mod column_map {
    use super::*;
    type C = pouch::UnsortedColumnMap<heapless::Vec<u32, N>, heapless::Vec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn cm_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.try_insert(black_box(k), black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn cm_get(c: &C, k: u32) -> u32 {
        black_box(c.get(&black_box(k)).copied().unwrap_or(0))
    }
    #[no_mangle]
    pub extern "C" fn cm_remove(c: &mut C, k: u32) -> u32 {
        black_box(c.remove(&black_box(k)).unwrap_or(0))
    }
}

#[cfg(feature = "sorted_column_map")]
mod sorted_column_map {
    use super::*;
    type C = pouch::SortedColumnMap<heapless::Vec<u32, N>, heapless::Vec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn scm_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.try_insert(black_box(k), black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn scm_get(c: &C, k: u32) -> u32 {
        black_box(c.get(&black_box(k)).copied().unwrap_or(0))
    }
    #[no_mangle]
    pub extern "C" fn scm_remove(c: &mut C, k: u32) -> u32 {
        black_box(c.remove(&black_box(k)).unwrap_or(0))
    }
}

// --- Entry API roots --------------------------------------------------------
// Measured *on top of* the basic family (build `--features <fam>,<fam>_entry`):
// the marginal `.text` is the entry surface a collection adds over its own
// insert/get/remove. Each exercises the realistic bounded-backend trio —
// `or_try_insert` (the `or_insert` half is `Unbounded`-gated, unreachable on
// heapless), an `and_modify` update, and removal through the entry.

#[cfg(feature = "sorted_map_entry")]
mod sorted_map_entry {
    use pouch::Entry;

    use super::*;
    type C = pouch::SortedMap<heapless::Vec<(u32, u32), N>>;
    #[no_mangle]
    pub extern "C" fn sm_e_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.entry(black_box(k)).or_try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn sm_e_modify(c: &mut C, k: u32, v: u32) -> u8 {
        let e = c
            .entry(black_box(k))
            .and_modify(|x| *x = x.wrapping_add(black_box(v)))
            .or_try_insert(black_box(v));
        black_box(e.is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn sm_e_remove(c: &mut C, k: u32) -> u32 {
        match c.entry(black_box(k)) {
            Entry::Occupied(e) => black_box(e.remove()),
            Entry::Vacant(_) => black_box(0),
        }
    }
}

#[cfg(feature = "unsorted_map_entry")]
mod unsorted_map_entry {
    use pouch::Entry;

    use super::*;
    type C = pouch::UnsortedMap<heapless::Vec<(u32, u32), N>>;
    #[no_mangle]
    pub extern "C" fn um_e_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.entry(black_box(k)).or_try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn um_e_modify(c: &mut C, k: u32, v: u32) -> u8 {
        let e = c
            .entry(black_box(k))
            .and_modify(|x| *x = x.wrapping_add(black_box(v)))
            .or_try_insert(black_box(v));
        black_box(e.is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn um_e_remove(c: &mut C, k: u32) -> u32 {
        match c.entry(black_box(k)) {
            Entry::Occupied(e) => black_box(e.remove()),
            Entry::Vacant(_) => black_box(0),
        }
    }
}

#[cfg(feature = "column_map_entry")]
mod column_map_entry {
    use pouch::ColumnEntry;

    use super::*;
    type C = pouch::UnsortedColumnMap<heapless::Vec<u32, N>, heapless::Vec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn cm_e_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.entry(black_box(k)).or_try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn cm_e_modify(c: &mut C, k: u32, v: u32) -> u8 {
        let e = c
            .entry(black_box(k))
            .and_modify(|x| *x = x.wrapping_add(black_box(v)))
            .or_try_insert(black_box(v));
        black_box(e.is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn cm_e_remove(c: &mut C, k: u32) -> u32 {
        match c.entry(black_box(k)) {
            ColumnEntry::Occupied(e) => black_box(e.remove()),
            ColumnEntry::Vacant(_) => black_box(0),
        }
    }
}

#[cfg(feature = "sorted_column_map_entry")]
mod sorted_column_map_entry {
    use pouch::ColumnEntry;

    use super::*;
    type C = pouch::SortedColumnMap<heapless::Vec<u32, N>, heapless::Vec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn scm_e_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.entry(black_box(k)).or_try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn scm_e_modify(c: &mut C, k: u32, v: u32) -> u8 {
        let e = c
            .entry(black_box(k))
            .and_modify(|x| *x = x.wrapping_add(black_box(v)))
            .or_try_insert(black_box(v));
        black_box(e.is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn scm_e_remove(c: &mut C, k: u32) -> u32 {
        match c.entry(black_box(k)) {
            ColumnEntry::Occupied(e) => black_box(e.remove()),
            ColumnEntry::Vacant(_) => black_box(0),
        }
    }
}

#[cfg(feature = "arrayvec_set")]
mod arrayvec_set {
    use super::*;
    type C = pouch::SortedSet<arrayvec::ArrayVec<u32, N>>;
    #[no_mangle]
    pub extern "C" fn avs_insert(c: &mut C, v: u32) -> u8 {
        black_box(c.try_insert(black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn avs_contains(c: &C, v: u32) -> u8 {
        black_box(c.contains(&black_box(v)) as u8)
    }
    #[no_mangle]
    pub extern "C" fn avs_remove(c: &mut C, v: u32) -> u8 {
        black_box(c.remove(&black_box(v)) as u8)
    }
}

#[cfg(feature = "cmp_handrolled")]
mod cmp_handrolled {
    use super::*;
    type C = heapless::Vec<u32, N>;
    #[no_mangle]
    pub extern "C" fn hr_insert(c: &mut C, v: u32) -> u8 {
        let v = black_box(v);
        match c.binary_search(&v) {
            Ok(_) => black_box(1),
            Err(i) => black_box(c.insert(i, v).is_ok() as u8),
        }
    }
    #[no_mangle]
    pub extern "C" fn hr_contains(c: &C, v: u32) -> u8 {
        black_box(c.binary_search(&black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn hr_remove(c: &mut C, v: u32) -> u8 {
        match c.binary_search(&black_box(v)) {
            Ok(i) => {
                c.remove(i);
                black_box(1)
            }
            Err(_) => black_box(0),
        }
    }
}

#[cfg(feature = "cmp_linear")]
mod cmp_linear {
    use super::*;
    type C = heapless::LinearMap<u32, u32, N>;
    #[no_mangle]
    pub extern "C" fn lin_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.insert(black_box(k), black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn lin_get(c: &C, k: u32) -> u32 {
        black_box(c.get(&black_box(k)).copied().unwrap_or(0))
    }
    #[no_mangle]
    pub extern "C" fn lin_remove(c: &mut C, k: u32) -> u32 {
        black_box(c.remove(&black_box(k)).unwrap_or(0))
    }
}

#[cfg(feature = "cmp_fnv")]
mod cmp_fnv {
    use super::*;
    type C = heapless::index_map::FnvIndexMap<u32, u32, N>;
    #[no_mangle]
    pub extern "C" fn fnv_insert(c: &mut C, k: u32, v: u32) -> u8 {
        black_box(c.insert(black_box(k), black_box(v)).is_ok() as u8)
    }
    #[no_mangle]
    pub extern "C" fn fnv_get(c: &C, k: u32) -> u32 {
        black_box(c.get(&black_box(k)).copied().unwrap_or(0))
    }
    #[no_mangle]
    pub extern "C" fn fnv_remove(c: &mut C, k: u32) -> u32 {
        black_box(c.remove(&black_box(k)).unwrap_or(0))
    }
}
