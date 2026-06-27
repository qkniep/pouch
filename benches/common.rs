//! Shared benchmark fixtures — the deterministic PRNG and the set/map key sets,
//! kept in one place so every bench draws inputs from the same distribution.
//!
//! Included into each bench binary with `mod common;`. Each binary uses a
//! different subset (the backend/population benches roll their own input shapes
//! over [`splitmix64`] and never touch [`Keys`]), hence the blanket `dead_code`
//! allow.

#![allow(dead_code)]

/// SplitMix64 — a bijection, so iterating it from a fixed seed yields
/// all-distinct outputs. The deterministic source under every benchmark's
/// inputs.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Present / sorted / miss key sets for the set & map benches. The first `n`
/// [`splitmix64`] outputs are the present keys, the next `n` are
/// guaranteed-absent misses (the whole `2n`-prefix is distinct because
/// splitmix64 is a bijection).
pub(crate) struct Keys {
    pub(crate) random: Vec<u64>, // present keys, pseudo-random insertion order
    pub(crate) sorted: Vec<u64>, // the same present keys, ascending
    pub(crate) misses: Vec<u64>, // `n` keys guaranteed absent
}

pub(crate) fn keys(n: usize) -> Keys {
    let mut state = 0u64;
    let random: Vec<u64> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let misses: Vec<u64> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let mut sorted = random.clone();
    sorted.sort_unstable();
    Keys {
        random,
        sorted,
        misses,
    }
}
