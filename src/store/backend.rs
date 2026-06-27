//! Backend implementations of the store traits — one file per backend.
//!
//! Each backend is gated by its feature *here*, on the `mod` line, so a disabled
//! backend's module never compiles and the impls inside need no per-item
//! `#[cfg]`. Nothing is re-exported: these modules exist only for their trait
//! impls on the concrete container types.
//!
//! Prefer a backend's native shifting `insert`/`remove` — every backend here has
//! one. `heapless.rs`'s module note also documents the `push`/`pop` +
//! `rotate_right(1)`/`rotate_left(1)` fallback for a hypothetical store that
//! exposes only `push`/`pop`; copy whichever fits when adding a backend.

// The read-only `&[T]` backend needs no dependency and no `alloc`, so it alone
// is ungated — available in every build, including `--no-default-features`.
mod slice;

#[cfg(feature = "alloc")]
mod vec;

#[cfg(feature = "smallvec")]
mod smallvec;

#[cfg(feature = "tinyvec")]
mod tinyvec;

#[cfg(feature = "arrayvec")]
mod arrayvec;

#[cfg(feature = "heapless")]
mod heapless;
