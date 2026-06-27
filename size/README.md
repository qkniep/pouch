# `size/` — binary-size probe

A standalone helper crate (not a workspace member, not published) that measures how
much `.text` pouch adds per collection on an embedded target. It backs the
[Binary size](../BENCHMARKS.md#binary-size-embedded) numbers and is for spotting
*regressions* — run it after touching the `Store` traits, a backend, or the
collection layer.

```sh
just size          # or: bash size/measure.sh
```

## What it does

`src/lib.rs` exposes one `#[no_mangle] extern "C"` root per collection op
(`K = V = u32`, `heapless::Vec` cap 64). `measure.sh` builds the staticlib once per
family for `thumbv7em-none-eabihf` with `opt-level = "z"` + fat LTO, so LTO keeps
exactly the code reachable from those roots. It then diffs the defined symbols of
each build against a bare baseline (panic handler only) with `llvm-nm`: a symbol the
family has and the baseline lacks is code that family pulled in — its roots plus the
monomorphizations they trigger. Shared `core` / `compiler_builtins` objects (which a
staticlib bundles wholesale) are in both builds and cancel out.

The entry-API roots are gated on a separate `<family>_entry` feature and reported as
a *delta*: `measure.sh` builds `<family>,<family>_entry` and subtracts the plain
`<family>` build, so the printed number is the marginal `.text` the `entry` surface
adds over that collection's insert/get/remove (the shared slot lookup cancels).

## Why it isn't a CI gate

Library code size is *probe-defined* (it depends on which types/backends/ops you
measure, not just on pouch) and *toolchain-volatile* (rustc/LLVM inlining and
`core`'s own codegen move it). A byte threshold would either be too loose to catch
anything or flap red on every toolchain bump. The regressions worth catching are
coarse step-changes (a whole helper like `ptr_rotate` getting pulled in), which are
obvious the moment you run this. Treat the numbers as ballpark.

Requirements: `rustup target add thumbv7em-none-eabihf` and
`rustup component add llvm-tools` (both included in `just setup`).
