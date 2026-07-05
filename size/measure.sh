#!/usr/bin/env bash
# Marginal binary size (.text) pouch adds per collection, for an embedded target.
#
# Method: build this staticlib once per family for `thumbv7em-none-eabihf` with
# `opt-level="z"` + fat LTO, then diff its defined symbols against a bare baseline
# (panic handler only). A symbol present in a family build but not the baseline is
# code that family pulled in — its `extern "C"` roots plus the monomorphizations
# they trigger; shared `core` / `compiler_builtins` objects the staticlib bundles
# are in both, so they cancel. Numbers are toolchain/target/opt-level dependent —
# ballpark, not a contract: CI (no-std.yml) runs this script so the probe can't
# bit-rot, but deliberately never gates on the byte counts themselves.
set -euo pipefail

cd "$(dirname "$0")"

TARGET="thumbv7em-none-eabihf"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
NM="$(rustc --print sysroot)/lib/rustlib/$HOST/bin/llvm-nm"
OUT="nm"

# Preflight: the tool needs the embedded target and the llvm-tools component.
if ! rustc --print target-list | grep -qx "$TARGET"; then
    echo "error: rustc has no target '$TARGET'." >&2; exit 1
fi
if ! rustup target list --installed 2>/dev/null | grep -qx "$TARGET"; then
    echo "error: target '$TARGET' not installed. Run: rustup target add $TARGET" >&2; exit 1
fi
if [ ! -x "$NM" ]; then
    echo "error: llvm-nm not found at $NM. Run: rustup component add llvm-tools" >&2; exit 1
fi

mkdir -p "$OUT"

# name:features  ("" = baseline). Order: baseline, pouch families, the "all" roll-up,
# then the alternatives shown for context.
VARIANTS=(
    "baseline:"
    "sorted_set:sorted_set"
    "unsorted_set:unsorted_set"
    "sorted_map:sorted_map"
    "unsorted_map:unsorted_map"
    "sorted_column_map:sorted_column_map"
    "column_map:column_map"
    "all:all"
    # basic+entry per map — diffed against the basic build for the entry delta
    "sorted_map_both:sorted_map,sorted_map_entry"
    "unsorted_map_both:unsorted_map,unsorted_map_entry"
    "column_map_both:column_map,column_map_entry"
    "sorted_column_map_both:sorted_column_map,sorted_column_map_entry"
    "arrayvec_set:arrayvec_set"
    "cmp_handrolled:cmp_handrolled"
    "cmp_linear:cmp_linear"
    "cmp_fnv:cmp_fnv"
)

echo "building ${#VARIANTS[@]} variants for $TARGET (opt-level=z, fat LTO)…" >&2
for v in "${VARIANTS[@]}"; do
    name="${v%%:*}"; feats="${v#*:}"
    args=(--release --target "$TARGET" --locked)
    [ -n "$feats" ] && args+=(--no-default-features --features "$feats")
    if ! out="$(cargo build "${args[@]}" 2>&1)"; then
        printf '%s\n' "$out" >&2
        echo "error: build failed for variant '$name'" >&2
        exit 1
    fi
    ar="$(ls "target/$TARGET/release"/*.a | head -1)"
    # mangled names (unique per instantiation), decimal sizes (portable awk).
    "$NM" --print-size --defined-only --radix=d "$ar" > "$OUT/$name.nm"
done

# marginal text bytes of a variant over baseline: symbols it has that baseline lacks.
text_of() {
    awk '
        NR==FNR { if (NF>=4) seen[$4]=1; next }
        NF>=4 && !($4 in seen) && ($3=="t" || $3=="T") { txt += ($2 + 0) }
        END { printf "%d", txt + 0 }
    ' "$OUT/baseline.nm" "$OUT/$1.nm"
}

row() { printf '  %-22s %5d B\n' "$1" "$(text_of "$2")"; }
# entry delta: (basic+entry) over (basic) — the marginal cost of the entry surface.
drow() { printf '  %-22s %5d B\n' "$1" "$(( $(text_of "$2") - $(text_of "$3") ))"; }

echo
echo "Marginal .text over a bare no_std baseline — heapless::Vec, K=V=u32, cap 64:"
echo
echo "pouch:"
row "SortedSet"        sorted_set
row "UnsortedSet"      unsorted_set
row "SortedMap"        sorted_map
row "UnsortedMap"      unsorted_map
row "SortedColumnMap"    sorted_column_map
row "UnsortedColumnMap"  column_map
row "all six together" all
echo
echo "entry API (marginal over the same collection's insert/get/remove):"
drow "SortedMap"        sorted_map_both        sorted_map
drow "UnsortedMap"      unsorted_map_both      unsorted_map
drow "SortedColumnMap"    sorted_column_map_both sorted_column_map
drow "UnsortedColumnMap"  column_map_both        column_map
echo
echo "for context (same setup):"
row "SortedSet/arrayvec"   arrayvec_set
row "hand-rolled heapless" cmp_handrolled
row "heapless::LinearMap"  cmp_linear
row "heapless::FnvIndexMap" cmp_fnv
