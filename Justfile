# Test threads for nextest. Override with `just jobs=8 test` or JOBS=8.
jobs := env_var_or_default("JOBS", "16")

# List available recipes.
default:
    @just --list

# Install the dev tools `just check` (and the optional recipes) need.
setup:
    rustup toolchain install nightly --component rustfmt
    rustup component add llvm-tools                 # `just size` (llvm-nm)
    rustup target add thumbv7em-none-eabihf         # `just size` (embedded target)
    cargo install --locked cargo-nextest cargo-deny cargo-machete typos-cli cargo-hack

# Full local check suite (mirrors the core CI).
check: _check-tools fmt clippy build test doc deny machete typos

# Check formatting (nightly rustfmt).
fmt:
    cargo +nightly fmt --all -- --check

# Apply formatting (nightly rustfmt).
fmt-fix:
    cargo +nightly fmt --all

# Lint with clippy, denying warnings.
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Apply clippy's machine-applicable suggestions.
clippy-fix:
    cargo clippy --fix --all-targets --all-features --allow-dirty --allow-staged

# Build all targets in release mode.
build: _lockfile
    cargo build --release --all-targets --locked

# Run unit, integration, and doc tests.
test: _lockfile
    cargo nextest run --all-targets --all-features --locked --test-threads={{jobs}}
    cargo test --doc --all-features --locked

# Build the documentation, denying warnings (mirrors CI and docs.rs).
doc: _lockfile
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --document-private-items --locked

# Check the dependency tree (advisories, licenses, bans, sources).
deny:
    cargo deny check

# Find unused dependencies.
machete:
    cargo machete

# Check spelling.
typos:
    typos

# Lint the GitHub Actions workflows (also enforced by the actionlint CI job).
actionlint:
    actionlint

# `--no-dev-deps` checks the public feature surface in isolation, catching a missing
# `#[cfg(feature = "...")]` gate that a dev-dependency would otherwise mask. Dev
# targets are out of scope here anyway: smoke and every bench carry
# `required-features`, so partial feature sets skip them rather than break.
# Type-check every feature combination with cargo-hack (mirrors feature-powerset CI).
hack: _lockfile
    cargo hack --feature-powerset --no-dev-deps check

# Marginal binary size (.text) pouch adds per collection, for an embedded target.
# Manual tool, not part of `check` — see size/README.md. Needs llvm-tools + the
# thumbv7em-none-eabihf target (`just setup` installs both).
size:
    bash size/measure.sh

# List unfinished work: todo!/unimplemented! macros and TODO-style comments.
todo:
    -rg 'todo!\(\)|unimplemented!\(\)' --iglob='!Justfile'
    -rg 'TODO|XXX|HACK|PERF|FIXME|BUG' --iglob='!Justfile'

# Ensure a Cargo.lock exists before the `--locked` recipes run (hidden helper).
_lockfile:
    [ -f Cargo.lock ] || cargo generate-lockfile

# Report any tools `just check` needs that aren't installed (hidden helper).
_check-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    command -v cargo-nextest >/dev/null 2>&1 || missing+=("cargo-nextest")
    command -v cargo-deny    >/dev/null 2>&1 || missing+=("cargo-deny")
    command -v cargo-machete >/dev/null 2>&1 || missing+=("cargo-machete")
    command -v typos         >/dev/null 2>&1 || missing+=("typos-cli")
    cargo +nightly fmt --version >/dev/null 2>&1 || missing+=("nightly rustfmt")
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Missing tools: ${missing[*]}" >&2
        echo "Run \`just setup\` to install all (or install them manually)." >&2
        exit 1
    fi
