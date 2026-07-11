# Contributing

Thanks for taking the time to contribute!

## Getting started

```sh
git clone https://github.com/qkniep/pouch
cd pouch
cargo install just   # bootstraps the command runner
just setup           # installs the nightly toolchain + dev tools
just check           # runs the full local check suite (see below)
```

The crate builds on stable Rust, but formatting requires the **nightly**
toolchain because `rustfmt.toml` enables unstable options; `just setup`
installs it, along with cargo-nextest, cargo-deny, cargo-machete, and
typos-cli.

## Before opening a pull request

Run `just check`, which mirrors CI:

- `cargo +nightly fmt --all -- --check` — formatting
- `cargo clippy --all-targets --all-features -- -D warnings` — lints
- `cargo build --release --all-targets`
- `cargo nextest run` and `cargo test --doc` — tests
- `cargo doc` — docs build cleanly
- `cargo deny check`, `cargo machete`, `typos` — supply chain & spelling

Because behavior is gated behind features, a green default run isn't enough:
`just hack` (the cargo-hack feature powerset) is the real gate for anything
touching trait impls or feature gates. Benchmarks aren't run in CI; compile them
with `cargo bench --no-run` and run them locally with `cargo bench`.

If any tool is missing, `just check` says so and points you back to `just setup`.
Run `just` with no arguments to list all available recipes.

## Pull request titles

Pull requests are squash-merged, so the **PR title** becomes the single commit
message on `main`. This project releases with [release-plz](https://release-plz.dev),
which derives version bumps and the changelog from those commit messages, so give
each PR a [Conventional Commits](https://www.conventionalcommits.org) title —
prefix it with `feat:`, `fix:`, `docs:`, `chore:`, etc. Individual commits within a
PR can be anything; they're squashed away on merge.

A CI check (`pr-title.yml`) enforces this. While a PR is still in progress, prefix
its title with `[WIP] ` to mark it as such: the check then stays pending rather
than failing, and clears once you drop the prefix and give the PR a real type.
(GitHub draft PRs work too.)

## Licensing

By contributing you agree that your contributions are dual licensed under the
MIT and Apache-2.0 licenses, as described in the [README](README.md).
