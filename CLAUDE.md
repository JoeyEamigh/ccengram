# CCEngram Build Agent

You are building **CCEngram** - intelligent memory and code search for Claude Code.

## Architecture Summary

**Single daemon + thin clients** via Unix socket IPC
**Storage: LanceDB** (per-project isolation)
**Search: Pure vector** - Semantic similarity via embeddings

## Project Structure

```
crates
├── backend (name: ccengram)
├── benchmark
├── cli
└── llm
```

### Dependency Graph

```
benchmark
├── ccengram
│   └── llm
└── llm

ccengram
└── llm

cli
└── ccengram
    └── llm

llm

```

## Docs

there are some docs in the `docs/` folder. search for relevant topics.

## Sample Commands

```bash
cargo check -p llm        # check llm crate
cargo check -p ccengram   # check backend crate
cargo check -p cli        # check cli crate
cargo check --all         # check all crates (recommended when changing backend)
cargo nextest run         # Run tests (use nextest over cargo test)
cargo xfmt                # Format (yes, this is intentionally `xfmt` not `fmt`)
```

## Test Guidelines

All tests can reasonably expect Openrouter to be available.

NEVER test trivial things. Focus on integration and e2e tests. asserts should have descriptive messages.

### What NOT to Test

- **Default values** - If they're wrong, you'll notice immediately
- **Type conversions** - The compiler checks these
- **Display/FromStr** - These are formatting, not logic
- **Struct field assignment** - This is what Rust's type system is for
- **Trivial math/comparisons/etc** - `50/100 = 0.5` doesn't need a test
- **Library behavior** - `HashMap::insert` works, so does `serde_json`'s `to_value` and `to_string`
- **Trivial functions** - simple functions with no calls to other functions or complex logic

### What to Test

- **Integration tests** - Test how components work together
- **E2E tests** - Test full workflows from start to finish
- **Edge cases** - Test unusual or extreme inputs

When a test fails, don't immediately go and edit the test. First, investigate if the failure indicates a real bug in the code. If a test fails, it often means there's a bug in the code, not the test itself. Only when you are certain the test is incorrect should you modify it. **NEVER SIMPLIFY A TEST TO MAKE IT PASS.** I would MUCH rather you leave a test failing than make it pass incorrectly.

When writing a test, think to yourself: "Is this complex enough that it could have a bug that wouldn't be obvious? Am I testing multiple behaviors?" If the answer is no, reconsider if the test is necessary.

## Code Rules

- **NO `any`**. NO `unwrap()` or `expect()` in library code. Use `?` and proper error types.
- Use `thiserror` for error enums
- use `anyhow` only in binary code, not in library code
- validate all inputs at boundaries using serde derives
- do not globally pub anything additional from lib.rs **THIS IS CRITICAL!!!**
- do not "flatten" exports - ie `pub use` from submods in the mod.rs. just import from the path. the mods are organized how they are for a reason. **THIS IS CRITICAL!!!**
- always use message passing over shared state
- use `tracing` for logging, using `trace!`, `debug!`, `info!`, `warn!`, and `error!` as appropriate (`trace!` and `debug!` can be especially useful). you should use `#[tracing::instrument]` at a trace level (for the span close mainly) on functions when interacting with external systems or performing significant operations (io, db access, etc)
- NEVER use `std::io` or `std::fs` directly. use `tokio::fs` and `tokio::io` only. all io MUST be async.
- do NOT mute warnings.
- do NOT use excessive comments. well-written code is self-explanatory. avoid inline comments unless necessary for clarity (non-obvious behavior).
