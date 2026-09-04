# Contributing

## Development setup

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the development environment, checks, and qualification lanes.

## SPDX headers

New project-authored files need only the license identifier in their header:

```text
# SPDX-License-Identifier: Apache-2.0
```

Declare project copyright in `REUSE.toml`; do not add a project copyright line
to new file headers. Adapted third-party files retain their upstream copyright
and SPDX license lines. Non-commentable files use `REUSE.toml` annotations.

## Commits

Commit messages follow Conventional Commits with the scopes in
[`docs/commits.md`](docs/commits.md); `committed.toml` enforces them.

## Third-party code

Never copy third-party code without retaining its upstream license and
attribution. Preserve or add an SPDX license header and add the required entry
to `NOTICE`. Code written after a reference implementation without copying it must say so in
the file header (`Implemented after ...`) and credit the reference in `NOTICE`.
