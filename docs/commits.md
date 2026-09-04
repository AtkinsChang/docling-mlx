# Commit scopes

Commit messages follow Conventional Commits. `committed.toml` enforces the allowed types and the
scope list below on pull requests. A scope names the component a changelog reader recognizes, not a
directory; omit the scope when a change spans three or more scopes.

| Scope             | Covers                                                                          |
| ----------------- | ------------------------------------------------------------------------------- |
| `layout/heron`    | RT-DETR-v2 engine, Heron presets, `tools/layout_heron`, `docs/layout-heron`     |
| `layout/egret`    | D-FINE engine, Egret presets, `tools/layout_egret`, `docs/layout-egret`         |
| `tableformer/v1`  | TableFormer v1 engine, stage, converter, and documents                          |
| `tableformer/v2`  | TableFormerV2 engine, stage, converter, and documents                           |
| `granite-vision`  | Corrected Granite Vision engine and the Granite table and chart stages          |
| `document-figure` | EfficientNet engine and the picture classification stage                        |
| `engines`         | Code shared by engines: `engines/_shared`, detector types and post-processing   |
| `stages`          | Code shared by stages: `_compat`, runtime guards, artifact resolution           |
| `pipeline`        | `pipeline.py`, `plugins.py`, `presets.py`, `examples/`                          |
| `tools`           | Benchmarks, DPBench comparison, release qualification, `tools/_common`          |
| `licensing`       | `NOTICE`, `REUSE.toml`, `LICENSES/`, SPDX headers                               |
| `deps`            | Dependency bounds and `uv.lock`                                                 |
| `packaging`       | Wheel and sdist contents, entry points, package metadata                        |

- `docs` and `test` commits use the scope of the component they describe.
- `stages/layout.py` serves both layout engines: use the scope of the engine mainly affected, or
  `stages` when both are affected equally.
- `ci` commits carry no scope.
- Breaking changes append `!` to the scope and add a `BREAKING CHANGE:` footer.
