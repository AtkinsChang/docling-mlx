# Changelog

## [0.1.1](https://github.com/AtkinsChang/docling-mlx/compare/v0.1.0...v0.1.1) (2026-09-05)


### Bug Fixes

* **engines:** admit one thread at a time into each compiled backbone ([288884f](https://github.com/AtkinsChang/docling-mlx/commit/288884ff36c18049da9d6e9aa3d2d88f49c334c8))
* **tools:** run the official Granite benchmark on Transformers 5.16 ([bfdeae8](https://github.com/AtkinsChang/docling-mlx/commit/bfdeae8b918147f8a30f7c8abf2987c8b82d8367))


### Performance

* **tableformer/v1:** double buffer the greedy decode loop ([f90feee](https://github.com/AtkinsChang/docling-mlx/commit/f90feeecf258912b8b9f782a9698c5b749c5711c))
* **tableformer/v2:** double buffer the greedy decode loop ([4341a2a](https://github.com/AtkinsChang/docling-mlx/commit/4341a2a2766f625d3384e20d6d8ba76e363f360d))
* **tableformer/v2:** read the emitted token without a new op ([5bbedaa](https://github.com/AtkinsChang/docling-mlx/commit/5bbedaa54ac9a1fdf0ee8d0c8a0479d184a9a855))

## 0.1.0 (2026-09-04)


### Features

* **document-figure:** add the EfficientNet engine and picture classification stage ([0c4cf00](https://github.com/AtkinsChang/docling-mlx/commit/0c4cf00700ccf46148c985d86fb068f77fc5dc44))
* **granite-vision:** add the corrected Granite Vision table and chart stages ([ea81b11](https://github.com/AtkinsChang/docling-mlx/commit/ea81b1125c1b98a4de5f6f53762c2df6096dff81))
* **layout/egret:** add the D-FINE engine and Egret layout presets ([9be154f](https://github.com/AtkinsChang/docling-mlx/commit/9be154fef49c159d4582d6341770510575c2d878))
* **layout/heron:** add the RT-DETR-v2 engine and Heron layout stage ([d03b491](https://github.com/AtkinsChang/docling-mlx/commit/d03b49156648457b9bbf82d453cc0493c1fe0851))
* **pipeline:** add the all-MLX standard pipeline and plugin entry point ([0d93866](https://github.com/AtkinsChang/docling-mlx/commit/0d93866f890337f00ffef64ef48ae6e94d0acf9b))
* **tableformer/v1:** add the TableFormer v1 engine and table structure stage ([591571e](https://github.com/AtkinsChang/docling-mlx/commit/591571e7edfc37c1702ce18f5cdd01635413dc85))
* **tableformer/v2:** add the TableFormerV2 engine and table structure stage ([68bddef](https://github.com/AtkinsChang/docling-mlx/commit/68bddefc96e76dfb6992ba71586fccb5ac4a54c9))
* **tools:** add the benchmark and DPBench comparison tools ([7386fc9](https://github.com/AtkinsChang/docling-mlx/commit/7386fc9dd236f2475e21404703819467771da292))
