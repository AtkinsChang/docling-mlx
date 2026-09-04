# `picture_classification.pdf` provenance

`picture_classification.pdf` is vendored from
[`docling-project/docling`](https://github.com/docling-project/docling), tag
[`v2.122.0`](https://github.com/docling-project/docling/tree/v2.122.0), commit
`facdd9bae5882c24c37d4e4c3ecb6e5510667c4d`, at
`tests/data/pdf/sources/picture_classification.pdf`.

- Upstream Git blob SHA-1: `230f74fd41a83e270f9ec806a38b1da7814f6e61`
- Vendored file SHA-256: `dc38947ee802b7bcf82915804ccb8d04c4611e1b17491c415972821a329352fb`
- License: MIT, Copyright The Docling Contributors. The complete upstream
  license text is retained in the repository
  [`LICENSES/MIT.txt`](../../../LICENSES/MIT.txt); see the repository
  [`NOTICE`](../../../NOTICE) for attribution.

The two PNGs in `reference_images/` are derived by
`tools.document_figure.capture_reference` from the PDF's image XObjects: it uses
`pypdfium2` with `render=True` and `scale_to_original=True`, then converts the
result to RGB. This applies each XObject's own transform and alpha mask at
native image resolution; it does not rasterize a page or crop a rendered page.
