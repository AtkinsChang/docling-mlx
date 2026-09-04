# TableFormer v1 matching fixture provenance

`basin_table_1_text.pdf` uses the public-domain USGS-derived
`../tableformer_v2/basin_table_1.png` image documented in
[`../tableformer_v2/UPSTREAM.md`](../tableformer_v2/UPSTREAM.md).

The PDF keeps that image as its visible page and adds thirteen invisible Helvetica
tokens. `A1` through `D3` form a 4-by-3 grid; `OV` starts at x=`505`, y=`460` and
overlaps two raw predicted cells, requiring pinned unique assignment and bbox repair.
The page is `1332 × 579.6` points. Grid token origins are x=`120,420,720,1020`
and y=`460,300,140`; all text uses font size 12 and PDF text rendering mode 3.
At 2x rendering the RGB pixels are byte-identical to `../tableformer_v2/basin_table_1.pdf`.

SHA-256: `44d87b7adb7ea0a14256adb2c5ecda078fa56607d0e645c80221b002e37c500a`.

The corresponding Fast expected output is stored in
[`../../golden/tableformer_v1/fast_basin_table_1.json`](../../golden/tableformer_v1/fast_basin_table_1.json).
It was captured with pinned Torch CPU Heron R50
`8f39ad3c0b4c58e9c2d2c84a38465abf757272d8` and upstream TableFormer Fast
`fc0f2d45e2218ea24bce5045f58a389aed16dc23`, with `parser_threads=1`.
