# TableFormerV2 real-table fixture provenance

These three lossless RGB crops are derived from the pocket tables in U.S.
Geological Survey Open-File Report 1958-87, *Geology of the Basin quadrangle,
Montana*. USGS-published report material is public domain in the United States;
credit is retained here as requested by the USGS publications guidance.

Credit: U.S. Geological Survey.

| Fixture | Source | Source PDF SHA-256 | 100 DPI crop box | Output size | PNG SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `basin_table_1.png` | [`Table-1.pdf`](https://pubs.usgs.gov/of/1958/0087/Table-1.pdf) | `103b7234056cf98fc1271b15b66c67e0b22b970882f82184a8d56426b39259d1` | `(75, 55, 1925, 860)` | `1850×805` | `c65c283bc9c6cc76477302af500aa634dbfb7e297e6d6330dd3e630f374db51a` |
| `basin_table_2.png` | [`Table-2.pdf`](https://pubs.usgs.gov/of/1958/0087/Table-2.pdf) | `632fcef9bf5328341bbd1da0525b4a4ca4381785da7a585a2621778a30f14ae8` | `(65, 55, 1875, 930)` | `1810×875` | `4a5474a85a4c7239fe3f15f071e174236b81f737936c5dff419249b9704a6c4a` |
| `basin_table_3.png` | [`Table-3.pdf`](https://pubs.usgs.gov/of/1958/0087/Table-3.pdf) | `9dcd424adf96edc1dffa701d51cdcfd8145a280843adeb7f50afdfb700073e3f` | `(65, 50, 2320, 1360)` | `2255×1310` | `df6559e23b98a2f4d0d14992c81710406703e829202bc336dfd7f6734fd9d174` |

The source PDFs are rendered with Poppler `pdftoppm -r 100 -png -singlefile`.
Each rendered RGB image is cropped with Pillow using the half-open crop box
listed above and saved as an optimized lossless PNG. The source PDFs are not
vendored.

`basin_table_1.pdf` is a deterministic one-page PDF wrapper around
`basin_table_1.png` for the offline Docling plugin smoke. Pillow first writes
the RGB image at 100 DPI with JPEG quality 90 and optimization enabled; QPDF
then removes file information and assigns a deterministic document ID with
`qpdf --remove-info --deterministic-id`. Its SHA-256 is
`95fc273e6794e34ac5f0b1022b4eedb6dcb94c68855c83cb94874baa1c07f669`.

References:

- [USGS report directory and title](https://pubs.usgs.gov/of/1958/0087/)
- [USGS Publications Warehouse copyright and reuse FAQ](https://pubs.usgs.gov/documentation/faq)
