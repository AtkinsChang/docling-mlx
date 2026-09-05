# Granite Vision 4.1 adaptor validation

Requalified on 2026-09-05 on an Apple M4 Pro against Docling 2.126.0. The qualification covers the
corrected engine and the table/chart adaptor boundary using the pinned official artifact. It does not
claim Torch-to-MLX tensor parity, general document accuracy, or a speedup over another backend.

## Qualified environment

- Apple M4 Pro, 48 GiB unified memory; macOS 26.5.2 arm64.
- Python 3.13.13; MLX 0.32.2; Docling/Docling Slim 2.126.0.
- `uv.lock` resolves MLX 0.32.2, `mlx-vlm==0.6.17`, `mlx-audio==0.5.1`, Transformers 5.16.1,
  Torch 2.14.0, torchvision 0.29.0, and OpenCV 5.0.0.93.
- Official model: `ibm-granite/granite-vision-4.1-4b` revision
  `dd48e97503de471803850df70843cf9eb5da8712`, unchanged BF16 artifact.

**Adaptation source versus pinned reference.** The adapted OTSL and chart parser sources from
Docling 2.123.1 are unchanged in currently pinned Docling 2.126.0, and this component is validated
against that version. The mlx-vlm `granite4_vision` vision, processing, and model modules are
unchanged in locked 0.6.17; its language module now builds RoPE through `initialize_rope`, which
returns the same `nn.RoPE` for this artifact because its config declares no rope scaling.

## Adaptor contracts

Canned-engine tests cover table crop and batch order, open/closed/self-closing OTSL, row and column
spans, invalid pages, empty clusters, malformed output, all three chart prompts, image-major batch
ordering, CSV/code parsing, metadata preservation, classification prerequisites, no-active-prompt
behavior, and independent task failure. Factory tests prove the custom table options dispatch only
to the custom plugin stage while built-in options retain the built-in stage. The focused Granite
suite passed 106 tests, deselected 3, in 29.91 seconds.

Shared-engine tests construct one corrected engine and inject it into both stages. The model is
initialized once, stage cleanup does not close the external engine, and application cleanup closes
it once after table and chart work.

## Corrected-engine qualification

The real offline release lane loaded the unchanged official BF16 artifact through
`MlxGraniteVision41Engine`. Three release tests passed in 60.10 seconds.

The same-crop regression recorded exact input arrays and generated output:

- `input_ids`: `0acb5307c1a000680fa3318de8a2ab44ccb5e07a7538742b3e13bb10cc94e893`;
- `pixel_values`: `a92fa19f562747c7470a93d985d023ced54d20293ae7a477ed8f1347ee4aff47`;
- `attention_mask`: `f4799abdb0cef6e988b8b67c31ae73ddc99cf210fe48c246cc6bc62bceb55be0`;
- `image_sizes`: `13170ab3b94a8bc51246aa1b0cae22b87676d826c2836cf36a33f4c401aa89d9`;
- all 106 generated token IDs matched exactly between the stored Torch reference path and committed
  corrected MLX path;
- generated CSV SHA-256: `d7e980c84fd734150af90f432267e6a7e345098610a8d3bdabb67e6ba346d47b`.

Corrected MLX model loading took 2.317099 seconds and generation took 4.882252 seconds. The
official artifact remains unchanged and no converted model is used.

- simple, merged-cell, and two-table images produced parseable Docling tables with expected text
  and spans;
- bar, line, and pie charts produced nonempty tabular metadata;
- one bar chart produced nonempty CSV, summary, and syntactically valid Python metadata.

The clean `[vlm]` wheel environment independently loaded the same local artifact with network
access blocked and completed corrected-engine chart CSV generation.

## Backend comparison

Each row used a fresh batch-size-one process: one construction-plus-inference
warm-up, then three timed rounds for MLX and `official-mps`, and one for
official CPU Granite. MLX used its stage-owned `VlmModelSpec.max_new_tokens=4096` unchanged;
the direct official calls were capped at 4096, and no measured request reached
that limit. The stock official Docling stage leaves generation unbounded, so
the explicit cap keeps the measurements finite. The table task used the first
five of 63 DPBench ground-truth table crops in dataset order. The chart task
used five ground-truth picture crops selected by the highest-confidence MLX
classifier top-1 in bar, line, or pie chart.

MLX used Metal. Official Docling Granite only permits CPU or CUDA, so the
supported official comparison used Torch CPU. The `official-mps` rows are a
direct stage-gate bypass for diagnostic measurement, not a supported official
Docling configuration; they show that the official implementation also varies
between CPU and MPS. Image tensors and prompts were previously verified
identical. The remaining MLX-versus-CPU differences are greedy-decoding
divergences between Metal BF16 and CPU BF16 execution of the same weights and
prompts; the positions below are the first divergent generated token ids.

The pinned Granite artifact ships its own modeling code, which Docling's official Granite stages
load with `trust_remote_code`, and that code calls `transformers.masking_utils.create_causal_mask`
with a `cache_position` argument that Transformers 5.8 accepted and ignored and 5.9 removed. The
official rows below therefore ran through a benchmark-side wrapper in
`tools/compare_backends.py` that drops that argument; without it Docling's Granite stages raise
`TypeError` on Transformers 5.9 and newer. The wrapper is faithful within the divergence this
record already documents: the official table OTSL is byte-identical to the 2026-09-03 official
run on 4 of the 5 crops, and the fifth is the crop where MLX and official already disagreed on
both stacks (similarity 0.98 against the earlier output). MLX Granite outputs are byte-identical
across mlx-vlm 0.6.4 and 0.6.17 on all 5 table crops.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.126.0, docling-ibm-models 4.0.2, MLX 0.32.2, mlx-vlm 0.6.17, Torch
2.14.0, Transformers 5.16.1; measured on 2026-09-05 with `tools/compare_backends.py` schema 2 and
`MLX_ENABLE_TF32=0`.

### Table (`<tables_otsl>`)

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | generated tokens/request (median) | tokens/s | GT exact OTSL | GT tree TEDS | OTSL identity vs CPU |
| -------------- | --------- | --------------------: | ------------: | -------: | --------------------------------: | -------: | ------------: | -----------: | -------------------: |
| mlx            | mlx-metal |             15374.099 |     24492.679 | 8.13 GiB |                               430 |   28.300 |      0.000000 |     0.846189 |             0.800000 |
| official       | torch-cpu |             70907.970 |     77519.633 | 9.05 GiB |                               430 |    5.488 |      0.000000 |     0.859141 |            reference |
| official-mps   | torch-mps |             19472.907 |     29148.690 | 8.12 GiB |                               430 |   21.998 |      0.000000 |     0.748714 |             0.800000 |

### Chart (`<chart2csv>`, `<chart2summary>`, `<chart2code>`)

| implementation | device    | warm ms/item (median) | first-call ms | peak RSS | generated tokens/request (median) | tokens/s | CSV identity vs CPU | summary identity vs CPU | code identity vs CPU |
| -------------- | --------- | --------------------: | ------------: | -------: | --------------------------------: | -------: | ------------------: | ----------------------: | -------------------: |
| mlx            | mlx-metal |             18656.438 |     21496.794 | 8.12 GiB |                               166 |   28.422 |            1.000000 |                0.200000 |             0.600000 |
| official       | torch-cpu |             73454.518 |     74700.242 | 8.99 GiB |                               166 |    6.323 |           reference |               reference |            reference |
| official-mps   | torch-mps |             23905.271 |     27857.360 | 8.12 GiB |                               166 |   18.962 |            0.800000 |                0.000000 |             0.600000 |

Selected ground-truth chart crops, in classifier-confidence order:

- `doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001.png` /
  `picture-01` (pie_chart, 0.9999870062)
- `doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001.png` /
  `picture-00` (pie_chart, 0.9999837875)
- `doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001.png` /
  `picture-00` (pie_chart, 0.9999713898)
- `doc_20c3068a37794e1e2db40c6227d57172ad308c8a81815c4c08970219586ef4ee_page_000001.png` /
  `picture-00` (pie_chart, 0.9999668598)
- `doc_a012af9597cad209a43dc6a6a4543994e857a56fac9fdd3eed92a5e1b93f79b2_page_000001.png` /
  `picture-00` (bar_chart, 0.9997484088)

### MLX versus official CPU differences

```diff
First divergent generated token id (1-based): 9.
--- official-cpu/doc_0592b0a1dff0c0b4a48b4504c035336bc9b30831fa2ac2ca8ef8b43a9b5df271_page_000001-table-00/<tables_otsl>
+++ mlx/doc_0592b0a1dff0c0b4a48b4504c035336bc9b30831fa2ac2ca8ef8b43a9b5df271_page_000001-table-00/<tables_otsl>
@@ -1 +1 @@
-[<otsl><ched>1. Changing Practices, Shifting Sites<ched>7<nl><ched>2. Core and Periphery of Play<ched>12<nl><ched>Part I: New Children, Different Toys<ched>21<nl><fcel>3. The Child as Consumer<fcel>26<nl><fcel>4. Domesticating Play<fcel>30<nl><fcel>5. The Child in the City<fcel>35<nl><fcel>6. Toys as Containers, Mediators and Promoters<fcel>39<nl><ched>Part II: From Solitary to Networked Geographies of Play<ched>45<nl><fcel>7. LEGO Toys: from Wooden Blocks to Plastic Bricks<fcel>50<nl><fcel>8. Brand Extension & Product Differentiation<fcel>58<nl><fcel>9. Bringing the Fans into the Company<fcel>62<nl><fcel>10. Many-to-Many Geographies of Play<fcel>66<nl><ched>Part III: Commercial Geographies of Play<ched>71<nl><fcel>11. Toy Towns and Simulated Cities<fcel>73<nl><fcel>12. A 21st-century Dollhouse: The Sims<fcel>83<nl><fcel>13. Unwanted Play Practices in The Sims Online<fcel>94<nl><fcel>14. Commodified Geographies of Play<fcel>103<nl><ched>Part IV: Serious Geographies of Play<ched>107<nl><fcel>15. Participation Tools<fcel>111<nl><fcel>16. Participation Processes<fcel>119<nl><fcel>17. Purposeful Play<fcel>122<nl><fcel>18. Serious Geographies of Play<fcel>124<nl><ched>Conclusion<ched>127<nl><fcel>19. Changing Geographies of Play<fcel>127<nl><fcel>20. Making Do<fcel>132<nl><ched>Notes<fcel>137<nl><ched>Bibliography<fcel>139<nl><ched>Index<fcel>153<nl></otsl>]
+[<otsl><ched>1.<ched>Changing Practices, Shifting Sites<ched>7<nl><fcel>2.<fcel>Core and Periphery of Play<fcel>12<nl><fcel>Part I: New Children, Different Toys<lcel><fcel>21<nl><fcel>3.<fcel>The Child as Consumer<fcel>26<nl><fcel>4.<fcel>Domesticating Play<fcel>30<nl><fcel>5.<fcel>The Child in the City<fcel>35<nl><fcel>6.<fcel>Toys as Containers, Mediators and Promoters<fcel>39<nl><fcel>Part II: From Solitary to Networked Geographies of Play<lcel><fcel>45<nl><fcel>7.<fcel>LEGO Toys: from Wooden Blocks to Plastic Bricks<fcel>50<nl><fcel>8.<fcel>Brand Extension & Product Differentiation<fcel>58<nl><fcel>9.<fcel>Bringing the Fans into the Company<fcel>62<nl><fcel>10.<fcel>Many-to-Many Geographies of Play<fcel>66<nl><fcel>Part III: Commercial Geographies of Play<lcel><fcel>71<nl><fcel>11.<fcel>Toy Towns and Simulated Cities<fcel>73<nl><fcel>12.<fcel>A 21st-century Dollhouse: The Sims<fcel>83<nl><fcel>13.<fcel>Unwanted Play Practices in The Sims Online<fcel>94<nl><fcel>14.<fcel>Commodified Geographies of Play<fcel>103<nl><fcel>Part IV: Serious Geographies of Play<lcel><fcel>107<nl><fcel>15.<fcel>Participation Tools<fcel>111<nl><fcel>16.<fcel>Participation Processes<fcel>119<nl><fcel>17.<fcel>Purposeful Play<fcel>122<nl><fcel>18.<fcel>Serious Geographies of Play<fcel>124<nl><fcel>Conclusion<lcel><fcel>127<nl><fcel>19.<fcel>Changing Geographies of Play<fcel>127<nl><fcel>20.<fcel>Making Do<fcel>132<nl><fcel>Notes<lcel><fcel>137<nl><fcel>Bibliography<lcel><fcel>139<nl><fcel>Index<lcel><fcel>153<nl></otsl>]
```

```diff
First divergent generated token id (1-based): 40.
--- official-cpu/doc_20c3068a37794e1e2db40c6227d57172ad308c8a81815c4c08970219586ef4ee_page_000001-picture-00/<chart2summary>
+++ mlx/doc_20c3068a37794e1e2db40c6227d57172ad308c8a81815c4c08970219586ef4ee_page_000001-picture-00/<chart2summary>
@@ -1,7 +1,7 @@
-The provided chart image is a 3D pie chart that visually represents the distribution of three categories: Cutting raw woods, Fabrication, and Transportation. The chart does not have a specified title or axis labels, but it effectively uses color coding to differentiate between the categories. Cutting raw woods is depicted in blue, Fabrication in orange, and Transportation in gray.
+The provided chart image is a 3D pie chart that visually represents the distribution of three categories: Cutting raw woods, Fabrication, and Transportation. The chart does not have a specified title or labels for the axes, as it is a pie chart, which typically does not require axis labels.
 
-The pie chart is divided into three segments, each corresponding to one of the categories. The largest segment, colored blue, represents Cutting raw woods, which occupies approximately 60% of the chart. This indicates that Cutting raw woods is the most significant category in the data set. The second-largest segment, colored orange, represents Fabrication, which accounts for about 30% of the chart. The smallest segment, colored gray, represents Transportation, making up roughly 10% of the chart.
+The pie chart is divided into three segments, each corresponding to one of the categories mentioned. The largest segment, colored in blue, represents "Cutting raw woods" and occupies approximately 60% of the chart. This indicates that the majority of the data falls under this category. The second-largest segment, colored in orange, represents "Fabrication" and makes up about 30% of the chart. The smallest segment, colored in gray, represents "Transportation" and accounts for roughly 10% of the chart.
 
-The chart includes a legend that clearly identifies the colors associated with each category, aiding in the interpretation of the data. The 3D effect of the pie chart adds depth to the visualization, making it more engaging and easier to understand the proportions of each category.
+The chart includes a legend that clearly identifies each category with its corresponding color: blue for Cutting raw woods, orange for Fabrication, and gray for Transportation. The use of different colors helps to distinguish between the categories and makes the chart easy to interpret.
 
-In summary, the 3D pie chart illustrates that Cutting raw woods is the dominant category, followed by Fabrication, with Transportation being the smallest. The use of distinct colors and a 3D effect enhances the clarity and visual appeal of the chart, providing a clear and concise representation of the data distribution.
+In summary, the 3D pie chart effectively illustrates the distribution of the three categories, with "Cutting raw woods" being the most significant, followed by "Fabrication," and "Transportation" being the least significant. The chart's design and color coding enhance its readability and provide a clear visual representation of the data.
```

```diff
First divergent generated token id (1-based): 70.
--- official-cpu/doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-00/<chart2code>
+++ mlx/doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-00/<chart2code>
@@ -2,7 +2,7 @@
 
 labels = 'Unknown', 'Decreasing', 'Same', 'Increasing', 'Gone'
 sizes = [45, 33, 12, 5, 5]
-colors = ['#3A5A40', '#A3B1B8', '#8F9A6B', '#B58A5A', '#D8D8D8']
+colors = ['#4a5d23', '#a3b1a4', '#8f9c6a', '#c49c44', '#d7b5a6']
 explode = (0.1, 0, 0, 0, 0)
 
 plt.pie(sizes, explode=explode, labels=labels, colors=colors,
```

```diff
First divergent generated token id (1-based): 139.
--- official-cpu/doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-00/<chart2summary>
+++ mlx/doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-00/<chart2summary>
@@ -1,9 +1,7 @@
 The provided chart image is a pie chart that illustrates the distribution of various categories, although a specific title is not given. The chart is divided into five distinct segments, each representing a different category with corresponding percentages.
 
-The largest segment, colored in dark green, represents the "Unknown" category, which constitutes 45% of the total. This is the most significant portion of the chart, indicating that nearly half of the data falls into this category. The next largest segment, shown in light green, is labeled "Decreasing" and accounts for 33% of the data. This is followed by a segment in brown, labeled "Same," which makes up 12% of the total.
+The largest segment, colored in dark green, represents the "Unknown" category, which constitutes 45% of the total. This is the most significant portion of the chart, indicating that nearly half of the data falls into this category. The next largest segment, shown in light green, is labeled "Decreasing" and accounts for 33% of the data. This is followed by a segment in brown, labeled "Same," which makes up 12% of the chart. The two smallest segments are "Increasing" and "Gone," each represented in different shades and both comprising 5% of the total.
 
-The two smallest segments are "Increasing" and "Gone," each representing 5% of the data. The "Increasing" segment is colored in a light brown shade, while the "Gone" segment is in a light gray color. These two categories are the least represented in the chart, each contributing equally to the total.
+The pie chart uses a color scheme that includes dark green, light green, brown, and two shades of gray to differentiate between the categories. The percentages are clearly labeled on each segment, making it easy to understand the distribution at a glance. The total of all categories sums up to 100%, ensuring that the chart accurately represents the entire dataset.
 
-The pie chart uses different colors to distinguish between the categories, making it easy to visually compare their proportions. The percentages are clearly labeled on each segment, providing precise information about the distribution of the data.
-
-In summary, the pie chart effectively shows that the "Unknown" category is the most prevalent, followed by "Decreasing" and "Same," with "Increasing" and "Gone" being the least common categories. The total of all categories sums up to 100%, ensuring that the chart provides a complete representation of the data.
+In summary, this pie chart effectively visualizes the distribution of five categories, with "Unknown" being the most prevalent at 45%, followed by "Decreasing" at 33%, "Same" at 12%, and both "Increasing" and "Gone" at 5% each. The use of distinct colors and clear labeling enhances the readability and comprehension of the data presented.
```

```diff
First divergent generated token id (1-based): 217.
--- official-cpu/doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-01/<chart2summary>
+++ mlx/doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-01/<chart2summary>
@@ -2,6 +2,6 @@
 
 The pie chart is divided into six distinct segments, each representing a different category of concern. The largest segment, which occupies 69% of the chart, is labeled "Least concern" and is depicted in a light gray color. This indicates that the majority of the data falls into this category. The next largest segment, colored in a darker shade of gray, represents "Data deficient" and accounts for 15% of the total. Following this, the "Vulnerable" category, shown in a medium gray color, makes up 9% of the chart. The "Near threatened" category, in a lighter gray shade, constitutes 5% of the data. The smallest segments, each representing 1% of the chart, are "Critically endangered" and "Endangered," both depicted in the darkest shades of gray.
 
-The chart uses a color gradient from light to dark gray to differentiate between the categories, with the lightest shade representing the "Least concern" category and the darkest shades representing the "Critically endangered" and "Endangered" categories. The percentages for each category are clearly labeled within their respective segments, making it easy to understand the distribution at a glance.
+The chart does not include any legends or additional annotations, and the labels for each category are directly placed on the corresponding segments. The color scheme is consistent, with varying shades of gray used to differentiate between the categories. The total percentage of all categories sums up to 100%, ensuring that the chart accurately represents the entire dataset.
 
-In summary, the pie chart effectively communicates that the majority of the data falls under the "Least concern" category, with smaller proportions in the "Data deficient," "Vulnerable," "Near threatened," and "Critically endangered" categories. The use of a color gradient and clear labeling enhances the readability and interpretability of the chart.
+In summary, this pie chart effectively communicates the distribution of concern levels, with the majority of the data falling into the "Least concern" category, followed by "Data deficient" and "Vulnerable," and the smallest proportions in the "Critically endangered" and "Endangered" categories.
```

```diff
First divergent generated token id (1-based): 28.
--- official-cpu/doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00/<chart2code>
+++ mlx/doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00/<chart2code>
@@ -1,14 +1,15 @@
 <code><loc_0><loc_0><loc_500><loc_500>import matplotlib.pyplot as plt
-import numpy as np
 
-labels = ['Domestic logs\nand wood\nchips', 'Domestic wood pellets', 'Import pellets,\nchips', 'PKS', 'Construction\nwood waste', 'Waste\nmaterials', 'Others']
+labels = 'Domestic logs\nand wood\nchips', 'Domestic wood pellets', 'Import pellets,\nchips', 'PKS', 'Construction\nwood waste', 'Waste\nmaterials', 'Others'
 sizes = [55, 20, 20, 15, 5, 5, 5]
 colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#fdc086', '#ffff99']
+explode = (0.1, 0, 0, 0, 0, 0, 0)
 
-fig1, ax1 = plt.subplots()
-ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
-ax1.axis('equal')
+plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
+        shadow=False, startangle=90, textprops={'fontsize': 10})
 
-plt.title('Wood fuel sources in Finland in 2018')
-plt.show()
+plt.axis('equal')
+plt.title('Wood fuel sources in Finland in 2018', fontsize=14, pad=20)
+plt.tight_layout()
+plt.savefig('output.png')
 </code>
```

```diff
First divergent generated token id (1-based): 211.
--- official-cpu/doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00/<chart2summary>
+++ mlx/doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00/<chart2summary>
@@ -2,8 +2,8 @@
 
 The pie chart is divided into six distinct segments, each representing a different category of wood-based materials. The largest segment, colored in green, represents "Domestic logs and wood chips," which constitutes the majority of the distribution at 50%. This is followed by a yellow segment labeled "Import pellets, chips," which accounts for 20% of the total. The third-largest segment, colored in orange, is "Domestic wood pellets," making up 15% of the distribution.
 
-The remaining categories are smaller in comparison. The "PKS" category, shown in a light brown color, represents 10% of the total. The "Construction wood waste" category, depicted in a light gray color, accounts for 5%. The smallest segment, colored in blue, is labeled "Waste materials" and "Others," together making up 10% of the distribution.
+The remaining categories are smaller in comparison. The "PKS" category, shown in a light brown color, represents 10% of the total. The "Construction wood waste" category, depicted in a light gray color, accounts for 5%. The smallest segment, colored in blue, is labeled "Others" and constitutes 0.5% of the distribution.
 
-The chart uses different colors to distinguish between the categories, making it easy to visually compare their proportions. The labels for each category are clearly marked, and the percentages are visually represented by the size of each segment. The total distribution across all categories sums up to 100%.
+The chart uses different colors to distinguish between the categories, making it easy to visually compare their proportions. The labels for each category are clearly marked, and the percentages are indicated next to each segment, providing a quick and clear understanding of the distribution.
 
-In summary, the pie chart effectively communicates the dominance of "Domestic logs and wood chips" in the wood-based materials sector, with "Import pellets, chips" and "Domestic wood pellets" also playing significant roles. The smaller categories, including "PKS," "Construction wood waste," and "Waste materials" and "Others," contribute less to the overall distribution.
+In summary, the pie chart effectively communicates that "Domestic logs and wood chips" are the most significant component of the wood-based materials, followed by "Import pellets, chips" and "Domestic wood pellets." The categories of "PKS," "Construction wood waste," and "Others" represent smaller proportions of the total distribution.
```

### Official MPS versus official CPU differences

The direct MPS bypass differed from CPU on table
`doc_0592b0a1dff0c0b4a48b4504c035336bc9b30831fa2ac2ca8ef8b43a9b5df271_page_000001-table-00`
at generated token id 9, the same crop and position where MLX diverges. Its chart first divergent
generated token ids were:

| crop                                                                                          | prompt            | position |
| --------------------------------------------------------------------------------------------- | ----------------- | -------: |
| `doc_20c3068a37794e1e2db40c6227d57172ad308c8a81815c4c08970219586ef4ee_page_000001-picture-00` | `<chart2summary>` |       40 |
| `doc_a012af9597cad209a43dc6a6a4543994e857a56fac9fdd3eed92a5e1b93f79b2_page_000001-picture-00` | `<chart2summary>` |       51 |
| `doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-00` | `<chart2code>`    |       31 |
| `doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-00` | `<chart2summary>` |      139 |
| `doc_aa00d01e639cf879439b5f4c9313f1647bba27070a833569dee0b3eeb3476731_page_000001-picture-01` | `<chart2summary>` |       68 |
| `doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00` | `<chart2code>`    |      220 |
| `doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00` | `<chart2csv>`     |       36 |
| `doc_f0ab823f66709631e6226a937a8d68b52bde1e7ff6ef28086563c0646867c769_page_000001-picture-00` | `<chart2summary>` |       92 |

### Logit-level check

This check is a separate teacher-forced experiment that predates the 2026-09-05 comparison
above and is retained as it was recorded; the greedy divergence positions it cites are that
earlier run's.

For the five manifest table crops and five manifest chart crops, the `<tables_otsl>` and `<chart2csv>`
inputs were built through the MLX stage builder/template and the official `compare_backends.py`
processor/template path. The MLX, official CPU, and official MPS BF16 paths then ran prefill and 64
teacher-forced steps on the same CPU-BF16 token prefix, with official CPU FP32 as an optional reference.

The canonical `input_ids` (int64) and `pixel_values` (float32) SHA-256 values matched across MLX
Metal, official Torch CPU BF16, official Torch MPS BF16, and official Torch CPU FP32 for every pair.
The raw per-pair hashes are retained in the scratchpad JSON run artifact.

Prefill entries are `mean/max` absolute logit differences over the full vocabulary; the argmax column
is `MLX/MPS/FP32` agreement with CPU BF16.

| pair                   |           MLX-CPU |           MPS-CPU |           MLX-MPS |          MLX-FP32 |          MPS-FP32 |          CPU-FP32 | argmax agrees | CPU margin |
| ---------------------- | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | ------------- | ---------: |
| `0592b0a1/table-00`    |     0.1202/0.7031 |     0.1141/0.6250 |     0.1231/0.7500 |     0.0900/0.5047 |     0.1212/0.8116 |     0.0894/0.5287 | 1/1/1         |     14.000 |
| `321a4283/table-00`    |     0.3026/1.5625 |     0.2212/1.1797 |     0.1365/0.7500 |     0.1160/0.8623 |     0.0923/0.5922 |     0.2484/1.3275 | 1/1/1         |      9.250 |
| `00f0adaa/table-00`    |     0.1164/0.5625 |     0.1006/0.6250 |     0.0854/0.6250 |     0.0723/0.4644 |     0.0700/0.4462 |     0.0840/0.4145 | 1/1/1         |     15.250 |
| `d276ca9a/table-00`    |     0.0837/0.5000 |     0.1033/0.6250 |     0.1044/0.6250 |     0.0819/0.5292 |     0.1040/0.5984 |     0.0675/0.5055 | 1/1/1         |     16.500 |
| `934fbf53/table-00`    |     0.1002/0.6250 |     0.1185/0.8750 |     0.1442/0.8750 |     0.0668/0.4627 |     0.1241/0.7368 |     0.0781/0.4913 | 1/1/1         |     16.750 |
| `aa00d01e/picture-01`  |     0.0781/0.5000 |     0.1115/0.5000 |     0.0832/0.5000 |     0.0561/0.3358 |     0.0558/0.3312 |     0.0900/0.4409 | 1/1/1         |      7.125 |
| `aa00d01e/picture-00`  |     0.0811/0.5000 |     0.0908/0.5625 |     0.0821/0.6562 |     0.0503/0.3167 |     0.0743/0.7200 |     0.0758/0.4872 | 1/1/1         |      4.000 |
| `f0ab823f/picture-00`  |     0.0723/0.6250 |     0.0579/0.3750 |     0.0860/0.6562 |     0.0509/0.3260 |     0.0706/0.4376 |     0.0595/0.4262 | 1/1/1         |      6.375 |
| `20c3068a/picture-00`  |     0.0946/0.4375 |     0.0415/0.3125 |     0.0884/0.5000 |     0.0463/0.3681 |     0.0595/0.3785 |     0.0671/0.4100 | 1/1/1         |      3.000 |
| `a012af95/picture-00`  |     0.0798/0.5000 |     0.0742/0.4062 |     0.0501/0.3750 |     0.0515/0.2558 |     0.0490/0.2563 |     0.1012/0.4943 | 1/1/1         |      5.750 |
| **all 10 (aggregate)** | **0.1129/1.5625** | **0.1034/1.1797** | **0.0983/0.8750** | **0.0682/0.8623** | **0.0821/0.8116** | **0.0961/1.3275** | **10/10/10**  |          — |

The 64-step teacher-forced summary uses 640 scored steps (10 pairs × 64); the last column compares the
minimum CPU margin at a disagreement with the global p95 absolute delta.

| backend        | steps disagreeing with CPU |      p50 |      p95 |      max | min CPU margin / p95 abs Δlogit |
| -------------- | -------------------------: | -------: | -------: | -------: | ------------------------------: |
| MLX Metal BF16 |                      4/640 | 0.046875 | 0.187500 | 1.562500 |                0.000 / 0.187500 |
| Torch MPS BF16 |                      6/640 | 0.046875 | 0.183594 | 2.250000 |                0.000 / 0.183594 |

Teacher-forced through the first divergent raw greedy token, `0592b0a1` diverged at 1-based step 9
(zero-based generated-token index 8): CPU margin 0.000 with CPU `13=22.750, 16134=22.750`, MLX
`16134=22.875, 13=22.750`, and MPS `13=22.750, 16134=22.750`. The `<chart2csv>` crop
`f0ab823f/...picture-00` first diverged at step 36: CPU margin 0.000 with CPU `54=26.375,
51918=26.375`, MLX `51918=26.625, 54=26.250`, and MPS `54=26.375, 51918=26.375`.

MLX−CPU and MPS−CPU differences are the same order of magnitude (identical p50 values of 0.046875
and near-identical p95 values of 0.187500 versus 0.183594), and every disagreement had a CPU margin
no larger than that step's observed maximum |Δlogit|, although the 0.25-margin cases are above the
global p95 band.

### Full pipeline (`pipeline_granite`)

The `pipeline_granite` component runs Docling's full standard PDF pipeline with the Granite Vision
4.1 table structure stage and Granite Vision 4.1 chart extraction enabled, so the adaptors are
measured inside a real conversion instead of on isolated crops. Both sides use the Heron-default
layout preset, the DocumentFigure picture classifier, and Docling's default chart tasks
(`<chart2csv>` only); OCR, code, formula, and picture description stay disabled. MLX keeps the
stage-owned `VlmModelSpec.max_new_tokens=4096`; the official stages keep their own unbounded
`tokenizer.model_max_length`. Docling's official Granite stages accept CPU or CUDA only, so the
official pipeline runs on the `auto` accelerator: layout and picture classification on Torch MPS
and Granite forced to CPU. Because a Granite CPU table crop costs about 71 s and a chart crop about
73 s, the component uses the first 50 pages of the pinned manifest with one
construction-plus-inference warm-up and one timed round. That window is the first prefix of the
manifest that carries both table and chart work: each side detected 6 table clusters and classified
14 pictures as a bar, line, or pie chart, so both Granite stages were exercised.

The official rows use the same `create_causal_mask` wrapper described above.

Machine: Apple M4 Pro, 48 GiB unified memory, macOS 26.5.2; Python 3.13.13;
Docling 2.126.0, docling-ibm-models 4.0.2, MLX 0.32.2, mlx-vlm 0.6.17, Torch
2.14.0, Transformers 5.16.1; measured on 2026-09-05 with `tools/compare_backends.py` schema 2 and
`MLX_ENABLE_TF32=0`.

| implementation | device        | warm ms/item (median) | warm ms/item (mean) | timed round s | first-call ms |  peak RSS | tables | charts | markdown identity | layout cluster agreement at IoU >= 0.5 | layout unmatched (MLX/official) | table structure exact | table unmatched (MLX/official) |
| -------------- | ------------- | --------------------: | ------------------: | ------------: | ------------: | --------: | -----: | -----: | ----------------: | -------------------------------------: | ------------------------------- | --------------------: | ------------------------------ |
| mlx            | mlx-metal     |               102.102 |            2769.909 |         138.5 |      3855.916 | 16.03 GiB |      6 |     14 |          0.980000 |                               1.000000 | 0 / 0                           |              1.000000 | 0 / 0                          |
| official       | torch-mps+cpu |               162.016 |           14331.059 |         716.6 |      5237.575 | 17.80 GiB |      6 |     14 |         reference |                              reference | reference                       |             reference | reference                      |

The MLX process took 142.4 s of wall clock and the official process 721.8 s. Both runs converted all
50 pages with `success` status. Every layout cluster and all 6 table structures agreed, so the
Granite table crops decoded to the same OTSL on Metal BF16 and CPU BF16 here. Markdown identity is
0.980000 because 1 of the 50 pages differs, inside a chart-derived CSV table on a page with no
document table: three header cells differ only in capitalization and one value is `7.6` versus
`7.7`. That is the expected BF16 greedy-decoding divergence documented above, not a structural
difference. Each raw report records `input_count` 50 together with the selected page ids, so the
window is reproducible without re-deriving it.

MLX peak RSS was 16.03 GiB against 5.13 GiB for the same component on 2026-09-04, and the official
side was 17.80 GiB against 17.71 GiB. These are measurements across two dependency stacks; the run
does not isolate a cause for the MLX change.

The median per page hides the Granite work: 34 of the 50 pages carry no table or chart, so both
Granite stages run on the remaining 16 pages, and the per-page mean and the timed-round total are
the comparable figures. Split by page kind, from the per-item wall clock in the raw reports:

| Pages                               | MLX p50 | MLX max | MLX total | Official p50 | Official max | Official total |
| ----------------------------------- | ------: | ------: | --------: | -----------: | -----------: | -------------: |
| 34 without table or chart work      |   87 ms |   0.2 s |     3.3 s |       118 ms |        0.3 s |          4.7 s |
| 16 with tables or charts (20 crops) |   6.4 s |  18.2 s |   135.2 s |       42.7 s |       90.9 s |        711.8 s |
| timed round                         |         |         |   138.5 s |              |              |        716.6 s |

## Clean-wheel qualification

The historical clean-wheel lane passed 9 tests in 160.35 seconds. The current boundary keeps the
base lazy import plus pure wheel metadata checks in the default lane, and one Egret offline
inference smoke remains artifact- and Metal-gated.

The wheel was `docling_mlx-0.1.0-py3-none-any.whl`, 142741 bytes with 81 members, SHA-256
`7bcd74aa548464b1a7defc5187c8fe216e497c1a368a52363b605e58e1bcec99`. It contained no weights,
tests, tools, goldens, docs, reports, or reference environments.

## Repository qualification

Repository-wide lane commands and required artifacts are maintained in
[`DEVELOPMENT.md`](../../DEVELOPMENT.md). Generated qualification and packaging JSON remains under
ignored `reports/` or CI artifacts rather than being committed as a second copy of this report.
