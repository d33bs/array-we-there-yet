Picture a table where each row is one sample, such as a well of cells or a
document, and each column is one number that describes it. Image-based profiling
and embedding models make tables like this, with hundreds or thousands of
columns. We call those numbers *features*.

When you save a table like this, you have a choice. You can keep one column for
each feature, like a spreadsheet. This is often called a [**wide**](https://doi.org/10.1145/362384.362685) layout. Or you can pack
all the features of a row into one field, like a short list. This is often called
an [**array-like**](https://doi.org/10.1145/588111.588133) layout.

<figure class="layout-diagram">
<svg viewBox="0 0 640 136" role="img" aria-labelledby="layout-diagram-title" xmlns="http://www.w3.org/2000/svg">
<title id="layout-diagram-title">A wide layout compared with an array-like layout</title>
<g transform="translate(8,8)">
<text x="130" y="12" text-anchor="middle" class="ld-heading">Wide layout</text>
<rect x="0" y="26" width="66" height="26"></rect>
<rect x="66" y="26" width="41" height="26"></rect>
<rect x="107" y="26" width="41" height="26"></rect>
<rect x="148" y="26" width="41" height="26"></rect>
<rect x="189" y="26" width="41" height="26"></rect>
<text x="33" y="43" text-anchor="middle">sample</text>
<text x="86" y="43" text-anchor="middle">f1</text>
<text x="127" y="43" text-anchor="middle">f2</text>
<text x="168" y="43" text-anchor="middle">f3</text>
<text x="209" y="43" text-anchor="middle">…</text>
<rect x="0" y="52" width="66" height="26"></rect>
<rect x="66" y="52" width="41" height="26" class="ld-cell"></rect>
<rect x="107" y="52" width="41" height="26" class="ld-cell"></rect>
<rect x="148" y="52" width="41" height="26" class="ld-cell"></rect>
<rect x="189" y="52" width="41" height="26"></rect>
<text x="33" y="69" text-anchor="middle">A</text>
<rect x="0" y="78" width="66" height="26"></rect>
<rect x="66" y="78" width="41" height="26" class="ld-cell"></rect>
<rect x="107" y="78" width="41" height="26" class="ld-cell"></rect>
<rect x="148" y="78" width="41" height="26" class="ld-cell"></rect>
<rect x="189" y="78" width="41" height="26"></rect>
<text x="33" y="95" text-anchor="middle">B</text>
<text x="130" y="120" text-anchor="middle" class="ld-note">one column per feature</text>
</g>
<text x="320" y="98" text-anchor="middle" class="ld-vs">vs</text>
<g transform="translate(368,8)">
<text x="120" y="12" text-anchor="middle" class="ld-heading">Array-like layout</text>
<rect x="0" y="26" width="66" height="26"></rect>
<rect x="66" y="26" width="180" height="26"></rect>
<text x="33" y="43" text-anchor="middle">sample</text>
<text x="156" y="43" text-anchor="middle">features</text>
<rect x="0" y="52" width="66" height="26"></rect>
<rect x="66" y="52" width="180" height="26" class="ld-cell"></rect>
<text x="33" y="69" text-anchor="middle">A</text>
<text x="156" y="69" text-anchor="middle" class="ld-array">[f1, f2, f3, …]</text>
<rect x="0" y="78" width="66" height="26"></rect>
<rect x="66" y="78" width="180" height="26" class="ld-cell"></rect>
<text x="33" y="95" text-anchor="middle">B</text>
<text x="156" y="95" text-anchor="middle" class="ld-array">[f1, f2, f3, …]</text>
<text x="120" y="120" text-anchor="middle" class="ld-note">all features in one field</text>
</g>
</svg>
<figcaption>A wide layout gives each feature its own column. An array-like layout packs every feature of a row into one field.</figcaption>
</figure>

Does the choice matter? [No single layout serves every workload](https://doi.org/10.1109/ICDE.2005.1), so this page tests seven common storage formats: CSV,
Parquet, DuckDB, Zarr, TileDB, Vortex, and Lance. It measures how long it takes
to save, load, and slice the data, how big the files are, and what it costs to
move them around. If you load features into [NumPy](https://doi.org/10.1038/s41586-020-2649-2)
or a model, or you share data
with other people, the results can help you pick a layout.

The [fairest test](https://doi.org/10.1145/5666.5673) compares each array-like layout with the wide layout in the
same format. We also show every layout against CSV wide, the most common way to
share this kind of data. CSV is plain text, so those gains look bigger than they
would against another binary format.
