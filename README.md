# Supplementary material for “Maximum softly penalized likelihood in
factor analysis”
Philipp Sterzinger, Ioannis Kosmidis, Irini Moustaki
October 5, 2025

# Directory structure

The directory `code/` contains the Julia scripts that reproduce all the
numerical results and figures in the manuscript

> Sterzinger P, Kosmidis I and Moustaki I (2025). Maximum softly
> penalized likelihood in factor analysis. \*\* add arxiv link \*\*

and the Supplementary Material document
[`fapl-supplementary.pdf`](fapl-supplementary.pdf).

The directory `code/methods/` contains methods that the scripts in
`code/` use.

The directory `results/` will be populated by files that store the
numerical results the scripts produce. Due to storage constraints, it is
left empty.

The directory `figures/` is populated by graphics that the scripts
produce.

# R version and contributed packages

All results are reproducible using Julia version 1.11.7 and the
contributed packages

<table style="width:39%;">
<colgroup>
<col style="width: 23%" />
<col style="width: 15%" />
</colgroup>
<thead>
<tr class="header">
<th>Package</th>
<th>Version</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>CairoMakie</td>
<td>0.12.9</td>
</tr>
<tr class="even">
<td>ColorSchemes</td>
<td>3.31.0</td>
</tr>
<tr class="odd">
<td>DataFrames</td>
<td>1.8.0</td>
</tr>
<tr class="even">
<td>Distributed</td>
<td>nothing</td>
</tr>
<tr class="odd">
<td>Distributions</td>
<td>0.25.122</td>
</tr>
<tr class="even">
<td>FiniteDiff</td>
<td>2.28.1</td>
</tr>
<tr class="odd">
<td>ForwardDiff</td>
<td>1.2.1</td>
</tr>
<tr class="even">
<td>GeometryBasics</td>
<td>0.4.11</td>
</tr>
<tr class="odd">
<td>JLD2</td>
<td>0.6.2</td>
</tr>
<tr class="even">
<td>Latexify</td>
<td>0.16.10</td>
</tr>
<tr class="odd">
<td>LaTeXStrings</td>
<td>1.4.0</td>
</tr>
<tr class="even">
<td>LinearAlgebra</td>
<td>1.11.0</td>
</tr>
<tr class="odd">
<td>LineSearches</td>
<td>7.4.0</td>
</tr>
<tr class="even">
<td>Optim</td>
<td>1.13.2</td>
</tr>
<tr class="odd">
<td>PrettyTables</td>
<td>2.4.0</td>
</tr>
<tr class="even">
<td>Printf</td>
<td>nothing</td>
</tr>
<tr class="odd">
<td>Random</td>
<td>1.11.0</td>
</tr>
<tr class="even">
<td>Statistics</td>
<td>1.11.1</td>
</tr>
</tbody>
</table>

# Reproducing the results

## Path

All scripts specify the path to the supplementary material path as
`supp_path`. This is currently set to
`abspath(joinpath(@__DIR__, ".."))` assuming that the working directory
for Julia is set to the `code` folder of the current `git` repository.
If this is not the case for your setup, you should set `supp_path`
appropriately.

## Parallel computation

In each Julia scripts relying on parallel computation, `num_workers`
(currently set to `8`) sets the number of cores to use.

Parallel computing in the Julia scripts uses the Julia packages
`Distributed`. See [Parallel
Computing](https://docs.julialang.org/en/v1/manual/parallel-computing/)
in Julia’s documentation for more details.

## Details

The following table lists the R and Julia scripts that need to be
executed in order to reproduce the results. The table also lists the
outputs from each script, and their label if they are shown in the main
text or the Supplementary Material document. Some of the outputs are
intermediate results, so the scripts should be executed in the order
shown.

<table>
<colgroup>
<col style="width: 48%" />
<col style="width: 45%" />
<col style="width: 5%" />
</colgroup>
<thead>
<tr class="header">
<th>Script</th>
<th>Output</th>
<th>Label</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><a
href="code/01-1-factor-model-sim.jl">01-1-factor-model-sim.jl</a></td>
<td><a href="results/">01-1-fapl-sim-*.jld2</a></td>
<td></td>
</tr>
<tr class="even">
<td><a href="code/01-2-aggregate-sim.jl">01-2-aggregate-sim.jl</a></td>
<td><a href="results/">01-2-full-sim.jld2</a></td>
<td></td>
</tr>
<tr class="odd">
<td><a href="code/01-3-perc-HW-plot.jl">01-3-perc-HW-plot.jl</a></td>
<td><a href="figures/perc-HW.pdf">perc-HW.pdf</a></td>
<td>Figure 1</td>
</tr>
<tr class="even">
<td><a href="code/02-violin-plots.jl">02-violin-plots.jl</a></td>
<td><a href="figures/violin-q3.pdf">violin-q3.pdf</a></td>
<td>Figure 2</td>
</tr>
<tr class="odd">
<td><a href="code/02-violin-plots.jl">02-violin-plots.jl</a></td>
<td><a href="figures/violin-q5.pdf">violin-q5.pdf</a></td>
<td>Figure S1</td>
</tr>
<tr class="even">
<td><a href="code/02-violin-plots.jl">02-violin-plots.jl</a></td>
<td><a href="figures/violin-q8.pdf">violin-q8.pdf</a></td>
<td>Figure S2</td>
</tr>
<tr class="odd">
<td><a
href="code/03-1-model-select-sim.jl">03-1-model-select-sim.jl</a></td>
<td><a href="results/">03-1-fapl-MS-sim-*.jld2</a></td>
<td></td>
</tr>
<tr class="even">
<td><a
href="code/03-2-aggregate-model-select-sim.jl">03-2-aggregate-model-select-sim.jl</a></td>
<td><a href="results/">03-2-full-sim-model-select.jld2</a></td>
<td></td>
</tr>
<tr class="odd">
<td><a
href="code/03-3-model-select-plot.jl">03-3-model-select-plot.jl</a></td>
<td><a
href="figures/model-select-perc-correct.pdf">model-select-perc-correct.pdf</a></td>
<td>Figure 3</td>
</tr>
<tr class="even">
<td><a
href="code/04-model-select-table.jl">04-model-select-table.jl</a></td>
<td></td>
<td>Table 2</td>
</tr>
<tr class="odd">
<td><a
href="code/05-numerical-examples.jl">05-numerical-examples.jl</a></td>
<td></td>
<td>Table 3</td>
</tr>
<tr class="even">
<td><a
href="code/05-numerical-examples.jl">05-numerical-examples.jl</a></td>
<td></td>
<td>Table S1</td>
</tr>
<tr class="odd">
<td><a
href="code/05-numerical-examples.jl">05-numerical-examples.jl</a></td>
<td></td>
<td>Table S2</td>
</tr>
<tr class="even">
<td><a
href="code/05-numerical-examples.jl">05-numerical-examples.jl</a></td>
<td></td>
<td>Table S3</td>
</tr>
</tbody>
</table>
