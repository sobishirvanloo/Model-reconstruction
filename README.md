# MeshBench

**Point-cloud surface reconstruction & comparison** — a desktop
application (PyQt5, VTK, Open3D) that rebuilds a surface from a point
cloud with five algorithms (Poisson, Ball Pivoting, Alpha Shape, RIMLS,
Signed Distance), tunes their parameters, and measures the result against
the original model with seven distance methods.

**Author:** Sobhan Shirvanlou · © 2026 Sobhan Shirvanlou. All rights
reserved. See [LICENSE](LICENSE) and [How to cite](#how-to-cite).

## Features

| Tab | Purpose |
|---|---|
| **Import mesh** | Import 3D geometry (PLY mesh or point cloud / OBJ / STL / PCD), view it in VTK, rotate it with sliders/spinboxes or the mouse, then **send it to Processing / Reconstruction** or export points + normals. Duplicate STL vertices are merged on import. |
| **Point Cloud Processing** | Import a point cloud or receive one from *Import mesh*, shuffle the points, reduce density with **voxel downsampling** (`0` = original), then **send it to Reconstruction** or export it (normals are estimated if none were imported). |
| **Reconstruction** | Import a point cloud with normals or receive one from another tab, reconstruct a surface with **Poisson**, **Ball Pivoting** or **Alpha Shape** (Open3D), measure its quality and export the result as STL / OBJ. |
| **Compare** | Compare the reconstruction with the original mesh using 7 distance methods and error statistics (see below). |

### Optimising the parameters (tab 3 → *Optimize parameters…*)

Searches the settings that fit the reference (ground-truth) mesh best,
for one algorithm or **all of them** (each tuned on its own, then
compared fairly):

* **Objective** — two-sided, so dropping the hard parts of the surface
  does not pay off: F-score @ 1 % of the diagonal (default), Chamfer, or
  the one-sided accuracy RMS of tab 4 together with a **minimum recall**
  (default 80 %; runs below it are listed but cannot win).
* **Search** — coordinate descent: a coarse pass over each parameter
  (Poisson: depth, density filter, smoothing, scale; the others: their
  radius / α factor and smoothing), then refinement around the best value
  (±1, ×0.8 / ×1.25, midpoints) until nothing improves or the run budget
  (default 25 per algorithm) is spent. Each setting runs once; the
  preprocessing cache is shared. Runs in the background, cancellable.
* **Check** — the starting and the best settings are re-run on a noisy
  copy (σ = 0.25 % of the diagonal), showing whether the gain is real or
  only fits this data set.
* **Result** — table of all runs (best starred), convergence chart,
  *Apply best to the tab* (or the selected row), *Apply & Send to
  Compare*, CSV / Markdown.

On the bunny, Poisson went from F 0.892 / Chamfer 0.42 % (depth 8,
density 2 %) to F 0.933 / 0.33 % (depth 6, no density filter: fewer
holes, recall 80 → 89 %), and the gain held on noisy data (0.893 → 0.932).
Without a reference mesh the fit to the input points is optimised, which
is weaker; send the data from tab 1 or load a reference first.

### Comparing a reconstruction with the original (tab 4)

*Send to Compare* in tab 3 hands over the mesh and its ground truth (or
open both files, or take the reference from tab 1). The two models always
overlap: the rotation / position given to the view applies to both, so a
model opened second appears in the same pose as the first; and when the
files are not in the same coordinates (e.g. the reconstruction was saved
rotated) the reconstruction is **aligned onto the reference** on opening —
FPFH features + RANSAC from any starting pose, then point-to-plane ICP
(about 1 s; random test poses come back within 0.005 units on the
175-unit bunny). Models that already match are left untouched, so their
measured error is not reduced by a refit. *Align now* forces it; the
applied rotation / move is shown in Object info and saved in the CSV. *Run comparison*
computes, on a background thread:

| Method | Distance of each compared point to … | Signed |
|---|---|---|
| Point-to-point | the nearest reference vertex | no |
| Point-to-plane | the tangent plane of the nearest reference vertex | yes |
| Point-to-mesh | the exact closest point on the reference triangles | yes |
| Cloud-to-cloud | the nearest point of a dense uniform sampling of the reference (both surfaces sampled, *Samples* points each) | no |
| Adaptive cloud-to-cloud | a least-squares plane through 6 / 12 / 24 reference samples — per point the largest neighbourhood that is still planar | yes |
| Cloud-to-mesh | the reference triangles, for the dense sampling of the compared surface | yes |
| M3C2 | Lague et al. 2013: normal at the most planar of 4 scales, mean of each cloud inside a cylinder along it; with a 95 % level of detection | yes |

Point-to-point and cloud-to-cloud also measure the sampling gap between
points, so they read higher than the surface methods. Signed: + outside,
− inside the reference. Mesh methods are skipped when the reference is a
point cloud. Nearest neighbours use Open3D's multithreaded KD-tree, exact
surface distances its BVH `RaycastingScene`, plane fits vectorised PCA;
on the bunny all seven take ≈ 6 s. Keep the M3C2 *depth* just above the
expected error: a long cylinder reaches the other side of thin parts.

**Errors** table per method: mean e, SD, MAE, RMSE, R² (= 1 − Σe² /
Σ‖q − q̄‖², q the matched reference points), median / P95 / max |e|,
time; the best value of each column is bold. **Distribution**: error
histograms of all methods overlaid, or the selected one. **View**: one
method on the reconstruction (click a table row to switch) or **all
methods in a grid** of linked views with one colour scale; the reference
can be overlaid translucent; |e| or signed colours; colour max = P95 or
your value. Export the table as CSV or copy it as Markdown; the
screenshot button saves the whole grid (every method, titles and legend).

**Colours** (tab 4): the reconstruction and the reference each get their
own colour (theme colours by default), the reference's opacity is
adjustable, and *Colour the reconstruction by its error* can be switched
off to keep both models in their own colours after a comparison;
*Default colours* resets them. The choice is remembered.

### User interface

* **Light theme by default, dark theme** with *View ▸ Dark Theme* (Ctrl+T);
  the choice is remembered. All colours live in `app/modules/theme.py`
  (no hard-coded style sheets in `main.ui`), so dialogs — including the file
  dialog — always match.
* **Drag & drop** a file onto the window, or use *File ▸ Open Recent*; file
  dialogs start in the last folder used. Meshes open in *Import mesh*,
  tables in *Process points*, PLY / PCD in the current tab.
* **Empty views explain what to do**; *File ▸ Open Sample* (Ctrl+Shift+O)
  loads the Stanford bunny.
* **Status bar**: current file, point / triangle counts, a progress bar while
  reconstructing or sweeping, and coloured success / error messages
  (serious errors also open a dialog).
* **Resizable layout** — drag the dividers (three dots) between the 3D
  view, the side panel and the console; the others adjust, the panel
  contents stretch, and a larger window gives its extra space to the 3D
  view. Double-click a divider for its default size, *View ▸ Reset
  Layout* for all; sizes are remembered per tab.
* **Object info** next to the log (drag the divider between them):
  pose; the object box (size, diagonal, volume, surface, centre, the 8
  corners in world coordinates); the axis-aligned world box; points
  (count, centroid, mean / min spacing, duplicates, density); normals
  (unit length, share pointing outward, mean direction, coherence); mesh
  (triangles, edges, surface area, volume, closed / boundary / non-manifold
  edges, edge lengths, Euler characteristic, how much of the box it
  fills); plus processing / reconstruction details per tab. Updated live
  while the model turns or moves; *Copy* puts it on the clipboard.
* **Toolbar** above every view: **Reset view** (rotation and position
  back to 0 and the model framed), **◀ 3 / 10 ▶** checkpoints, **View ▾**
  (Front / Top / Side / Iso), **Object ▾**, **Background ▾** and
  Screenshot (PNG, rendered off-screen, guides included).
* **Checkpoints** — each tab keeps its last **10** states: opening /
  receiving data, voxel size or degradation, shuffling, each finished
  reconstruction and every rotation / move (once it settles). ◀ / ▶
  (Alt+Left / Alt+Right) step through them, the badges on the arrows show
  the checkpoint they lead to, and the counter lists them all to jump to
  one. A checkpoint holds the whole state — data, mesh, parameters,
  quality results, rotation and position — and restoring it recomputes
  nothing. Going back and changing something drops the later checkpoints
  (like undo); *Remove data* clears them so the memory is freed.
* **Object ▾ → Colour & display…** — *original colours* (the file's vertex
  colours, which follow the points through processing and onto the
  reconstructed mesh; the theme colour when the file has none) or a custom
  colour; style (surface, surface + mesh lines, wireframe, points),
  opacity and point size. An error heatmap always takes precedence.
* **Background ▾ → Colour & guides…** — theme / solid / gradient
  background, and guides for reading the model: a floor grid (square size
  shown), a ruler with tick labels in model units, the **object axes and
  box** (they turn and move with the model, in step with the rotation /
  position values and the corner axes; the box's 8 corners are labelled
  with their current x, y, z — labels that would overlap are skipped), and a read-out of rotation /
  position relative to the pose the model was loaded in.
* Both windows have **Default**, Cancel, Apply and Save; settings are
  saved (kept after a restart) and apply to all tabs. *Restore default*
  in each menu resets at once.
* **Persian interface** — *View ▸ Language ▸ فارسی* (restarts the
  program): menus, buttons, labels, tooltips, dialogs, table headers and
  the object info in Persian, laid out right to left; numbers, tables,
  sliders and the log stay left to right, and technical names
  (algorithms, methods, RMSE …) stay English (`modules/i18n.py`).
* **Start screen** — while tab 1 is empty it shows the three steps
  (open → reconstruct → compare) with *Try the sample bunny* and
  *Open a file…* instead of an empty 3D view.
* **Stop reconstruction** — tab 3's *Stop reconstruction* button (or Esc)
  ends a running build at once; the last finished mesh stays.
* **One zero for all tabs** — *Send to …* hands the data over in its own
  (file) coordinates together with the six rotation / position values, and
  the next tab shows it with the same values: a model at X = 10° in one tab
  is at X = 10° in the next, not a new 0. Saved files still contain the
  model as it is shown.
* **Pose bar** — rotation and position X / Y / Z (number box + slider,
  with their reset buttons) sit in a compact bar under every 3D view
  instead of the side panel; the **Pose** button on the view toolbar or
  *View ▸ Pose Bar* hides / shows it in all tabs (remembered).
* **Themed icons** — one consistent line-icon set (`modules/icons.py`)
  replaces the platform's built-in icons on buttons, menus and toolbars;
  it is drawn in the theme colours, so it follows light / dark (white on
  the blue main buttons, red on *Remove data*).
* **Collapsible sections** — click a section header in a side panel
  (▾ / ▸) to fold it, right-click for *Expand / Collapse all*; the state is
  remembered per tab (*View ▸ Reset Layout* opens them all again). Folding
  only hides the section: every control keeps its value and state.
* Side panels are split into sections with tooltips on every parameter;
  rotation and **position** (X/Y/Z, model units) sliders act **live**
  (*Reset rotation* / *Reset position* set them back to 0).
* **Mouse, anywhere in the view** (the model need not be under the cursor):
  left drag rotates, right (or middle) drag moves the model with the mouse,
  the wheel zooms. Panel values and the corner axes follow the model.
  A small guide under each view shows this; ✕ hides it (*View ▸ Mouse
  Guide* brings it back, and it shows again on the next start).
* Tabs read as steps (1 Import → 2 Process → 3 Reconstruct) and get a
  check mark once they hold data.
* **🗑 Remove data** in every tab (or *File ▸ Remove Data*, Ctrl+W / for
  all tabs Ctrl+Shift+W) empties the tab and frees the RAM it used — points,
  mesh, reference, preprocessing / tetrahedralisation caches; the run
  history is kept. Files on disk are never touched.

| Shortcut | Action |
|---|---|
| Ctrl+O / Ctrl+Shift+O | Open file / open the sample bunny |
| Ctrl+S / Ctrl+Shift+S | Export / save a screenshot |
| Ctrl+W / Ctrl+Shift+W | Remove the data of this tab / of all tabs |
| Ctrl+0, Ctrl+1…4 | Reset view (rotation, position, zoom), Front / Top / Side / Iso |
| Alt+Left / Alt+Right | Previous / next checkpoint |
| Ctrl+R | Quality report |
| Ctrl+T | Light / dark theme |
| F1 | All shortcuts |

### Reconstruction algorithms

| Algorithm | Idea | Parameter | Good at | Weak at |
|---|---|---|---|---|
| **Poisson** | fits a smooth implicit surface | depth (or width), scale, density filter | closed, smooth surfaces, noise | smooths fine detail, can add "bubbles" |
| **Ball Pivoting** | rolls balls of radius r, 2r, 4r over the points | ball radius × mean point spacing (default 2) | interpolates the points exactly | holes where sampling is sparse |
| **Alpha Shape** | keeps Delaunay triangles smaller than α | α × mean point spacing (default 10) | very fast once tetrahedralised | many small pieces, not manifold |
| **RIMLS** | robust implicit moving least squares (Öztireli et al. 2009): local fits re-weighted to ignore outliers and the far side of thin parts, zero level extracted | kernel radius h × mean spacing (default 6) | sharp features, outliers, one clean surface | slowest (≈ 10 s on the bunny) |
| **Signed Distance** | volumetric signed distance to the oriented points over a radius (VTK `vtkSignedDistance`, Curless & Levoy style) | influence radius × mean spacing (default 5) | fast, best on noisy data, no debris | slightly smoother than Poisson |

Ball radius and α are multiples of the mean nearest-neighbour spacing, so the
defaults work for any model size. The Delaunay tetrahedralisation is cached,
so changing α is almost instant.

Measured on the Stanford bunny (no preprocessing or smoothing, compared with
the original mesh):

| Algorithm | Chamfer | F-score | Pieces | Time |
|---|---|---|---|---|
| Poisson, depth 8 | 0.36% | 0.91 | 4 | 5.0 s |
| Ball Pivoting, ×2 | 0.25% | 0.95 | 4 | 2.6 s |
| Alpha Shape, ×10 | 0.29% | 0.94 | 1813 | 3.2 s |

The two implicit methods evaluate their function only in a thin band of
grid cells around the points (cell = 1.5 × spacing) and extract the zero
level with `vtkExtractSurface`, which never closes a surface against an
empty cell; pieces smaller than 1 % of the largest are removed.
Measured on the bunny after preprocessing, against the original mesh
(with 0.5 % Gaussian noise in brackets):

| Algorithm | Chamfer | F-score | Pieces | Time |
|---|---|---|---|---|
| Poisson, depth 8 | 0.42% (0.50%) | 0.89 (0.89) | 1 (537) | 4 s |
| Ball Pivoting, ×2 | 0.51% (0.83%) | 0.86 (0.85) | 4 (844) | 2 s (9 s) |
| RIMLS, ×6 | 0.43% (0.59%) | 0.89 (0.88) | 2 (2) | 11 s (19 s) |
| Signed Distance, ×5 | 0.44% (0.53%) | 0.89 (0.89) | 2 (2) | 4 s (6 s) |

Poisson stays the most accurate; on noisy data Signed Distance is nearly as
accurate and gives one clean surface instead of hundreds of pieces. Hoppe
1992 (`vtkSurfaceReconstructionFilter`) was tried as well: same quality,
150 s, so it was left out.

### Quality evaluation (📊 Quality Report)

After every reconstruction the mesh is compared with a **reference**:

* the **original mesh** — automatically, when the data was sent from
  *Import mesh* (it travels with the points, rotation included), or loaded
  with *Load reference mesh…* (must be in the same coordinates);
* otherwise the **input points** (how well the surface fits the data).

| Metric | Meaning |
|---|---|
| Chamfer-L1 | mean of accuracy (mesh → reference) and completeness (reference → mesh) distances, also in % of the bbox diagonal |
| Hausdorff / Hausdorff-95 | worst / 95th-percentile distance (the latter ignores outliers) |
| F-score @ τ | harmonic mean of precision and recall within τ = 1% of the diagonal |
| Normal consistency | mean \|n_mesh · n_reference\| at the closest points (1 = perfect) |
| Closed, manifold, components, area, volume | mesh topology / size |
| Timings | preprocess / reconstruction / post-processing / evaluation |

Distances to surfaces are exact point-to-triangle distances (Open3D
`RaycastingScene`). The window also offers an **error heatmap** on the mesh,
a **run history** to compare parameter settings, *Copy as Markdown* and
*Save history CSV* (every parameter and metric per run). A warning is shown
when the input points do not lie on the reference mesh (median distance > 1%
of the diagonal), i.e. the reference is probably not aligned.

With only input points as reference, accuracy is pessimistic where the
points are sparse — prefer a reference mesh for final numbers.

### Robustness experiments

**By hand — *Processing* tab → 🧪 Degrade data…**: add position noise
(σ in % of the bbox diagonal), normal noise (σ in degrees), spherical holes,
or non-uniform sampling (density falling along an axis). Settings are seeded
(reproducible) and applied after voxel downsampling. The ground-truth mesh
that travelled with the data stays clean, and the degradation is recorded in
the *Data* column of the run history.

**Automatically — *Quality Report* → 🧪 Robustness sweep…**: pick one
perturbation, a list of levels and the algorithms; every combination is
reconstructed and evaluated against the ground truth (or the clean input
points). The result is a level × algorithm table (copy as Markdown) plus one
history row per run (save as CSV). Runs can be cancelled; a failing run is
reported without stopping the sweep.

### Moving data between tabs

The **➡ Send to …** buttons hand the points and their normals to the next tab
in memory and switch to it, so no intermediate files are needed. Points and
normals always travel together (shuffling permutes both, downsampling
averages both), so they cannot get mismatched.

### Point cloud files

Points and normals are saved together in **one file**:

| Format | Content |
|---|---|
| `.ply` / `.pcd` (default) | point cloud with normals, binary |
| `.txt` / `.csv` | 6 columns `x y z nx ny nz` |
| `.npy` | `(N, 6)` array |

The old two-file layout (`points.txt` + `normals.txt`, row-aligned) can still
be imported and is available on export as *legacy*.

**Exports always match the view:** the rotation shown in the viewport (sliders
or mouse) is applied to exported and sent points, normals and meshes.

## Stability and performance

* **No crash on unexpected errors** — PyQt5 aborts the program when an
  exception escapes a slot; a global hook logs it (`errors.log`), shows it
  once and keeps the program running. Native crashes write their stack to
  `crash.log`, VTK warnings go to `vtk.log` (the log folder is shown in
  *Help ▸ About*; logs are capped at 2 MB).
* **Safe exit** — closing the window cancels sweeps / optimisations and
  waits for running jobs before the OpenGL windows are released.
* **All reconstruction work in a helper process** — Open3D keeps
  Python's global lock (GIL) during its C++ calls, so the window froze for
  up to 4.5 s per reconstruction even on a background thread (most of an
  optimisation run), and it occasionally crashes or never returns inside
  the library (Ball Pivoting with a large radius on noisy data ran for over
  30 minutes). The whole pipeline - preprocessing, the five algorithms,
  quality evaluation (`modules/pipeline.py`) and the seven comparison
  methods - now runs in a long-lived helper process (`modules/isolated.py`,
  started in the background at launch). The window stays fluid (longest
  stall while optimising: 0.08 s); if the library crashes only the helper
  dies, a new one starts and the run is retried once.
* **Time limits and instant cancel** — an optimisation run that takes far
  longer than the slowest one so far (6 ×, at least 90 s; 10 min for the
  first) is stopped and listed as "too slow", and the search goes on; sweep
  runs stop after 15 min. *Cancel* and closing the window stop the running
  library call at once (the helper is ended and restarted for the next
  job). Changing a parameter while a build of the old ones has run for
  more than 8 s stops that build and starts the new one.
* **GPU** — rendering always runs on the graphics card (VTK / OpenGL);
  nearest-neighbour searches (comparison, quality metrics, RIMLS, colours,
  object info) run on an NVIDIA GPU when Open3D has CUDA (*View ▸ Use GPU
  for Computations*) and fall back to all CPU cores automatically, also
  when the GPU runs out of memory.
* **Large coordinates** — float32 KD-trees / BVHs work on centred
  coordinates: georeferenced data (around 10⁶) gives the same errors as
  data at the origin (it used to be 6–16 × too large).
* **Memory** — volumetric grids shrink their resolution when they would
  take more than a quarter of the free RAM; consoles keep the last 5000
  lines; *Remove data* frees everything a tab holds.
* **Bad input** — NaN / infinite coordinates are dropped on import.
* **Responsive UI** — heavy work runs on background threads, including
  the object statistics of large models.

## Requirements

* Python **3.14** (tested with numpy 2, VTK 9.7, Open3D 0.20, PyQt5 5.15).
* See [`requirements.txt`](requirements.txt). If your PyPI mirror fails on a
  Jupyter package while installing Open3D, use the three commands at the top
  of that file instead (Open3D does not need Jupyter for this app).

## Setup

```bash
# 1) Create and activate a virtual environment
python3.14 -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the application
cd app
python main.py
```

> The UI file is loaded relative to `main.py` (`app/ui/main.ui`), but it is
> still recommended to launch from inside `app/`.

## Pipeline used by the Reconstruction tab

The reconstruction follows [`models/test.py`](models/test.py):

1. Optional **voxel downsampling** (set *Voxel size* > 0 to enable).
2. **Statistical outlier removal**.
3. **Consistent normal orientation** (`orient_normals_consistent_tangent_plane`),
   keeping the global direction of the imported normals (no inside-out meshes).
4. **Poisson reconstruction** (`TriangleMesh.create_from_point_cloud_poisson`).
   *Width* `0` uses *Depth*; *Width* > 0 derives the depth from that finest
   cell size (Open3D itself ignores `width` whenever `depth` is set).
5. **Density filtering** — removes vertices with low Poisson density
   (adjust the *Density filter %* quantile; `0` disables it).
6. **Crop to the point-cloud bounding box** (removes Poisson "bubbles").
7. Optional **Taubin smoothing** (set *Smoothing iterations* > 0).

Reconstruction runs on a **background thread** so the UI stays responsive;
parameter changes are debounced (400 ms) so a burst of clicks runs once.

The preprocessing result (outlier removal + normal orientation, ~3 s on the
bunny) is **cached** (last 4 combinations of data, *Denoise* and *Voxel
size*), so changing a reconstruction parameter or switching algorithm only
reruns the reconstruction itself — e.g. Alpha Shape 3.3 s → 0.4 s,
Poisson depth change 7.7 s → 4.3 s.

Note: Open3D's multithreaded Poisson is not bit-for-bit reproducible —
re-running with identical input gives geometrically identical meshes
(Chamfer < 0.01%) but slightly different vertex values/order.

`models/test.py` runs the same pipeline standalone:
`python models/test.py [points.txt] [normals.txt]`.

## Sample data

`models/` contains test data used by `models/test.py`:

* `points.txt` / `normals.txt` — the Stanford bunny's 56,203 vertices and
  their vertex normals (row-aligned, legacy two-file layout). Previously these
  held every STL triangle corner separately (336,044 rows, ~83% duplicates);
  the duplicates were merged and their normals averaged.
* `Stanford_Bunny.stl`, `Raven Tools.stl` — example meshes.
* `reconstructed_mesh.stl` / `.obj` — previous reconstruction outputs.

## License

MeshBench is © 2026 Sobhan Shirvanlou. **All rights reserved** — no
licence is granted to copy, modify, publish or distribute it without the
author's written permission, and it may not be presented, in whole or in
part, as someone else's work. Full terms: [LICENSE](LICENSE).

## How to cite

If you use MeshBench or results obtained with it, cite:

> Sobhan Shirvanlou. *MeshBench: point-cloud surface reconstruction and
> comparison*, version 1.0, 2026.
> https://github.com/sobishirvanloo/Model-reconstruction

GitHub's *Cite this repository* button gives the same in BibTeX / APA
(from [`CITATION.cff`](CITATION.cff)).

