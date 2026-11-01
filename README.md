# Model Reconstruction

A PyQt5 desktop application for **surface reconstruction from point clouds**
using the **Poisson Reconstruction** algorithm (Open3D) with 3D rendering via **VTK**.

## Features

| Tab | Purpose |
|---|---|
| **Import mesh** | Import 3D geometry (PLY mesh or point cloud / OBJ / STL / PCD), view it in VTK, rotate it with sliders/spinboxes or the mouse, then **send it to Processing / Reconstruction** or export points + normals. Duplicate STL vertices are merged on import. |
| **Point Cloud Processing** | Import a point cloud or receive one from *Import mesh*, shuffle the points, reduce density with **voxel downsampling** (`0` = original), then **send it to Reconstruction** or export it (normals are estimated if none were imported). |
| **Reconstruction** | Import a point cloud with normals or receive one from another tab, reconstruct a surface with **Poisson**, **Ball Pivoting** or **Alpha Shape** (Open3D), measure its quality and export the result as STL / OBJ. |

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
* **Camera toolbar** above every view: Reset, Front, Top, Side, Iso and
  Screenshot (PNG, rendered off-screen).
* Side panels are split into sections with tooltips on every parameter;
  rotation sliders act **live** (*Reset rotation* sets them back to 0°).
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
| Ctrl+0, Ctrl+1…4 | Reset view, Front / Top / Side / Iso |
| Ctrl+R | Quality report |
| Ctrl+T | Light / dark theme |
| F1 | All shortcuts |

### Reconstruction algorithms

| Algorithm | Idea | Parameter | Good at | Weak at |
|---|---|---|---|---|
| **Poisson** | fits a smooth implicit surface | depth (or width), scale, density filter | closed, smooth surfaces, noise | smooths fine detail, can add "bubbles" |
| **Ball Pivoting** | rolls balls of radius r, 2r, 4r over the points | ball radius × mean point spacing (default 2) | interpolates the points exactly | holes where sampling is sparse |
| **Alpha Shape** | keeps Delaunay triangles smaller than α | α × mean point spacing (default 10) | very fast once tetrahedralised | many small pieces, not manifold |

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