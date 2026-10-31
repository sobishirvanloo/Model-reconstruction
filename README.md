# Model Reconstruction

A PyQt5 desktop application for **surface reconstruction from point clouds**
using the **Poisson Reconstruction** algorithm (Open3D) with 3D rendering via **VTK**.

## Features

| Tab | Purpose |
|---|---|
| **Import mesh** | Import 3D geometry (PLY mesh or point cloud / OBJ / STL / PCD), view it in VTK, rotate it with sliders/spinboxes or the mouse, then **send it to Processing / Reconstruction** or export points + normals. Duplicate STL vertices are merged on import. |
| **Point Cloud Processing** | Import a point cloud or receive one from *Import mesh*, shuffle the points, reduce density with **voxel downsampling** (`0` = original), then **send it to Reconstruction** or export it (normals are estimated if none were imported). |
| **Reconstruction** | Import a point cloud with normals or receive one from another tab, reconstruct a surface with **Poisson**, **Ball Pivoting** or **Alpha Shape** (Open3D), measure its quality and export the result as STL / OBJ. |

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

* Python **3.11** recommended (the project was built with 3.11; newer Python
  versions may lack wheels for the C-extension dependencies).
* See [`requirements.txt`](requirements.txt).

## Setup

```bash
# 1) Create and activate a virtual environment
python3.11 -m venv myenv
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

* `points.txt` / `normals.txt` — matching point cloud + normals (x y z / nx ny nz).
* `points_changed.txt` — a shuffled copy of `points.txt` **without** matching
  normals; do not pair it with `normals.txt`.
* `Stanford_Bunny.stl`, `Raven Tools.stl` — example meshes.
* `reconstructed_mesh.stl` / `.obj` — previous reconstruction outputs.