# Model Reconstruction

A PyQt5 desktop application for **surface reconstruction from point clouds**
using the **Poisson Reconstruction** algorithm (Open3D) with 3D rendering via **VTK**.

## Features

| Tab | Purpose |
|---|---|
| **Import mesh** | Import 3D geometry (PLY / OBJ / STL / PCD), view it in VTK, rotate it with sliders/spinboxes or the mouse, export `points.txt` + `normals.txt`. |
| **Point Cloud Processing** | Import a point cloud (TXT), shuffle the points, reduce density with **voxel downsampling**, export the modified points (+ normals when available). |
| **Reconstruction** | Import point + normal files, then run **Poisson Reconstruction** through Open3D with configurable parameters, and export the result as STL / OBJ. |

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
3. **Consistent normal orientation** (`orient_normals_consistent_tangent_plane`).
4. **Poisson reconstruction** (`TriangleMesh.create_from_point_cloud_poisson`).
5. **Density filtering** — removes vertices with low Poisson density
   (adjust the *Density filter %* quantile; `0` disables it).
6. **Crop to the point-cloud bounding box** (removes Poisson "bubbles").
7. Optional **Taubin smoothing** (set *Smoothing iterations* > 0).

Reconstruction runs on a **background thread** so the UI stays responsive.

## Sample data

`models/` contains test data used by `models/test.py`:

* `points.txt` / `normals.txt` / `points_changed.txt` — point cloud (x y z [+ nx ny nz]).
* `Stanford_Bunny.stl`, `Raven Tools.stl` — example meshes.
* `reconstructed_mesh.stl` / `.obj` — previous reconstruction outputs.