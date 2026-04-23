# himawari-cloud

Efficient cloud-optimised access to **Himawari-9 AHI Level-2 cloud products** for the
**Great Barrier Reef (GBR)** region, using kerchunk virtual datasets to avoid downloading
full 120 MB satellite files when only a small geographic subset is needed.

---

## Background

[Himawari-9](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v13.pdf)
is a Japanese Meteorological Agency geostationary satellite stationed at 140.7 °E, providing
full-disk imagery of the Asia-Pacific region every 10 minutes via its Advanced Himawari Imager
(AHI). NOAA distributes three AHI Level-2 cloud products on AWS Open Data
(`s3://noaa-himawari9`, anonymous public access):

| Product | Contents |
|---------|----------|
| **CMSK** | Cloud mask (4-way flag), cloud probability, quality flag |
| **CHGT** | Cloud-top height (m), temperature (K), pressure (hPa), quality flag |
| **CPHS** | Cloud phase flag, cloud type flag |

Each file covers the full 5500 × 5500 pixel disk at ~2 km nadir resolution and is stored as a
compressed NetCDF4/HDF5 file (~120 MB). The GBR footprint spans roughly 770 × 704 pixels — just
6% of the full disk. Without optimisation, opening even one variable in xarray streams the entire
120 MB; with the kerchunk approach used here, only the ~20 HDF5 chunks that overlap the GBR are
fetched (~3 MB per variable), a **40× data reduction**.

---

## How it works

```
S3 keys   s3://noaa-himawari9/AHI-L2-FLDK-Clouds/{YYYY}/{MM}/{DD}/{HHMM}/AHI-{PROD}_...nc
    │
    ▼  list_product_files()
Per-file S3 keys
    │
    ▼  build_references_parallel()   ──►  cached JSON files (~2 MB each)
Kerchunk refs   {"CloudMask/5.3": ["s3://...", offset, length], ...}
    │
    ▼  build_combined_reference()    ──►  cached Parquet store dir
Virtual Parquet store   combined/{product}_{start}_{end}/ (*.parq per variable)
    │
    ▼  open_virtual_dataset()  +  subset_to_gbr()
Lazy xr.Dataset   dims: (time=N, Rows=1170, Columns=1104)
    │
    ▼  .compute()   ← only now does S3 I/O happen (~3 MB per variable)
In-memory arrays
```

### Kerchunk reference files

A kerchunk JSON reference maps every HDF5 chunk in a NetCDF4 file to a
`[url, byte_offset, length]` triple. `fsspec` translates these into HTTP Range requests
so xarray only downloads the chunks that intersect the GBR bounding box — regardless of
whether a single time step or a full day is being read.

### Multi-file time-series via Parquet store

For a time series, `MultiZarrToZarr` concatenates N per-file references along a new `time`
dimension. The combined chunk manifest is written as a **Parquet store** (one `.parq` file per
variable) so that chunk references for large collections can be read lazily without loading the
entire manifest into RAM. Latitude and Longitude arrays (identical across files) are stored once.

### GBR pixel bounds

The pre-computed constants cover 142–155 °E, 10–25 °S with a one-HDF5-chunk (200 px) buffer:

```python
GBR_ROW_SLICE = slice(3095, 4265)   # 1170 rows
GBR_COL_SLICE = slice(2615, 3719)   # 1104 cols
```

These were derived once from the embedded `Latitude`/`Longitude` arrays and are valid for all
Himawari-9 files (the geostationary projection is fixed). A helper `compute_pixel_bounds()` is
provided for custom regions.

---

## Installation

Requires Python ≥ 3.13. Install with [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:diodon/himawari-cloud.git
cd himawari-cloud

# Core pipeline only
uv sync

# Core + JupyterLab (to run the workflow notebook)
uv sync --group notebook
uv run jupyter lab
```

Or with pip:

```bash
pip install -e .
pip install jupyterlab   # to run the workflow notebook
```

No AWS credentials are required — the NOAA Himawari-9 bucket is publicly readable.

---

## Quick start

### High-level wrapper

```python
from datetime import datetime
from himawari_gbr.access import load_gbr_cloud_data

ds = load_gbr_cloud_data(
    products=["CMSK", "CHGT"],
    start_dt=datetime(2024, 1, 1,  0, 0),
    end_dt  =datetime(2024, 1, 1,  0, 50),  # 6 time steps
)
# ds is fully lazy — dims: (time=6, Rows=1170, Columns=1104)

cloud_mask = ds["CloudMask"].compute()     # downloads ~3 MB of GBR chunks
cloud_top  = ds["CldTopHght"].compute()    # metres, NaN where clear
```

`load_gbr_cloud_data()` runs the full pipeline and caches everything under
`~/.cache/himawari-gbr`. Subsequent calls for the same time window return instantly.

### Step-by-step API

```python
from pathlib import Path
from datetime import datetime
from himawari_gbr.access import (
    list_product_files,
    build_references_parallel,
    build_combined_reference,
    open_virtual_dataset,
    subset_to_gbr,
)

CACHE = Path("~/.cache/himawari-gbr").expanduser()
START = datetime(2024, 1, 1, 0, 0)
END   = datetime(2024, 1, 1, 0, 50)

# 1. List files on S3
keys = list_product_files("CMSK", START, END)           # 6 keys

# 2. Generate kerchunk JSON references (parallel, cached)
refs = build_references_parallel(keys, CACHE, max_workers=4)

# 3. Combine into a Parquet virtual store
store = CACHE / "combined" / "CMSK_demo"
build_combined_reference(refs, keys, store)             # cached on disk

# 4. Open as a lazy xarray Dataset
ds = open_virtual_dataset(store, variables=["CloudMask", "CloudProbability"])

# 5. Subset to GBR before computing (fetches only ~36 chunks per variable)
ds_gbr = subset_to_gbr(ds)
cm = ds_gbr["CloudMask"].compute()
```

---

## API reference

### `list_product_files(product, start_dt, end_dt)`
List all AHI L2 S3 keys for a product and UTC time window. Iterates 10-minute slots.

### `build_references_parallel(s3_keys, cache_dir, max_workers=8)`
Generate (or load cached) kerchunk JSON references for multiple files using a thread pool.
Each worker uses its own S3 connection; recommend ≤ 8 workers to avoid throttling.

### `build_combined_reference(ref_paths, s3_keys, output_dir)`
Combine per-file references into a Parquet virtual store via `MultiZarrToZarr`.
Injects a `time` coordinate (nanoseconds since epoch) from each filename timestamp.
`Latitude` and `Longitude` are stored once as `identical_dims`.

### `open_virtual_dataset(combined_ref, variables=None)`
Open a Parquet store as a Dask-backed lazy `xr.Dataset`. No S3 I/O occurs until `.compute()`.

### `open_single_file(s3_key, cache_dir, variables=None)`
Open a single NetCDF4 file via kerchunk without building a combined store. Useful for
quick inspection.

### `subset_to_gbr(ds, row_slice=GBR_ROW_SLICE, col_slice=GBR_COL_SLICE)`
Slice a full-disk Dataset to the GBR bounding box using `.isel()`. Must be called **before**
`.compute()` to avoid downloading the full 5500 × 5500 array.

### `compute_pixel_bounds(s3_key, cache_dir, lon_range, lat_range, buffer_chunks=1)`
One-time utility to derive pixel bounds for a custom geographic region from the embedded
`Latitude`/`Longitude` arrays.

### `load_gbr_cloud_data(products, start_dt, end_dt, cache_dir=None, ...)`
Full-pipeline convenience wrapper. Runs all steps above, merges multiple products along
shared dimensions, and returns a lazy GBR-subsetted Dataset.

---

## Variables

| Product | Variable | Description | Dtype |
|---------|----------|-------------|-------|
| CMSK | `CloudMask` | 0 = clear, 1 = prob. clear, 2 = prob. cloudy, 3 = cloudy | int8 |
| CMSK | `CloudProbability` | Bayesian cloud probability [0–1] | float32 |
| CMSK | `CloudMaskQualFlag` | Quality flag | int8 |
| CHGT | `CldTopHght` | Cloud-top height (m above sea level) | float32 |
| CHGT | `CldTopTemp` | Cloud-top temperature (K) | float32 |
| CHGT | `CldTopPres` | Cloud-top pressure (hPa) | float32 |
| CHGT | `CloudHgtQF` | Quality flag | int8 |
| CPHS | `CloudPhase` | Cloud phase (ice / liquid / mixed / unknown) | int8 |
| CPHS | `CloudType` | Cloud type classification | int8 |
| All | `Latitude` | Pixel latitude (°N) — 2-D, identical across time | float32 |
| All | `Longitude` | Pixel longitude (°E) — 2-D, identical across time | float32 |

Fill values (off-disk / no-retrieval pixels) are encoded as `NaN` after `mask_and_scale=True`
(the xarray default used here).

---

## Cache layout

```
~/.cache/himawari-gbr/
├── refs/
│   └── AHI-L2-FLDK-Clouds/2024/01/01/0000/
│       ├── AHI-CMSK_v1r1_h09_s202401010000...json   (~2 MB per file)
│       └── ...
└── combined/
    └── CMSK_20240101T0000_20240101T0050/
        ├── .complete                                  (sentinel: build complete)
        ├── CloudMask/refs.0.parq
        ├── CloudProbability/refs.0.parq
        └── ...
```

Kerchunk JSON references (~2 MB each) are reused across different time-window queries.
Combined Parquet stores are keyed by product and time window; use `force_rebuild=True` to
regenerate.

---

## Known issues and fixes

**zarr ≥ 2.17 / kerchunk 0.2.x fill-value incompatibility**
HDF5 files store `_FillValue` attributes as 1-element NumPy arrays. zarr 2.17+
`encode_fill_value()` requires a Python scalar and silently skips affected variables.
`access.py` applies a one-line monkey-patch at import time:
```python
zarr.meta.encode_fill_value = lambda v, dtype, oc=None: orig(
    v.flat[0] if isinstance(v, np.ndarray) else v, dtype, oc
)
```

**zarr 3.x incompatibility**
`pyproject.toml` pins `zarr>=2.17.0,<3.0`. zarr 3 restructured its internals and is
not yet compatible with the kerchunk reference filesystem interface used here.

---

## Workflow notebook

`himawari_gbr_workflow.ipynb` walks through the entire pipeline with detailed explanations,
including:

- S3 path structure and file naming conventions
- Kerchunk reference internals (chunk key format, fill-value patch)
- GBR pixel bounds derivation and chunk reduction math
- Parallel reference generation and Parquet store building
- Multi-product merge and time-series analysis
- Cloud fraction time-series and cloud-top height scatter plots
- Cache management and performance tips

---

## Dependencies

| Package | Role |
|---------|------|
| `kerchunk` | HDF5→zarr reference generation, `MultiZarrToZarr` combine |
| `zarr` | Virtual store backend (pinned < 3.0) |
| `fsspec` / `s3fs` | S3 anonymous access, `ReferenceFileSystem` |
| `xarray` | Lazy Dataset interface |
| `dask` | Deferred chunk computation |
| `fastparquet` | Reading/writing Parquet chunk manifests |
| `numpy` / `h5py` / `h5netcdf` | Array handling, HDF5 I/O |

---

## Licence

MIT
