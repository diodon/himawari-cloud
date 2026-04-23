"""
Efficient access to Himawari-9 AHI L2 cloud products for the Great Barrier Reef.

Pipeline
--------
1. :func:`list_product_files`        – enumerate CMSK/CHGT/CPHS S3 keys for a date range
2. :func:`build_references_parallel` – generate per-file kerchunk JSON references (cached)
3. :func:`build_combined_reference`  – merge refs into a Parquet virtual dataset via MultiZarrToZarr
4. :func:`open_virtual_dataset`      – open the Parquet ref as a lazy xarray Dataset
5. :func:`subset_to_gbr`             – slice to GBR pixel bounding box before .compute()

Convenience wrapper
-------------------
:func:`load_gbr_cloud_data` runs the full pipeline and returns a ready-to-use Dataset.

Notes
-----
The NOAA Himawari-9 bucket (``s3://noaa-himawari9``) is publicly readable without
AWS credentials.  All S3 access is done anonymously.

The files store 5500 × 5500 full-disk arrays with HDF5 chunk size 200 × 200.
With kerchunk we fetch only the ~20 GBR-region chunks per variable (~3 MB) instead
of the full 121 MB array.

A compatibility patch is applied at import time to fix a zarr 2.17+ / kerchunk 0.2.x
incompatibility that caused HDF5 variables with numpy-array fill values to be silently
skipped during reference generation.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import fsspec
import numpy as np
import s3fs
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility patch – zarr ≥2.17 vs kerchunk 0.2.x fill-value encoding
# ---------------------------------------------------------------------------
# HDF5 attributes store _FillValue as 1-element numpy arrays.  zarr 2.17+
# expects a Python scalar in encode_fill_value(), raising TypeError and
# causing kerchunk to silently skip the variable.  We patch the function once
# at import time so every subsequent SingleHdf5ToZarr call works correctly.

import zarr.meta as _zmeta

_orig_encode_fill_value = _zmeta.encode_fill_value


def _patched_encode_fill_value(v, dtype, object_codec=None):
    if isinstance(v, np.ndarray):
        v = v.flat[0]
    return _orig_encode_fill_value(v, dtype, object_codec)


_zmeta.encode_fill_value = _patched_encode_fill_value

# ---------------------------------------------------------------------------
# Types and constants
# ---------------------------------------------------------------------------

Product = Literal["CMSK", "CHGT", "CPHS"]

BUCKET = "noaa-himawari9"
PREFIX = "AHI-L2-FLDK-Clouds"
ANON: dict = {"anon": True}

# GBR geographic bounding box (degrees)
GBR_LON: tuple[float, float] = (142.0, 155.0)  # °E
GBR_LAT: tuple[float, float] = (-25.0, -10.0)  # °N (negative = south)

# Pre-computed pixel bounds for the 5500 × 5500 full-disk grid.
# Centre of the disk (row 2750, col 2750) corresponds to SSP 140.71 °E, ~0 °N.
# Derived by checking the embedded Latitude/Longitude arrays; padded by one
# 200-pixel HDF5 chunk on each edge to avoid edge effects.
GBR_ROW_SLICE: slice = slice(3095, 4265)  # 1170 rows (770 GBR + 2 × 200 buffer)
GBR_COL_SLICE: slice = slice(2615, 3719)  # 1104 cols (704 GBR + 2 × 200 buffer)

# Key 2D scientific variables per product (AWIPS projections excluded)
PRODUCT_DATA_VARS: dict[str, list[str]] = {
    "CMSK": ["CloudMask", "CloudProbability", "CloudMaskQualFlag"],
    "CHGT": ["CldTopHght", "CldTopTemp", "CldTopPres", "CloudHgtQF"],
    "CPHS": ["CloudPhase", "CloudType"],
}

# Lat/lon coordinate variables present in every product
COORD_VARS: list[str] = ["Latitude", "Longitude"]

# Dimensions shared across all files (kerchunk identical_dims)
_IDENTICAL_DIMS: list[str] = ["Latitude", "Longitude"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_fs() -> s3fs.S3FileSystem:
    """Return an anonymous S3 filesystem configured for noaa-himawari9."""
    return s3fs.S3FileSystem(anon=True, default_block_size=2**22)  # 4 MB read-ahead


def _parse_start_time(s3_key: str) -> np.datetime64:
    """Extract scan-start time from a Himawari filename.

    Filename convention::

        AHI-CMSK_v1r1_h09_s<YYYYMMDDHHMMSSS>_e..._c....nc
                             └── 13-char start timestamp
    """
    m = re.search(r"_s(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})\d{3}", s3_key)
    if not m:
        raise ValueError(f"Cannot parse start time from: {s3_key!r}")
    yr, mo, dy, hr, mn = m.groups()
    return np.datetime64(f"{yr}-{mo}-{dy}T{hr}:{mn}:00", "ns")


def _ref_cache_path(s3_key: str, cache_dir: Path) -> Path:
    """Return the local JSON cache path for a given S3 key."""
    rel = s3_key.removeprefix(f"{BUCKET}/")
    return cache_dir / "refs" / Path(rel).with_suffix(".json")


def _expand_refs_for_time(
    refs: dict,
    s3_key: str,
    skip_time_expand: frozenset[str] = frozenset(),
) -> dict:
    """Transform a per-file kerchunk reference so all (Rows, Columns) arrays gain
    a leading ``time`` dimension of size 1, ready for MultiZarrToZarr concat.

    Variables listed in *skip_time_expand* (typically ``identical_dims`` like
    ``Latitude`` and ``Longitude``) are left as-is so that ``MultiZarrToZarr``
    can copy them verbatim from the first file without a dimension-size conflict.

    This modifies:

    * ``.zarray`` – prepends 1 to ``shape`` and ``chunks``
    * ``.zattrs`` – prepends ``"time"`` to ``_ARRAY_DIMENSIONS``
    * chunk keys  – renames ``<var>/i.j`` → ``<var>/0.i.j``

    A scalar ``time`` variable (dtype ``int64``, nanoseconds since epoch) is also
    injected as inline data so MultiZarrToZarr can read the coordinate values.
    """
    t_ns: int = int(_parse_start_time(s3_key).astype("int64"))

    new_refs: dict = {"version": refs.get("version", 1), "refs": {}}
    old = refs["refs"]

    # ── Passthrough: non-variable top-level keys ──────────────────────────
    for key in (".zgroup", ".zattrs"):
        if key in old:
            new_refs["refs"][key] = old[key]

    # ── Inject scalar time variable ──────────────────────────────────────
    new_refs["refs"]["time/.zarray"] = json.dumps(
        {
            "chunks": [1],
            "compressor": None,
            "dtype": "<i8",
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "shape": [1],
            "zarr_format": 2,
        }
    )
    new_refs["refs"]["time/.zattrs"] = json.dumps(
        {
            "_ARRAY_DIMENSIONS": ["time"],
            "calendar": "proleptic_gregorian",
            "units": "nanoseconds since 1970-01-01",
        }
    )
    new_refs["refs"]["time/0"] = (
        "base64:" + base64.b64encode(np.int64(t_ns).tobytes()).decode()
    )

    # ── Collect variable names ────────────────────────────────────────────
    var_names: set[str] = {
        k.split("/")[0]
        for k in old
        if "/" in k and not k.startswith(".")
    }
    var_names.discard("time")

    # ── Per-variable transformation ───────────────────────────────────────
    for var in var_names:
        zarr_key = f"{var}/.zarray"
        attr_key = f"{var}/.zattrs"

        if zarr_key not in old:
            # Passthrough unknown entries unchanged
            for k, v in old.items():
                if k.startswith(f"{var}/"):
                    new_refs["refs"][k] = v
            continue

        zarray = json.loads(old[zarr_key])
        zattrs = json.loads(old.get(attr_key, "{}"))
        dims: list[str] = zattrs.get("_ARRAY_DIMENSIONS", [])

        if len(dims) == 2 and dims == ["Rows", "Columns"] and var not in skip_time_expand:
            # Expand: prepend time dimension to shape, chunks, and dims
            new_zarray = dict(zarray)
            new_zarray["shape"] = [1] + zarray["shape"]
            new_zarray["chunks"] = [1] + zarray["chunks"]
            new_refs["refs"][zarr_key] = json.dumps(new_zarray)

            new_zattrs = dict(zattrs)
            new_zattrs["_ARRAY_DIMENSIONS"] = ["time"] + dims
            new_refs["refs"][attr_key] = json.dumps(new_zattrs)

            # Rename chunk keys: '<var>/i.j' → '<var>/0.i.j'
            for k, v in old.items():
                if k.startswith(f"{var}/") and k not in (zarr_key, attr_key):
                    chunk_part = k[len(var) + 1 :]
                    new_refs["refs"][f"{var}/0.{chunk_part}"] = v
        else:
            # Copy scalar / 1D / non-spatial variables unchanged
            for k, v in old.items():
                if k.startswith(f"{var}/"):
                    new_refs["refs"][k] = v

    return new_refs


# ---------------------------------------------------------------------------
# Step 1 – List S3 files
# ---------------------------------------------------------------------------


def list_product_files(
    product: Product,
    start_dt: datetime,
    end_dt: datetime,
    fs: s3fs.S3FileSystem | None = None,
) -> list[str]:
    """List all AHI L2 S3 keys for *product* between *start_dt* and *end_dt*.

    Himawari-9 observes at 10-minute cadence.  The function iterates over every
    10-minute slot in the requested window and lists matching files.

    Parameters
    ----------
    product:
        ``"CMSK"`` (cloud mask), ``"CHGT"`` (cloud height), or ``"CPHS"``
        (cloud phase).
    start_dt, end_dt:
        UTC datetimes (inclusive on both ends).
    fs:
        Optional pre-built :class:`s3fs.S3FileSystem`.  A new anonymous
        instance is created if omitted.

    Returns
    -------
    list[str]
        S3 keys (without the ``s3://`` scheme prefix) sorted chronologically.
    """
    if fs is None:
        fs = _make_fs()

    # Snap start time back to the nearest 10-minute boundary
    snap_start = start_dt.replace(
        minute=(start_dt.minute // 10) * 10, second=0, microsecond=0
    )

    paths: list[str] = []
    current = snap_start
    while current <= end_dt:
        slot = current.strftime("%H%M")
        day_prefix = f"{BUCKET}/{PREFIX}/{current:%Y/%m/%d}/{slot}/"
        try:
            for entry in fs.ls(day_prefix, detail=False):
                fname = entry.split("/")[-1]
                if f"AHI-{product}_" in fname and fname.endswith(".nc"):
                    paths.append(entry)
        except FileNotFoundError:
            logger.debug("No files at %s", day_prefix)
        current += timedelta(minutes=10)

    return sorted(paths)


# ---------------------------------------------------------------------------
# Step 2 – Per-file kerchunk reference generation
# ---------------------------------------------------------------------------


def generate_reference(
    s3_key: str,
    cache_dir: Path,
    fs: s3fs.S3FileSystem | None = None,
    force: bool = False,
) -> Path:
    """Generate (or load cached) a kerchunk JSON reference for a single NetCDF file.

    The reference is a JSON dictionary that maps every HDF5/NetCDF chunk to its
    byte range in the S3 object, allowing chunk-level access without downloading
    the full file.

    Parameters
    ----------
    s3_key:
        S3 key without the ``s3://`` prefix.
    cache_dir:
        Root directory for caching.  References are stored under
        ``{cache_dir}/refs/…`` mirroring the S3 path structure.
    fs:
        Optional anonymous S3 filesystem.
    force:
        Regenerate even when a cached file already exists.

    Returns
    -------
    Path
        Local path to the cached JSON reference.
    """
    from kerchunk.hdf import SingleHdf5ToZarr

    ref_path = _ref_cache_path(s3_key, cache_dir)
    if ref_path.exists() and not force:
        return ref_path

    ref_path.parent.mkdir(parents=True, exist_ok=True)

    if fs is None:
        fs = _make_fs()

    url = f"s3://{s3_key}"
    logger.debug("Generating reference for %s", s3_key.split("/")[-1])

    with fs.open(s3_key, "rb") as fobj:
        h5chunks = SingleHdf5ToZarr(fobj, url, inline_threshold=100)
        refs = h5chunks.translate()

    with open(ref_path, "w") as out:
        json.dump(refs, out, separators=(",", ":"))

    logger.info("Cached: %s", ref_path.name)
    return ref_path


def build_references_parallel(
    s3_keys: list[str],
    cache_dir: Path,
    max_workers: int = 8,
    force: bool = False,
) -> list[Path]:
    """Generate kerchunk JSON references for multiple files in parallel.

    Thread-safe: each worker uses its own S3FileSystem connection.

    Parameters
    ----------
    s3_keys:
        Ordered list of S3 keys to process.
    cache_dir:
        Local cache directory root.
    max_workers:
        Maximum parallel threads.  Recommend ≤8 to avoid S3 throttling.
    force:
        Regenerate even when cached files exist.

    Returns
    -------
    list[Path]
        Reference file paths in the same order as *s3_keys*.
    """
    # Check which files are already cached to avoid spawning unnecessary threads
    uncached = [k for k in s3_keys if force or not _ref_cache_path(k, cache_dir).exists()]
    logger.info("%d/%d references need generating", len(uncached), len(s3_keys))

    if uncached:
        fs = _make_fs()
        errors: dict[str, Exception] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(generate_reference, key, cache_dir, fs, force): key
                for key in uncached
            }
            for future in as_completed(futures):
                key = futures[future]
                exc = future.exception()
                if exc is not None:
                    logger.error("Failed to generate reference for %s: %s", key, exc)
                    errors[key] = exc

        if errors:
            raise RuntimeError(
                f"Reference generation failed for {len(errors)} file(s): "
                + ", ".join(errors)
            )

    return [_ref_cache_path(k, cache_dir) for k in s3_keys]


# ---------------------------------------------------------------------------
# Step 3 – Combine references into a Parquet virtual dataset
# ---------------------------------------------------------------------------


def build_combined_reference(
    ref_paths: list[Path],
    s3_keys: list[str],
    output_dir: Path,
    identical_vars: list[str] | None = None,
) -> Path:
    """Combine per-file kerchunk references into a Parquet virtual dataset store.

    Uses :class:`kerchunk.combine.MultiZarrToZarr` to concatenate along a new
    ``time`` dimension.  Before combining, each per-file reference is expanded
    by :func:`_expand_refs_for_time` so that (Rows, Columns) spatial arrays gain
    a leading ``time`` dimension of size 1.

    The result is written via :func:`kerchunk.df.refs_to_dataframe` to a
    directory of Parquet files (one per zarr variable).  This scales to thousands
    of files without loading all chunk references into memory at once.

    Parameters
    ----------
    ref_paths:
        Per-file JSON reference paths, in chronological order.
    s3_keys:
        Corresponding S3 keys (used to extract timestamps).
    output_dir:
        Output directory for the Parquet store.  Will be created if missing.
        Pass the same path back to :func:`open_virtual_dataset` to open the result.
    identical_vars:
        Variables that are identical across all files (only first file's chunks
        are retained).  Defaults to ``["Latitude", "Longitude"]``.

    Returns
    -------
    Path
        Path to the written Parquet store directory.
    """
    from kerchunk.combine import MultiZarrToZarr
    from kerchunk.df import refs_to_dataframe

    if identical_vars is None:
        identical_vars = list(_IDENTICAL_DIMS)

    if len(ref_paths) != len(s3_keys):
        raise ValueError("ref_paths and s3_keys must have the same length")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Expanding %d references for time concatenation…", len(ref_paths))
    skip = frozenset(identical_vars)
    expanded: list[dict] = []
    for ref_path, key in zip(ref_paths, s3_keys):
        with open(ref_path) as fh:
            refs = json.load(fh)
        expanded.append(_expand_refs_for_time(refs, key, skip_time_expand=skip))

    logger.info("Running MultiZarrToZarr over %d files…", len(expanded))
    mzz = MultiZarrToZarr(
        expanded,
        remote_protocol="s3",
        remote_options=ANON,
        concat_dims=["time"],
        identical_dims=identical_vars,
        coo_dtypes={"time": "M8[ns]"},
    )
    # translate() with no filename returns the combined reference dict
    combined = mzz.translate()

    # Write to Parquet store (directory of per-variable .parquet files)
    refs_to_dataframe(combined, str(output_dir))

    logger.info("Written Parquet store: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Step 4 – Open virtual dataset and subset to GBR
# ---------------------------------------------------------------------------


def open_virtual_dataset(
    combined_ref: Path,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Open a combined Parquet virtual store as a lazy, Dask-backed xarray Dataset.

    No data is downloaded from S3 until ``.compute()`` is called on a variable.

    Parameters
    ----------
    combined_ref:
        Path to the Parquet store **directory** produced by
        :func:`build_combined_reference`.
    variables:
        If given, restrict the dataset to these variable names.

    Returns
    -------
    xr.Dataset
        Fully lazy Dataset.  Dimensions: ``(time, Rows, Columns)``.
    """
    from fsspec.implementations.reference import LazyReferenceMapper, ReferenceFileSystem

    combined_ref = Path(combined_ref)
    local_fs = fsspec.filesystem("file")

    # LazyReferenceMapper reads chunk pointers from the Parquet store.
    # Wrapping in ReferenceFileSystem gives it the remote_protocol/options
    # it needs to actually fetch chunk byte ranges from S3.
    lazy_mapper = LazyReferenceMapper(str(combined_ref), fs=local_fs)
    ref_fs = ReferenceFileSystem(
        fo=lazy_mapper,
        remote_protocol="s3",
        remote_options=ANON,
        target_protocol="file",
    )
    mapper = ref_fs.get_mapper("")

    ds = xr.open_dataset(
        mapper,
        engine="zarr",
        chunks={},             # Dask-backed lazy arrays
        consolidated=False,
        mask_and_scale=True,
    )
    if variables:
        available = [v for v in variables if v in ds]
        missing = set(variables) - set(available)
        if missing:
            logger.warning("Variables not found in dataset: %s", missing)
        ds = ds[available]
    return ds


def open_single_file(
    s3_key: str,
    cache_dir: Path,
    variables: list[str] | None = None,
    fs: s3fs.S3FileSystem | None = None,
) -> xr.Dataset:
    """Open a single Himawari file as a lazy xarray Dataset via kerchunk.

    Generates (or loads cached) the kerchunk reference first, then opens it
    without building a combined Parquet reference.  Useful for quick inspection
    of individual files.

    Parameters
    ----------
    s3_key:
        S3 key for a single AHI L2 NetCDF file.
    cache_dir:
        Local cache directory for the JSON reference.
    variables:
        If given, restrict to these variable names.
    fs:
        Optional S3 filesystem.

    Returns
    -------
    xr.Dataset
        Lazy 2-D (Rows × Columns) Dataset.
    """
    ref_path = generate_reference(s3_key, cache_dir, fs=fs)

    mapper = fsspec.get_mapper(
        "reference://",
        fo=str(ref_path),
        remote_protocol="s3",
        remote_options=ANON,
        target_protocol="file",
    )
    ds = xr.open_dataset(
        mapper,
        engine="zarr",
        chunks={},
        consolidated=False,
        mask_and_scale=True,
    )
    if variables:
        available = [v for v in variables if v in ds]
        ds = ds[available]
    return ds


def subset_to_gbr(
    ds: xr.Dataset,
    row_slice: slice = GBR_ROW_SLICE,
    col_slice: slice = GBR_COL_SLICE,
) -> xr.Dataset:
    """Slice a full-disk Dataset to the GBR pixel bounding box.

    Slicing happens **before** ``.compute()``, so only the ~20 GBR-region
    HDF5 chunks (≈3 MB per variable) are fetched from S3 instead of the
    full 5500 × 5500 array (≈121 MB per float32 variable).

    Parameters
    ----------
    ds:
        Lazy full-disk Dataset (from :func:`open_virtual_dataset` or
        :func:`open_single_file`).
    row_slice:
        Row index slice in the 5500-row full-disk grid.
        Default: :data:`GBR_ROW_SLICE`.
    col_slice:
        Column index slice in the 5500-column full-disk grid.
        Default: :data:`GBR_COL_SLICE`.

    Returns
    -------
    xr.Dataset
        Spatially subsetted lazy Dataset.  Trigger a download with
        ``ds.compute()`` or ``ds.load()``.

    Raises
    ------
    ValueError
        If the dataset does not contain the expected ``Rows`` / ``Columns``
        dimension names.
    """
    indexers: dict[str, slice] = {}
    if "Rows" in ds.dims:
        indexers["Rows"] = row_slice
    if "Columns" in ds.dims:
        indexers["Columns"] = col_slice

    if not indexers:
        raise ValueError(
            "Dataset lacks expected 'Rows'/'Columns' dimensions. "
            f"Found: {list(ds.dims)}"
        )
    return ds.isel(**indexers)


# ---------------------------------------------------------------------------
# Utility – derive pixel bounds from lat/lon (one-time setup helper)
# ---------------------------------------------------------------------------


def compute_pixel_bounds(
    s3_key: str,
    cache_dir: Path,
    lon_range: tuple[float, float] = GBR_LON,
    lat_range: tuple[float, float] = GBR_LAT,
    buffer_chunks: int = 1,
    fs: s3fs.S3FileSystem | None = None,
) -> dict[str, int]:
    """Compute GBR pixel row/column bounds by reading the embedded Lat/Lon arrays.

    This is a relatively expensive one-time operation (~10–15 MB downloaded for
    the latitude and longitude arrays of the GBR region).  The default
    :data:`GBR_ROW_SLICE` / :data:`GBR_COL_SLICE` constants in this module
    were derived with this function and should not need recomputing.

    Parameters
    ----------
    s3_key:
        Any AHI L2 NetCDF key (CMSK/CHGT/CPHS all carry the same geolocation).
    cache_dir:
        Cache directory for the kerchunk reference.
    lon_range, lat_range:
        Geographic bounding box in degrees.
    buffer_chunks:
        Number of 200-pixel HDF5 chunks to pad around the detected bounds.
    fs:
        Optional S3 filesystem.

    Returns
    -------
    dict
        Keys: ``row_min``, ``row_max``, ``col_min``, ``col_max`` (all int).
    """
    ds = open_single_file(s3_key, cache_dir, variables=["Latitude", "Longitude"], fs=fs)
    lat = ds["Latitude"].isel(Rows=slice(2500, 5000), Columns=slice(2000, 4000)).values
    lon = ds["Longitude"].isel(Rows=slice(2500, 5000), Columns=slice(2000, 4000)).values

    mask = (
        (lon >= lon_range[0])
        & (lon <= lon_range[1])
        & (lat >= lat_range[0])
        & (lat <= lat_range[1])
    )
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise RuntimeError(
            f"No pixels found within lon={lon_range}, lat={lat_range}. "
            "Check that the bounding box overlaps the Himawari-9 full disk."
        )

    chunk = 200 * buffer_chunks
    # Offsets for the pre-sliced sub-region (Rows 2500, Cols 2000)
    return {
        "row_min": max(0, int(rows.min()) + 2500 - chunk),
        "row_max": min(5500, int(rows.max()) + 2500 + chunk + 1),
        "col_min": max(0, int(cols.min()) + 2000 - chunk),
        "col_max": min(5500, int(cols.max()) + 2000 + chunk + 1),
    }


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def load_gbr_cloud_data(
    products: list[Product],
    start_dt: datetime,
    end_dt: datetime,
    cache_dir: Path | None = None,
    max_workers: int = 8,
    variables: list[str] | None = None,
    force_rebuild: bool = False,
) -> xr.Dataset:
    """Load GBR-subsetted Himawari-9 AHI L2 cloud data as a lazy xarray Dataset.

    Full pipeline:

    1. List matching S3 files for each product.
    2. Generate kerchunk JSON references in parallel (cached under *cache_dir*).
    3. Build a combined Parquet virtual dataset per product (cached).
    4. Open each product as a lazy xarray Dataset.
    5. Spatially subset to the GBR bounding box.
    6. Merge products along shared dimensions.

    After the first call (which populates the cache) the same time window can
    be re-opened without any S3 metadata requests.

    Parameters
    ----------
    products:
        List of one or more products, e.g. ``["CMSK", "CHGT", "CPHS"]``.
    start_dt, end_dt:
        UTC time window (inclusive).
    cache_dir:
        Root directory for kerchunk JSON references and Parquet combined
        references.  Defaults to ``~/.cache/himawari-gbr``.
    max_workers:
        Threads for parallel reference generation.
    variables:
        Explicit variable list.  If ``None``, the default scientific variables
        for each product are loaded (see :data:`PRODUCT_DATA_VARS`).
    force_rebuild:
        If ``True``, regenerate all cached files from scratch.

    Returns
    -------
    xr.Dataset
        Lazy, GBR-subsetted Dataset with dimensions
        ``(time, Rows, Columns)``.  Call ``.compute()`` on a variable to
        trigger S3 downloads (only the GBR chunks are fetched).

    Raises
    ------
    RuntimeError
        If no files are found for the requested products and time window.

    Examples
    --------
    >>> from datetime import datetime
    >>> from himawari_gbr.access import load_gbr_cloud_data
    >>> ds = load_gbr_cloud_data(
    ...     ["CMSK", "CHGT"],
    ...     start_dt=datetime(2024, 1, 1, 0, 0),
    ...     end_dt=datetime(2024, 1, 1, 1, 0),
    ... )
    >>> cloud_mask = ds["CloudMask"].compute()   # downloads ~3 MB of chunks
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "himawari-gbr"
    cache_dir = Path(cache_dir).expanduser().resolve()

    fs = _make_fs()
    product_datasets: list[xr.Dataset] = []

    for product in products:
        logger.info(
            "Processing %s  [%s – %s]",
            product,
            start_dt.isoformat(),
            end_dt.isoformat(),
        )

        # 1. List files
        keys = list_product_files(product, start_dt, end_dt, fs=fs)
        if not keys:
            logger.warning("No %s files found for requested window.", product)
            continue
        logger.info("Found %d %s files.", len(keys), product)

        # 2. Per-file references
        ref_paths = build_references_parallel(
            keys, cache_dir, max_workers=max_workers, force=force_rebuild
        )

        # 3. Combined Parquet store (directory)
        ts = start_dt.strftime("%Y%m%dT%H%M")
        te = end_dt.strftime("%Y%m%dT%H%M")
        store_dir = cache_dir / "combined" / f"{product}_{ts}_{te}"

        # The store sentinel file confirms a complete build
        sentinel = store_dir / ".complete"
        if not sentinel.exists() or force_rebuild:
            if store_dir.exists() and force_rebuild:
                import shutil
                shutil.rmtree(store_dir)
            build_combined_reference(ref_paths, keys, store_dir)
            sentinel.touch()

        # 4. Open virtual dataset
        load_vars = variables or (PRODUCT_DATA_VARS[product] + COORD_VARS)
        ds = open_virtual_dataset(store_dir, variables=load_vars)

        # 5. GBR spatial subset
        ds_gbr = subset_to_gbr(ds)
        product_datasets.append(ds_gbr)

    if not product_datasets:
        raise RuntimeError(
            "No data found for the requested products and time window. "
            f"products={products}, start={start_dt}, end={end_dt}"
        )

    if len(product_datasets) == 1:
        return product_datasets[0]

    # Merge multiple products that share (time, Rows, Columns)
    return xr.merge(product_datasets, join="inner")
