"""Test MultiZarrToZarr time-concat pipeline with 2 CMSK files."""
import logging
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from himawari_gbr.access import (
    list_product_files,
    build_references_parallel,
    build_combined_reference,
    open_virtual_dataset,
    subset_to_gbr,
    _expand_refs_for_time,
    _parse_start_time,
)
import json
import numpy as np

CACHE = Path("/tmp/himawari-gbr-test-cache")
START = datetime(2024, 1, 1, 0, 0)
END = datetime(2024, 1, 1, 0, 10)  # 2 time slots

print("1. Listing CMSK files (2 slots)…")
keys = list_product_files("CMSK", START, END)
print(f"   Found {len(keys)} files")
for k in keys:
    print(f"     {k.split('/')[-1]}")
    print(f"     → time: {_parse_start_time(k)}")

print("\n2. Building references…")
ref_paths = build_references_parallel(keys, CACHE, max_workers=2)
print(f"   References: {[str(p.name) for p in ref_paths]}")

print("\n3. Testing _expand_refs_for_time on first file…")
with open(ref_paths[0]) as f:
    orig_refs = json.load(f)
expanded = _expand_refs_for_time(orig_refs, keys[0])

# Check structure
cm_zarray = json.loads(expanded["refs"]["CloudMask/.zarray"])
cm_zattrs = json.loads(expanded["refs"]["CloudMask/.zattrs"])
print(f"   CloudMask shape after expansion: {cm_zarray['shape']}")
print(f"   CloudMask dims after expansion:  {cm_zattrs['_ARRAY_DIMENSIONS']}")
print(f"   CloudMask chunks after expansion:{cm_zarray['chunks']}")

# Check time variable
time_zarray = json.loads(expanded["refs"]["time/.zarray"])
time_val_b64 = expanded["refs"]["time/0"]
time_bytes = __import__("base64").b64decode(time_val_b64.replace("base64:", ""))
time_ns = int(np.frombuffer(time_bytes, dtype="<i8")[0])
print(f"   Time value (ns): {time_ns} → {np.datetime64(time_ns, 'ns')}")

# Check chunk key renaming
orig_chunks = [k for k in orig_refs["refs"] if k.startswith("CloudMask/")
               and k not in ("CloudMask/.zarray", "CloudMask/.zattrs")]
exp_chunks  = [k for k in expanded["refs"] if k.startswith("CloudMask/")
               and k not in ("CloudMask/.zarray", "CloudMask/.zattrs")]
print(f"   Chunk keys before: {orig_chunks[:3]} … ({len(orig_chunks)} total)")
print(f"   Chunk keys after:  {exp_chunks[:3]} … ({len(exp_chunks)} total)")

print("\n4. Building combined Parquet reference (MultiZarrToZarr)…")
store_dir = CACHE / "combined" / "CMSK_test_2files"
store_dir.parent.mkdir(parents=True, exist_ok=True)
build_combined_reference(ref_paths, keys, store_dir)
# Count parquet files in store
pfiles = list(store_dir.rglob("*.parq*"))
print(f"   Written: {store_dir}  ({len(pfiles)} .parq files)")

print("\n5. Opening combined dataset…")
ds = open_virtual_dataset(store_dir, variables=["CloudMask", "Latitude", "Longitude"])
print(f"   Variables: {list(ds.data_vars.keys())}")
print(f"   Dims: {dict(ds.dims)}")
print(f"   time: {ds['time'].values if 'time' in ds else 'NOT FOUND'}")

print("\n6. GBR subset + compute first time step…")
ds_gbr = subset_to_gbr(ds)
print(f"   CloudMask shape (lazy, 2 time steps): {ds_gbr['CloudMask'].shape}")
cm0 = ds_gbr["CloudMask"].isel(time=0).values
cm1 = ds_gbr["CloudMask"].isel(time=1).values
print(f"   t=0 cloud fraction: {(cm0[cm0>=0] >= 2).mean():.1%}")
print(f"   t=1 cloud fraction: {(cm1[cm1>=0] >= 2).mean():.1%}")

print("\n✓ Combine + time-series pipeline PASSED")
