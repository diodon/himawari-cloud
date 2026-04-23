"""End-to-end test: list → reference → subset → compute for one 10-minute slot."""
import logging
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from himawari_gbr.access import (
    list_product_files,
    generate_reference,
    open_single_file,
    subset_to_gbr,
    GBR_ROW_SLICE,
    GBR_COL_SLICE,
)

CACHE = Path("/tmp/himawari-gbr-test-cache")
START = datetime(2024, 1, 1, 0, 0)
END = datetime(2024, 1, 1, 0, 0)  # single slot

print("=" * 60)
print("1. Listing CMSK files for 2024-01-01 00:00 UTC…")
keys = list_product_files("CMSK", START, END)
print(f"   Found {len(keys)} file(s):")
for k in keys:
    print(f"   {k.split('/')[-1]}")

print("\n2. Generating kerchunk reference…")
ref_path = generate_reference(keys[0], CACHE)
print(f"   Cached at: {ref_path}")

print("\n3. Opening as lazy xarray Dataset via ReferenceFileSystem…")
ds = open_single_file(keys[0], CACHE, variables=["CloudMask", "CloudProbability",
                                                   "CloudMaskQualFlag", "Latitude", "Longitude"])
print(f"   Variables: {list(ds.data_vars.keys())}")
print(f"   Dims: {dict(ds.dims)}")

print("\n4. Subsetting to GBR region…")
ds_gbr = subset_to_gbr(ds)
print(f"   CloudMask shape (lazy): {ds_gbr['CloudMask'].shape}")
print(f"   GBR_ROW_SLICE={GBR_ROW_SLICE}, GBR_COL_SLICE={GBR_COL_SLICE}")

print("\n5. Computing (downloads only GBR chunks from S3)…")
import time
t0 = time.perf_counter()
cm = ds_gbr["CloudMask"].compute().values
t1 = time.perf_counter()

import numpy as np
valid = cm[cm >= 0]
print(f"   Elapsed: {t1-t0:.1f}s")
print(f"   CloudMask shape: {cm.shape}")
print(f"   Flag values: {dict(zip(*np.unique(valid, return_counts=True)))}")
print(f"   Cloud fraction: {(valid >= 2).mean():.1%}")

lat = ds_gbr["Latitude"].values
lon = ds_gbr["Longitude"].values
valid_lat = lat[lat > -900]
valid_lon = lon[lon > -900]
print(f"   Lat range: {valid_lat.min():.2f}°N – {valid_lat.max():.2f}°N")
print(f"   Lon range: {valid_lon.min():.2f}°E – {valid_lon.max():.2f}°E")

print("\n✓ End-to-end test PASSED")
