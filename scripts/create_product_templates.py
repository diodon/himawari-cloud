"""Generate one kerchunk JSON reference per AHI L2 product for use as a reusable template.

Run this once (or whenever a new software version is deployed by NOAA) to create
CMSK_template.json, CHGT_template.json, and CPHS_template.json in
<cache_dir>/templates/.

The pipeline reads these templates in build_references_parallel(templates_dir=...)
and derives per-file references by substituting the S3 URL instead of parsing the
HDF5 metadata of every file.  This reduces reference generation from O(N) S3 round-
trips to O(0) — milliseconds instead of minutes for any number of files.

Usage
-----
    python scripts/create_product_templates.py
    python scripts/create_product_templates.py --date 2024-06-01
    python scripts/create_product_templates.py --cache-dir /fast/ssd/cache

The script uses the 00:00 UTC file for the given date.  Any date with complete
data works; the templates are independent of time and valid for all files produced
by the same NOAA software version.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from himawari_gbr.access import BUCKET, _make_fs, generate_reference, list_product_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

PRODUCTS = ("CMSK", "CHGT", "CPHS")
DEFAULT_DATE = "2024-01-01"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "himawari-gbr"


def create_product_templates(
    date_str: str = DEFAULT_DATE,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Path]:
    """Generate and save a template reference JSON for each AHI L2 product.

    Parameters
    ----------
    date_str:
        Date in YYYY-MM-DD format.  The 00:00 UTC slot is used.
    cache_dir:
        Root cache directory.  Templates are saved to ``cache_dir/templates/``.

    Returns
    -------
    dict
        Maps product name to the written template path.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    templates_dir = cache_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    fs = _make_fs()
    results: dict[str, Path] = {}

    for product in PRODUCTS:
        logging.info("─── %s ───────────────────────────────────────────", product)

        keys = list_product_files(product, dt, dt, fs=fs)
        if not keys:
            logging.error("No %s files found for %s — skipping.", product, date_str)
            continue

        source_key = keys[0]
        logging.info("Source file: %s", source_key.split("/")[-1])

        ref_path = generate_reference(source_key, cache_dir, fs=fs)

        with open(ref_path) as fh:
            refs = json.load(fh)

        # Embed the source S3 key (without s3://) so the pipeline can identify
        # which URL to replace when deriving per-file references.
        refs["_template_source"] = source_key

        template_path = templates_dir / f"{product}_template.json"
        with open(template_path, "w") as fh:
            json.dump(refs, fh, separators=(",", ":"))

        size_kb = template_path.stat().st_size // 1024
        logging.info("Template written: %s  (%d KB)", template_path, size_kb)
        results[product] = template_path

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate kerchunk JSON templates for CMSK, CHGT, and CPHS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "After running this script, pass templates_dir to build_references_parallel:\n\n"
            "    from himawari_gbr.access import build_references_parallel\n"
            "    from pathlib import Path\n\n"
            "    TEMPLATES = Path('~/.cache/himawari-gbr/templates').expanduser()\n"
            "    refs = build_references_parallel(keys, cache_dir, templates_dir=TEMPLATES)\n"
        ),
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        metavar="YYYY-MM-DD",
        help=f"Date to fetch template files from (default: {DEFAULT_DATE})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        metavar="DIR",
        help=f"Root cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    results = create_product_templates(date_str=args.date, cache_dir=cache_dir)

    if results:
        print("\nTemplates ready:")
        for product, path in results.items():
            print(f"  {product}: {path}  ({path.stat().st_size // 1024} KB)")
        print(
            "\nTo use them, pass templates_dir to build_references_parallel() "
            "or load_gbr_cloud_data()."
        )
    else:
        print("No templates were created.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
