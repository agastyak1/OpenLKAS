#!/usr/bin/env python3
"""
Model download utility for OpenLCWS Forward Collision Warning.
Downloads MobileNet-SSD v2 Caffe model files.
"""

import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_FILES = {
    "MobileNetSSD_deploy.prototxt": {
        "url": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    },
    "MobileNetSSD_deploy.caffemodel": {
        "url": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
    },
}


def download_file(url: str, dest: str):
    """Download a file with progress indication."""
    logger.info(f"Downloading {os.path.basename(dest)}...")
    logger.info(f"  URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def download_models(force: bool = False):
    """
    Download all required model files.

    Args:
        force: Re-download even if files already exist.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_ok = True
    for filename, info in MODEL_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)

        if os.path.exists(dest) and not force:
            logger.info(f"✓ {filename} already exists. Use --force to re-download.")
            continue

        try:
            download_file(info["url"], dest)
            size = os.path.getsize(dest)
            logger.info(f"✓ {filename} downloaded ({size / (1024*1024):.1f} MB)")
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            all_ok = False

    if all_ok:
        logger.info("\n🎉 All model files ready!")
    else:
        logger.error("\n❌ Some downloads failed. Please retry.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FCW model files for OpenLCWS")
    parser.add_argument("--force", action="store_true", help="Force re-download of all files")
    args = parser.parse_args()
    download_models(force=args.force)
