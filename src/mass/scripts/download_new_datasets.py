#!/usr/bin/env python3
"""Download helper for the datasets that are not yet part of the n20 benchmark."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError as exc:  # pragma: no cover - keep runtime friendly
    raise SystemExit(
        "This script requires the `requests` package. "
        "Install it with `pip install requests` and run again."
    ) from exc


# Keep the metadata close to this script so it's easy to audit or extend later on.
DATASETS: List[Dict[str, str]] = [
    {
        "name": "Vegetables",
        "type": "kaggle_dataset",
        "identifier": "misrakahmed/vegetable-image-dataset",
        "homepage": "https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset",
    },
    {
        "name": "Kvasir-V2",
        "type": "kaggle_dataset",
        "identifier": "yasserhessein/the-kvasir-dataset",
        "homepage": "https://www.kaggle.com/datasets/yasserhessein/the-kvasir-dataset",
    },
    {
        "name": "Intel-Images",
        "type": "kaggle_dataset",
        "identifier": "puneet6060/intel-image-classification",
        "homepage": "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    },
    {
        "name": "Weather",
        "type": "http",
        "url": "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/M8JQCR",
        "filename": "weather-dataset.zip",
        "homepage": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M8JQCR",
    },
    {
        "name": "Cats & Dogs",
        "type": "kaggle_competition",
        "identifier": "dogs-vs-cats",
        "homepage": "https://www.kaggle.com/competitions/dogs-vs-cats",
    },
    {
        "name": "MangoLeafBD",
        "type": "http",
        "url": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hxsnvwty3r-1.zip",
        "homepage": "https://data.mendeley.com/datasets/hxsnvwty3r/1",
    },
    {
        "name": "beans",
        "type": "huggingface",
        "identifier": "AI-Lab-Makerere/beans",
        "homepage": "https://huggingface.co/datasets/AI-Lab-Makerere/beans",
    },
    {
        "name": "Dogs",
        "type": "http",
        "url": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        "homepage": "http://vision.stanford.edu/aditya86/ImageNetDogs/",
    },
    {
        "name": "Landscape",
        "type": "kaggle_dataset",
        "identifier": "utkarshsaxenadn/landscape-recognition-image-dataset-12k-images",
        "homepage": "https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images",
    },
    {
        "name": "Flowers",
        "type": "kaggle_dataset",
        "identifier": "alxmamaev/flowers-recognition",
        "homepage": "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition",
    },
    {
        "name": "CUB-200-2011",
        "type": "http",
        "url": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        "homepage": "http://www.vision.caltech.edu/datasets/cub_200_2011/",
    },
    {
        "name": "KenyanFood13",
        "type": "http",
        "url": "https://www.dropbox.com/scl/fi/hk1llnnv6bpjw153epfxo/Food13.zip?rlkey=o7iq83g4g0xjeif45ibxd9kkb&dl=1",
        "filename": "KenyanFood13.zip",
        "homepage": "https://www.dropbox.com/scl/fi/hk1llnnv6bpjw153epfxo/Food13.zip",
    },
    # {
    #     "name": "Animal-10N",
    #     "type": "http",
    #     "url": "http://dm.kaist.ac.kr/datasets/animal-10n/animal-10n.zip",
    #     "homepage": "http://dm.kaist.ac.kr/datasets/animal-10n/",
    # },
    {
        "name": "Garbage",
        "type": "kaggle_dataset",
        "identifier": "asdasdasasdas/garbage-classification",
        "homepage": "https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification",
    },
    {
        "name": "Fruits-360",
        "type": "kaggle_dataset",
        "identifier": "moltean/fruits",
        "homepage": "https://www.kaggle.com/datasets/moltean/fruits",
    },
]

DATASET_INDEX: Dict[str, Dict[str, str]] = {entry["name"]: entry for entry in DATASETS}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "storage" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the datasets that are not part of the original n20 benchmark.\n"
            "Kaggle downloads require the Kaggle API credentials (kaggle.json) to be configured."
        )
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to store the datasets (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(DATASET_INDEX),
        help="Optional list of dataset names to download (defaults to every new dataset).",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archive files instead of deleting them after extraction.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Download again even if the dataset directory already contains files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even when archives or extracted folders already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list the supported dataset names and exit.",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("Supported datasets:")
        for name in sorted(DATASET_INDEX):
            print(f" - {name}")
        return
    
    print(len(args.datasets))

    missing = [name for name in args.datasets if name not in DATASET_INDEX]
    if missing:
        raise SystemExit(f"Unknown dataset(s): {', '.join(missing)}")

    target_root = args.target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        entry = DATASET_INDEX[name]
        dataset_root = target_root / name
        dataset_root.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and not args.force and _has_files(dataset_root):
            print(f"[skip] {name} already exists in {dataset_root}")
            continue

        print(f"[download] {name} -> {dataset_root}")
        if entry["type"] == "http":
            download_http(entry, dataset_root, keep_archive=args.keep_archives, force=args.force)
        elif entry["type"] == "kaggle_dataset":
            download_kaggle_dataset(entry, dataset_root, keep_archive=args.keep_archives)
        elif entry["type"] == "kaggle_competition":
            download_kaggle_competition(entry, dataset_root, keep_archive=args.keep_archives)
        elif entry["type"] == "huggingface":
            download_huggingface(entry, dataset_root)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"Unsupported dataset type: {entry['type']}")


def _has_files(path: Path) -> bool:
    return any(path.iterdir())


def download_http(entry: Dict[str, str], target: Path, *, keep_archive: bool, force: bool) -> None:
    url = entry["url"]
    filename = entry.get("filename") or _filename_from_url(url, entry["name"])
    archive_path = target / filename

    if archive_path.exists() and not force:
        print(f"  > reusing existing archive {archive_path.name}")
    else:
        print(f"  > streaming {url}")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)

    if not extract_archive(archive_path, target):
        print(f"  > no extraction performed for {archive_path.name}")
    elif not keep_archive:
        archive_path.unlink(missing_ok=True)


def download_kaggle_dataset(entry: Dict[str, str], target: Path, *, keep_archive: bool) -> None:
    api = _kaggle_api()
    identifier = entry["identifier"]
    api.dataset_download_files(identifier, path=str(target), unzip=True, quiet=False)
    zip_name = identifier.split("/")[-1] + ".zip"
    archive_path = target / zip_name
    if archive_path.exists() and not keep_archive:
        archive_path.unlink()


def download_kaggle_competition(entry: Dict[str, str], target: Path, *, keep_archive: bool) -> None:
    api = _kaggle_api()
    identifier = entry["identifier"]
    api.competition_download_files(identifier, path=str(target), quiet=False)
    archive_path = target / f"{identifier}.zip"
    if archive_path.exists():
        extract_archive(archive_path, target)
        if not keep_archive:
            archive_path.unlink()


def download_huggingface(entry: Dict[str, str], target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - runtime convenience
        raise SystemExit(
            "Downloading Hugging Face datasets requires `huggingface_hub`. "
            "Install it with `pip install huggingface_hub datasets`."
        ) from exc

    snapshot_download(
        repo_id=entry["identifier"],
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )


def _filename_from_url(url: str, fallback: str) -> str:
    tail = url.split("?")[0].rstrip("/").split("/")[-1]
    return tail or f"{fallback}.bin"


def extract_archive(archive_path: Path, target: Path) -> bool:
    if not archive_path.exists():
        return False

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)
        return True

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(target)
        return True

    return False


def _kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - runtime convenience
        raise SystemExit(
            "Downloading Kaggle datasets requires the `kaggle` package. "
            "Install it with `pip install kaggle` and ensure kaggle.json is configured."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


if __name__ == "__main__":
    main()
