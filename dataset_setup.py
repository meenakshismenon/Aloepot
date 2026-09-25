from pathlib import Path
import shutil

import kagglehub

DATASET_SLUG = "souvikrana17/indoor-plant-health-and-growth-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALOE_KEYWORDS = (
    "aloe",
    "aloevera",
    "aloe-vera",
    "aloe vera",
    "aloe_plant",
    "aloe vera plant",
)


def download_dataset() -> Path:
    """Download the Kaggle plant dataset and return its local path."""
    try:
        dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    except Exception as exc:  # pragma: no cover - user environment specific
        raise RuntimeError(
            "Unable to download the Kaggle dataset. "
            "Make sure your Kaggle API credentials are configured in ~/.kaggle/kaggle.json."
        ) from exc

    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path


def list_dataset_contents(dataset_path: Path):
    """Print the dataset layout and image file counts."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    print("\nTop-level contents:")
    for child in sorted(dataset_path.iterdir()):
        print(f"- {child.name}")

    image_files = []
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)

    print(f"\nImage files found: {len(image_files)}")
    for image_file in image_files[:10]:
        print(f"- {image_file.relative_to(dataset_path)}")

    if len(image_files) > 10:
        print("- ...")


def is_aloe_related(path: Path) -> bool:
    """Return True when a path looks like it belongs to an aloe class."""
    text = path.name.lower() + " " + str(path).lower()
    return any(keyword in text for keyword in ALOE_KEYWORDS)


def create_aloe_subset(dataset_path: Path, output_dir: str = "aloe_subset") -> Path:
    """Copy only aloe-related images into a separate folder."""
    output_path = Path(output_dir)
    if output_path.exists():
        for child in sorted(output_path.iterdir(), reverse=True):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    aloe_images = []
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            if is_aloe_related(file_path):
                aloe_images.append(file_path)

    if not aloe_images:
        print("No aloe-related images were found in this dataset.")
        return output_path

    for source_file in aloe_images:
        relative = source_file.relative_to(dataset_path)
        destination = output_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)

    print(f"\nAloe subset created at: {output_path}")
    print(f"Aloe images copied: {len(aloe_images)}")
    return output_path


if __name__ == "__main__":
    dataset_path = download_dataset()
    list_dataset_contents(dataset_path)
    create_aloe_subset(dataset_path)
