import csv
from pathlib import Path


DATASET_ROOT = Path(
    "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
)
TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
OUTPUT_CSV = Path("data/class_analysis.csv")


def count_jpg_images(class_folder):
    """Count .jpg images directly inside a class folder."""
    return sum(
        1
        for file_path in class_folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".jpg"
    )


def split_class_name(class_name):
    """Split a class folder name into crop and disease/status parts."""
    crop, disease_or_status = class_name.split("___", 1)
    crop = crop.replace("_", " ")
    disease_or_status = disease_or_status.strip("_").replace("_", " ")
    return crop, disease_or_status


def build_rows():
    """Build one CSV row for each actual class folder in the train directory."""
    class_folders = sorted(folder for folder in TRAIN_DIR.iterdir() if folder.is_dir())
    rows = []

    for class_id, class_folder in enumerate(class_folders, start=1):
        class_name = class_folder.name
        valid_class_folder = VALID_DIR / class_name
        train_count = count_jpg_images(class_folder)
        valid_count = count_jpg_images(valid_class_folder) if valid_class_folder.is_dir() else 0
        crop, disease_or_status = split_class_name(class_name)
        category = "Healthy" if disease_or_status.lower() == "healthy" else "Diseased"

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "crop": crop,
                "disease_or_status": disease_or_status,
                "category": category,
                "train_image_count": train_count,
                "valid_image_count": valid_count,
                "total_image_count": train_count + valid_count,
                "selected_for_project": "NO",
                "reason": "",
            }
        )

    return rows


def write_csv(rows):
    """Write all class analysis rows to data/class_analysis.csv."""
    fieldnames = [
        "class_id",
        "class_name",
        "crop",
        "disease_or_status",
        "category",
        "train_image_count",
        "valid_image_count",
        "total_image_count",
        "selected_for_project",
        "reason",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not TRAIN_DIR.is_dir() or not VALID_DIR.is_dir():
        raise FileNotFoundError(f"Expected train and valid folders inside {DATASET_ROOT}")

    rows = build_rows()
    write_csv(rows)

    total_train = sum(row["train_image_count"] for row in rows)
    total_valid = sum(row["valid_image_count"] for row in rows)
    total_images = sum(row["total_image_count"] for row in rows)

    print(f"CSV file created: {OUTPUT_CSV}")
    print(f"Total classes: {len(rows)}")
    print(f"Total train images: {total_train}")
    print(f"Total valid images: {total_valid}")
    print(f"Total images: {total_images}")


if __name__ == "__main__":
    main()
