import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_dataset_root(raw_dir):
    """Find the folder that contains the train and valid dataset folders."""
    for folder in [raw_dir, *raw_dir.rglob("*")]:
        if folder.is_dir() and (folder / "train").is_dir() and (folder / "valid").is_dir():
            return folder
    return None


def get_image_files(folder):
    """Return image files directly inside one class folder."""
    return [
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_image_dimensions(image_path):
    """Read one image with OpenCV and return its dimensions."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    return width, height


def analyze_split(split_dir, check_all_images):
    """Analyze class folders inside one split, such as train or valid."""
    class_folders = sorted([folder for folder in split_dir.iterdir() if folder.is_dir()])
    image_counts = {}
    empty_classes = []
    unreadable_images = []
    extensions = set()
    sample_dimensions = {}

    for class_folder in class_folders:
        print(f"Scanning {split_dir.name}/{class_folder.name}...", flush=True)
        image_files = get_image_files(class_folder)
        image_counts[class_folder.name] = len(image_files)

        if not image_files:
            empty_classes.append(class_folder.name)

        for index, image_path in enumerate(image_files):
            extensions.add(image_path.suffix.lower())

            should_read_image = check_all_images or index == 0
            if should_read_image:
                dimensions = read_image_dimensions(image_path)
                if dimensions is None:
                    unreadable_images.append(str(image_path))
                    continue
            else:
                dimensions = None

            if dimensions is not None and class_folder.name not in sample_dimensions:
                sample_dimensions[class_folder.name] = dimensions

    return {
        "class_names": [folder.name for folder in class_folders],
        "image_counts": image_counts,
        "empty_classes": empty_classes,
        "unreadable_images": unreadable_images,
        "extensions": extensions,
        "sample_dimensions": sample_dimensions,
    }


def print_report(dataset_root, split_reports, check_all_images):
    """Print a beginner-friendly summary of the dataset."""
    all_class_names = sorted(
        set().union(*(report["class_names"] for report in split_reports.values()))
    )
    total_images = sum(
        sum(report["image_counts"].values()) for report in split_reports.values()
    )
    all_extensions = sorted(
        set().union(*(report["extensions"] for report in split_reports.values()))
    )
    all_empty_classes = {
        split_name: report["empty_classes"]
        for split_name, report in split_reports.items()
        if report["empty_classes"]
    }
    all_unreadable_images = [
        image_path
        for report in split_reports.values()
        for image_path in report["unreadable_images"]
    ]

    print("Dataset Analysis Report")
    print("=" * 24)
    print(f"Dataset root path: {dataset_root}")
    print(f"Total number of classes: {len(all_class_names)}")

    print("\nAll class names:")
    for class_name in all_class_names:
        print(f"  - {class_name}")

    print("\nNumber of images in each class:")
    for split_name, report in split_reports.items():
        print(f"\n{split_name}:")
        for class_name in all_class_names:
            count = report["image_counts"].get(class_name, 0)
            print(f"  {class_name}: {count}")

    print(f"\nTotal number of images: {total_images}")

    print("\nImage file extensions found:")
    for extension in all_extensions:
        print(f"  - {extension}")

    print("\nSample image dimensions:")
    for split_name, report in split_reports.items():
        print(f"\n{split_name}:")
        for class_name, dimensions in report["sample_dimensions"].items():
            width, height = dimensions
            print(f"  {class_name}: {width} x {height}")

    print("\nEmpty class folders:")
    if all_empty_classes:
        for split_name, class_names in all_empty_classes.items():
            print(f"\n{split_name}:")
            for class_name in class_names:
                print(f"  - {class_name}")
    else:
        print("  None found")

    print("\nImages that OpenCV cannot read:")
    if not check_all_images:
        print("  Full check not run. Use --check-all to test every image with OpenCV.")
    elif all_unreadable_images:
        for image_path in all_unreadable_images:
            print(f"  - {image_path}")
    else:
        print("  None found")


def main():
    parser = argparse.ArgumentParser(description="Analyze the plant disease dataset.")
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Use OpenCV to read every image. This is slower but finds unreadable files.",
    )
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    dataset_root = find_dataset_root(raw_dir)

    if dataset_root is None:
        print("Could not find a dataset root with both train and valid folders.")
        return

    split_reports = {}
    for split_name in ["train", "valid"]:
        split_dir = dataset_root / split_name
        if split_dir.is_dir():
            split_reports[split_name] = analyze_split(split_dir, args.check_all)

    print_report(dataset_root, split_reports, args.check_all)


if __name__ == "__main__":
    main()
