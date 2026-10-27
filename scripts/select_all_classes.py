import csv
from pathlib import Path


CSV_PATH = Path("data/class_analysis.csv")
REASON = (
    "Included to support multi-crop healthy and disease classification "
    "using all actual dataset classes."
)


def main():
    with CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError("CSV header was not found.")

    for row in rows:
        row["selected_for_project"] = "YES"
        row["reason"] = REASON

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    crops = {row["crop"] for row in rows}
    healthy_count = sum(1 for row in rows if row["category"] == "Healthy")
    diseased_count = sum(1 for row in rows if row["category"] == "Diseased")
    total_images = sum(int(row["total_image_count"]) for row in rows)

    print(f"Updated CSV: {CSV_PATH}")
    print(f"Total selected classes: {len(rows)}")
    print(f"Total supported crops: {len(crops)}")
    print(f"Healthy classes: {healthy_count}")
    print(f"Diseased classes: {diseased_count}")
    print(f"Total selected images: {total_images}")


if __name__ == "__main__":
    main()
