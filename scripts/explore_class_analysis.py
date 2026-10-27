from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/class_analysis.csv")


def main():
    print("=" * 60)
    print("SMART AGRICULTURE - PANDAS DATA EXPLORATION")
    print("=" * 60)

    # Step 2: Load data into DataFrame
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    df = pd.read_csv(CSV_PATH)

    # 1. First 5 rows
    print("\n--- 1. First 5 Rows (df.head()) ---")
    print(df.head())

    # 2. DataFrame shape
    print("\n--- 2. DataFrame Shape (df.shape) ---")
    print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

    # 3. Column names
    print("\n--- 3. Column Names (df.columns) ---")
    print(list(df.columns))

    # 4. Basic information using .info()
    print("\n--- 4. DataFrame Information (df.info()) ---")
    df.info()

    # 5. Number of classes
    num_classes = len(df)
    print(f"\n--- 5. Number of Classes: {num_classes} ---")

    # 6. Number of unique crops
    num_crops = df["crop"].nunique()
    print(f"\n--- 6. Number of Unique Crops (df['crop'].nunique()): {num_crops} ---")

    # 7. List of unique crops
    unique_crops = df["crop"].unique()
    print("\n--- 7. List of Unique Crops (df['crop'].unique()) ---")
    for crop in sorted(unique_crops):
        print(f"  - {crop}")

    # 8. Number of healthy classes
    healthy_df = df[df["category"] == "Healthy"]
    num_healthy = len(healthy_df)
    print(f"\n--- 8. Number of Healthy Classes: {num_healthy} ---")

    # 9. Number of diseased classes
    diseased_df = df[df["category"] == "Diseased"]
    num_diseased = len(diseased_df)
    print(f"\n--- 9. Number of Diseased Classes: {num_diseased} ---")

    # 10. Total training images
    total_train = int(df["train_image_count"].sum())
    print(f"\n--- 10. Total Training Images: {total_train:,} ---")

    # 11. Total validation images
    total_valid = int(df["valid_image_count"].sum())
    print(f"\n--- 11. Total Validation Images: {total_valid:,} ---")

    # 12. Total images
    total_images = int(df["total_image_count"].sum())
    print(f"\n--- 12. Total Images: {total_images:,} ---")

    # 13. Minimum images in a class
    min_images = int(df["total_image_count"].min())
    print(f"\n--- 13. Minimum Images in a Class: {min_images:,} ---")

    # 14. Maximum images in a class
    max_images = int(df["total_image_count"].max())
    print(f"\n--- 14. Maximum Images in a Class: {max_images:,} ---")

    # 15. Average images per class
    avg_images = float(df["total_image_count"].mean())
    print(f"\n--- 15. Average Images per Class: {avg_images:,.2f} ---")

    # 16. Classes grouped by crop
    print("\n--- 16. Classes Grouped by Crop (df.groupby('crop').size()) ---")
    print(df.groupby("crop").size())

    # 17. Healthy classes
    print("\n--- 17. Healthy Classes (df[df['category'] == 'Healthy']) ---")
    print(healthy_df[["class_id", "crop", "class_name", "total_image_count"]].to_string(index=False))

    # 18. Diseased classes
    print("\n--- 18. Diseased Classes (df[df['category'] == 'Diseased']) ---")
    print(diseased_df[["class_id", "crop", "disease_or_status", "total_image_count"]].to_string(index=False))

    # 19. Verify that all 38 classes have selected_for_project == 'YES'
    all_yes = bool((df["selected_for_project"] == "YES").all())
    selected_count = int((df["selected_for_project"] == "YES").sum())
    print(f"\n--- 19. Selected For Project Check: {selected_count}/{num_classes} marked 'YES' ---")

    # Step 4: Simple validation
    print("\n" + "=" * 40)
    print("PANDAS DATA ANALYSIS VERIFICATION")
    print(f"Classes: {'PASS' if num_classes == 38 else f'FAIL (actual: {num_classes})'}")
    print(f"Crops: {'PASS' if num_crops == 14 else f'FAIL (actual: {num_crops})'}")
    print(f"Healthy classes: {'PASS' if num_healthy == 12 else f'FAIL (actual: {num_healthy})'}")
    print(f"Diseased classes: {'PASS' if num_diseased == 26 else f'FAIL (actual: {num_diseased})'}")
    print(f"Total images: {'PASS' if total_images == 87867 else f'FAIL (actual: {total_images})'}")
    print(f"Selection status: {'PASS' if all_yes and num_classes == 38 else f'FAIL (actual: {selected_count} YES)'}")
    print("=" * 40)


if __name__ == "__main__":
    main()
