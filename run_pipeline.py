import os
import time
import pandas as pd
from pipeline import run_pipeline


INPUT_FOLDER = "incoming_data"
OUTPUT_FOLDER = "processed_data"


# Ensure folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_file(file_path):

    print(f"Processing file: {file_path}")

    df = run_pipeline(file_path)

    # Create week column
    df["week"] = df["timestamp"].dt.to_period("W")

    # Split dataset week-wise
    for week, group in df.groupby("week"):

        # convert week to safe string
        week_str = str(week).replace("/", "-").replace(" ", "")

        output_file = os.path.join(
            OUTPUT_FOLDER,
            f"processed_week_{week_str}.csv"
        )

        group.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")


def watch_folder():

    processed_files = set()

    print("Watching folder for new files...")

    while True:

        files = os.listdir(INPUT_FOLDER)

        for file in files:

            if file.endswith(".csv"):

                file_path = os.path.join(INPUT_FOLDER, file)

                if file_path not in processed_files:

                    print(f"New file detected: {file}")

                    try:
                        process_file(file_path)
                        processed_files.add(file_path)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        time.sleep(5)


if __name__ == "__main__":
    watch_folder()