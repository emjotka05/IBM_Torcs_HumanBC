import json
import os

from train import LEGACY_FEATURE_DIM, FEATURE_DIM, normalize_dataset

MAIN_FILE = 'driving_data.json'
TEMP_FILE = 'driving_data_toCombine.json'


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_data(data, source_name):
    prepared, _, legacy_count = normalize_dataset(data, source_name)
    normalized = [
        {'state': row['state'], 'action': row['action']}
        for row in prepared
    ]
    if legacy_count:
        print(
            f"Converted {legacy_count} legacy samples "
            f"{LEGACY_FEATURE_DIM}D -> {FEATURE_DIM}D."
        )
    return normalized


def main():
    print("=== TORCS Data Combiner ===")

    if not os.path.exists(TEMP_FILE):
        print(f"Missing temporary file '{TEMP_FILE}'. Nothing to do.")
        return

    print(f"Loading '{TEMP_FILE}'...")
    try:
        temp_data = load_json(TEMP_FILE)
    except Exception as e:
        print(f"Failed to load '{TEMP_FILE}': {e}")
        return

    print(f"Read {len(temp_data)} new samples.")
    if not temp_data:
        print("Temporary file is empty. Removing it.")
        os.remove(TEMP_FILE)
        return

    try:
        temp_data = normalize_data(temp_data, TEMP_FILE)
    except ValueError as e:
        print(f"Invalid temporary data: {e}")
        return

    if os.path.exists(MAIN_FILE):
        print(f"Loading '{MAIN_FILE}'...")
        try:
            main_data = load_json(MAIN_FILE)
        except Exception as e:
            print(f"Failed to load '{MAIN_FILE}': {e}")
            return

        print(f"Main file has {len(main_data)} samples.")
        try:
            main_data = normalize_data(main_data, MAIN_FILE)
        except ValueError as e:
            print(f"Invalid main data: {e}")
            return

        main_data.extend(temp_data)
    else:
        print(f"Main file '{MAIN_FILE}' does not exist. Creating it.")
        main_data = temp_data

    print(f"Saving {len(main_data)} normalized {FEATURE_DIM}D samples to '{MAIN_FILE}'...")
    with open(MAIN_FILE, 'w') as f:
        json.dump(main_data, f)

    print(f"Removing '{TEMP_FILE}'...")
    os.remove(TEMP_FILE)
    print("Combine complete.")


if __name__ == '__main__':
    main()
