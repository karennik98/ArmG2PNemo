import json
import os
import re
import random


def clean_phonetic_transcription(phoneme):
    """
    Clean phonetic transcription by removing extra spaces
    """
    return re.sub(r'\s+', '', phoneme).strip()


def load_and_shuffle_dataset(input_file):
    """
    Load dataset and shuffle entries
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip header if present
        lines = f.readlines()[1:]

    # Shuffle the lines
    random.seed(42)  # For reproducibility
    random.shuffle(lines)

    return lines


def convert_to_manifest(lines, output_file):
    """
    Convert lines to JSON manifest
    """
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for line in lines:
            # Split line by tab, handling potential whitespace
            parts = line.strip().split('\t')

            # Ensure we have both grapheme and phoneme
            if len(parts) == 2:
                grapheme = parts[0]
                phoneme = clean_phonetic_transcription(parts[1])

                # Create JSON entry
                entry = {
                    "text_graphemes": grapheme,
                    "text": phoneme
                }

                # Write as JSON line
                json.dump(entry, out_f, ensure_ascii=False)
                out_f.write('\n')

    # Print stats
    print(f"Converted dataset saved to {output_file}")
    print(f"Total entries: {len(lines)}")


def split_dataset(input_file):
    """
    Randomly split dataset into train, validation, and test sets
    """
    # Load and shuffle dataset
    all_lines = load_and_shuffle_dataset(input_file)
    total_entries = len(all_lines)

    # Calculate split indices
    train_split = int(total_entries * 0.8)
    val_split = train_split + int(total_entries * 0.1)

    # Split datasets
    train_lines = all_lines[:train_split]
    val_lines = all_lines[train_split:val_split]
    test_lines = all_lines[val_split:]

    # Convert to manifests
    convert_to_manifest(train_lines, "armenian_g2p_train_manifest.json")
    convert_to_manifest(val_lines, "armenian_g2p_val_manifest.json")
    convert_to_manifest(test_lines, "armenian_g2p_test_manifest.json")

    # Verify splits
    print("\nDataset Split:")
    print(f"Total entries: {total_entries}")
    print(f"Training set: {len(train_lines)} ({len(train_lines) / total_entries * 100:.2f}%)")
    print(f"Validation set: {len(val_lines)} ({len(val_lines) / total_entries * 100:.2f}%)")
    print(f"Test set: {len(test_lines)} ({len(test_lines) / total_entries * 100:.2f}%)")

    # Sample entries verification
    print("\nVerifying output files:")
    for manifest in ["armenian_g2p_train_manifest.json",
                     "armenian_g2p_val_manifest.json",
                     "armenian_g2p_test_manifest.json"]:
        with open(manifest, 'r', encoding='utf-8') as f:
            print(f"\nFirst 3 entries in {manifest}:")
            for _ in range(3):
                print(f.readline().strip())


# Main execution
if __name__ == "__main__":
    input_file = "armenian_g2p_dataset.txt"
    split_dataset(input_file)