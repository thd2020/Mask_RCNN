import os
import json
from collections import defaultdict

# Path to the directory containing all the JSON files
json_dir = 'data/jsons'

# List of expected labels in English (you can add more as needed)
expected_labels = {
    "胎盘": "placenta",
    "胎盘植入部分": "placenta_accreta",
    "膀胱": "bladder",
    "子宫肌层": "uterine_myometrium"
}

# Initialize statistics counters
json_stats = {}
label_counts = defaultdict(int)
label_combinations = defaultdict(int)

# Process each JSON file in the directory
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_dir, json_file)
        
        # Load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the label information
        labels_found = set()  # Set to store unique labels
        for shape in data['shapes']:
            label = shape['label']
            if label in expected_labels:
                labels_found.add(expected_labels[label])
        
        # Save the information
        num_labels = len(labels_found)
        json_stats[json_file] = {
            "num_labels": num_labels,
            "labels": list(labels_found)
        }

        # Update overall stats
        label_combinations[num_labels] += 1  # Count how many JSONs have 4, 3, 2, etc. labels
        for label in labels_found:
            label_counts[label] += 1  # Count how many times each label is found

# Output the statistics for each JSON file
for json_file, stats in json_stats.items():
    print(f"File: {json_file}")
    print(f"  Number of labels: {stats['num_labels']}")
    print(f"  Labels: {', '.join(stats['labels'])}")
    print("-" * 40)

# Output summary statistics
print("\nSummary Statistics:")
print(f"  JSONs with 4 labels: {label_combinations[4]}")
print(f"  JSONs with 3 labels: {label_combinations[3]}")
print(f"  JSONs with 2 labels: {label_combinations[2]}")
print(f"  JSONs with 1 label: {label_combinations[1]}")
print(f"  JSONs with no labels: {label_combinations[0]}")

# Most frequent label
most_common_label = max(label_counts, key=label_counts.get)
print(f"\nMost common label: {most_common_label} ({label_counts[most_common_label]} occurrences)")

# Output label statistics
print("\nLabel Distribution:")
for label, count in label_counts.items():
    print(f"  {label}: {count} occurrences")