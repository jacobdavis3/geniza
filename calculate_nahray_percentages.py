#!/usr/bin/env python3
"""
Calculate average percentages for Nahray b Nissim distributions
across claude and gpt folders.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Goldberg's 10 Categories
CATEGORIES = [
    "Transactions",
    "Behavior of Associates",
    "Information",
    "Correspondence",
    "Travels",
    "Personal, Familial, Communal News",
    "Accounts",
    "Advice",
    "Legal Actions",
    "Government Relations"
]

def load_json_files(folder_path):
    """Load all JSON files from a folder and return combined data."""
    data = {}
    folder = Path(folder_path)
    
    for json_file in sorted(folder.glob("*.json")):
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            data.update(file_data)
    
    return data

def calculate_averages(data, model_name):
    """Calculate average percentages for each category across all documents."""
    category_sums = defaultdict(float)
    document_count = 0
    
    for doc_id, doc_data in data.items():
        if "codings" in doc_data and model_name in doc_data["codings"]:
            document_count += 1
            codings = doc_data["codings"][model_name]
            
            for category in CATEGORIES:
                if category in codings:
                    category_sums[category] += codings[category]
    
    # Calculate averages
    averages = {}
    for category in CATEGORIES:
        if document_count > 0:
            averages[category] = category_sums[category] / document_count
        else:
            averages[category] = 0.0
    
    return averages, document_count

def main():
    base_path = Path("NahraybNissim")
    
    # Process Claude folder
    print("Processing Claude folder...")
    claude_path = base_path / "claude"
    claude_data = load_json_files(claude_path)
    claude_averages, claude_count = calculate_averages(claude_data, "claude35")
    
    print(f"\nClaude: Processed {claude_count} documents")
    
    # Process GPT folder
    print("\nProcessing GPT folder...")
    gpt_path = base_path / "gpt"
    gpt_data = load_json_files(gpt_path)
    gpt_averages, gpt_count = calculate_averages(gpt_data, "gpt5")
    
    print(f"\nGPT: Processed {gpt_count} documents")
    
    # Create output in the format shown in the image
    output = {
        "claude35": claude_averages,
        "gpt5": gpt_averages
    }
    
    # Print results
    print("\n" + "="*60)
    print("CLAUDE35 AVERAGES:")
    print("="*60)
    for category, value in claude_averages.items():
        print(f"{category}: {value}")
    
    print("\n" + "="*60)
    print("GPT5 AVERAGES:")
    print("="*60)
    for category, value in gpt_averages.items():
        print(f"{category}: {value}")
    
    # Save to JSON file
    output_file = "nahray_percentages.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

