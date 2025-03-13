import os
import json
import re

# Define the output directory where Phase 2 discussion JSON files are stored
OUTPUT_DIR = os.path.join(os.getcwd(), "phase2/phase2_results_csqa")

def load_and_filter_critique(filepath):
    """
    Loads a discussion results JSON file.
    For each example and agent, parses the critique string (expected to be a JSON string),
    then filters out any key-value pair where the value is exactly "N/A" (after converting to a string and stripping).
    Returns a dictionary of the filtered critiques.
    """
    filtered_results = {}
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Data format: { example_index: { agent_id: critique_str } }
    for ex_idx, agent_dict in data.items():
        if ex_idx not in filtered_results:
            filtered_results[ex_idx] = {}
        for agent_id, critique_str in agent_dict.items():
            try:
                critique_json = json.loads(critique_str)
            except Exception as e:
                print(f"Error parsing critique for example {ex_idx} agent {agent_id}: {e}")
                continue
            filtered = {k: v for k, v in critique_json.items() if str(v).strip() != "N/A"}
            if filtered:
                filtered_results[ex_idx][agent_id] = filtered
    return filtered_results

def merge_all_critics(output_dir):
    """
    Loads all JSON files in the output directory that start with "discussion_results_"
    and merges them into a single dictionary.
    """
    merged = {}
    for filename in os.listdir(output_dir):
        if filename.startswith("discussion_results_") and filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            file_results = load_and_filter_critique(filepath)
            for ex_idx, agents in file_results.items():
                if ex_idx not in merged:
                    merged[ex_idx] = {}
                merged[ex_idx].update(agents)
    return merged

def main():
    final_critics = merge_all_critics(OUTPUT_DIR)
    output_filepath = os.path.join(OUTPUT_DIR, "phase2_critics_csqa.json")
    with open(output_filepath, "w") as f:
        json.dump(final_critics, f, indent=2)
    print(f"Phase 2 critics saved to {output_filepath}")

if __name__ == "__main__":
    main()
