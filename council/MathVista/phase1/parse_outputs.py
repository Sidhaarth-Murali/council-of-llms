import json
import re

def parse_number_from_text(text: str):
    match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', text)
    if match:
        num_str = match.group(0)
        return float(num_str) if '.' in num_str else int(num_str)
    return None

def clean_mathvista_overall_early_exit(data: dict) -> dict:
    cleaned_data = {}
    for key, record in data.items():
        predictions = record.get("model_predictions", [])
        cleaned_predictions = [parse_number_from_text(pred) for pred in predictions]
        
        # Set exit to True if the cleaned predictions list is non-empty and all values are identical
        if cleaned_predictions:
            all_same = all(pred == cleaned_predictions[0] for pred in cleaned_predictions)
        else:
            all_same = False
        
        cleaned_data[key] = {
            "exit": all_same,
            "answer": record.get("answer"),
            "model_predictions": cleaned_predictions
        }
    return cleaned_data

if __name__ == "__main__":
    input_filename = "phase1_results/phase1_results_mathvista.json"
    output_filename = "phase1_results/early_exits.json"

    with open(input_filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    overall = data.get("overall_early_exit", {})
    cleaned_overall = clean_mathvista_overall_early_exit(overall)
    output_data = {"overall_early_exit": cleaned_overall}
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=2, ensure_ascii=False)
    print(f"Cleaned output has been written to {output_filename}")
