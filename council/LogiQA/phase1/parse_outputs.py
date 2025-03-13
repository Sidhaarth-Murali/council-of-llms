import json
import re
from collections import Counter

def extract_option(text: str) -> str:
    m = re.search(r'^\s*([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'([A-D])\s*:', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None

def clean_logiqa_overall_early_exit(data: dict) -> dict:
    cleaned_data = {}
    for key, record in data.items():
        predictions = record.get("model_predictions", [])
        # Extract options from each prediction
        extracted = [extract_option(pred) for pred in predictions]
        valid_options = [opt for opt in extracted if opt is not None]
        
        # Determine the majority option (if available)
        majority = None
        if valid_options:
            counts = Counter(valid_options)
            majority = counts.most_common(1)[0][0]
        
        # Replace missing or invalid options with the majority option
        filled = [opt if opt is not None and opt in {"A", "B", "C", "D"} else majority for opt in extracted]
        
        # Determine the exit value:
        # If the cleaned predictions list is non-empty and all predictions are identical, set exit to True.
        if filled:
            all_same = all(option == filled[0] for option in filled)
        else:
            all_same = False
        
        cleaned_data[key] = {
            "exit": all_same,
            "answer": record.get("answer"),
            "model_predictions": filled
        }
    return cleaned_data

if __name__ == "__main__":
    input_filename = "phase1_results/phase1_results_logiqa.json"
    output_filename = "phase1_results/overall_exits.json"
    
    with open(input_filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    
    overall = data.get("overall_early_exit", {})
    cleaned_overall = clean_logiqa_overall_early_exit(overall)
    output_data = {"overall_early_exit": cleaned_overall}
    
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=2, ensure_ascii=False)
    
    print(f"Cleaned output has been written to {output_filename}")
