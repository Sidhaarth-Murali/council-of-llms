import json
import re

def extract_csqa_option(text: str) -> str:
    m = re.search(r'^\s*([A-E])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'([A-E])\s*:', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "NA"

def clean_csqa_overall_early_exit(data: dict) -> dict:
    cleaned_data = {}
    for key, record in data.items():
        predictions = record.get("model_predictions", [])
        cleaned_preds = [extract_csqa_option(pred) for pred in predictions]
        new_exit = True if cleaned_preds and all(x == cleaned_preds[0] and x != "NA" for x in cleaned_preds) else False
        cleaned_data[key] = {
            "exit": new_exit,
            "answer": record.get("answer"),
            "model_predictions": cleaned_preds
        }
    return cleaned_data

if __name__ == "__main__":
    input_filename = "CSQA/phase1_results/phase1_results_commonsenseqa.json"
    output_filename = "CSQA/phase1_results/phase1_results_commonsenseqa_early_exits.json"
    with open(input_filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    overall = data.get("overall_early_exit", {})
    cleaned_overall = clean_csqa_overall_early_exit(overall)
    output_data = {"overall_early_exit": cleaned_overall}
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=2, ensure_ascii=False)
    print(f"Cleaned output has been written to {output_filename}")
