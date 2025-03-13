import json
import re

def parse_prediction(prediction: str) -> bool:
    pred = prediction.lower().strip()
    pred = re.sub(r"[()]", "", pred)
    
    if "true" in pred:
        return True
    if "false" in pred:
        return False

    if "a" in pred:
        return False
    if "b" in pred:
        return True

    raise ValueError(f"Cannot parse prediction: {prediction}")

def clean_overall_early_exit(data: dict) -> dict:
    cleaned_data = {}
    for key, record in data.items():
        predictions = record.get("model_predictions", [])
        parsed_preds = []
        for pred in predictions:
            try:
                parsed_preds.append(parse_prediction(pred))
            except ValueError:
                parsed_preds.append(None)

        new_exit = all(val == parsed_preds[0] for val in parsed_preds if val is not None) if parsed_preds else False

        cleaned_data[key] = {
            "exit": new_exit,
            "answer": record.get("answer"),
            "model_predictions": parsed_preds
        }
    return cleaned_data

if __name__ == "__main__":
    input_filename = "PangeaBench/phase1_results/phase1_results_pangea_bench.json"
    output_filename = "PangeaBench/phase1_results/phase1_results_pangea_bench_cleaned.json"

    with open(input_filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    overall_early_exit = data.get("overall_early_exit", {})
    cleaned_overall = clean_overall_early_exit(overall_early_exit)

    output_data = {"overall_early_exit": cleaned_overall}
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=2, ensure_ascii=False)

    print(f"Cleaned output has been written to {output_filename}")
