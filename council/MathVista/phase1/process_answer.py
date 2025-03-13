import os
import json
import re

# Define the same output directory used in batch submission
OUTPUT_DIR = os.path.join(os.getcwd(), "phase1_results")

def parse_final_answer(json_text):
    if json_text is None:
        return ""
    try:
        data = json.loads(json_text)
        final_str = data.get("Answer", "")
        match = re.search(r"Answer:\s*(.+)", final_str)
        return match.group(1).strip() if match else final_str.strip()
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return json_text.strip() if json_text is not None else ""


def extract_confidence(text):
    if not text:
        return None
    match = re.search(r"Confidence:\s*(\d+)%", text)
    return int(match.group(1)) if match else None


def parse_batch_results(results_filename):
    """
    Reads a batch results JSONL file and returns a dictionary:
      { example_index: { agent_id: { "raw": ..., "predicted": ..., "confidence": ... } } }
    """
    example_results = {}
    with open(results_filename, "r") as f:
        for line in f:
            res = json.loads(line.strip())
            custom_id = res["custom_id"]
            parts = custom_id.split("_")
            idx = parts[1]
            agent_id = parts[2]

            raw_output = res["response"]["body"]["choices"][0]["message"]["content"]
            predicted = parse_final_answer(raw_output)
            confidence = extract_confidence(raw_output)
            if idx not in example_results:
                example_results[idx] = {}
            example_results[idx][agent_id] = {
                "raw": raw_output,
                "predicted": predicted,
                "confidence": confidence
            }
    return example_results

def determine_early_exit(example_results):
    """
    Checks if all models and agents agree on an answer for each problem.
    For each problem instance, returns an object containing:
      - exit: True if all predictions agree, False otherwise.
      - answer: The consensus answer (if exit is True) or None.
      - model_predictions: List of all predictions.
    """
    exit_results = {}
    all_problem_indices = set(example_results.keys())
    sorted_indices = sorted(
        all_problem_indices,
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x)
    )
    for idx in sorted_indices:
        predictions = []
        for agent_id, agent_output in example_results[idx].items():
            predictions.append(agent_output["predicted"])
        if len(set(predictions)) == 1:
            exit_results[idx] = {
                "exit": True,
                "answer": predictions[0],
                "model_predictions": predictions
            }
            print(f"Example {idx} early exit with answer: {predictions[0]}")
        else:
            exit_results[idx] = {
                "exit": False,
                "answer": None,
                "model_predictions": predictions
            }
    return exit_results

def merge_results(all_results):
    """
    Merges results across models.
    all_results is expected to be a dict { model: { example_index: ... } }.
    Returns a merged dictionary keyed by example_index.
    """
    merged_results = {}
    for model_results in all_results.values():
        for idx, agent_outputs in model_results.items():
            if idx not in merged_results:
                merged_results[idx] = {}
            merged_results[idx].update(agent_outputs)
    return merged_results

def process_all_results():
    """
    Processes all batch result files in OUTPUT_DIR,
    merges them, computes early exit, and writes the final JSON output.
    """
    all_results = {}
    # Look for result files starting with "batch_job_results_phase1_"
    for filename in os.listdir(OUTPUT_DIR):
        if filename.startswith("batch_job_results_phase1_") and filename.endswith(".jsonl"):
            filepath = os.path.join(OUTPUT_DIR, filename)
            # Extract model name from filename if needed
            model = filename.replace("batch_job_results_phase1_", "").replace(".jsonl", "")
            print(f"Processing results for model: {model}")
            results = parse_batch_results(filepath)
            all_results[model] = results

    merged_results = merge_results(all_results)
    overall_early_exit = determine_early_exit(merged_results)

    final_output = {
        "example_results": all_results,
        "overall_early_exit": overall_early_exit
    }
    output_filename = os.path.join(OUTPUT_DIR, "phase1_results_mathvista.json")
    with open(output_filename, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Processing complete. Final results saved to '{output_filename}'.")

if __name__ == "__main__":
    process_all_results()
