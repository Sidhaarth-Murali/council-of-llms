# role_assignment.py

import os
import json
import time
import re
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
client = OpenAI(api_key=openai_api_key)

# Define and create output directory for Phase 2 results
OUTPUT_DIR = os.path.join(os.getcwd(), "phase2/phase2_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_confidence(text):
    """Extracts a confidence score from text using the pattern 'Confidence: X%'."""
    match = re.search(r"Confidence:\s*(\d+)%", text)
    return int(match.group(1)) if match else None

def load_phase1_results(phase1_filepath, overall_exit_filepath):
    """
    Loads Phase 1 results and overall exit info from separate files.
    Expected structure for phase1_filepath (merged agent outputs):
      {
         "example_results": {
              "batch_tasks_phase1_<model>": {
                  "0": { "A1": { "raw": "...", "predicted": "...", "confidence": ... }, ... },
                  "1": { ... },
                  ...
              }
         }
      }
    Overall exit file (overall_exit_filepath) is expected to have:
      {
          "0": { "exit": true/false, "answer": ..., "model_predictions": [ ... ] },
          "1": { "exit": false, "answer": ..., "model_predictions": [ ... ] },
          ...
      }
    Returns a tuple: (merged_agent_outputs, overall_exit)
    """
    with open(phase1_filepath, "r") as f:
        phase1_data = json.load(f)
    merged = {}
    # Assuming phase1_data has key "example_results"
    for key, model_data in phase1_data.get("example_results", {}).items():
        # Skip overall exit info if accidentally included
        if key == "overall_early_exit":
            continue
        for idx, agent_dict in model_data.items():
            if idx not in merged:
                merged[idx] = {}
            merged[idx].update(agent_dict)
    
    with open(overall_exit_filepath, "r") as f:
        overall_exit = json.load(f)
    
    return merged, overall_exit

# Fallback hierarchy for roles if all confidences are missing/0.
FALLBACK_ROLES = ["Solver", "Fact-Checker", "Devil's Advocate"]

def assign_roles(sample_results):
    """
    For a given sample (dict of agent outputs keyed by agent_id), assign roles based on confidence.
    If all confidences are 0 or missing, use FALLBACK_ROLES (in order).
    Otherwise, sort agents by confidence:
      - Lowest: Devil's Advocate
      - Middle: Solver
      - Highest: Fact-Checker
    Returns a dict mapping agent_id to role.
    """
    agents = [a for a in list(sample_results.keys()) if a.lower() != "exit"]
    confs = {agent: sample_results[agent].get("confidence") or 0 for agent in agents}
    if all(conf == 0 for conf in confs.values()):
        roles = {agent: FALLBACK_ROLES[i] for i, agent in enumerate(agents)}
    else:
        sorted_agents = sorted(agents, key=lambda a: confs[a])
        roles = {}
        if len(sorted_agents) >= 3:
            roles[sorted_agents[0]] = "Devil's Advocate"
            roles[sorted_agents[1]] = "Solver"
            roles[sorted_agents[2]] = "Fact-Checker"
        else:
            for i, agent in enumerate(sorted_agents):
                roles[agent] = ["Solver", "Critic", "Verifier"][i]
    return roles

def create_prompt_aquarat_discussion(agent_id, role, peer_outputs):
    """
    Constructs a discussion prompt for the AQUARAT dataset Phase 2.
    Each agent sees its peers' Phase 1 raw outputs (in JSON format) in separate sections.
    Instruct the agent to provide step-by-step critiques for each peer's reasoning.
    For each step, if an issue is detected, include the peer agent's ID in the key (e.g., "A2_Step1_critique").
    If a step is correct, output "N/A".
    Return the prompt string.
    """
    prompt = (
        f"You are Agent {agent_id} with role {role} (a math expert). "
        "Review the following peer outputs from Phase 1 (provided separately). "
        "For each peer's chain-of-thought, provide step-by-step critiques. "
        "For each step, label your critique with the peer agent's ID (e.g., 'A2_Step1_critique'). "
        "If the step is correct, output \"N/A\". "
        "Return your critiques as a valid JSON object with keys like 'A2_Step1_critique', 'A2_Step2_critique', ..., and a final key 'FinalCritique' summarizing your overall assessment.\n\n"
    )
    for peer_id, output in peer_outputs.items():
        prompt += f"---\nFrom Agent {peer_id}:\n{output}\n\n"
    return prompt

def create_batch_tasks_aquarat_discussion(unsolved_data, agents_info):
    """
    Create batch tasks for Phase 2 (discussion) for the AQUARAT dataset.
    For each unsolved example (where overall_exit is false and predictions differ) and for each agent,
    generate a discussion prompt that shows only the peer outputs.
    Returns a dict: { model: list_of_tasks }.
    """
    tasks_by_model = {}
    for idx, sample in unsolved_data.items():
        roles = assign_roles(sample)
        for agent_id, model in agents_info:
            if agent_id not in sample:
                continue
            # Build peer outputs: include only outputs from other agents.
            peer_outputs = {other: sample[other]["raw"] for other in sample if other.lower() != agent_id.lower() and other.lower() != "exit"}
            user_prompt = create_prompt_aquarat_discussion(agent_id, roles.get(agent_id, "Unknown"), peer_outputs)
            task = {
                "custom_id": f"problem_{idx}_{agent_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ]
                }
            }
            tasks_by_model.setdefault(model, []).append(task)
    return tasks_by_model

def write_tasks_to_jsonl_per_model(tasks_by_model):
    """
    Writes separate JSONL files for Phase 2 tasks per model.
    Returns a dict: { model: jsonl_filename }.
    """
    task_files = {}
    for model, tasks in tasks_by_model.items():
        filename = os.path.join(OUTPUT_DIR, f"batch_tasks_phase2_{model.replace('/', '_')}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        task_files[model] = filename
    return task_files

def upload_and_run_batch_job_per_model(tasks_file):
    """
    Uploads and runs a batch job for the given JSONL file using the batch API.
    Saves the resulting output file to OUTPUT_DIR.
    """
    batch_file = client.files.create(file=open(tasks_file, "rb"), purpose="batch")
    print(f"Batch file {tasks_file} uploaded: {batch_file.id}")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print("Batch job created:", batch_job.id)

    while True:
        status = client.batches.retrieve(batch_job.id)
        print(f"Batch job status for {tasks_file}: {status.status}")
        if status.status == "completed":
            break
        elif status.status == "failed":
            raise RuntimeError(f"Batch job {tasks_file} failed!")
        time.sleep(10)
    
    result_content = client.files.content(status.output_file_id).content
    results_filename = os.path.join(OUTPUT_DIR, f"batch_job_results_phase2_{os.path.basename(tasks_file)}.jsonl")
    with open(results_filename, "wb") as f:
        f.write(result_content)
    print(f"Batch results saved to {results_filename}")
    return results_filename

def parse_batch_results(results_filename):
    """
    Parses the batch result JSONL file and returns a dictionary:
      { example_index: { agent_id: critique_output } }
    """
    example_results = {}
    with open(results_filename, "r") as f:
        for line in f:
            res = json.loads(line.strip())
            custom_id = res["custom_id"]  # Format: problem_{idx}_{agent_id}
            parts = custom_id.split("_")
            idx = parts[1]
            agent_id = parts[2]
            raw_output = res["response"]["body"]["choices"][0]["message"]["content"]
            if idx not in example_results:
                example_results[idx] = {}
            example_results[idx][agent_id] = raw_output
    return example_results

def get_unsolved_indices(phase1_agent_data, overall_exit):
    """
    Returns a dictionary of unsolved examples (by example index) that require discussion:
      - Only include examples where the overall_exit flag (loaded from a separate file) is absent or False.
      - And, where not all agent predicted answers agree.
    """
    unsolved = {}
    for idx, sample in phase1_agent_data.items():
        exit_info = overall_exit.get("overall_early_exit").get(idx, {})
        if exit_info.get("exit", False):
            continue  # Skip if early exit is True
        # Collect predicted values from sample (excluding any "exit" key)
        pred_list = [output.get("predicted") for key, output in sample.items() if key.lower() != "exit"]
        if len(set(pred_list)) > 1:
            unsolved[idx] = sample
    return unsolved

def main():
    phase1_filepath = os.path.join(os.getcwd(), "phase1/phase1_results/phase1_results_aqua_rat.json")
    overall_exit_filepath = os.path.join(os.getcwd(), "phase1/phase1_results/early_exits.json")
    
    phase1_agent_data, overall_exit = load_phase1_results(phase1_filepath, overall_exit_filepath)
    unsolved = get_unsolved_indices(phase1_agent_data, overall_exit)
    print(f"Found {len(unsolved)} unsolved examples for discussion.")
    
    agents_info = [
        ("A1", "gpt-4o-mini"),
        ("A2", "gpt-4o"),
        ("A3", "gpt-3.5-turbo")
    ]

    tasks_by_model = create_batch_tasks_aquarat_discussion(unsolved, agents_info)
    
    task_files = write_tasks_to_jsonl_per_model(tasks_by_model)
    for model, task_file in task_files.items():
        print(f"\nSubmitting batch for model: {model}")
        results_filename = upload_and_run_batch_job_per_model(task_file)
        discussion_results = parse_batch_results(results_filename)
        output_model_file = os.path.join(OUTPUT_DIR, f"discussion_results_{model.replace('/', '_')}.json")
        with open(output_model_file, "w") as f:
            json.dump(discussion_results, f, indent=2)
        print(f"Discussion results for model {model} saved to {output_model_file}")

if __name__ == "__main__":
    main()
