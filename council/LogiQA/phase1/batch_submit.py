import os
import json
import time
import re
import math
from datasets import load_dataset
from openai import OpenAI

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=openai_api_key)

# Define and create output directory for phase1 results
OUTPUT_DIR = os.path.join(os.getcwd(), "phase1_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_prompt_logiqa(agent_id, role, context, query, options):
    """
    Constructs a prompt for LogiQA.
    The prompt includes the context, the query, and a formatted list of options.
    """
    # Format options: if it's a dict, format as "key: value"; if list, enumerate as A:, B:, etc.
    if isinstance(options, dict):
        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    elif isinstance(options, list):
        options_str = "\n".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(options)])
    else:
        options_str = str(options)
        
    prompt = (
        f"You are Agent {agent_id} and your role is {role} (a logical reasoning expert). "
        "Your task is to analyze the following context and query, and choose the best answer from the provided options. "
        "Provide your chain-of-thought reasoning in a step-by-step manner in JSON format. "
        "Your output must be a valid JSON object with keys like 'Step1', 'Step2', etc., "
        "and include a final key 'Answer' formatted as 'Answer: {final_answer}', as well as a key 'Confidence' "
        "formatted as 'Confidence: {confidence_value}%'.\n\n"
        f"Context: {context}\n"
        f"Query: {query}\n"
        f"Options:\n{options_str}\n"
        "Please choose the best option."
    )
    return prompt

def create_batch_tasks_logiqa(dataset, sample_size, agents_info):
    """
    Create batch tasks for the LogiQA dataset.
    Expects each example to have 'context', 'query', and 'options' fields.
    Returns a dict: {model: list_of_tasks}
    """
    tasks_by_model = {}
    for idx, example in enumerate(dataset.select(range(sample_size))):
        context = example.get("context")
        query = example.get("query")
        options = example.get("options")
        if not context or not isinstance(context, str) or not context.strip():
            continue
        if not query or not isinstance(query, str) or not query.strip():
            continue
        if not options:
            continue
        for agent_id, role, model in agents_info:
            user_prompt = create_prompt_logiqa(agent_id, role, context, query, options)
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
    Writes separate JSONL files per model.
    Returns a dictionary {model_name: jsonl_filename}.
    """
    task_files = {}
    for model, tasks in tasks_by_model.items():
        filename = os.path.join(OUTPUT_DIR, f"batch_tasks_phase1_{model}.jsonl")
        with open(filename, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        task_files[model] = filename
    return task_files

def upload_and_run_batch_job_per_model(tasks_file):
    """
    Uploads and runs a batch job for a given JSONL file.
    Saves the resulting output file to the OUTPUT_DIR.
    """
    batch_file = client.files.create(file=open(tasks_file, "rb"), purpose="batch")
    print(f"Batch file {tasks_file} uploaded:", batch_file.id)

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
        time.sleep(30)
    result_content = client.files.content(status.output_file_id).content
    results_filename = os.path.join(OUTPUT_DIR, f"batch_job_results_phase1_{os.path.basename(tasks_file)}.jsonl")
    with open(results_filename, "wb") as f:
        f.write(result_content)
    print(f"Batch results saved to {results_filename}")
    return results_filename

def phase1_batch_pipeline_logiqa(sample_size=300):
    agents_info = [
        ("A1", "Critic", "gpt-4o-mini"),
        ("A2", "Solver", "gpt-4o"),
        ("A3", "Verifier", "gpt-3.5-turbo")
    ]
    # Load the LogiQA dataset (test split)
    dataset = load_dataset("lucasmccabe/logiqa", split="test")
    
    # Create batch tasks using context, query, and options
    tasks_by_model = create_batch_tasks_logiqa(dataset, sample_size, agents_info)
    
    # Write separate JSONL files per model
    task_files = write_tasks_to_jsonl_per_model(tasks_by_model)
    
    # Submit batch jobs for each model
    for model, task_file in task_files.items():
        print(f"\nSubmitting batch for model: {model}")
        upload_and_run_batch_job_per_model(task_file)

if __name__ == "__main__":
    phase1_batch_pipeline_logiqa(sample_size=300)
