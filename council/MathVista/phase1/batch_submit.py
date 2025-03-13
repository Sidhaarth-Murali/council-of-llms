import os
import json
import time
import re
from io import BytesIO
import base64
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

def encode_image_to_base64(image):
    """
    Converts a PIL image to a Base64-encoded PNG string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_str

def create_prompt_mathvista(agent_id, role, question):
    """
    Constructs a prompt for MathVista.
    Instruct the agent (a math expert) to analyze the attached image and the question.
    The agent is expected to provide a detailed chain-of-thought in JSON format.
    """
    prompt = (
        f"You are Agent {agent_id} and your role is {role} (a math expert). "
        "Your task is to analyze the attached image and answer the following question. "
        "Provide your chain-of-thought reasoning in a step-by-step manner in JSON format. "
        "Your output must be a valid JSON object with keys such as 'Step1', 'Step2', etc., "
        "and include a final key 'Answer' formatted as 'Answer: {final_answer}', as well as a key 'Confidence' "
        "formatted as 'Confidence: {confidence_value}%'.\n\n"
        f"Question: {question}\n"
        "Please provide your final answer."
    )
    return prompt

def create_batch_tasks_mathvista(dataset, sample_size, agents_info):
    """
    Creates batch tasks for the MathVista dataset.
    Expects each example to have a 'question' and a 'decoded_image' field.
    Filters out examples where 'choices' is not null (i.e. keeps non-MCQ ones only).
    Returns a dict: {model: list_of_tasks}
    """
    tasks_by_model = {}
    for idx, example in enumerate(dataset.select(range(sample_size))):
        # Filter out examples with non-null choices (i.e. MCQ ones)
        if "choices" in example and example["choices"] is not None:
            continue

        question = example.get("question")
        image = example.get("decoded_image")
        if not question or not isinstance(question, str) or not question.strip():
            continue
        if image is None:
            continue

        # Encode the image
        encoded_image = encode_image_to_base64(image)
        base64_url = f"data:image/png;base64,{encoded_image}"
        
        for agent_id, role, model in agents_info:
            user_prompt = create_prompt_mathvista(agent_id, role, question)
            task = {
                "custom_id": f"problem_{idx}_{agent_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": base64_url}}
                            ]
                        }
                    ]
                }
            }
            tasks_by_model.setdefault(model, []).append(task)
    return tasks_by_model

def write_tasks_to_jsonl_per_model(tasks_by_model):
    """
    Writes separate JSONL files per model.
    Returns a dictionary {model: jsonl_filename}.
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
        time.sleep(10)
    
    result_content = client.files.content(status.output_file_id).content
    results_filename = os.path.join(OUTPUT_DIR, f"batch_job_results_phase1_{os.path.basename(tasks_file)}.jsonl")
    with open(results_filename, "wb") as f:
        f.write(result_content)
    print(f"Batch results saved to {results_filename}")
    return results_filename

def phase1_batch_pipeline_mathvista(sample_size=300):
    agents_info = [
        ("A1", "Critic", "gpt-4o-mini"),
        ("A2", "Solver", "gpt-4o"),
        ("A3", "Verifier", "gpt-4-turbo")
    ]
    # Load the MathVista dataset (testmini split)
    dataset = load_dataset("AI4Math/MathVista", split="test")

    # Create batch tasks; only include non-MCQ examples (choices is null)
    tasks_by_model = create_batch_tasks_mathvista(dataset, sample_size, agents_info)

    # Write JSONL task files per model
    task_files = write_tasks_to_jsonl_per_model(tasks_by_model)

    # Submit batch jobs for each model
    for model, task_file in task_files.items():
        print(f"\nSubmitting batch for model: {model}")
        upload_and_run_batch_job_per_model(task_file)

if __name__ == "__main__":
    phase1_batch_pipeline_mathvista(sample_size=300)