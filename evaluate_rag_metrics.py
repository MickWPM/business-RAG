import requests
import json
import os
import time

# --- Configuration ---
# The file containing the results from the RAG test script.
RESULTS_FILE = "rag_test_results.json"
# The URL for the raw Ollama API endpoint (not the RAG server).
# This assumes you are running a standard Ollama instance.
OLLAMA_API_URL = "http://localhost:11434/api/generate" 
# The model to use for evaluation.
EVALUATOR_LLM_MODEL = "llama3:8b"

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for Retrieval-Augmented Generation systems.
Your task is to assess the quality of a generated answer based on a question and a ground truth (expected) answer.
Evaluate the generated answer on two metrics:
1.  **Correctness**: How factually accurate is the generated answer when compared to the ground truth answer? Does it contain any information that contradicts the ground truth?
2.  **Relevance**: How well does the generated answer directly address the user's question? Is it on-topic and to the point?

Provide a score from 1 to 5 for each metric (1=Poor, 5=Excellent) and a brief justification for each score.

**Question**: "{question}"
**Ground Truth Answer**: "{expected_answer}"
**Generated Answer**: "{generated_answer}"

Provide your evaluation *only* in the following JSON format. Do not include any other text or explanations outside of the JSON structure.

{{
  "correctness_score": <score_integer>,
  "correctness_justification": "<justification_text>",
  "relevance_score": <score_integer>,
  "relevance_justification": "<justification_text>"
}}
"""

def check_ollama_readiness(url, retries=3, delay=3):
    """
    Checks if the raw Ollama server is ready.
    """
    print(f"Checking if raw Ollama server is ready at {url}...")
    server_root_url = url.replace("/api/generate", "/")
    for i in range(retries):
        try:
            # A standard Ollama instance responds to a GET at the root.
            response = requests.get(server_root_url, timeout=2)
            if response.status_code == 200 and "Ollama is running" in response.text:
                print("Raw Ollama Server is up.")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Connection attempt {i+1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def evaluate_with_llm(question, expected, generated):
    """
    Uses a raw LLM to evaluate the correctness and relevance of a generated answer.
    """
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        expected_answer=expected,
        generated_answer=generated
    )
    
    payload = {
        "model": EVALUATOR_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Request JSON output from Ollama
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        response.raise_for_status()
        response_data = response.json()
        # The actual JSON content is in the 'response' key, as a string.
        evaluation_json = json.loads(response_data.get("response", "{}"))
        return evaluation_json
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to Ollama server: {e}")
        return None
    except json.JSONDecodeError:
        print(f"\n[ERROR] Failed to decode JSON from LLM evaluator.")
        print(f"Raw response from evaluator: {response_data.get('response')}")
        return None

def run_metrics_evaluation():
    """
    Loads results and uses an LLM to evaluate them.
    """
    if not os.path.exists(RESULTS_FILE):
        print(f"[FATAL] Results file not found: '{RESULTS_FILE}'")
        print("Please run the 'lisabot_test_script.py' first to generate the results.")
        return

    if not check_ollama_readiness(OLLAMA_API_URL):
        print("\n[FATAL] Raw Ollama server not ready.")
        print("Please ensure Ollama is running (e.g., 'ollama serve').")
        return

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"\n--- Starting RAG Metrics Evaluation using {EVALUATOR_LLM_MODEL} ---")
    
    total_correctness = 0
    total_relevance = 0
    evaluated_count = 0

    for i, result in enumerate(results):
        print(f"\n--- Evaluating Result {i+1}/{len(results)} ---")
        print(f"Question: {result['question']}")
        
        evaluation = evaluate_with_llm(
            result['question'],
            result['expected_answer'],
            result['generated_answer']
        )
        
        if evaluation:
            corr_score = evaluation.get('correctness_score', 0)
            corr_just = evaluation.get('correctness_justification', 'N/A')
            rel_score = evaluation.get('relevance_score', 0)
            rel_just = evaluation.get('relevance_justification', 'N/A')

            print(f"  - Correctness Score: {corr_score}/5")
            print(f"    Justification: {corr_just}")
            print(f"  - Relevance Score:   {rel_score}/5")
            print(f"    Justification: {rel_just}")
            
            total_correctness += corr_score
            total_relevance += rel_score
            evaluated_count += 1
        else:
            print("  - Evaluation failed for this item.")
        
        time.sleep(1)

    print("\n--- Final Metrics Summary ---")
    if evaluated_count > 0:
        avg_correctness = total_correctness / evaluated_count
        avg_relevance = total_relevance / evaluated_count
        print(f"Average Correctness Score: {avg_correctness:.2f} / 5.0")
        print(f"Average Relevance Score:   {avg_relevance:.2f} / 5.0")
    else:
        print("No items were successfully evaluated.")
    print("-----------------------------")

if __name__ == "__main__":
    run_metrics_evaluation()
