import requests
import json
import os
import time

# --- Configuration ---
# The results file from the test script.
RESULTS_FILE = "rag_test_results.json"
# The URL for the raw Ollama API endpoint.
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# The model to use for evaluation.
EVALUATOR_LLM_MODEL = "llama3:8b"

EVALUATION_PROMPT_TEMPLATE = """
You are a meticulous and impartial evaluator for Retrieval-Augmented Generation (RAG) systems.
Your task is to assess a generated answer based on the user's question, a ground truth answer, the context that the RAG system retrieved, and the ideal context it should have retrieved.

Evaluate on four metrics using a strict 1-5 scale:
1.  **Context Relevance**: How well does the `Retrieved Context` cover the key information outlined in the `Ground Truth Quotes`?
    - 5: The retrieved context contains all the information from the ground truth quotes.
    - 3: The retrieved context contains some, but not all, of the key information from the ground truth quotes.
    - 1: The retrieved context is completely irrelevant and contains none of the key information.
2.  **Faithfulness**: Is the `Generated Answer` fully and exclusively supported by the `Retrieved Context`?
    - 5: The answer is 100% supported by the context and contains no extra information.
    - 3: The answer is mostly supported by the context but makes a minor claim not found in the context.
    - 1: The answer contains significant information not found in the context (hallucination).
3.  **Correctness**: How factually accurate is the `Generated Answer` when compared to the `Ground Truth Answer`?
    - 5: The generated answer is a perfect match or a perfect superset of the ground truth.
    - 4: The generated answer is mostly correct but misses a minor detail from the ground truth.
    - 3: The generated answer has some correct information but also contains a noticeable factual error.
    - 1: The generated answer is completely factually incorrect.
4.  **Answer Relevance**: How well does the `Generated Answer` directly address the user's `User Question`?
    - 5: The answer is perfectly on-topic and concise.
    - 3: The answer is on-topic but contains some unnecessary, un-asked for information.
    - 1: The answer is completely off-topic.

**User Question**: "{question}"
**Ground Truth Answer**: "{ground_truth_answer}"
**Ground Truth Quotes (Ideal Context)**:
```json
{ground_truth_quotes}
```
**Retrieved Context**:
```json
{retrieved_contexts}
```
**Generated Answer**: "{generated_answer}"

Provide your evaluation *only* in the following JSON format. Do not include any other text, explanations, or markdown formatting outside of the JSON structure.

{{
  "context_relevance_score": <score_integer from 1-5>,
  "context_relevance_justification": "<brief justification for the context relevance score>",
  "faithfulness_score": <score_integer from 1-5>,
  "faithfulness_justification": "<brief justification for the faithfulness score>",
  "correctness_score": <score_integer from 1-5>,
  "correctness_justification": "<brief justification for the correctness score>",
  "answer_relevance_score": <score_integer from 1-5>,
  "answer_relevance_justification": "<brief justification for the answer relevance score>"
}}
"""

def check_ollama_readiness(url, retries=3, delay=3):
    """Checks if the raw Ollama server is ready."""
    print(f"Checking if raw Ollama server is ready at {url}...")
    server_root_url = url.replace("/api/generate", "/")
    for i in range(retries):
        try:
            response = requests.get(server_root_url, timeout=2)
            if response.status_code == 200 and "Ollama is running" in response.text:
                print("Raw Ollama Server is up.")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Connection attempt {i+1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def calculate_retrieval_recall(ground_truth_path, retrieved_contexts):
    """
    Calculates the recall for retrieved documents based on UNIQUE documents.
    This version is robust to extra descriptive text in retrieved filenames.
    """
    if not ground_truth_path:
        return 1.0

    unique_expected_doc_ids = {item['doc_id'] for item in ground_truth_path}
    if not unique_expected_doc_ids:
        return 1.0

    unique_retrieved_doc_files = set()
    for item in retrieved_contexts:
        base_name = os.path.splitext(item['source_file'])[0]
        identifier = base_name.split()[0]
        unique_retrieved_doc_files.add(identifier)

    found_docs = unique_expected_doc_ids.intersection(unique_retrieved_doc_files)
    recall = len(found_docs) / len(unique_expected_doc_ids)
    return recall


def evaluate_with_llm(question, ground_truth_answer, generated_answer, retrieved_contexts, ground_truth_path):
    """Uses a raw LLM to evaluate the generated answer against multiple metrics."""
    context_str = json.dumps(retrieved_contexts, indent=2)
    quotes_str = json.dumps(ground_truth_path, indent=2)

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        generated_answer=generated_answer,
        retrieved_contexts=context_str,
        ground_truth_quotes=quotes_str
    )

    payload = {"model": EVALUATOR_LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"}

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        evaluation_json = json.loads(response_data.get("response", "{}"))
        return evaluation_json
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not connect to Ollama server: {e}")
        return None
    except json.JSONDecodeError:
        print(f"\n[ERROR] Failed to decode JSON from LLM evaluator. Raw response: {response_data.get('response')}")
        return None

def run_metrics_evaluation():
    """Loads results, uses an LLM to evaluate them, and provides a categorized summary."""
    if not os.path.exists(RESULTS_FILE):
        print(f"[FATAL] Results file not found: '{RESULTS_FILE}'")
        return

    if not check_ollama_readiness(OLLAMA_API_URL):
        print("\n[FATAL] Raw Ollama server not ready. Please ensure it's running.")
        return

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"\n--- Starting RAG Metrics Evaluation using {EVALUATOR_LLM_MODEL} ---")

    all_evaluations = []
    for i, result in enumerate(results):
        print(f"\n--- Evaluating Result {i+1}/{len(results)} (ID: {result.get('question_id', 'N/A')}) ---")
        print(f"Question: {result['question_text']}")

        retrieval_recall = calculate_retrieval_recall(
            result.get('ground_truth_retrieval_path', []),
            result.get('retrieved_contexts', [])
        )
        print(f"  - Document Recall: {retrieval_recall:.2%}")

        if result.get("question_type") == "UNANSWERABLE":
            result['evaluation_scores'] = {'retrieval_recall': retrieval_recall}
            all_evaluations.append(result)
            continue

        llm_evaluation = evaluate_with_llm(
            result['question_text'],
            result['ground_truth_answer'],
            result['generated_answer'],
            result['retrieved_contexts'],
            result.get('ground_truth_retrieval_path', [])
        )

        if llm_evaluation:
            print(f"  - Context Relevance: {llm_evaluation.get('context_relevance_score', 0)}/5 | Faithfulness: {llm_evaluation.get('faithfulness_score', 0)}/5 | Correctness: {llm_evaluation.get('correctness_score', 0)}/5")
            llm_evaluation['retrieval_recall'] = retrieval_recall
            result['evaluation_scores'] = llm_evaluation
        else:
            print("  - LLM-based evaluation failed for this item.")
            result['evaluation_scores'] = {'retrieval_recall': retrieval_recall}

        all_evaluations.append(result)
        time.sleep(1)

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_evaluations, f, indent=4)
    print(f"\nUpdated results with evaluation scores have been saved to '{RESULTS_FILE}'")

    print("\n" + "="*25 + " FINAL METRICS SUMMARY " + "="*25)
    
    simple_types = ["SIMPLE_RETRIEVAL", "LIST_EXTRACTION"]
    complex_types = ["SYNTHESIS", "CHAINED_RETRIEVAL"]

    metrics = {
        'overall': {'c': 0, 'r': 0, 'f': 0, 'recall': 0, 'context': 0, 'count': 0, 'llm_count': 0},
        'simple': {'c': 0, 'r': 0, 'f': 0, 'recall': 0, 'context': 0, 'count': 0, 'llm_count': 0},
        'complex': {'c': 0, 'r': 0, 'f': 0, 'recall': 0, 'context': 0, 'count': 0, 'llm_count': 0}
    }

    for result in all_evaluations:
        scores = result.get('evaluation_scores')
        if not scores: continue
        
        q_type = result.get('question_type')
        category = 'simple' if q_type in simple_types else 'complex' if q_type in complex_types else None

        metrics['overall']['recall'] += scores.get('retrieval_recall', 0)
        metrics['overall']['count'] += 1
        if category:
            metrics[category]['recall'] += scores.get('retrieval_recall', 0)
            metrics[category]['count'] += 1

        if q_type != "UNANSWERABLE" and 'correctness_score' in scores:
            metrics['overall']['c'] += scores.get('correctness_score', 0)
            metrics['overall']['r'] += scores.get('answer_relevance_score', 0)
            metrics['overall']['f'] += scores.get('faithfulness_score', 0)
            metrics['overall']['context'] += scores.get('context_relevance_score', 0)
            metrics['overall']['llm_count'] += 1
            if category:
                metrics[category]['c'] += scores.get('correctness_score', 0)
                metrics[category]['r'] += scores.get('answer_relevance_score', 0)
                metrics[category]['f'] += scores.get('faithfulness_score', 0)
                metrics[category]['context'] += scores.get('context_relevance_score', 0)
                metrics[category]['llm_count'] += 1

    if metrics['overall']['count'] > 0:
        print("\n--- Overall Performance ---")
        print(f"Average Document Recall:      {metrics['overall']['recall'] / metrics['overall']['count']:.2%}")
        if metrics['overall']['llm_count'] > 0:
            print(f"Average Context Relevance:    {metrics['overall']['context'] / metrics['overall']['llm_count']:.2f} / 5.0")
            print(f"Average Faithfulness Score:   {metrics['overall']['f'] / metrics['overall']['llm_count']:.2f} / 5.0")
            print(f"Average Correctness Score:    {metrics['overall']['c'] / metrics['overall']['llm_count']:.2f} / 5.0")
            print(f"Average Answer Relevance:     {metrics['overall']['r'] / metrics['overall']['llm_count']:.2f} / 5.0")

    else:
        print("No items were successfully evaluated.")

    print("\n--- Performance by Query Type ---")
    for cat_name, cat_types in [('Simple', simple_types), ('Complex', complex_types)]:
        cat_metrics = metrics[cat_name.lower()]
        if cat_metrics['count'] > 0:
            print(f"\n  {cat_name} Queries ({', '.join(cat_types)}):")
            print(f"    - Avg Document Recall:   {cat_metrics['recall'] / cat_metrics['count']:.2%}")
            if cat_metrics['llm_count'] > 0:
                print(f"    - Avg Context Relevance: {cat_metrics['context'] / cat_metrics['llm_count']:.2f} / 5.0")
                print(f"    - Avg Faithfulness:      {cat_metrics['f'] / cat_metrics['llm_count']:.2f} / 5.0")
                print(f"    - Avg Correctness:       {cat_metrics['c'] / cat_metrics['llm_count']:.2f} / 5.0")
                print(f"    - Avg Answer Relevance:  {cat_metrics['r'] / cat_metrics['llm_count']:.2f} / 5.0")
        else:
            print(f"\n  No '{cat_name}' type questions were evaluated.")
    
    print("\n" + "="*73)

if __name__ == "__main__":
    run_metrics_evaluation()
