import requests
import json
import time
import os

# --- Configuration ---
# The URL of your local lisabot server's API endpoint.
LISABOT_API_URL = "http://localhost:8000/query"
# The input file with evaluation questions.
QUESTIONS_FILE = "evaluation_questions.json"
# The output file to save the full evaluation results.
RESULTS_FILE = "rag_test_results.json"

def check_server_readiness(url, retries=5, delay=3):
    """
    Checks if the LISABot server is ready to accept connections.
    """
    print(f"Checking if LISABot server is ready at {url}...")
    server_root_url = url.replace("/query", "/")
    for i in range(retries):
        try:
            response = requests.get(server_root_url, timeout=2)
            if response.status_code == 200:
                print("LISABot Server is up.")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Connection attempt {i+1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def run_evaluation_test():
    """
    Loads questions, sends them to the LISABot API, and saves the
    comprehensive results, including retrieved contexts, to a JSON file.
    """
    if not os.path.exists(QUESTIONS_FILE):
        print(f"[FATAL] Questions file not found: '{QUESTIONS_FILE}'")
        print("Please create this file with your evaluation questions.")
        return

    if not check_server_readiness(LISABOT_API_URL):
        print("\n[FATAL] LISABot Server did not become ready. Aborting tests.")
        print("Please ensure 'lisabot_server.py' is running.")
        return

    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    test_results = []
    print(f"\n--- Starting LISABot Evaluation Test ---")
    print(f"Loaded {len(questions)} questions from '{QUESTIONS_FILE}'")

    for i, item in enumerate(questions):
        question_text = item.get("question_text")
        if not question_text:
            print(f"Skipping invalid item at index {i} (missing 'question_text')")
            continue

        print(f"\n--- Test Case {i+1}/{len(questions)} (ID: {item.get('question_id', 'N/A')}) ---")
        print(f"Question: {question_text}")

        payload = {"query": question_text}

        try:
            response = requests.post(LISABOT_API_URL, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            generated_answer = response_data.get("answer", "N/A").strip()
            # Capture the source documents retrieved by the RAG system
            retrieved_contexts = response_data.get("source_documents", [])

            print(f"RAG Answer: {generated_answer}")
            print(f"Retrieved {len(retrieved_contexts)} source documents.")

            # Build the full result object based on the new structure
            result_to_save = {
                "question_id": item.get("question_id"),
                "question_type": item.get("question_type"),
                "question_text": question_text,
                "ground_truth_answer": item.get("ground_truth_answer"),
                "ground_truth_retrieval_path": item.get("retrieval_path", []),
                "generated_answer": generated_answer,
                "retrieved_contexts": retrieved_contexts
            }
            test_results.append(result_to_save)
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] An error occurred during the request: {e}")
            continue
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON response from server. Raw Response: {response.text}")

    # Save the comprehensive results to a file
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4)
    print(f"\n--- Evaluation Test Complete ---")
    print(f"Comprehensive results saved to '{RESULTS_FILE}'")

if __name__ == "__main__":
    run_evaluation_test()
