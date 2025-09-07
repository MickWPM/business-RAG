import requests
import json
import time
import os
import argparse
import logging

# --- Configuration ---
# Set up basic logging to see the progress and any potential issues.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


RAGBOT_API_URL = "http://localhost:8000/query"
RESULTS_FILE = "rag_test_results.json"


def load_master_test_plan(folder_path):
    if not os.path.isdir(folder_path):
        logging.error(f"Evaluation folder not found at: {folder_path}")
        return []

    master_suite = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files in '{folder_path}'.")

    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    master_suite.extend(data)
                    logging.info(f"Successfully loaded and added {len(data)} test cases from {filename}.")
                else:
                    logging.warning(f"File {filename} does not contain a JSON list. Skipping.")
        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from {filename}. Skipping.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading {filename}: {e}")

    return master_suite

def check_server_readiness(url, retries=5, delay=3):
    logging.info(f"Checking if RAG Bot server is ready at {url}...")
    server_root_url = url.replace("/query", "/")
    for i in range(retries):
        try:
            response = requests.get(server_root_url, timeout=2)
            if response.status_code == 200:
                logging.info("RAG Bot Server is up.")
                return True
        except requests.exceptions.ConnectionError:
            logging.warning(f"Connection attempt {i+1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def run_evaluation_test(eval_folder='eval'):
    #Evaluation consists of the following steps:
        #1. Collate all test cases into a single master test plan
        #2. Send the questions to the Ragbot API
        #3. Save all results including sources and original ground truth to test_results.json file

    master_test_plan = load_master_test_plan(eval_folder)
    if not master_test_plan:
        logging.fatal(f"No test cases loaded from '{eval_folder}'. Aborting.")
        return

    if not check_server_readiness(RAGBOT_API_URL):
        logging.fatal("\RAG Bot Server did not become ready. Aborting tests.")
        logging.fatal("Please ensure 'ragbot_server.py' is running.")
        return

    test_results = []
    total_questions = len(master_test_plan)
    logging.info(f"\n--- Starting RAG Bot Evaluation Test ---")
    logging.info(f"Consolidated {total_questions} questions from '{eval_folder}'")

    for i, item in enumerate(master_test_plan):
        question_text = item.get("question_text")
        if not question_text:
            logging.warning(f"Skipping invalid item at index {i} (missing 'question_text')")
            continue

        logging.info(f"\n--- Test Case {i+1}/{total_questions} (ID: {item.get('question_id', 'N/A')}) ---")
        logging.info(f"Question: {question_text}")

        payload = {"query": question_text}

        try:
            response = requests.post(RAGBOT_API_URL, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            generated_answer = response_data.get("answer", "N/A").strip()
            retrieved_contexts = response_data.get("source_documents", [])

            logging.info(f"RAG Answer: {generated_answer}")
            logging.info(f"Retrieved {len(retrieved_contexts)} source documents.")

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
            logging.error(f"[ERROR] An error occurred during the request: {e}")
            continue
        except json.JSONDecodeError:
            logging.error(f"[ERROR] Failed to decode JSON response from server. Raw Response: {response.text}")

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4)
    logging.info(f"\n--- Evaluation Test Complete ---")
    logging.info(f"Comprehensive results saved to '{RESULTS_FILE}'")

def run_from_command_line():
    parser = argparse.ArgumentParser(description="Run a RAG evaluation suite against a RAG Bot server.")
    parser.add_argument(
        "--eval_folder",
        type=str,
        default="eval", 
        help="Path to the folder containing the evaluation .json files. Defaults to './eval'."
    )
    args = parser.parse_args()
    
    run_evaluation_test(args.eval_folder)

#Added ability to run from command line with eval folder parameter. Most of the time this is not used but allows for future alternative evaluation approaches. 
if __name__ == "__main__":
    run_from_command_line()
