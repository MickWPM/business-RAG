import requests
import json
import time
import os

# --- Configuration ---
# The URL of your local lisabot server's API endpoint.
LISABOT_API_URL = "http://localhost:8000/query"
# The file to save the evaluation results to.
RESULTS_FILE = "rag_test_results.json"

# The JSON data containing the evaluation questions and answers
# based on the LISA SEMP document.
EVALUATION_QUESTIONS_JSON = """
[
  {
    "question": "What is the primary purpose of the LISA mission?",
    "expected_answer": "The primary purpose of the LISA mission is to detect and study gravitational waves in the low-frequency range (10^-4 to 10^-1 Hz) from galactic and extra-galactic binary systems.",
    "question_type": "simple_retrieval"
  },
  {
    "question": "What are the four major segments that compose the overall LISA system?",
    "expected_answer": "The four major segments are the Flight segment, Launch segment, Ground Operations segment, and the Science Data Processing segment.",
    "question_type": "list_extraction"
  },
  {
    "question": "Who is the chairperson of the Integrated Design Team (IDT)?",
    "expected_answer": "The Mission System Engineering Manager (MSEM) chairs the IDT.",
    "question_type": "simple_retrieval"
  },
  {
    "question": "Which project phase includes the System Requirements Review (SRR), and what is the review's purpose?",
    "expected_answer": "The System Requirements Review (SRR) is held during Phase B. Its purpose is to ensure that the objectives and requirements of the LISA mission are understood and that the system-level requirements meet the mission objectives.",
    "question_type": "synthesis"
  },
  {
    "question": "List the working groups that the Integrated Design Team provides oversight for.",
    "expected_answer": "The Integrated Design Team provides oversight for the Requirements Analysis Working Group, Software Architecture Working Group, Modeling Working Group, and the Integration and Test (I&T) Working Group.",
    "question_type": "list_extraction"
  }
]
"""

def check_server_readiness(url, retries=5, delay=3):
    """
    Checks if the server is ready to accept connections before running tests.
    """
    print(f"Checking if LISABot server is ready at {url}...")
    # The lisabot_server has a GET endpoint at the root now.
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
    Sends each evaluation question to the LISABot API, prints the results,
    and saves them to a JSON file.
    """
    if not check_server_readiness(LISABOT_API_URL):
        print("\n[FATAL] Server did not become ready. Aborting tests.")
        print("Please ensure 'lisabot_server.py' is running.")
        return

    test_results = []
    try:
        questions = json.loads(EVALUATION_QUESTIONS_JSON)
        
        print("\n--- Starting LISABot Evaluation Test ---")
        print(f"Targeting API: {LISABOT_API_URL}\n")

        for i, item in enumerate(questions):
            question_text = item.get("question")
            expected_answer = item.get("expected_answer")

            if not question_text or not expected_answer:
                print(f"Skipping invalid item at index {i}")
                continue

            print(f"--- Test Case {i+1} ---")
            print(f"Question: {question_text}")
            
            payload = {"query": question_text}

            try:
                response = requests.post(LISABOT_API_URL, json=payload, timeout=60)
                response.raise_for_status()
                response_data = response.json()
                actual_answer = response_data.get("answer", "N/A").strip()
                
                print(f"RAG Answer:   {actual_answer}")
                
                # Store the results for saving later
                test_results.append({
                    "question": question_text,
                    "expected_answer": expected_answer,
                    "generated_answer": actual_answer
                })
                print("-" * 20)
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR] An error occurred during the request: {e}")
                # Decide whether to stop or continue
                continue # Continue with the next question
            except json.JSONDecodeError:
                print(f"\n[ERROR] Failed to decode JSON response from server.")
                print(f"Raw Response: {response.text}")

        # Save the results to a file
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=4)
        print(f"\n--- Evaluation Test Complete ---")
        print(f"Results saved to '{RESULTS_FILE}'")

    except json.JSONDecodeError:
        print("[ERROR] The provided JSON string for questions is invalid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_evaluation_test()
