import time
import lisa_test as tester
import evaluate_rag_metrics as eval

def format_time(seconds):
    """Converts a duration in seconds to a formatted string."""
    # Round to the nearest whole number of seconds
    seconds = round(seconds)
    
    # Use divmod to get minutes and remaining seconds
    minutes, seconds = divmod(seconds, 60)
    
    # Use divmod again to get hours and remaining minutes
    hours, minutes = divmod(minutes, 60)
    
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

# --- Main Execution and Timing ---

# 1. Time the entire process
total_start_time = time.time()

# 2. Time the first component (test)
test_start_time = time.time()
tester.run_evaluation_test()
test_time = time.time() - test_start_time

# 3. Time the second component (evaluation)
eval_start_time = time.time()
eval.run_metrics_evaluation()
eval_time = time.time() - eval_start_time

# 4. Calculate the total time elapsed
total_time = time.time() - total_start_time

# --- Print Formatted Results ---

print("\n" + "="*40)
print("--- 📊 Execution Time Report ---")
print("="*40)
print(f"Test Run Time:      {format_time(test_time)}")
print(f"Evaluation Run Time: {format_time(eval_time)}")
print("-"*40)
print(f"Total Execution Time: {format_time(total_time)}")
print("="*40)