import time
import run_rag_tests as tester
import evaluate_rag_metrics as eval

def format_time(seconds):
    # Formats time from seconds to human-readable hours, minutes, and seconds. 
    seconds = round(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

#Keeping track of time for an awareness of compute impost on different parts of the test. 
total_start_time = time.time()


test_start_time = time.time()
tester.run_evaluation_test()
test_time = time.time() - test_start_time

eval_start_time = time.time()
eval.run_metrics_evaluation()
eval_time = time.time() - eval_start_time


total_time = time.time() - total_start_time
print("\n" + "="*40)
print('--- Test and Evaluation Complete ---\n')
print(f"Test Run Time:      {format_time(test_time)}")
print(f"Evaluation Run Time: {format_time(eval_time)}")
print("-"*40)
print(f"Total Execution Time: {format_time(total_time)}")
print("="*40)