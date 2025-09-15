from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from dotenv import load_dotenv

load_dotenv()

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="correctness",
    )
    
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    
    return eval_result

if __name__ == "__main__":
    inputs = {"question": "What is 2 + 2?"}
    outputs = {"answer": "4"}
    reference_outputs = {"expected_answer": "4"}
    
    result = correctness_evaluator(inputs, outputs, reference_outputs)
    print("Evaluation result:", result)