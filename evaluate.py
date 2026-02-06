import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer

def evaluate_gsm8k(model_name, max_new_tokens=1024, load_in_4bit=True, limit=None):
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit:
        dataset = dataset.select(range(limit))
    
    # Qwen chat template
    # We will use a standard prompt for zero-shot CoT or just direct question
    # "Answer the following grade school math problem. Show your work."
    
    correct = 0
    total = 0
    
    results = []
    
    for item in tqdm(dataset):
        question = item['question']
        ground_truth = item['answer']
        
        # Format prompt
        # Using chat template if available, else standard formatting
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step."},
            {"role": "user", "content": question}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids = inputs, 
            max_new_tokens = max_new_tokens,
            use_cache = True
        )
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (part after system/user) - simple extraction
        # Qwen usually handles this well. We'll look for the last assistant response.
        # But tokenizer.decode strips roles usually if not carefully handled?
        # Let's just assume the generation adds to the prompt.
        
        # Unsloth/HF generate returns input + output. 
        # We need to slice the output.
        generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Check answer
        # GSM8K ground truth usually has the answer after ####
        gt_value = ground_truth.split("####")[-1].strip()
        
        # Simple number extraction from generated text
        # Look for the last number or #### pattern
        match = re.search(r"####\s*(-?\d+\.?\d*)", generated_text)
        if match:
            pred_value = match.group(1)
        else:
            # Check for \boxed{...} format (common in CoT models)
            boxed_match = re.findall(r"\\boxed\{([^}]*)\}", generated_text)
            if boxed_match:
                # Take the last boxed value
                pred_value = boxed_match[-1].strip()
            else:
                # Fallback: extract last number
                numbers = re.findall(r"-?\d+\.?\d*", generated_text.replace(',', ''))
                pred_value = numbers[-1] if numbers else ""
            
        is_correct = (pred_value == gt_value)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "generated": generated_text,
            "ground_truth": gt_value,
            "prediction": pred_value,
            "correct": is_correct
        })
        
        if total % 10 == 0:
            print(f"Propcessed {total}: Current Acc: {correct/total:.2%}")

    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.2%}")
    return accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_gsm8k(args.model_name, limit=args.limit)
