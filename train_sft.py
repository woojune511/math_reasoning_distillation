from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
max_seq_length = 2048 # Supports RoPE Scaling internally, so choose any!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 2. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 4. Load Dataset & Formatting
# Bespoke-Stratos-17k structure check needed. Assuming typical columns or mapped.
# Based on search: might have 'question', 'reasoning', 'answer' or similar.
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split = "train")

def formatting_prompts_func(examples):
    prompts = []
    
    # Dataset has 'conversations' column
    convs = examples["conversations"]
    
    for conv in convs:
        # Extract user and assistant messages
        user_msg = ""
        assistant_msg = ""
        
        for turn in conv:
            if turn["from"] == "user":
                user_msg = turn["value"]
            elif turn["from"] == "assistant":
                assistant_msg = turn["value"]
        
        if not user_msg or not assistant_msg:
            continue
            
        # Transform tags to DeepSeek-R1 style <think> ... </think>
        # Current: <|begin_of_thought|> ... <|end_of_thought|> ... <|begin_of_solution|> ... <|end_of_solution|>
        # Target: <think> ... </think> ...
        
        assistant_msg = assistant_msg.replace("<|begin_of_thought|>", "<think>")
        assistant_msg = assistant_msg.replace("<|end_of_thought|>", "</think>")
        assistant_msg = assistant_msg.replace("<|begin_of_solution|>", "")
        assistant_msg = assistant_msg.replace("<|end_of_solution|>", "")
        
        # Apply Qwen Chat Template manually or via tokenizer
        # <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n{msg}<|im_end|>
        
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ], tokenize=False, add_generation_prompt=False)
        
        prompts.append(text)
        
    return { "text" : prompts }

dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can speed up training for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 1, # Train on full dataset (approx 2125 steps)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use wandb if needed
    ),
)

trainer.train()

# 6. Save
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Model saved to lora_model")
