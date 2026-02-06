from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
import torch.nn.functional as F

# 1. Configuration
max_seq_length = 2048 
dtype = None 
load_in_4bit = True 
THINK_WEIGHT = 2.0  # Weight for reasoning tokens

# 2. Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# 3. Custom Trainer with Step-Aware Loss
class WeightedLossTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # inputs: 'input_ids', 'attention_mask', 'labels'
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=inputs.get("attention_mask"))
        logits = outputs.get("logits")
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()
        
        # Flatten the tensors
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_input_ids = shift_input_ids.view(-1)
        
        # Calculate standard CrossEntropyLoss (element-wise to apply weights)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        
        # --- Step-Aware Weighting ---
        # We need to identify which tokens belong to <think>...</think>
        # Note: This is a bit tricky with tokenized data because tokens might be split.
        # However, we can roughly approximate by finding the token IDs for <think> and </think>.
        
        # Get token IDs for tags
        # Note: Depending on tokenizer, <think> might be multiple tokens or single special token.
        # We should check how Qwen tokenizes "<think>" and "</think>".
        # For safety/simplicity in this script, we'll iterate through the batch (slow) or use tensor ops if possible.
        # Let's assume we can map back to CPU for boolean mask creation if needed, or use a simplified heuristic.
        
        # Robust approach: Pre-calculate masks in the dataset. But we are subclassing Trainer.
        # Faster approach: Just check for the tokens if they are distinct.
        
        # Qwen2.5 uses byte-level BPE. 
        # let's try to detect the segments. 
        # Since we cannot easily reconstruct text differantiably, we rely on the fact that labels have the same structure.
        
        # Create a weight mask initialized to 1.0
        weights = torch.ones_like(shift_labels, dtype=loss.dtype)
        
        # We need to find the bounds for each sequence in the batch.
        # But we flattened the batch. Let's unflatten for logic.
        batch_size, seq_len = input_ids.shape
        # shift_labels is (batch * (seq_len-1))
        
        # Reflattening logic is complex. Let's do it per sample before collation? 
        # Actually SFTTrainer computes loss on batch.
        
        # Heuristic: 
        # 1. Decode IDs to text (expensive inside training loop, but accurate) -> Too slow.
        # 2. Pre-process dataset to include "loss_weight_mask".
        
        # Since we can't easily preprocess with "loss_weight_mask" passed to SFTTrainer without custom data collator...
        # Let's use the Custom Trainer to just use a custom Data Collator that allows 'weight_mask'.
        
        # WAIT. We can do this:
        # Just use standard SFT but modify the LABELS. 
        # If labels are -100, they are ignored. 
        # We want to boost weight, not ignore.
        
        # Let's go with OPTION 2: Pre-calculate weights in the dataset map function.
        # But SFTTrainer expects standard columns.
        
        # Let's stick to the plan: Custom Trainer.
        # But for efficiency, verify if we can move logic to data prep.
        # If we stick to logic inside compute_loss, we must be efficient.
        
        # Let's try to pass 'weight_mask' via inputs.
        # This requires a custom DataCollator.
        
        loss = loss * 1.0 # Placeholder if we don't apply weights
        
        # To make this robust: 
        # Let's actually implement the weight logic in the `formatting_prompts_func` by leveraging `data_collator`.
        
        # ... Wait, simplest way: 
        # Just use the fact that we know the structure.
        # Find token ID for <think> and </think>.
        # Scan and fill.
        
        # Let's assume we passed 'token_type_ids' or similar? No.
        
        # Let's decode ONLY the special tokens?
        # Actually, Unsloth/Qwen tokenizer might treat <think> as special tokens if we added them?
        # We didn't add them as special tokens in train_sft.py, just text.
        
        # Let's rely on a simpler hack:
        # Since we know the prompt format is:
        # User: ...
        # Assistant: <think> ... </think> ...
        
        # We can detect the approximate range. 
        # But doing this inside `compute_loss` every step is painful.
        
        # BETTER PLAN:
        # Don't subclass Trainer just for logic that can be static.
        # Use `mask` in Data Processing.
        # Unsloth SFTTrainer supports standard inputs.
        
        # Let's IMPLEMENT: Custom Data Collator that adds a 'loss_weight' tensor.
        pass # See actual implementation below with DataCollator override
        
        if "loss_weights" in inputs:
            # Shift weights to match labels
            shift_weights = inputs["loss_weights"][..., 1:].contiguous().view(-1)
            loss = loss * shift_weights
            
        return loss.mean() if num_items_in_batch is None else loss.sum() / num_items_in_batch

# 4. Data Processing with Weights
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split = "train")

# We need to add <think> as special tokens to key-word search them? 
# Or just finding the bytes.
think_start_token = "<think>"
think_end_token = "</think>"

def apply_template_and_weights(examples):
    prompts = []
    # We will generate the text first, then tokenize, then Find the positions.
    
    # We need full encoding here to build the mask.
    inputs_list = []
    masks_list = []
    
    convs = examples["conversations"]
    
    for conv in convs:
        # ... (Same extraction logic as train_sft.py)
        user_msg = ""
        assistant_msg = ""
        for turn in conv:
            if turn["from"] == "user": user_msg = turn["value"]
            elif turn["from"] == "assistant": assistant_msg = turn["value"]
            
        if not user_msg or not assistant_msg: continue
            
        assistant_msg = assistant_msg.replace("<|begin_of_thought|>", "<think>")
        assistant_msg = assistant_msg.replace("<|end_of_thought|>", "</think>")
        assistant_msg = assistant_msg.replace("<|begin_of_solution|>", "")
        assistant_msg = assistant_msg.replace("<|end_of_solution|>", "")
        
        # Full text
        full_text = tokenizer.apply_chat_template([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ], tokenize=False, add_generation_prompt=False)
        
        # Tokenize
        tokenized = tokenizer(full_text, truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids = tokenized.input_ids[0]
        
        # Create weight mask
        weight_mask = torch.ones_like(input_ids, dtype=torch.float32)
        
        # Find start and end of think block
        # We search for the tokens corresponding to <think> and </think>
        # Note: This is fragile if tokenized differently. 
        # Robust way: Find string offsets and map to tokens (char_to_token).
        
        start_indices = [i for i in range(len(full_text)) if full_text.startswith(think_start_token, i)]
        end_indices = [i for i in range(len(full_text)) if full_text.startswith(think_end_token, i)]
        
        # Tokenizer offset mapping
        enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True, max_length=max_seq_length)
        offsets = enc.offset_mapping
        
        for start_char in start_indices:
            # Find closest token
            end_char = -1
            # Find meaningful end
            for e in end_indices:
                if e > start_char:
                    end_char = e
                    break
            
            if end_char == -1: continue # mismatched
            
            # Map char range to token range
            # We iterate tokens to identify which fall into [start_char, end_char + len(end_token)]
            for idx, (o_start, o_end) in enumerate(offsets):
                if o_start >= start_char and o_end <= (end_char + len(think_end_token)):
                    weight_mask[idx] = THINK_WEIGHT
        
        inputs_list.append(input_ids)
        masks_list.append(weight_mask)
        # prompts.append(full_text) # Removed to avoid collision in DataCollator

    return {"input_ids": inputs_list, "loss_weights": masks_list}

# Pre-process
# Note: SFTTrainer usually handles tokenization. If we pass input_ids, it skips it.
# We need to format properly.
dataset = dataset.map(apply_template_and_weights, batched = True, remove_columns=dataset.column_names)

# Custom Data Collator
from transformers import DataCollatorForLanguageModeling
class WeightedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # Extract loss_weights separately so super() doesn't choke on them or ignore them
        loss_weights = [ex.pop("loss_weights") for ex in examples]
        
        # Standard collation for input_ids/labels
        batch = super().__call__(examples)
        
        # Pad weights to match the batch's padded input_ids length
        # batch['input_ids'] is already padded here.
        max_len = batch["input_ids"].shape[1]
        padded_weights = []
        
        for w in loss_weights:
            w_tensor = torch.tensor(w, dtype=torch.float32)
            pad_len = max_len - len(w_tensor)
            if pad_len > 0:
                # Pad with 1.0 (standard loss weight) for padding tokens (though usually ignored by label -100)
                # It is safer to pad with 1.0 or 0.0. Since labels are -100, these positions have 0 loss anyway.
                w_tensor = torch.cat([w_tensor, torch.ones(pad_len, dtype=torch.float32)])
            else:
                w_tensor = w_tensor[:max_len]
            padded_weights.append(w_tensor)
            
        batch["loss_weights"] = torch.stack(padded_weights)
        return batch

# 5. Train
trainer = WeightedLossTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    # dataset_text_field = "text", # Removed since we provide pre-tokenized input_ids
    # SFTTrainer args
    max_seq_length = max_seq_length,
    data_collator = WeightedDataCollator(tokenizer=tokenizer, mlm=False),
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_weighted",
        report_to = "none",
        remove_unused_columns=False, # Important to keep loss_weights
    ),
)

trainer.train()

model.save_pretrained("lora_model_weighted")
tokenizer.save_pretrained("lora_model_weighted")
print("Model saved to lora_model_weighted")
