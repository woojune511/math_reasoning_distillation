from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
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
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        outputs = model(input_ids=input_ids, attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, self.model.config.vocab_size)
        flat_labels = shift_labels.view(-1)

        valid_mask = flat_labels.ne(-100)
        safe_labels = flat_labels.masked_fill(~valid_mask, 0)
        per_token_loss = F.cross_entropy(flat_logits, safe_labels, reduction="none")

        if "loss_weights" in inputs:
            flat_weights = inputs["loss_weights"][..., 1:].contiguous().view(-1).to(per_token_loss.device)
        else:
            flat_weights = torch.ones_like(per_token_loss)

        effective_weights = flat_weights * valid_mask.to(per_token_loss.dtype)
        normalizer = effective_weights.sum().clamp_min(1.0)
        loss = (per_token_loss * effective_weights).sum() / normalizer

        if return_outputs:
            return loss, outputs
        return loss

# 4. Data Processing with Weights
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split = "train")

# We need to add <think> as special tokens to key-word search them? 
# Or just finding the bytes.
think_start_token = "<think>"
think_end_token = "</think>"

def apply_template_and_weights(examples):
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
        
        start_indices = [i for i in range(len(full_text)) if full_text.startswith(think_start_token, i)]
        end_indices = [i for i in range(len(full_text)) if full_text.startswith(think_end_token, i)]

        enc = tokenizer(full_text, return_offsets_mapping=True, truncation=True, max_length=max_seq_length)
        offsets = enc.offset_mapping
        
        for start_char in start_indices:
            end_char = -1
            for e in end_indices:
                if e > start_char:
                    end_char = e
                    break
            
            if end_char == -1:
                continue

            think_end_char = end_char + len(think_end_token)
            for idx, (o_start, o_end) in enumerate(offsets):
                if o_start == o_end:
                    continue
                if o_end > start_char and o_start < think_end_char:
                    weight_mask[idx] = THINK_WEIGHT
        
        inputs_list.append(input_ids)
        masks_list.append(weight_mask)

    return {"input_ids": inputs_list, "loss_weights": masks_list}

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
