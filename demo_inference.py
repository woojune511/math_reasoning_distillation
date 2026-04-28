from unsloth import FastLanguageModel
from transformers import TextStreamer

def demo():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device

    # GSM8K Example
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    messages = [
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to(device)

    print("Generating...")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

if __name__ == "__main__":
    demo()
