import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_na_prompt, get_bnb_config, get_zero_shot_prompt, get_few_shot_prompt, get_prompt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="zake7749/gemma-2-2b-it-chinese-kyara-dpo", help="Path to the base model.")
    parser.add_argument("--peft_path", type=str, default="", help="Path to the saved PEFT checkpoint.")
    parser.add_argument("--test_data_path", type=str, default="hw3/data/private_test.json", help="Path to test data.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save the output files.")
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()
    attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()

    generated_conversations = []
    for item in tqdm(data, desc="Generating conversations"):
        prompt = get_prompt(item["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_ids"], max_length=512)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        generated_conversations.append({
            "id": item["id"],
            "instruction": item["instruction"],
            "output": generated_text
        })

    generated_output_path = os.path.join(args.output_dir)
    with open(generated_output_path, "w") as f:
        json.dump(generated_conversations, f, ensure_ascii=False, indent=4)

    print(f"Generated conversations saved to {generated_output_path}")
