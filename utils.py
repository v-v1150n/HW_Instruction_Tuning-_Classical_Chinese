from transformers import BitsAndBytesConfig
import torch

# #ver1
def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"指令: 將以下文言文翻譯成白話文（或 將以下白話文翻譯成文言文） 內容: {instruction} 翻譯: "

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

def get_na_prompt(instruction: str) -> str:
    return f"{instruction}"

def get_zero_shot_prompt(instruction: str) -> str:
    # Zero-shot prompt with task description
    return f"指令：將以下文言文翻譯成白話文（或將以下白話文翻譯成文言文）內容：{instruction} 翻譯："

def get_few_shot_prompt(instruction: str) -> str:
    # Two in-context examples
    examples = [
        {
            "instruction": "正月，甲子朔，罷至，太后享通天宮；放天下，改元。\n把這句話翻譯成現代文。",
            "output": "聖曆元年正月，甲子朔，罷至，太后在通天宮祭祀；大赦天下，更改年號。"
        },
        {
            "instruction": "文言文翻譯：\n明日，趙用賢疏入。",
            "output": "答案：第二天，趙用賢的疏奏上。"
        }
    ]
    
    # Format examples into prompt
    few_shot_prompt = "請根據以下範例翻譯文言文： "
    for example in examples:
        few_shot_prompt += f"範例 - 指令：{example['instruction']} 範例 - 翻譯結果：{example['output']} "

    # Add the new instruction for the model to translate
    few_shot_prompt += f"指令：{instruction} 翻譯："
    return few_shot_prompt
