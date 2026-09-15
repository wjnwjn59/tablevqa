import os
import random
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed

MODEL_PATH = "/mnt/pretrained_fm/Qwen_Qwen3-VL-2B-Instruct" 
SYSTEM_PROMPT = "Answer concisely"

def seed_everything(seed=42):
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model_path: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def infer_image(image_path: str, question: str, model, processor) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path, 
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=300,
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()


def main():
    seed_everything(seed=42)
    
    tasks = [
        {
            "image_path": "/home/binhdt/TableVQA/charts_only_table.png",
            "question": "How many green bars are in the Win/Loss column for Report ID ITM-10005?",
            "answer": "5"
        }
    ]
    
    model, processor = load_model(MODEL_PATH)
    
    for task in tasks:
        img_path = task["image_path"]
        question = task["question"]
        answer = infer_image(img_path, question, model, processor)
        print(f"Question: {question}")
        print(f"Predict: {answer}")
        print(f"Answer: {task['answer']}\n" + "-"*40)

if __name__ == "__main__":
    main()