import os
import json
import csv
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed

# --- CẤU HÌNH ---
MODEL_PATH = "/mnt/pretrained_fm/Qwen_Qwen3-VL-2B-Instruct" 
DATASET_JSON = "natural_qa_dataset.json"
SYSTEM_PROMPT = "Answer concisely without any extra explanation."

def seed_everything(seed=42):
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model_path: str):
    print(f"🔄 Loading model from {model_path}...")
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
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def export_report(results):
    csv_file = f"evaluation_report.csv"
    
    correct_count = 0
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Task Type", "Image File", "Question", "Ground Truth", "Prediction", "Is Correct"])
        
        for res in results:
            gt = str(res['ground_truth']).strip().lower()
            pred = str(res['prediction']).strip().lower()
            
            is_correct = gt in pred
            if is_correct: 
                correct_count += 1
            
            writer.writerow([
                res['task_type'], os.path.basename(res['image_path']), 
                res['question'], res['ground_truth'], res['prediction'], is_correct
            ])
            
    print("\n" + "="*50)
    print(f"🎉 INFERENCE COMPLETED!")
    print(f"📊 Accuracy: {correct_count}/{len(results)} ({(correct_count/len(results))*100:.2f}%)")
    print(f"💾 Report saved to: {csv_file}")
    print("="*50 + "\n")

def main():
    seed_everything(seed=42)
    
    if not os.path.exists(DATASET_JSON):
        print(f"❌ Error: Cannot find '{DATASET_JSON}'. Please run the generation script first.")
        return
        
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    
    print(f"✅ Loaded {len(tasks)} tasks from {DATASET_JSON}")

    model, processor = load_model(MODEL_PATH)
    results = []
    
    for idx, task in enumerate(tasks, 1):
        print(f"[{idx}/{len(tasks)}] Processing {task['task_type']}...")
        img_path = task["image_path"]
        
        if not os.path.exists(img_path):
            print(f"   ⚠️ Warning: Image '{img_path}' not found. Skipping...")
            continue
            
        prediction = infer_image(img_path, task["question"], model, processor)
        
        print(f"   Q: {task['question']}")
        print(f"   Pred: {prediction}")
        print(f"   GT  : {task['answer']}\n")
        
        results.append({
            "task_type": task["task_type"],
            "image_path": task["image_path"],
            "question": task["question"],
            "ground_truth": task["answer"],
            "prediction": prediction
        })
        
    if results:
        export_report(results)

if __name__ == "__main__":
    main()