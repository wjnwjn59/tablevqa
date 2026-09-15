import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# --- CẤU HÌNH CƠ BẢN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448


MODEL_PATH = "/mnt/pretrained_fm/OpenGVLab_InternVL3_5-2B"
SYSTEM_PROMPT = "Answer concisely"
DATA_DIR = "/data"


# --- CÁC HÀM TIỀN XỬ LÝ ẢNH ---
def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list, width: int, height: int, image_size: int) -> tuple:
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False) -> list:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_ar = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_model(model_path: str):
    """Tải model và tokenizer."""
    print(f"Đang tải model từ: {model_path}...")
    model = AutoModel.from_pretrained(
        model_path, 
        dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True, 
        device_map="auto"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False
    )
    return model, tokenizer

def process_image_file(image_path: str, transform: T.Compose) -> torch.Tensor:
    """Đọc và tiền xử lý một file ảnh."""
    image = Image.open(image_path).convert('RGB')
    images = dynamic_preprocess(image, image_size=IMAGE_SIZE, use_thumbnail=True, max_num=12)
    return torch.stack([transform(img) for img in images])

def infer_image(image_path: str, question: str, model, tokenizer, transform) -> str:
    """Chạy inference cho một ảnh cụ thể."""
    pixel_values = process_image_file(image_path, transform).to(torch.bfloat16).to(device)
    prompt = f"{SYSTEM_PROMPT}\n<image>Question: {question}\nAnswer:"
    
    with torch.no_grad():
        response = model.chat(
            tokenizer, 
            pixel_values, 
            prompt,
            generation_config={"max_new_tokens": 100, "pad_token_id": tokenizer.eos_token_id}
        )

    return response.strip()


def main():
    tasks = [
        # --- CÁC CÂU HỎI CHO IMAGE 1 ---
        {
            "image_path": "data/image1.jpg",
            "question": "For the quarter that recorded the lowest Consolidated Net income in the year 2018, what was the corresponding Motorcycles Revenue?",
            "answer": "$955.6" 
            # Giải thích: Tìm Net income thấp nhất năm 2018 (0.5 ở Q4/Dec 31, 2018) -> Gióng lên lấy Motorcycles Revenue ($ 955.6)
        },
        {
            "image_path": "data/image1.jpg",
            "question": "Identify the specific quarter and year where the Motorcycles segment experienced an operating loss, and provide the exact loss amount.",
            "answer": "4th Quarter ending Dec 31, 2018, amount is $(59.5)"
            # Giải thích: Phải hiểu ký hiệu "( )" trong tài chính là số âm/loss, tìm đúng cột chứa $(59.5) và gióng lên header để lấy thông tin quý.
        },
        {
            "image_path": "data/image2.jpg",
            "question": "In the year where 'Total Debt as a % of Total Capitalization' reached its highest percentage, what was the exact amount of 'Research and development expenses'?",
            "answer": "70.8"
            # Giải thích: Quét tìm % cao nhất ở hàng "Total Debt as a %..." (52% ở năm 2017) -> Gióng lên hàng "Research and development expenses" (70.8).
        },
        {
            "image_path": "data/image2.jpg",
            "question": "Look at the year with the lowest number of 'Employees at Year-End'. What was the 'Net Income' for that same year?",
            "answer": "413.9"
            # Giải thích: Tìm số lượng nhân viên thấp nhất (4,145 ở năm 2014) -> Gióng lên tìm "Net Income" (413.9). Chú ý không nhầm với Net Income per share.
        }
    ]
    
    transform = build_transform(IMAGE_SIZE)
    model, tokenizer = load_model(MODEL_PATH)
    
    for task in tasks:
        img_path = task["image_path"]
        question = task["question"]
        
        answer = infer_image(img_path, question, model, tokenizer, transform)
        print(f"Question: {question}")
        print(f"Predict: {answer}")
        print(f"Answer: {task['answer']}\n" + "-"*40)

if __name__ == "__main__":
    main()