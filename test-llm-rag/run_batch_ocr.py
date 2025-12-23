import json
import os
import cv2
from test_ocr import extract_candles_with_ocr_cleanup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset.json')

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Found {len(data)} images to process.")

    for i, item in enumerate(data):
        relative_path = item['image']
        # images are relative to the current folder (test-llm-rag)
        image_path = os.path.join(BASE_DIR, relative_path)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue

        print(f"[{i+1}/{len(data)}] Processing {relative_path}...")
        try:
            # extract_candles_with_ocr_cleanup returns rgb, mask, final_result
            # final_result is in RGB format because test_ocr converts it early on.
            _, _, result_rgb = extract_candles_with_ocr_cleanup(image_path)
            
            # Convert back to BGR for saving with cv2
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            
            # Overwrite the file
            cv2.imwrite(image_path, result_bgr)
            print(f"✅ Overwritten {relative_path}")
            
        except Exception as e:
            print(f"❌ Error processing {relative_path}: {e}")

if __name__ == "__main__":
    main()
