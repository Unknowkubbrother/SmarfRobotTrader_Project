import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# โหลดโมเดล OCR (โหลดครั้งแรกจะนานหน่อย)
# gpu=True ถ้ามี NVIDIA GPU, ถ้าไม่มีให้แก้เป็น gpu=False
reader = easyocr.Reader(['en', 'th'], gpu=True) 

def extract_candles_with_ocr_cleanup(image_path):
    # 1. โหลดภาพ
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = img.shape

    # ==================================================
    # STEP 1: หาพื้นที่กราฟด้วยสี (วิธีเดิม)
    # ==================================================
    # Green
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([95, 255, 255])
    # Red
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    mask_color = cv2.inRange(hsv, lower_green, upper_green) | \
                 cv2.inRange(hsv, lower_red1, upper_red1) | \
                 cv2.inRange(hsv, lower_red2, upper_red2)
                 
    # ใช้ Vertical Kernel ช่วยเชื่อมไส้เทียน (เหมือนรอบที่แล้ว)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    mask_candles = cv2.dilate(mask_color, kernel_vertical, iterations=1)

    # ==================================================
    # STEP 2: ใช้ OCR หา "พื้นที่ต้องห้าม" (Text Mask)
    # ==================================================
    print("Running OCR... (might take a few seconds)")
    results = reader.readtext(img)

    mask_text = np.zeros((height, width), dtype=np.uint8)

    for (bbox, text, prob) in results:
        # bbox คือพิกัด 4 จุดของกล่องข้อความ [[tl], [tr], [br], [bl]]
        (tl, tr, br, bl) = bbox
        top_left = (int(tl[0]), int(tl[1]))
        bottom_right = (int(br[0]), int(br[1]))

        # วาดสี่เหลี่ยมทับลงไปใน mask_text
        cv2.rectangle(mask_text, top_left, bottom_right, 255, -1)

    # *** ขยายกล่อง OCR (Dilate) ***
    # เพราะ OCR มักจะจับแค่ตัวหนังสือ ไม่รวมขอบกล่อง หรือ Padding รอบๆ
    # เราจึงต้อง "เบ่ง" พื้นที่ OCR ให้กว้างขึ้นเพื่อกินขอบกล่องข้อความไปด้วย
    kernel_box = np.ones((15, 15), np.uint8) # ขยายออกเยอะหน่อย (ปรับเลขนี้ได้)
    mask_text_expanded = cv2.dilate(mask_text, kernel_box, iterations=2)

    # ==================================================
    # STEP 3: ลบพื้นที่ Text ออกจากพื้นที่กราฟ
    # ==================================================
    # Logic: เอา Mask กราฟ ตั้ง แล้ว "ลบ" ด้วย Mask ข้อความ
    # cv2.bitwise_not(mask_text_expanded) คือกลับขาวเป็นดำ ดำเป็นขาว
    
    final_mask = cv2.bitwise_and(mask_candles, mask_candles, mask=cv2.bitwise_not(mask_text_expanded))

    # Clean Noise เล็กๆ น้อยๆ ครั้งสุดท้าย
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(final_mask)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 3: # กรองจุดเล็กๆ ทิ้ง
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # สร้างภาพผลลัพธ์
    final_result = cv2.bitwise_and(rgb, rgb, mask=clean_mask)

    return rgb, mask_text_expanded, final_result

if __name__ == "__main__":
    # --- Run Code ---
    image_path = "datasets/chart21.png"
    # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่ก่อนรัน
    import os
    if os.path.exists(image_path):
        original, text_mask, result = extract_candles_with_ocr_cleanup(image_path)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(text_mask, cmap='gray')
        plt.title("Detected Text Zones (OCR)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title("Final Result (Color - OCR)")
        plt.axis("off")

        plt.show()
    else:
        print(f"File not found: {image_path}")