# ==========================================
# ไฟล์: llm_analyzer_comparison.py
# วัตถุประสงค์: เปรียบเทียบวิธีส่งรูปภาพแบบต่างๆ
# ==========================================

import os
import base64
import sys

def compare_image_sending_methods(image_path):
    """
    เปรียบเทียบขนาดและวิธีการส่งรูปภาพแบบต่างๆ
    
    Parameters:
    -----------
    image_path : str
        path ของไฟล์รูปภาพ
    """
    print("="*80)
    print("📊 Image Sending Methods Comparison")
    print("="*80)
    
    # 1. Binary (วิธีปัจจุบัน - ดีที่สุด)
    with open(image_path, 'rb') as f:
        binary_data = f.read()
    
    binary_size = len(binary_data)
    
    # 2. Base64
    base64_data = base64.b64encode(binary_data)
    base64_size = len(base64_data)
    
    # 3. Base64 String
    base64_string = base64_data.decode('utf-8')
    base64_string_size = len(base64_string)
    
    print(f"\n📁 Image: {image_path}")
    print(f"   File size: {binary_size:,} bytes ({binary_size/1024:.2f} KB)")
    
    print("\n" + "="*80)
    print("Method 1: Binary Data (ปัจจุบัน - แนะนำ)")
    print("="*80)
    print(f"✅ Size: {binary_size:,} bytes ({binary_size/1024:.2f} KB)")
    print(f"✅ Token cost: ~0 tokens (images don't count as text tokens)")
    print(f"✅ Overhead: 0%")
    print(f"✅ Speed: Fastest")
    print("\nCode example:")
    print("""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    image_part = {
        'mime_type': 'image/png',
        'data': image_data  # ← Binary bytes
    }
    """)
    
    print("\n" + "="*80)
    print("Method 2: Base64 Encoded")
    print("="*80)
    print(f"⚠️ Size: {base64_size:,} bytes ({base64_size/1024:.2f} KB)")
    print(f"⚠️ Token cost: ~0 tokens (still treated as image)")
    print(f"⚠️ Overhead: +{((base64_size - binary_size) / binary_size * 100):.1f}%")
    print(f"⚠️ Speed: Slower (encoding overhead)")
    print("\nCode example:")
    print("""
    with open(image_path, 'rb') as f:
        binary_data = f.read()
    
    base64_data = base64.b64encode(binary_data)
    
    image_part = {
        'mime_type': 'image/png',
        'data': base64_data  # ← Base64 bytes
    }
    """)
    
    print("\n" + "="*80)
    print("Method 3: Base64 String (สำหรับ REST API)")
    print("="*80)
    print(f"❌ Size: {base64_string_size:,} bytes ({base64_string_size/1024:.2f} KB)")
    print(f"❌ Token cost: ~0 tokens (if sent as image field)")
    print(f"❌ Overhead: +{((base64_string_size - binary_size) / binary_size * 100):.1f}%")
    print(f"❌ Speed: Slowest (encoding + string conversion)")
    print("\nCode example:")
    print("""
    with open(image_path, 'rb') as f:
        binary_data = f.read()
    
    base64_string = base64.b64encode(binary_data).decode('utf-8')
    
    # For REST API
    payload = {
        'image': f'data:image/png;base64,{base64_string}'
    }
    """)
    
    print("\n" + "="*80)
    print("📊 Summary")
    print("="*80)
    print(f"\n{'Method':<30} {'Size (KB)':<15} {'Overhead':<15} {'Recommended'}")
    print("-" * 80)
    print(f"{'Binary (current)':<30} {binary_size/1024:<15.2f} {'0%':<15} {'✅ YES'}")
    print(f"{'Base64 bytes':<30} {base64_size/1024:<15.2f} {f'+{((base64_size - binary_size) / binary_size * 100):.1f}%':<15} {'❌ NO'}")
    print(f"{'Base64 string':<30} {base64_string_size/1024:<15.2f} {f'+{((base64_string_size - binary_size) / binary_size * 100):.1f}%':<15} {'❌ NO'}")
    
    print("\n" + "="*80)
    print("💡 Recommendation")
    print("="*80)
    print("""
✅ KEEP USING BINARY METHOD (current implementation)

Reasons:
1. Smallest size (33% smaller than base64)
2. No encoding/decoding overhead
3. Faster transmission
4. Google Gemini API accepts binary data directly
5. No token cost for images (regardless of method)

❌ DON'T SWITCH TO BASE64 unless:
- You're using a different API that requires base64
- You need to embed images in JSON/text
- You're sending via REST API that doesn't support binary
    """)
    
    print("\n" + "="*80)
    print("🔍 Technical Details")
    print("="*80)
    print("""
Why Base64 is larger:
- Base64 uses 4 characters to represent 3 bytes
- This creates ~33% overhead
- Example: 3 bytes (24 bits) → 4 base64 chars (32 bits)

Token counting:
- Images are NOT counted as text tokens in Gemini API
- Whether you send binary or base64, the cost is the same
- Image cost is based on resolution, not encoding method

Current implementation (binary):
- Reads file as bytes: f.read()
- Sends directly to API: {'data': bytes}
- API handles it natively
- Most efficient method
    """)


def demonstrate_methods():
    """
    สาธิตวิธีการส่งรูปภาพแบบต่างๆ
    """
    # สร้างรูปตัวอย่าง
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title('Sample Chart')
    ax.grid(True)
    
    test_image = 'comparison_test_image.png'
    plt.savefig(test_image, dpi=150)
    plt.close()
    
    # เปรียบเทียบ
    compare_image_sending_methods(test_image)
    
    # ลบไฟล์ทดสอบ
    os.remove(test_image)


if __name__ == "__main__":
    demonstrate_methods()
