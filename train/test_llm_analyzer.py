# ==========================================
# ไฟล์: test_llm_analyzer.py
# วัตถุประสงค์: ทดสอบ LLM Analyzer ด้วยข้อมูลจำลอง
# ==========================================

import numpy as np
from llm_analyzer import LLMTrainingAnalyzer

def test_llm_analyzer():
    """
    ทดสอบ LLM Analyzer ด้วยข้อมูลจำลอง
    """
    print("="*80)
    print("🧪 Testing LLM Analyzer")
    print("="*80)
    
    # สร้าง analyzer
    analyzer = LLMTrainingAnalyzer()
    
    if not analyzer.enabled:
        print("\n⚠️ LLM Analyzer is disabled.")
        print("Please set GOOGLE_API_KEY environment variable:")
        print("  Windows (PowerShell): $env:GOOGLE_API_KEY='your-key'")
        print("  Windows (CMD): set GOOGLE_API_KEY=your-key")
        print("  Linux/Mac: export GOOGLE_API_KEY='your-key'")
        return
    
    print("\n📊 Generating simulated training metrics...")
    
    # จำลองการเทรน 25 iterations
    # Scenario: โมเดลกำลังเรียนรู้ได้ดี
    for i in range(1, 26):
        # Policy loss ลดลงเรื่อยๆ
        policy_loss = -0.02 - (i * 0.001) + np.random.normal(0, 0.005)
        
        # Value loss ค่อยๆ ลดลง
        value_loss = 0.05 - (i * 0.001) + np.random.normal(0, 0.003)
        
        # Entropy ลดลงเรื่อยๆ (policy converging)
        entropy_loss = -1.1 + (i * 0.02) + np.random.normal(0, 0.03)
        
        # Explained variance เพิ่มขึ้นเรื่อยๆ (เข้าใกล้ 1.0)
        explained_variance = 0.2 + (i * 0.03) + np.random.normal(0, 0.05)
        explained_variance = min(explained_variance, 1.0)  # ไม่เกิน 1.0
        
        metrics = {
            'loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'explained_variance': explained_variance,
            'approx_kl': 0.01 + np.random.normal(0, 0.002),
            'clip_fraction': 0.1 + np.random.normal(0, 0.02)
        }
        
        analyzer.log_metrics(i, metrics)
        
        if i % 5 == 0:
            print(f"  ✓ Logged metrics for iteration {i}/25")
    
    print("\n📈 Creating training chart...")
    chart_path = analyzer.create_training_chart('test_training_chart.png')
    
    if not chart_path:
        print("❌ Failed to create chart")
        return
    
    print(f"✅ Chart created: {chart_path}")
    
    print("\n🤖 Analyzing with LLM...")
    print("   (This may take a few seconds...)")
    
    # วิเคราะห์ด้วย LLM
    analysis = analyzer.analyze_with_llm(
        chart_path,
        current_iteration=25,
        total_iterations=50
    )
    
    # แสดงผล
    analyzer.print_analysis(analysis)
    
    # บันทึกผล
    analyzer.save_analysis(analysis, 'test_llm_analysis.json')
    
    print("\n" + "="*80)
    print("✅ Test completed successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 test_training_chart.png - Training progress chart")
    print("  📄 test_llm_analysis.json - LLM analysis results")
    print("\n")


if __name__ == "__main__":
    test_llm_analyzer()
