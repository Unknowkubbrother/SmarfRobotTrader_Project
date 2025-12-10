# ==========================================
# ไฟล์: llm_analyzer.py
# วัตถุประสงค์: ใช้ LLM Vision Model วิเคราะห์กราฟการเทรนและให้คำแนะนำ
# ==========================================

import os
import base64
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

class LLMTrainingAnalyzer:
    """
    คลาสสำหรับวิเคราะห์กราฟการเทรนด้วย LLM Vision Model
    - สร้างกราฟจาก training metrics
    - ส่งกราฟให้ LLM วิเคราะห์
    - รับคำแนะนำว่าควรทำอะไรต่อ
    """
    
    def __init__(self, api_key=None, model="gemini-2.0-flash-exp"):
        """
        สร้าง LLM Analyzer
        
        Parameters:
        -----------
        api_key : str
            Google AI API Key (ถ้าไม่ใส่จะใช้จาก environment variable)
        model : str
            ชื่อโมเดลที่จะใช้ (default: gemini-2.0-flash-exp)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = model
        self.metrics_history = []
        
        # ตรวจสอบว่ามี API key หรือไม่
        if not self.api_key:
            print("⚠️ Warning: GOOGLE_API_KEY not found. LLM analysis will be disabled.")
            print("   Set it with: export GOOGLE_API_KEY='your-api-key'")
            self.enabled = False
        else:
            self.enabled = True
            # Import Google AI SDK
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                print("✅ LLM Analyzer initialized successfully!")
            except ImportError:
                print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")
                self.enabled = False
    
    def log_metrics(self, iteration, metrics):
        """
        บันทึก metrics จากการเทรน
        
        Parameters:
        -----------
        iteration : int
            รอบการเทรนปัจจุบัน
        metrics : dict
            ข้อมูล metrics เช่น loss, reward, etc.
        """
        metrics['iteration'] = iteration
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
    
    def create_training_chart(self, save_path='training_progress.png'):
        """
        สร้างกราฟแสดงความคืบหน้าการเทรน
        
        Parameters:
        -----------
        save_path : str
            ที่อยู่ไฟล์ที่จะบันทึกกราฟ
            
        Returns:
        --------
        str : path ของไฟล์กราฟที่สร้าง
        """
        if not self.metrics_history:
            print("⚠️ No metrics to plot yet.")
            return None
        
        df = pd.DataFrame(self.metrics_history)
        
        # สร้างกราฟ 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # 1. Loss over time
        if 'loss' in df.columns:
            axes[0, 0].plot(df['iteration'], df['loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('Policy Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Value Loss
        if 'value_loss' in df.columns:
            axes[0, 1].plot(df['iteration'], df['value_loss'], 'r-', linewidth=2)
            axes[0, 1].set_title('Value Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Value Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Entropy Loss
        if 'entropy_loss' in df.columns:
            axes[1, 0].plot(df['iteration'], df['entropy_loss'], 'g-', linewidth=2)
            axes[1, 0].set_title('Entropy Loss (Exploration)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Explained Variance
        if 'explained_variance' in df.columns:
            axes[1, 1].plot(df['iteration'], df['explained_variance'], 'm-', linewidth=2)
            axes[1, 1].set_title('Explained Variance', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training chart saved to: {save_path}")
        return save_path
    
    def analyze_with_llm(self, chart_path, current_iteration, total_iterations):
        """
        ส่งกราฟให้ LLM วิเคราะห์และให้คำแนะนำ
        
        Parameters:
        -----------
        chart_path : str
            path ของไฟล์กราฟ
        current_iteration : int
            รอบการเทรนปัจจุบัน
        total_iterations : int
            จำนวนรอบทั้งหมด
            
        Returns:
        --------
        dict : ผลการวิเคราะห์จาก LLM
        """
        if not self.enabled:
            return {
                'status': 'disabled',
                'message': 'LLM analysis is disabled. Please set GOOGLE_API_KEY.'
            }
        
        try:
            # อ่านไฟล์กราฟและแปลงเป็น base64
            with open(chart_path, 'rb') as f:
                image_data = f.read()
            
            # สร้าง prompt สำหรับ LLM
            prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้าน Reinforcement Learning และ Trading AI

กราฟนี้แสดงความคืบหน้าการเทรนโมเดล PPO (Proximal Policy Optimization) สำหรับเทรด Forex

**ข้อมูลการเทรนปัจจุบัน:**
- Iteration ปัจจุบัน: {current_iteration}/{total_iterations} ({current_iteration/total_iterations*100:.1f}%)
- จำนวน metrics ที่บันทึก: {len(self.metrics_history)} รอบ

**กราฟประกอบด้วย:**
1. **Policy Loss** (บนซ้าย) - ค่า loss ของ policy network
2. **Value Loss** (บนขวา) - ค่า loss ของ value network
3. **Entropy Loss** (ล่างซ้าย) - ระดับการสำรวจ (exploration)
4. **Explained Variance** (ล่างขวา) - ความสามารถในการทำนายของ value function

**กรุณาวิเคราะห์และตอบเป็นภาษาไทยในรูปแบบ JSON:**

{{
    "overall_status": "excellent/good/fair/poor/critical",
    "analysis": {{
        "policy_loss": "การวิเคราะห์ policy loss (แนวโน้ม, ปัญหา, ข้อสังเกต)",
        "value_loss": "การวิเคราะห์ value loss",
        "entropy": "การวิเคราะห์ entropy (การสำรวจ)",
        "variance": "การวิเคราะห์ explained variance"
    }},
    "issues": [
        "ปัญหาที่พบ (ถ้ามี)"
    ],
    "recommendations": [
        "คำแนะนำที่ควรทำต่อ (เรียงตามความสำคัญ)"
    ],
    "should_continue": true/false,
    "reason": "เหตุผลว่าควรเทรนต่อหรือหยุด",
    "estimated_completion": "ประมาณการว่าควรเทรนต่ออีกกี่ iteration"
}}

**หมายเหตุ:**
- ถ้า loss ลดลงอย่างต่อเนื่อง = ดี
- ถ้า loss กระโดดขึ้นลงมาก = อาจมีปัญหา learning rate หรือ instability
- Entropy ควรลดลงค่อยๆ (แสดงว่า policy กำลัง converge)
- Explained Variance ควรเข้าใกล้ 1.0 (แสดงว่า value function ทำนายได้ดี)
"""
            
            # เรียก LLM
            model = self.genai.GenerativeModel(self.model)
            
            # Upload image
            image_part = {
                'mime_type': 'image/png',
                'data': image_data
            }
            
            response = model.generate_content([prompt, image_part])
            
            # Parse response
            response_text = response.text
            
            # ลองแยก JSON ออกจาก response
            try:
                # หา JSON block
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif '```' in response_text:
                    json_start = response_text.find('```') + 3
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text
                
                analysis = json.loads(json_text)
                analysis['raw_response'] = response_text
                analysis['status'] = 'success'
                
            except json.JSONDecodeError:
                # ถ้า parse ไม่ได้ ให้ส่ง raw text กลับไป
                analysis = {
                    'status': 'success',
                    'raw_response': response_text,
                    'overall_status': 'unknown',
                    'recommendations': ['ดูข้อมูลใน raw_response']
                }
            
            return analysis
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during LLM analysis: {str(e)}'
            }
    
    def print_analysis(self, analysis):
        """
        แสดงผลการวิเคราะห์ในรูปแบบที่อ่านง่าย
        
        Parameters:
        -----------
        analysis : dict
            ผลการวิเคราะห์จาก LLM
        """
        print("\n" + "="*80)
        print("🤖 LLM TRAINING ANALYSIS")
        print("="*80)
        
        if analysis['status'] == 'disabled':
            print(f"⚠️ {analysis['message']}")
            return
        
        if analysis['status'] == 'error':
            print(f"❌ {analysis['message']}")
            return
        
        # แสดงสถานะโดยรวม
        status_emoji = {
            'excellent': '🌟',
            'good': '✅',
            'fair': '⚠️',
            'poor': '⚠️',
            'critical': '🚨'
        }
        
        overall = analysis.get('overall_status', 'unknown')
        emoji = status_emoji.get(overall, '❓')
        print(f"\n{emoji} สถานะโดยรวม: {overall.upper()}")
        
        # แสดงการวิเคราะห์แต่ละส่วน
        if 'analysis' in analysis:
            print("\n📊 การวิเคราะห์:")
            for key, value in analysis['analysis'].items():
                print(f"  • {key}: {value}")
        
        # แสดงปัญหาที่พบ
        if 'issues' in analysis and analysis['issues']:
            print("\n⚠️ ปัญหาที่พบ:")
            for i, issue in enumerate(analysis['issues'], 1):
                print(f"  {i}. {issue}")
        
        # แสดงคำแนะนำ
        if 'recommendations' in analysis and analysis['recommendations']:
            print("\n💡 คำแนะนำ:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # แสดงข้อสรุป
        if 'should_continue' in analysis:
            should_continue = analysis['should_continue']
            reason = analysis.get('reason', 'ไม่ระบุเหตุผล')
            
            if should_continue:
                print(f"\n✅ ควรเทรนต่อ: {reason}")
            else:
                print(f"\n🛑 ควรหยุดเทรน: {reason}")
        
        if 'estimated_completion' in analysis:
            print(f"📈 ประมาณการ: {analysis['estimated_completion']}")
        
        print("\n" + "="*80 + "\n")
    
    def save_analysis(self, analysis, filepath='llm_analysis.json'):
        """
        บันทึกผลการวิเคราะห์ลงไฟล์
        
        Parameters:
        -----------
        analysis : dict
            ผลการวิเคราะห์
        filepath : str
            ที่อยู่ไฟล์ที่จะบันทึก
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"💾 Analysis saved to: {filepath}")


# ==========================================
# ฟังก์ชันตัวอย่างการใช้งาน
# ==========================================

def example_usage():
    """
    ตัวอย่างการใช้งาน LLM Analyzer
    """
    # สร้าง analyzer
    analyzer = LLMTrainingAnalyzer()
    
    # จำลองการบันทึก metrics
    for i in range(1, 26):
        metrics = {
            'loss': -0.02 + np.random.normal(0, 0.01),
            'value_loss': 0.03 + np.random.normal(0, 0.005),
            'entropy_loss': -1.0 + i * 0.02 + np.random.normal(0, 0.05),
            'explained_variance': 0.3 + i * 0.02 + np.random.normal(0, 0.05)
        }
        analyzer.log_metrics(i, metrics)
    
    # สร้างกราฟ
    chart_path = analyzer.create_training_chart()
    
    # วิเคราะห์ด้วย LLM
    if chart_path:
        analysis = analyzer.analyze_with_llm(chart_path, current_iteration=25, total_iterations=50)
        analyzer.print_analysis(analysis)
        analyzer.save_analysis(analysis)


if __name__ == "__main__":
    example_usage()
