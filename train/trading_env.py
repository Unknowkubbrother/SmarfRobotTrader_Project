# ==========================================
# ไฟล์: trading_env.py
# วัตถุประสงค์: สร้าง Custom Trading Environment สำหรับ Reinforcement Learning
# ==========================================

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Custom Trading Environment ที่ทำงานตาม gymnasium interface
    
    Environment นี้จำลองการเทรด Forex โดย:
    - Agent สามารถเลือก action: Neutral (0), Long (1), Short (2)
    - คำนวณ reward จากการเปลี่ยนแปลงของ net worth
    - จำลองค่า commission (spread) ในการเทรด
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission=0.0002):
        """
        สร้าง Trading Environment
        
        Parameters:
        -----------
        df : pandas.DataFrame
            ข้อมูลราคาและ features ที่ประมวลผลแล้ว
        initial_balance : float
            เงินทุนเริ่มต้น (default: $10,000)
        commission : float
            ค่า commission/spread ต่อการเทรด (default: 0.02% หรือ 0.0002)
        """
        super(TradingEnv, self).__init__()

        self.df = df
        # รีเซ็ต index ให้เริ่มจาก 0
        self.df = self.df.reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.commission = commission
        
        # ==========================================
        # Action Space: พื้นที่การกระทำ
        # ==========================================
        # 0 = Neutral (ไม่ถือ position)
        # 1 = Long (ซื้อ - คาดว่าราคาจะขึ้น)
        # 2 = Short (ขาย - คาดว่าราคาจะลง)
        self.action_space = spaces.Discrete(3)
        
        # ==========================================
        # Observation Space: พื้นที่สังเกตการณ์
        # ==========================================
        # กรองเอาเฉพาะ feature columns (ไม่รวม time, close, open, high, low, etc.)
        self.feature_cols = [col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'tick_volume', 'real_volume', 'spread', 'close']]
        
        # ตรวจสอบว่ามี features หรือไม่
        if not self.feature_cols:
             # ถ้าไม่มี features ให้ใช้ข้อมูลพื้นฐาน (แต่ปกติควรมี features จากการประมวลผล)
             pass

        # กำหนดขนาดของ observation (จำนวน features)
        self.obs_shape = (len(self.feature_cols),)
        # Observation เป็น array ของ features ที่มีค่าได้ตั้งแต่ -inf ถึง +inf
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        # ==========================================
        # State Variables: ตัวแปรสถานะ
        # ==========================================
        self.current_step = 0  # ขั้นตอนปัจจุบัน
        self.max_steps = len(self.df) - 1  # ขั้นตอนสูงสุด
        
        self.balance = self.initial_balance  # ยอดเงินคงเหลือ
        self.net_worth = self.initial_balance  # มูลค่าสุทธิ (เงิน + กำไร/ขาดทุนจาก position)
        self.current_position = 0  # Position ปัจจุบัน: 0=Neutral, 1=Long, -1=Short
        self.entry_price = 0  # ราคาที่เข้า position
        
        self.trades = []  # บันทึกการเทรดทั้งหมด (สำหรับวิเคราะห์)

    def reset(self, seed=None, options=None):
        """
        รีเซ็ต environment กลับไปสู่สถานะเริ่มต้น
        ฟังก์ชันนี้ถูกเรียกเมื่อเริ่ม episode ใหม่
        
        Returns:
        --------
        observation : array
            สถานะเริ่มต้นของ environment
        info : dict
            ข้อมูลเพิ่มเติม (ว่างเปล่า)
        """
        super().reset(seed=seed)
        
        # รีเซ็ตตัวแปรทั้งหมดกลับไปสู่ค่าเริ่มต้น
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_position = 0
        self.entry_price = 0
        self.trades = []
        
        return self._next_observation(), {}

    def _next_observation(self):
        """
        ดึงข้อมูล observation ของ step ปัจจุบัน
        
        Returns:
        --------
        observation : np.array
            array ของ features ที่ step ปัจจุบัน
        """
        obs = self.df.iloc[self.current_step][self.feature_cols].values
        return obs.astype(np.float32)

    def step(self, action):
        """
        ดำเนินการ action และอัพเดทสถานะของ environment
        
        Parameters:
        -----------
        action : int
            การกระทำที่ agent เลือก (0=Neutral, 1=Long, 2=Short)
            
        Returns:
        --------
        observation : np.array
            สถานะใหม่หลังจากทำ action
        reward : float
            รางวัลที่ได้รับจากการทำ action นี้
        terminated : bool
            จบ episode หรือไม่ (ถึงขั้นตอนสุดท้ายแล้ว)
        truncated : bool
            ถูกตัดทอนหรือไม่ (ไม่ได้ใช้ในที่นี้)
        info : dict
            ข้อมูลเพิ่มเติม (net_worth, price, position)
        """
        # บันทึก net worth ก่อนหน้าเพื่อคำนวณ reward
        prev_net_worth = self.net_worth
        
        # แปลง action เป็น target position
        # action 0 -> position 0 (Neutral)
        # action 1 -> position 1 (Long)
        # action 2 -> position -1 (Short)
        target_position = 0
        if action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1
        
        # ดึงราคาปัจจุบันและเวลา
        current_price = self.df.iloc[self.current_step]['close']
        time = self.df.iloc[self.current_step]['time'] if 'time' in self.df.columns else self.current_step
        
        # ==========================================
        # จัดการ Position (เปิด/ปิด/เปลี่ยน)
        # ==========================================
        
        trade_occurred = False  # ตัวแปรบอกว่ามีการเทรดเกิดขึ้นหรือไม่
        
        # ถ้า position เปลี่ยนแปลง (เช่น จาก Long -> Neutral หรือ Neutral -> Short)
        if self.current_position != target_position:
            # ==========================================
            # 1. ปิด Position เดิม (ถ้ามี)
            # ==========================================
            if self.current_position != 0:
                # คำนวณกำไร/ขาดทุนจากการปิด position
                if self.current_position == 1:
                    # ปิด Long: กำไร = (ราคาปัจจุบัน - ราคาเข้า) / ราคาเข้า
                    trade_pnl_pct = (current_price - self.entry_price) / self.entry_price
                else:
                    # ปิด Short: กำไร = (ราคาเข้า - ราคาปัจจุบัน) / ราคาเข้า
                    trade_pnl_pct = (self.entry_price - current_price) / self.entry_price
                
                # หัก commission จากกำไร/ขาดทุน
                trade_pnl_pct -= self.commission
                
                # บันทึกการเทรด (ปิด position)
                self.trades.append({
                    'step': self.current_step,
                    'time': time,
                    'type': 'sell' if self.current_position == 1 else 'cover',  # sell=ปิด Long, cover=ปิด Short
                    'price': current_price,
                    'pnl': trade_pnl_pct,
                    'net_worth': self.net_worth
                })
                
                trade_occurred = True

            # ==========================================
            # 2. เปิด Position ใหม่ (ถ้า target ไม่ใช่ Neutral)
            # ==========================================
            if target_position != 0:
                self.entry_price = current_price  # บันทึกราคาที่เข้า position
                
                # บันทึกการเทรด (เปิด position ใหม่)
                self.trades.append({
                    'step': self.current_step,
                    'time': time,
                    'type': 'buy' if target_position == 1 else 'short',  # buy=เปิด Long, short=เปิด Short
                    'price': current_price,
                    'pnl': 0,  # ยังไม่มีกำไร/ขาดทุน (เพิ่งเปิด)
                    'net_worth': self.net_worth
                })
                
                trade_occurred = True
            
            # อัพเดท position ปัจจุบัน
            self.current_position = target_position
        
        # ==========================================
        # คำนวณการเปลี่ยนแปลงของ Net Worth
        # ==========================================
        # Net Worth = เงินสด + กำไร/ขาดทุนที่ยังไม่ได้ปิด (Unrealized PnL)
        
        step_return = 0
        
        # ผลกระทบจากการเคลื่อนไหวของตลาด
        if self.current_position == 1:
             # Long: ถ้าราคาขึ้น -> กำไร, ถ้าราคาลง -> ขาดทุน
             price_change_pct = (current_price - self.df.iloc[self.current_step-1]['close']) / self.df.iloc[self.current_step-1]['close']
             self.net_worth *= (1 + price_change_pct)
        elif self.current_position == -1:
             # Short: ถ้าราคาลง -> กำไร, ถ้าราคาขึ้น -> ขาดทุน
             price_change_pct = (self.df.iloc[self.current_step-1]['close'] - current_price) / self.df.iloc[self.current_step-1]['close']
             self.net_worth *= (1 + price_change_pct)
             
        # ==========================================
        # หัก Commission (ค่าธรรมเนียม/Spread)
        # ==========================================
        # ถ้ามีการเทรดเกิดขึ้นในขั้นตอนนี้ ต้องหัก commission
        if trade_occurred:
            # หัก commission จาก net worth
            # ทุกครั้งที่เปลี่ยน position (เปิด/ปิด) จะต้องจ่าย spread
            self.net_worth *= (1 - self.commission)
             
        # ==========================================
        # คำนวณ Reward (รางวัล)
        # ==========================================
        # Reward = การเปลี่ยนแปลงของ net worth เป็นเปอร์เซ็นต์
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        
        # ปรับขนาด reward ให้เหมาะสมกับการเทรน PPO
        # คูณ 100 เพื่อแปลงเป็นเปอร์เซ็นต์ (เช่น 0.01 -> 1.0)
        reward *= 100
        
        # ==========================================
        # อัพเดท Step และตรวจสอบว่าจบ Episode หรือไม่
        # ==========================================
        self.current_step += 1
        terminated = self.current_step >= self.max_steps  # จบเมื่อถึงขั้นตอนสุดท้าย
        truncated = False  # ไม่มีการตัดทอน
        
        # ดึง observation ถัดไป (ถ้ายังไม่จบ)
        if not terminated:
            next_obs = self._next_observation()
        else:
            # ถ้าจบแล้ว ส่ง observation ว่างเปล่า
            next_obs = np.zeros(self.obs_shape, dtype=np.float32)
        
        # ข้อมูลเพิ่มเติมสำหรับ logging/debugging
        info = {
            'net_worth': self.net_worth,
            'current_price': current_price,
            'position': self.current_position
        }
        
        return next_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        แสดงสถานะปัจจุบันของ environment (สำหรับ debugging)
        """
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Pos: {self.current_position}')
