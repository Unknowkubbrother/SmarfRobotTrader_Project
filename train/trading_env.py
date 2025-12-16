import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

class TradingEnv(gym.Env):
    """A custom trading environment for Gymnasium"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = 10000
        self.transaction_cost = 0.00001 # ค่าธรรมเนียม 0.001% (สมมติ Spread ต่ำ) - ปรับให้เหมาะสมกับการ Scalping

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observations: คือ Features ทั้งหมดที่เราสร้าง
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(df.shape[1],), 
            dtype=np.float32
        )

    def _get_obs(self):
        # ดึงข้อมูล Features ในแถวปัจจุบัน
        return self.df.iloc[self.current_step].values

    def _get_info(self):
        # ส่งข้อมูลเพิ่มเติม (ถ้าต้องการ)
        return {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': self.df.iloc[self.current_step]['mid_price']
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0 # 0=none, 1=long, -1=short
        self.entry_price = 0
        self.current_step = 0

        return self._get_obs(), self._get_info()

    # ในไฟล์ trading_env.py

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['mid_price']
        done = False # Initialize done
        
        # --- คำนวณมูลค่าพอร์ตfolio ก่อนทำ Action ---
        prev_portfolio_value = self.balance
        if self.position == 1: # ถ้าถือ Buy
            prev_portfolio_value += (current_price - self.entry_price) * 100
        elif self.position == -1: # ถ้าถือ Sell
            prev_portfolio_value += (self.entry_price - current_price) * 100

        # --- ประมวลผล Action ปัจจุบัน ---
        trade_executed = False
        if action == 1: # Buy
            if self.position == 1:
                # ถ้ามีสถานะ Buy อยู่แล้ว ให้ถือต่อ (ไม่ทำอะไรค่า entry_price เดิม)
                pass
            elif self.position == -1: # ถ้ากำลัง Sell ให้ปิดก่อน แล้วเปิด Buy
                pnl = (self.entry_price - current_price) * 100
                self.balance += pnl - self.transaction_cost * abs(pnl)
                
                # เปิดสถานะใหม่
                self.position = 1
                self.entry_price = current_price
                trade_executed = True
            else: # Position == 0 (ไม่มีสถานะ)
                self.position = 1
                self.entry_price = current_price
                trade_executed = True
                
        elif action == 2: # Sell
            if self.position == -1:
                # ถ้ามีสถานะ Sell อยู่แล้ว ให้ถือต่อ
                pass
            elif self.position == 1: # ถ้ากำลัง Buy ให้ปิดก่อน แล้วเปิด Sell
                pnl = (current_price - self.entry_price) * 100
                self.balance += pnl - self.transaction_cost * abs(pnl)
                
                # เปิดสถานะใหม่
                self.position = -1
                self.entry_price = current_price
                trade_executed = True
            else: # Position == 0
                self.position = -1
                self.entry_price = current_price
                trade_executed = True

        # --- คำนวณมูลค่าพอร์ตfolio หลังทำ Action ---
        new_portfolio_value = self.balance
        if self.position == 1:
            new_portfolio_value += (current_price - self.entry_price) * 100
        elif self.position == -1:
            new_portfolio_value += (self.entry_price - current_price) * 100

        # --- Reward Calculation ---
        # 1. Main reward: portfolio value change
        reward = new_portfolio_value - prev_portfolio_value
        
        # 2. ลด transaction cost penalty มาก (เดิม 0.1 -> 0.01)
        if trade_executed:
            reward -= 0.01
        
        # 3. **FIX: เพิ่ม penalty เล็กน้อยสำหรับการ Hold โดยไม่มี position**
        # ทำให้โมเดลต้อง "ทำอะไรสักอย่าง" แทนที่จะนิ่งตลอด
        if action == 0 and self.position == 0:
            reward -= 0.02  # Small penalty for doing nothing
        
        # 4. Reward สำหรับ unrealized PnL (กำไรที่ยังไม่ได้ปิด)
        # ทำให้โมเดลเห็นว่าการถือ position ที่กำไรดีกว่า hold
        if self.position != 0:
            unrealized_pnl = 0
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * 100
            elif self.position == -1:
                unrealized_pnl = (self.entry_price - current_price) * 100
            
            # เพิ่ม reward เล็กน้อยสำหรับ unrealized profit
            reward += unrealized_pnl * 0.1  # 10% ของ unrealized PnL
        
        # 5. Normalize reward เพื่อป้องกันค่า reward ที่แปรปรวนมากเกินไป
        reward = reward / 10.0  # Scale down rewards

        # --- อัพเดทขั้นตอนถัดไป ---
        self.current_step += 1
        
        # เช็คว่าจบ Episode หรือยัง
        if self.current_step >= len(self.df) - 1:
            done = True
            # ปิดออเดอร์สุดท้าย
            if self.position == 1:
                self.balance += (current_price - self.entry_price) * 100
            elif self.position == -1:
                self.balance += (self.entry_price - current_price) * 100
        
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, done, False, info