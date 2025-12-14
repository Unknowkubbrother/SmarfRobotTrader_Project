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
        self.transaction_cost = 0.0005 # ตัวอย่างค่าธรรมเนียม 0.05%

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

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['mid_price']
        reward = 0
        done = False
        truncated = False

        # --- คำนวณ Reward จาก Action ที่ผ่านมา ---
        if self.position == 1: # ถ้ากำลัง Buy
            reward = (current_price - self.entry_price) * 100 # สมมติว่า 1 pip = 100 units
        elif self.position == -1: # ถ้ากำลัง Sell
            reward = (self.entry_price - current_price) * 100

        # --- ประมวลผล Action ปัจจุบัน ---
        if action == 1: # Buy
            if self.position == -1: # ถ้ากำลัง Sell ให้ปิดก่อน
                self.balance += (self.entry_price - current_price) * 100
                self.balance -= self.transaction_cost * abs(self.entry_price - current_price) * 100
            self.position = 1
            self.entry_price = current_price
        elif action == 2: # Sell
            if self.position == 1: # ถ้ากำลัง Buy ให้ปิดก่อน
                self.balance += (current_price - self.entry_price) * 100
                self.balance -= self.transaction_cost * abs(current_price - self.entry_price) * 100
            self.position = -1
            self.entry_price = current_price

        # --- อัพเดทขั้นตอน ---
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

        return obs, reward, done, truncated, info