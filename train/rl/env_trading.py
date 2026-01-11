import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.max_step = len(df) - 2

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.step_idx = 0
        self.position = 0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['return'],
            row['range'],
            row['delta_tick'],
            row['delta_price'],
            row['has_delta']
        ], dtype=np.float32)

    def step(self, action):
        prev_pos = self.position

        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        else:
            self.position = prev_pos

        next_ret = self.df.iloc[self.step_idx + 1]['return']
        reward = self.position * next_ret

        # transaction cost
        reward -= 0.0001 * abs(self.position - prev_pos)

        self.step_idx += 1
        done = self.step_idx >= self.max_step

        return self._get_obs(), reward, done, {}
