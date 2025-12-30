import gymnasium as gym
import numpy as np


class ProductionLineEnv(gym.Env):
    def __init__(self, df, predict_fn, window_size=10):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.predict_fn = predict_fn
        self.window_size = window_size
        self.target_col = df.columns[0]

        self.current_step = window_size

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 1),
            dtype=np.float32
        )

    def _obs(self):
        window = self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ][self.target_col].values

        return window.reshape(-1, 1).astype(np.float32)  # (T,1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        return self._obs(), {}

    def step(self, action):
        obs_window = self._obs()            # (T,1)
        true_val = float(
            self.df.iloc[self.current_step][self.target_col]
        )

        pred_val = float(self.predict_fn(obs_window))  # LSTM çağrısı

        # continuous action: [-1,1]
        a = float(np.asarray(action)[0])
        adjusted_pred = pred_val * (1.0 + a)

        error = abs(true_val - adjusted_pred)
        reward = -error

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        next_obs = self._obs() if not terminated else obs_window

        return next_obs, reward, terminated, truncated, {}
