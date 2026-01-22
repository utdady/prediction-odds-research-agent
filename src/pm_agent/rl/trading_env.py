"""Reinforcement Learning trading environment."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


if GYM_AVAILABLE:

    class TradingEnv(gym.Env):
        """Trading environment for reinforcement learning."""

        def __init__(self, features_df: pd.DataFrame, prices_df: pd.DataFrame):
            """
            Initialize trading environment.
            
            Args:
                features_df: DataFrame with features (indexed by date)
                prices_df: DataFrame with prices (columns: open, close, indexed by date)
            """
            super().__init__()

            self.features = features_df
            self.prices = prices_df
            self.current_step = 0

            # Action space: [0=hold, 1=buy, 2=sell]
            self.action_space = spaces.Discrete(3)

            # Observation space: features + position state
            n_features = len(features_df.columns) if not features_df.empty else 10
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_features + 3,),  # features + [position, pnl, days_held]
            )

            self.position = 0
            self.entry_price = 0.0
            self.pnl = 0.0
            self.days_held = 0

        def reset(self, seed=None, options=None):
            """Reset environment to initial state."""
            super().reset(seed=seed)
            self.current_step = 0
            self.position = 0
            self.entry_price = 0.0
            self.pnl = 0.0
            self.days_held = 0
            return self._get_observation(), {}

        def step(self, action: int):
            """
            Execute one step in the environment.
            
            Args:
                action: 0=hold, 1=buy, 2=sell
            
            Returns:
                observation, reward, done, info
            """
            if self.current_step >= len(self.features):
                return self._get_observation(), 0.0, True, {}

            current_price = float(self.prices.iloc[self.current_step]["close"])
            reward = 0.0

            if action == 1 and self.position == 0:  # Buy
                self.position = 1
                self.entry_price = current_price
                self.days_held = 0

            elif action == 2 and self.position == 1:  # Sell
                self.pnl = (current_price - self.entry_price) / self.entry_price
                reward = self.pnl * 100  # Scale reward
                self.position = 0
                self.days_held = 0

            elif self.position == 1:  # Holding
                self.days_held += 1
                # Small negative reward for holding (encourage active trading)
                reward = -0.01

            self.current_step += 1
            done = self.current_step >= len(self.features)

            obs = self._get_observation()

            info = {
                "position": self.position,
                "pnl": self.pnl,
                "days_held": self.days_held,
            }

            return obs, reward, done, False, info

        def _get_observation(self) -> np.ndarray:
            """Get current observation."""
            if self.current_step >= len(self.features):
                # Return zeros if out of bounds
                n_features = len(self.features.columns) if not self.features.empty else 10
                return np.zeros(n_features + 3)

            features = self.features.iloc[self.current_step].values
            state = np.array([self.position, self.pnl, self.days_held])
            return np.concatenate([features, state])

else:
    # Placeholder if gym not available
    TradingEnv = None

