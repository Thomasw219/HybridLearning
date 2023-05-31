from typing import Tuple, Union
import gym

class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_length: int, **kwargs):
        super().__init__(env, **kwargs)
        self._max_episode_length = max_episode_length
        self._current_episode_length = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action.copy())
        self._current_episode_length += 1

        if self._current_episode_length >= self._max_episode_length:
            done = True
        else:
            done = False

        return next_state, reward, done, info

    def reset(self, **kwargs):
        self._current_episode_length = 0
        return self.env.reset(**kwargs)
