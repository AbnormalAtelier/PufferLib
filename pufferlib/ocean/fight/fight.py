'''A fighting game environment with Tekken-style mechanics'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.fight import binding

class Fight(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1080, height=720, num_agents=2,
            render_mode=None, log_interval=128, buf=None, seed=0):
        # Fight environment has 2 fighters with complex observations
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(10,), dtype=np.float32)  # 10D observation per fighter
        self.single_action_space = gymnasium.spaces.Discrete(9)  # 9 actions: move, jump, attacks

        self.render_mode = render_mode
        self.num_agents = num_envs * num_agents  # 2 fighters per environment
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, width=width, height=height)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512

    env = Fight(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.n, size=(CACHE, env.num_agents))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Fight SPS:', int(steps / (time.time() - start)))
