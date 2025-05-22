import functools

import pufferlib

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='auto', buf=None, seed=0):
    '''Crafter creation function'''
    return MettaPuff(config, render_mode, buf)

def oc_divide(a, b):
    """
    Divide a by b, returning an int if both inputs are ints and result is a whole number,
    otherwise return a float.
    """
    result = a / b
    # If both inputs are integers and the result is a whole number, return as int
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result


class MettaPuff(pufferlib.PufferEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        self.render_mode = render_mode
        import mettagrid.mettagrid_env
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("div", oc_divide, replace=True)
        cfg = OmegaConf.load(config)

        from mettagrid.mettagrid_env import MettaGridEnv
        self.env = MettaGridEnv(cfg, render_mode=render_mode, buf=buf)

        #if render_mode == 'human':
        #    from mettagrid.gym_wrapper import RaylibRendererWrapper
        #    self.env = RaylibRendererWrapper(self.env, self.env._env_cfg)

        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.num_agents = self.env.num_agents
        super().__init__(buf)

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)

        if all(term) or all(trunc):
            self.reset()
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']
        else:
            info = []

        return obs, rew, term, trunc, [info]

    def reset(self, seed=None):
        obs = self.env.reset()
        self.tick = 0
        return obs, []

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
