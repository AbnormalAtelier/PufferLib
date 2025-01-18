import configparser
import argparse
import shutil
import glob
import uuid
import ast
import os
import random
import time
import torch

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl
   
def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])

def init_wandb(args, name, id=None, resume=True, tag=None):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=False,
        resume=resume,
        config=args,
        name=name,
        tags=[tag] if tag is not None else [],
    )
    return wandb

def init_neptune(args, name, id=None, resume=True, tag=None):
    import neptune
    run = neptune.init_run(
        project="pufferai/ablations",
        capture_hardware_metrics=False,
        capture_stdout=False,
        capture_stderr=False,
        capture_traceback=False,
        tags=[tag] if tag is not None else [],
    )
    return run

import numpy as np

def log_normal(mean, scale, clip):
    '''Samples normally spaced points on a log 10 scale.
    mean: Your center sample point
    scale: standard deviation in base 10 orders of magnitude
    clip: maximum standard deviations

    Example: mean=0.001, scale=1, clip=2 will produce data from
    0.1 to 0.00001 with most of it between 0.01 and 0.0001
    '''
    return 10**np.clip(
        np.random.normal(
            np.log10(mean),
            scale,
        ),
        a_min = np.log10(mean) - clip,
        a_max = np.log10(mean) + clip,
    )

def logit_normal(mean, scale, clip):
    '''log normal but for logit data like gamma and gae_lambda'''
    return 1 - log_normal(1 - mean, scale, clip)

def uniform_pow2(min, max):
    '''Uniform distribution over powers of 2 between min and max inclusive'''
    min_base = np.log2(min)
    max_base = np.log2(max)
    return 2**np.random.randint(min_base, max_base+1)

def uniform(min, max):
    '''Uniform distribution between min and max inclusive'''
    return np.random.uniform(min, max)

def int_uniform(min, max):
    '''Uniform distribution between min and max inclusive'''
    return np.random.randint(min, max+1)

def sample_hyperparameters(sweep_config):
    samples = {}
    for name, param in sweep_config.items():
        if name in ('method', 'name', 'metric'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            samples[name] = sample_hyperparameters(param)
        elif 'values' in param:
            assert 'distribution' not in param
            samples[name] = random.choice(param['values'])
        elif 'distribution' in param:
            if param['distribution'] == 'uniform':
                samples[name] = uniform(param['min'], param['max'])
            elif param['distribution'] == 'int_uniform':
                samples[name] = int_uniform(param['min'], param['max'])
            elif param['distribution'] == 'uniform_pow2':
                samples[name] = uniform_pow2(param['min'], param['max'])
            elif param['distribution'] == 'log_normal':
                samples[name] = log_normal(
                    param['mean'], param['scale'], param['clip'])
            elif param['distribution'] == 'logit_normal':
                samples[name] = logit_normal(
                    param['mean'], param['scale'], param['clip'])
            else:
                raise ValueError(f'Invalid distribution: {param["distribution"]}')
        else:
            raise ValueError('Must specify either values or distribution')

    return samples

from carbs import (
    CARBS,
    CARBSParams,
    ObservationInParam,
    Param,
    LinearSpace,
    Pow2Space,
    LogSpace,
    LogitSpace,
)

class PufferCarbs:
    def __init__(self,
            sweep_config: dict,
            max_suggestion_cost: float = None,
            resample_frequency: int = 5,
            num_random_samples: int = 10,
            max_suggestion_count: float = None,
        ):
        param_spaces = _carbs_params_from_puffer_sweep(sweep_config)
        flat_spaces = [e[1] for e in pufferlib.utils.unroll_nested_dict(param_spaces)]
        for e in flat_spaces:
            print(e.name, e.space)

        metric = sweep_config['metric']
        goal = metric['goal']
        assert goal in ['maximize', 'minimize'], f"Invalid goal {goal}"
        self.carbs_params = CARBSParams(
            better_direction_sign=1 if goal == 'maximize' else -1,
            is_wandb_logging_enabled=False,
            resample_frequency=5,
            num_random_samples=len(flat_spaces),
            max_suggestion_cost=max_suggestion_cost,
            is_saved_on_every_observation=False,
        )
        self.carbs = CARBS(self.carbs_params, flat_spaces)

    def suggest(self, args):
        self.suggestion = self.carbs.suggest().suggestion
        for k in ('train', 'env'):
            for name, param in args['sweep'][k].items():
                if name in self.suggestion:
                    args[k][name] = self.suggestion[name]

    def observe(self, score, cost, is_failure=False):
        self.carbs.observe(
            ObservationInParam(
                input=self.suggestion,
                output=score,
                cost=cost,
                is_failure=is_failure,
            )
        )

def _carbs_params_from_puffer_sweep(sweep_config):
    param_spaces = {}
    for name, param in sweep_config.items():
        if name in ('method', 'name', 'metric'):
            continue

        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = _carbs_params_from_puffer_sweep(param)
            continue
 
        assert 'distribution' in param
        distribution = param['distribution']
        search_center = param['mean']
        kwargs = dict(
            min=param['min'],
            max=param['max'],
            scale=param['scale'],
            mean=search_center,
        )
        if distribution == 'uniform':
            space = LinearSpace(**kwargs)
        elif distribution == 'int_uniform':
            space = LinearSpace(**kwargs, is_integer=True)
        elif distribution == 'uniform_pow2':
            space = Pow2Space(**kwargs, is_integer=True)
        elif distribution == 'log_normal':
            space = LogSpace(**kwargs)
        elif distribution == 'logit_normal':
            space = LogitSpace(**kwargs)
        else:
            raise ValueError(f'Invalid distribution: {distribution}')

        param_spaces[name] = Param(name=name, space=space, search_center=search_center)

    return param_spaces

def sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls):
    target_metric = args['sweep']['metric']['name']
    carbs = PufferCarbs(
        args['sweep'],
        resample_frequency=5,
        num_random_samples=10, # Should be number of params
        max_suggestion_cost=args['base']['max_suggestion_cost'],
    )
    for i in range(args['max_runs']):
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
 
        carbs.suggest(args)
        if args['train']['batch_size'] / args['train']['num_minibatches'] > 32_768:
            carbs.observe(score=0, cost=0, is_failure=True)
            continue

        target, uptime, _, _ = train(args, make_env, policy_cls, rnn_cls, target_metric)
        carbs.observe(score=target, cost=uptime)

def synthetic_carbs_observation(args):
    '''Simulates the outcome of an RL experiment by making
    some heavy handed assumptions about hyperparameters'''
    num_envs = args['env']['num_envs']
    train_args = args['train']
    total_timesteps = train_args['total_timesteps']
    batch_size = train_args['batch_size']
    num_minibatches = train_args['num_minibatches']
    learning_rate = train_args['learning_rate']
    gamma = train_args['gamma']
    gae_lambda = train_args['gae_lambda']
    update_epochs = train_args['update_epochs']
    bptt_horizon = train_args['bptt_horizon']
    minibatch_size = batch_size // num_minibatches

    # Base score maxes out at 1.0
    base_score = (
        - 100*abs(learning_rate - 0.001)
        - 100*abs(gamma - 0.99)
        - 100*abs(gae_lambda - 0.95)
        - abs(bptt_horizon - 16)/16.0
        + 20.0
    )

    score_mod = (
        np.log2(total_timesteps) / np.log2(100_000_000)
        * np.sqrt(batch_size)  / np.sqrt(65536)
        * np.sqrt(num_minibatches)
        * np.sqrt(update_epochs)
    )

    '''
    cost = (
        total_timesteps / 100_000_000
        * 16_384 / minibatch_size
        * 1024 / num_envs
        * update_epochs
        * num_minibatches
    )
    '''

    score = score_mod * base_score
    cost = 1

    return score, cost


def test_carbs(args, env_name, make_env, policy_cls, rnn_cls):
    target_metric = args['sweep']['metric']['name']
    carbs = PufferCarbs(
        args['sweep'],
        resample_frequency=5,
        num_random_samples=10, # Should be number of params
        max_suggestion_cost=args['base']['max_suggestion_cost'],
    )
    scores = []
    for i in range(args['max_runs']):
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
 
        carbs.suggest(args)

        # Optimal params
        '''
        train_args = args['train']
        train_args['total_timesteps'] = 1e9
        train_args['batch_size'] = 262144
        train_args['num_minibatches'] = 8
        train_args['learning_rate'] = 0.001
        train_args['gamma'] = 0.99
        train_args['gae_lambda'] = 0.95
        train_args['update_epochs'] = 4
        train_args['bptt_horizon'] = 16
        '''

        score, cost = synthetic_carbs_observation(args)

        '''
        neptune = init_neptune(args, env_name, id=args['exp_id'], tag=args['tag'])
        for k, v in pufferlib.utils.unroll_nested_dict(args):
            neptune[k].append(v)

        neptune['environment/score'].append(score)
        neptune['environment/uptime'].append(cost)
        neptune.stop()
        '''
 
        #stats, uptime, _, _ = train(args, make_env, policy_cls, rnn_cls)
        carbs.observe(score=score, cost=cost)

        import plotly.graph_objects as go
        scores.append(score)
        t = list(range(len(scores)))
        fig = go.Figure(data=go.Scatter(x=t, y=scores, mode='markers'))
        fig.update_layout(title='CARBS Synthetic Test', xaxis_title='Index', yaxis_title='Value')
        fig.show()


    import plotly.express as px
    y = [3, 7, 1, 5, 9]  # Replace with your values
    x = range(len(y))  # Generate x as range
    fig = px.scatter(x=x, y=y, labels={'x': 'Index', 'y': 'Value'})
    fig.show()


def sweep(args, env_name, make_env, policy_cls, rnn_cls):
    target_metric = args['sweep']['metric']['name']
    for i in range(args['max_runs']):
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))
        hypers = sample_hyperparameters(args['sweep'])
        args['train'].update(hypers['train'])
        train(args, make_env, policy_cls, rnn_cls, target_metric)

def train(args, make_env, policy_cls, rnn_cls, target_metric, min_eval_points=100,
        elos={'model_random.pt': 1000}, vecenv=None, wandb=None, neptune=None):

    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    elif args['vec'] == 'native':
        vec = pufferlib.environment.PufferEnv
    else:
        raise ValueError(f'Invalid --vector (serial/multiprocessing/ray/native).')

    if vecenv is None:
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=args['env'],
            num_envs=args['train']['num_envs'],
            num_workers=args['train']['num_workers'],
            batch_size=args['train']['env_batch_size'],
            zero_copy=args['train']['zero_copy'],
            overwork=args['vec_overwork'],
            backend=vec,
        )

    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)

    '''
    if env_name == 'moba':
        import torch
        os.makedirs('moba_elo', exist_ok=True)
        torch.save(policy, os.path.join('moba_elo', 'model_random.pt'))
    '''

    neptune = None
    wandb = None
    if args['neptune']:
        neptune = init_neptune(args, env_name, id=args['exp_id'], tag=args['tag'])
        for k, v in pufferlib.utils.unroll_nested_dict(args):
            neptune[k].append(v)
    elif args['wandb']:
        wandb = init_wandb(args, env_name, id=args['exp_id'], tag=args['tag'])

    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb, neptune=neptune)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    steps_evaluated = 0
    uptime = data.profile.uptime
    batch_size = args['train']['batch_size']
    while len(data.stats[target_metric]) < min_eval_points:
        stats, _ = clean_pufferl.evaluate(data)
        data.experience.sort_keys = []
        steps_evaluated += batch_size

    print(f'Evaluated {steps_evaluated} steps')
    clean_pufferl.mean_and_log(data)

    '''
    if env_name == 'moba':
        exp_n = len(elos)
        model_name = f'model_{exp_n}.pt'
        torch.save(policy, os.path.join('moba_elo', model_name))
        from evaluate_elos import calc_elo
        elos = calc_elo(model_name, 'moba_elo', elos)
        stats['elo'] = elos[model_name]
        if wandb is not None:
            wandb.log({'environment/elo': elos[model_name]})
    '''

    clean_pufferl.close(data)
    return stats[target_metric], uptime, elos, vecenv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--env', '--environment', type=str,
        default='puffer_squared', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval evaluate sweep sweep-carbs test-carbs autotune profile'.split())
    parser.add_argument('--vec', '--vector', '--vectorization', type=str,
        default='native', choices=['serial', 'multiprocessing', 'ray', 'native'])
    parser.add_argument('--vec-overwork', action='store_true',
        help='Allow vectorization to use >1 worker/core. Not recommended.')
    parser.add_argument('--eval-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--baseline', action='store_true',
        help='Load pretrained model from WandB if available')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--exp-id', '--exp-name', type=str,
        default=None, help='Resume from experiment')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--wandb', action='store_true', help='Track on WandB')
    parser.add_argument('--neptune', action='store_true', help='Track on Neptune')
    #parser.add_argument('--wandb-project', type=str, default='pufferlib')
    #parser.add_argument('--wandb-group', type=str, default='debug')
    args = parser.parse_known_args()[0]

    file_paths = glob.glob('config/**/*.ini', recursive=True)
    for path in file_paths:
        p = configparser.ConfigParser()
        p.read('config/default.ini')

        subconfig = os.path.join(*path.split('/')[:-1] + ['default.ini'])
        if subconfig in file_paths:
            p.read(subconfig)

        p.read(path)
        if args.env in p['base']['env_name'].split():
            break
    else:
        raise Exception('No config for env_name {}'.format(args.env))

    for section in p.sections():
        for key in p[section]:
            argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    parsed = parser.parse_args().__dict__
    args = {'env': {}, 'policy': {}, 'rnn': {}}
    env_name = parsed.pop('env')
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    package = args['base']['package']
    module_name = f'pufferlib.environments.{package}'
    if package == 'ocean':
        module_name = 'pufferlib.ocean'

    import importlib
    env_module = importlib.import_module(module_name)

    make_env = env_module.env_creator(env_name)
    policy_cls = getattr(env_module.torch, args['base']['policy_name'])
    
    rnn_name = args['base']['rnn_name']
    rnn_cls = None
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['base']['rnn_name'])

    if args['baseline']:
        assert args['mode'] in ('train', 'eval', 'evaluate')
        args['track'] = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args['exp_id'] = f'puf-{version}-{env_name}'
        args['wandb_group'] = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args["exp_id"]}', ignore_errors=True)
        run = init_wandb(args, args['exp_id'], resume=False)
        if args['mode'] in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args['eval_model_path'] = os.path.join(data_dir, model_file)
    if args['mode'] == 'train':
        target_metric = args['sweep']['metric']['name']
        train(args, make_env, policy_cls, rnn_cls, target_metric)
    elif args['mode'] in ('eval', 'evaluate'):
        vec = pufferlib.vector.Serial
        if args['vec'] == 'native':
            vec = pufferlib.environment.PufferEnv

        clean_pufferl.rollout(
            make_env,
            args['env'],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            backend=vec,
            model_path=args['eval_model_path'],
            render_mode=args['render_mode'],
            device=args['train']['device'],
        )
    elif args['mode'] == 'sweep':
        assert args['wandb'] or args['neptune'], 'Sweeps require either wandb or neptune'
        sweep(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'sweep-carbs':
        sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'test-carbs':
        test_carbs(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
    elif args['mode'] == 'profile':
        import cProfile
        cProfile.run('train(args, make_env, policy_cls, rnn_cls, wandb=None)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)


