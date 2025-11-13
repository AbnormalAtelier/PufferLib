# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PufferLib is a high-performance reinforcement learning library designed for compatibility with complex environments. It provides optimized parallel simulation, environments that run and train at 1M+ steps/second, and compatibility layers for popular RL frameworks like Gymnasium, PettingZoo, and CleanRL.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install with specific environment dependencies
pip install -e ".[atari]"        # For Atari environments
pip install -e ".[mujoco]"       # For MuJoCo environments
pip install -e ".[box2d]"        # For Box2D environments
# See pyproject.toml for full list of environment extras

# Build C/CUDA extensions (happens automatically during install)
python setup.py build_ext --inplace

# Debug build with symbols
DEBUG=1 python setup.py build_ext --inplace --force
```

### Running Tests
```bash
# Run individual test files
python tests/test_api.py
python tests/test_performance.py
python tests/test_puffernet.py

# Run memory tests
python tests/mem_test.py

# Run specific environment tests
python tests/test_env_binding.py
```

### Training and Evaluation
```bash
# Train an agent
puffer train [env_name] [options]
# Example: puffer train atari_pong

# Evaluate a trained agent
puffer eval [env_name] --load-model-path path/to/model.pt

# Run hyperparameter sweeps
puffer sweep [env_name] --wandb  # Requires wandb
puffer sweep [env_name] --neptune # Requires neptune

# Profile performance
puffer profile [env_name]

# Autotune vectorization parameters
puffer autotune [env_name]
```

## Architecture Overview

### Core Components

1. **PufferEnv** (`pufferlib/pufferlib.py`): Base class for native environments
   - Requires `single_observation_space`, `single_action_space`, `num_agents`
   - Handles internal vectorization and buffer management
   - Native environments run faster than emulated ones

2. **Vectorization** (`pufferlib/vector.py`): High-performance environment vectorization
   - `Serial`: Single-process vectorization for debugging
   - `Multiprocessing`: Multi-process with shared memory for performance
   - `Ray`: Distributed vectorization for clusters
   - `PufferEnv`: Native vectorization built into environments

3. **PuffeRL** (`pufferlib/pufferl.py`): Main training framework
   - PPO-based algorithm with advanced features
   - CUDA kernels for advantage computation
   - Support for RNN policies, distributed training
   - Integration with WandB and Neptune for logging

4. **Environment Emulation** (`pufferlib/emulation.py`): Compatibility layers
   - `GymnasiumPufferEnv`: Wraps Gymnasium environments
   - `PettingZooPufferEnv`: Wraps PettingZoo environments

### Environment Structure

Environments are organized in two main categories:

1. **Ocean** (`pufferlib/ocean/`): Native C/C++ environments for maximum performance
   - Includes games, simulations, and custom environments
   - Each has a `binding.c` file for Python integration
   - Examples: fight, nmmo, pokemon_red, etc.

2. **Environments** (`pufferlib/environments/`): Python wrappers for popular RL environments
   - Each environment package has `__init__.py`, `environment.py`, `torch.py`
   - Examples: atari, mujoco, box2d, pettingzoo, etc.

### Key Design Patterns

1. **Buffer Management**: Environments write directly to shared memory buffers to minimize copying
2. **Async Interface**: `async_reset()`, `send()`, `recv()` pattern for vectorized operations
3. **Agent Batching**: Multiple agents per environment with proper ID management
4. **Space Handling**: Custom space classes that extend Gymnasium spaces for multi-agent support

## Important Implementation Notes

### Performance Considerations
- Native environments (PufferEnv) are significantly faster than emulated ones
- Use `Multiprocessing` backend for best single-machine performance
- CUDA advantage computation requires nvcc compiler
- Shared memory vectorization minimizes data copying

### Common Pitfalls
- Always call `async_reset()` before `recv()` in vectorized environments
- Ensure `num_agents` is set correctly in environment constructors
- Use `single_observation_space` and `single_action_space` (not `observation_space`/`action_space`)
- Check that batch sizes are divisible by number of workers

### Extension Development
- New environments should inherit from `PufferEnv` for best performance
- Implement required methods: `reset()`, `step()`, `close()`
- Set required attributes in `__init__()` before calling `super().__init__()`
- Use provided buffer system for memory efficiency

## Configuration System

PufferLib uses INI files for configuration (`pufferlib/config/`):
- `default.ini`: Base configuration
- Environment-specific configs override defaults
- Command-line arguments override config files
- Configs are loaded hierarchically by environment name

## Dependencies and Build System

- **Build**: Custom setup.py with Cython/C++ extensions
- **Raylib**: Downloaded automatically for Ocean environments
- **Box2D**: Downloaded automatically for physics environments
- **PyTorch**: Required for training framework
- **Optional**: wandb, neptune for logging; ray for distributed training