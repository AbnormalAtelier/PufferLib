# Fight Environment Episode Termination Fixes

## Problem Analysis
Episodes weren't terminating because:
- **HP Max**: 180 (too high)
- **Damage per attack**: Jab=5, DF1=13, DF2=40, D2=14
- **Attacks needed to kill**: Jab=36 hits, DF1=14 hits, DF2=5 hits, D2=13 hits
- With blocking and no combo system, rounds take too long to end

## Quick Fix (Do This First)

### 1. Reduce Health Pool
In `sample.h`, line 144, change:
```c
f->hpMax = 80;  // Reduced from 180
f->hp = f->hpMax;
```

This makes termination much faster:
- Jab: 16 hits to kill (was 36)
- DF1: 7 hits to kill (was 14)
- DF2: 2 hits to kill (was 5)
- D2: 6 hits to kill (was 13)

## Additional Improvements

### 2. Add Maximum Episode Length

**Implementation Pattern from Other Ocean Files:**

Based on patterns from Cartpole, Pacman, and Trash Pickup environments:

**Option A: Constant-Based (Cartpole Pattern)**
```c
// At top of fight.h
#define MAX_STEPS 200

// In c_step function
bool truncated = env->current_step >= MAX_STEPS;
bool done = terminated || truncated;
```

**Option B: Environment Field (Trash Pickup Pattern)**
```c
// Add to Fight struct
int max_steps;      // Configurable maximum
int current_step;   // Current step counter

// In c_reset
env->max_steps = 1000;  // ~16 seconds at 60 FPS
env->current_step = 0;

// In c_step
env->current_step++;
if (env->current_step >= env->max_steps) {
    env->round_over = true;
    env->terminals[0] = 1;
    env->terminals[1] = 1;
    float health_diff = (float)(env->fighters[0].hp - env->fighters[1].hp) / 80.0f;
    env->rewards[0] = health_diff;
    env->rewards[1] = -health_diff;
    add_log(env);
    return;
}
```

**Common Patterns from Ocean Files:**
- **Cartpole**: Uses `#define MAX_STEPS 200` with `env->tick` counter
- **Pacman**: Uses `#define MAX_STEPS 10000` with `env->step_count` field
- **Trash Pickup**: Uses configurable `int max_steps` and `int current_step` fields
- **Step tracking**: Usually `env->tick++` or `env->current_step++` in `c_step()`
- **Termination**: `if (env->current_step >= MAX_STEPS)` pattern
- **Logging**: Track `max_steps_termination` in log metrics (Cartpole pattern)

### 3. Increase Damage Values (Optional)
```c
static const MoveStats MOVE_DATA[5] = {
    {0},
    {10, 2, 19, 70, 20, 0.1, 12, 28, 18},   // Jab: 5->12
    {12, 2, 23, 70, 20, 0.45, 25, 30, 25}, // df1: 13->25
    {15, 2, 31, 70, 20, 0.5, 50, 25, 29},  // df2: 40->50
    {18, 2, 31, 70, 20, 0.8, 20, 31, 18}   // d2: 14->20
};
```

### 4. Add Hit/Block Rewards (Recommended)
In CheckAttack function, after successful hit:
```c
env->rewards[attacker_index] += 0.1f;  // Small reward for landing hit
env->rewards[defender_index] -= 0.05f; // Small penalty for getting hit
```

For blocks:
```c
if (attack_was_blocked) {
    env->rewards[defender_index] += 0.05f; // Reward for good defense
}
```

## Implementation Order
1. **Start with health reduction** - this alone should fix termination
2. **Add max episode length** - ensures episodes always end for logging
3. **Add hit rewards** - improves learning signal
4. **Consider damage increases** - if still too slow

The health reduction from 180 to 80 is the most important change and should immediately solve your episode termination problem.

## RL Environment Best Practices

### Control System Design for Reinforcement Learning

For reinforcement learning environments, use the **classic discrete action system** rather than advanced contextual controls.

#### Why Discrete Actions Are Better for RL:

1. **Smaller Action Space**: 8 discrete actions vs potentially larger contextual space
2. **Clear Reward Attribution**: High actions (5,6,7,8) map directly to specific attacks
3. **Consistent State Transitions**: Same input always produces same result
4. **Better for Policy Gradient Methods**: Direct causal chain from policy → action → result
5. **Sample Efficiency**: Explores meaningful attacks immediately vs. wasting time on blocked inputs

#### Recommended RL Configuration:

```c
#define ACTIONS_CLASSIC // Use discrete actions for RL training

// RL-friendly state observations
float observations[] = {
    f->health / MAX_HEALTH,        // Health 0-1
    opp->health / MAX_HEALTH,      // Opponent health 0-1
    f->distance_to_opponent,       // Spatial information
    f->grounded,                   // Jump state
    f->in_recovery_frames,         // Action safety
    opp->in_recovery_frames        // Punish opportunities
};
```

#### When to Use Advanced System:
- Human gameplay (more intuitive)
- Final polish/testing (after basic RL training)
- Multi-agent scenarios with human opponents
- Transfer learning from human demonstrations

**Bottom Line**: Start with discrete action system for RL training, add advanced controls later for gameplay variety or human competition. The classic system converges to good policies much faster and more reliably.