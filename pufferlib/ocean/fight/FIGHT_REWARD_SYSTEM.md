# Fight Environment Reward System Documentation

## Overview
This document describes the reward system implementation for the PufferLib fight environment, including hit-based rewards, logging, and best practices.

## Reward Structure

### Per-Step Rewards
Rewards are calculated every time `c_step()` is called:

1. **Time Penalty**: `-0.01f` per step (encourages action)
2. **Hit Rewards**: `+0.1f` for successful hit, `-0.05f` for getting hit
3. **Episode End**: `±1.0f` for win/loss, `0.0f` for draw

### Reward Flow
```
Each Step:
├── Time penalty: -0.01f
├── Hit rewards (if attack lands): +0.1f/-0.05f
└── Episode end (if terminal): ±1.0f

Episode End:
└── add_log() called once to log cumulative metrics
```

## Implementation Guide

### 1. CheckAttack Function Modification
```c
static void CheckAttack(Fighter *att, Fighter *def, Fight *env, int attacker_idx) {
    // ... existing collision detection ...

    if (!CheckCollisionRecs(hb, defBox))
        return; /* whiff */

    // Successful hit landed!
    int defender_idx = 1 - attacker_idx;
    env->rewards[attacker_idx] += 0.1f;   // Reward for landing hit
    env->rewards[defender_idx] -= 0.05f;  // Penalty for getting hit

    // ... rest of blocking/damage logic ...
}
```

### 2. Update Function Calls in c_step()
```c
// Change from:
CheckAttack(&env->fighters[0], &env->fighters[1]);
CheckAttack(&env->fighters[1], &env->fighters[0]);

// To:
CheckAttack(&env->fighters[0], &env->fighters[1], env, 0);
CheckAttack(&env->fighters[1], &env->fighters[0], env, 1);
```

### 3. Correct add_log() Implementation
```c
void add_log(Fight *env) {
    float p1_health = (float)env->fighters[0].hp / env->fighters[0].hpMax;
    float p2_health = (float)env->fighters[1].hp / env->fighters[1].hpMax;

    env->log.perf += (p1_health > p2_health) ? 1.0f : 0.0f;
    env->log.score += p1_health - p2_health;
    env->log.episode_length += 1.0f;
    env->log.episode_return += env->rewards[0];  // Track actual rewards!
    env->log.n += 1.0f;
}
```

## Logging Metrics Explained

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `perf` | Win rate (0-1) | 1.0 if P1 wins, 0.0 if loses |
| `score` | Health differential | P1_health - P2_health |
| `episode_return` | Total rewards | Sum of all rewards received |
| `episode_length` | Steps taken | Always increments by 1 |

## Key Implementation Notes

### Reward Timing
- **Per-step rewards**: Set in `c_step()` every frame
- **Hit rewards**: Applied immediately in `CheckAttack()` when hit lands
- **Episode rewards**: Applied at terminal states (death/max steps)
- **Logging**: `add_log()` called once per episode at termination

### Multi-Agent Considerations
- Player 1 (index 0) is the training agent
- Player 2 (index 1) is the opponent/environment
- All logged metrics should be from Player 1's perspective
- `episode_return` tracks `env->rewards[0]` (Player 1's rewards)

### Common Pitfalls to Avoid
1. **Don't track health differential in episode_return** - track actual rewards
2. **Reset rewards properly** - ensure rewards are set each step, not accumulated
3. **Call add_log() only at episode end** - not every step
4. **Track both players' rewards** - even if only logging Player 1's metrics

## Debugging Tips

### Check Reward Values
```c
// Add debug prints to verify rewards
printf("Step %d: P1 reward=%.3f, P2 reward=%.3f\n",
       env->current_step, env->rewards[0], env->rewards[1]);
```

### Verify Hit Detection
```c
// In CheckAttack, after successful hit
printf("Hit landed! Attacker=%d, Defender=%d, Damage=%d\n",
       attacker_idx, defender_idx, MOVE_DATA[att->attackID].dmg);
```

### Monitor Episode Returns
```c
// In add_log, before logging
printf("Episode end: Return=%.3f, Length=%.0f, Perf=%.3f\n",
       env->log.episode_return, env->log.episode_length, env->log.perf);
```

## Future Enhancements

### Additional Reward Types
- **Combo rewards**: Bonus for multiple hits in sequence
- **Defense rewards**: Small reward for successful blocks
- **Positioning rewards**: Reward for maintaining optimal distance
- **Style rewards**: Bonus for using varied attacks

### Advanced Metrics
- **Hit rate**: Successful hits / total attacks attempted
- **Block rate**: Successful blocks / total attacks faced
- **Damage efficiency**: Damage dealt / time taken
- **Action diversity**: Entropy of action distribution

## Testing Checklist

- [ ] Hit rewards are applied immediately on successful contact
- [ ] Both attacker and defender receive appropriate rewards
- [ ] Episode return accumulates all rewards correctly
- [ ] Win/loss rewards still work at episode end
- [ ] Time penalties are applied every step
- [ ] add_log() is called only at episode termination
- [ ] All metrics are logged from Player 1's perspective

## References

- **Pong implementation**: `pufferlib/ocean/pong/pong.h` (lines 85-92)
- **Cartpole implementation**: `pufferlib/ocean/cartpole/cartpole.h` (lines 55-67)
- **Battle implementation**: `pufferlib/ocean/battle/battle.h` (lines 827-832)
- **PufferLib documentation**: See `CLAUDE.md` in repository root