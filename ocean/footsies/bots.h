// CPU bot policy for footsies: faithful C port of the Unity demo's scripted
// opponent, Assets/Script/BattleAI.cs from hifight/Footsies. Included by
// footsies.h AFTER the Fighter / Game structs (bots read sim state directly,
// same information model as ocean/robocode/bots.h) and BEFORE the Env struct
// so Env can embed BotMem.
//
// Fidelity notes (deliberate differences from the C# original):
//   * Unity's Random.Range is the engine-global PRNG; here we use rand_r on
//     the env's rng so rollouts stay seed-deterministic per env instance.
//   * BattleAI is hardwired to fighter2 (P2, right side) and reads fighter1
//     as "the opponent": forward = INPUT_LEFT, backward = INPUT_RIGHT.
//     Same here - the bot never plays the left side.
//   * AddBackwardDashInputQueue() in the original enqueues the FORWARD input
//     (copy-paste bug in BattleAI.cs). Kept verbatim - this is a port, not a
//     fix - so C and Unity bots whiff-punish identically.
//   * UpdateFightState's history shift is ascending in the original, which
//     smears the previous frame across all slots (1-frame reaction lag, not
//     5). Kept verbatim.
//   * isOpponentBlocking is computed into FightState but never read by
//     SelectMovement/SelectAttack in the original. Kept, for the same reason.
//   * C# Queues are unbounded; the longest pattern is 121 entries
//     (AddDelaySpecialAttack) and selection only runs on an empty queue, so
//     fixed 160-entry rings cannot overflow. Pushes past the cap are dropped.

#ifndef FOOTSIES_BOTS_H
#define FOOTSIES_BOTS_H

#define FOOTSIES_BOT_QUEUE_CAP 160
#define FOOTSIES_BOT_MAX_FIGHT_STATE_RECORD 10
#define FOOTSIES_BOT_FIGHT_STATE_READ_INDEX 5 /* C# fightStateReadIndex */

typedef enum {
    BOT_NONE = -1,      /* both fighters on the policy (self-play) */
    BOT_BATTLE_AI = 0,  /* Assets/Script/BattleAI.cs */
} FootsiesBotPolicy;

typedef struct {
    float distanceX;
    bool isOpponentDamage;
    bool isOpponentGuardBreak;
    bool isOpponentBlocking;
    bool isOpponentNormalAttack;
    bool isOpponentSpecialAttack;
} BotFightState;

typedef struct {
    int moveQueue[FOOTSIES_BOT_QUEUE_CAP];
    int moveHead, moveLen;   /* ring: pop at head, push at head+len */
    int attackQueue[FOOTSIES_BOT_QUEUE_CAP];
    int attackHead, attackLen;
    BotFightState fightStates[FOOTSIES_BOT_MAX_FIGHT_STATE_RECORD];
    int fightStateLen;       /* C# leaves entries null until filled */
} BotMem;

static inline void bot_mem_reset(BotMem *m)
{
    m->moveHead = m->moveLen = 0;
    m->attackHead = m->attackLen = 0;
    m->fightStateLen = 0;
}

/* --- input-queue primitives (C# moveQueue / attackQueue) --------------- */

static inline void bot_move_push(BotMem *m, int input)
{
    if (m->moveLen < FOOTSIES_BOT_QUEUE_CAP) {
        m->moveQueue[(m->moveHead + m->moveLen) % FOOTSIES_BOT_QUEUE_CAP] = input;
        m->moveLen++;
    }
}

static inline int bot_move_pop(BotMem *m)
{
    int input = m->moveQueue[m->moveHead];
    m->moveHead = (m->moveHead + 1) % FOOTSIES_BOT_QUEUE_CAP;
    m->moveLen--;
    return input;
}

static inline void bot_attack_push(BotMem *m, int input)
{
    if (m->attackLen < FOOTSIES_BOT_QUEUE_CAP) {
        m->attackQueue[(m->attackHead + m->attackLen) % FOOTSIES_BOT_QUEUE_CAP] = input;
        m->attackLen++;
    }
}

static inline int bot_attack_pop(BotMem *m)
{
    int input = m->attackQueue[m->attackHead];
    m->attackHead = (m->attackHead + 1) % FOOTSIES_BOT_QUEUE_CAP;
    m->attackLen--;
    return input;
}

/* --- queue-pattern builders (C# Add*InputQueue / Add*Attack) ------------ */
/* GetForwardInput() = Left and GetBackwardInput() = Right in BattleAI.cs:
 * the bot is fighter2, on the right side facing left. */

static inline void bot_add_forward_input_queue(BotMem *m, int frames)
{
    for (int i = 0; i < frames; i++)
        bot_move_push(m, INPUT_LEFT);
}

static inline void bot_add_backward_input_queue(BotMem *m, int frames)
{
    for (int i = 0; i < frames; i++)
        bot_move_push(m, INPUT_RIGHT);
}

static inline void bot_add_forward_dash_input_queue(BotMem *m)
{
    bot_move_push(m, INPUT_LEFT);
    bot_move_push(m, INPUT_NONE);
    bot_move_push(m, INPUT_LEFT);
}

/* Verbatim from BattleAI.cs: this "backward" dash enqueues the forward
 * input. See fidelity notes above. */
static inline void bot_add_backward_dash_input_queue(BotMem *m)
{
    bot_move_push(m, INPUT_LEFT);
    bot_move_push(m, INPUT_NONE);
    bot_move_push(m, INPUT_LEFT);
}

static inline void bot_add_neutral_movement(BotMem *m)
{
    for (int i = 0; i < 30; i++)
        bot_move_push(m, INPUT_NONE);
}

static inline void bot_add_far_approach_1(BotMem *m)
{
    bot_add_forward_input_queue(m, 40);
    bot_add_backward_input_queue(m, 10);
    bot_add_forward_input_queue(m, 30);
    bot_add_backward_input_queue(m, 10);
}

static inline void bot_add_far_approach_2(BotMem *m)
{
    bot_add_forward_dash_input_queue(m);
    bot_add_backward_input_queue(m, 25);
    bot_add_forward_dash_input_queue(m);
    bot_add_backward_input_queue(m, 25);
}

static inline void bot_add_mid_approach_1(BotMem *m)
{
    bot_add_forward_input_queue(m, 30);
    bot_add_backward_input_queue(m, 10);
    bot_add_forward_input_queue(m, 20);
    bot_add_backward_input_queue(m, 10);
}

static inline void bot_add_mid_approach_2(BotMem *m)
{
    bot_add_forward_dash_input_queue(m);
    bot_add_backward_input_queue(m, 30);
}

static inline void bot_add_fall_back_1(BotMem *m)
{
    bot_add_backward_input_queue(m, 60);
}

static inline void bot_add_fall_back_2(BotMem *m)
{
    bot_add_backward_dash_input_queue(m);
    bot_add_backward_input_queue(m, 60);
}

static inline void bot_add_no_attack(BotMem *m)
{
    for (int i = 0; i < 30; i++)
        bot_attack_push(m, INPUT_NONE);
}

static inline void bot_add_one_hit_immediate_attack(BotMem *m)
{
    bot_attack_push(m, INPUT_ATTACK);
    for (int i = 0; i < 18; i++)
        bot_attack_push(m, INPUT_NONE);
}

static inline void bot_add_two_hit_immediate_attack(BotMem *m)
{
    bot_attack_push(m, INPUT_ATTACK);
    for (int i = 0; i < 3; i++)
        bot_attack_push(m, INPUT_NONE);
    bot_attack_push(m, INPUT_ATTACK);
    for (int i = 0; i < 18; i++)
        bot_attack_push(m, INPUT_NONE);
}

static inline void bot_add_immediate_special_attack(BotMem *m)
{
    for (int i = 0; i < 60; i++)
        bot_attack_push(m, INPUT_ATTACK);
    bot_attack_push(m, INPUT_NONE);
}

static inline void bot_add_delay_special_attack(BotMem *m)
{
    for (int i = 0; i < 120; i++)
        bot_attack_push(m, INPUT_ATTACK);
    bot_attack_push(m, INPUT_NONE);
}

/* --- selection (C# SelectMovement / SelectAttack) ------------------------ */

static inline void bot_select_movement(BotMem *m, unsigned int *rng,
    const BotFightState *fs)
{
    if (fs->distanceX > 4.0f) {
        if (rand_r(rng) % 2 == 0)
            bot_add_far_approach_1(m);
        else
            bot_add_far_approach_2(m);
    } else if (fs->distanceX > 3.0f) {
        int rand = rand_r(rng) % 7;
        if (rand <= 1)
            bot_add_mid_approach_1(m);
        else if (rand <= 3)
            bot_add_mid_approach_2(m);
        else if (rand == 4)
            bot_add_far_approach_1(m);
        else if (rand == 5)
            bot_add_far_approach_2(m);
        else
            bot_add_neutral_movement(m);
    } else if (fs->distanceX > 2.5f) {
        int rand = rand_r(rng) % 5;
        if (rand == 0)
            bot_add_mid_approach_1(m);
        else if (rand == 1)
            bot_add_mid_approach_2(m);
        else if (rand == 2)
            bot_add_fall_back_1(m);
        else if (rand == 3)
            bot_add_fall_back_2(m);
        else
            bot_add_neutral_movement(m);
    } else if (fs->distanceX > 2.0f) {
        int rand = rand_r(rng) % 4;
        if (rand == 0)
            bot_add_fall_back_1(m);
        else if (rand == 1)
            bot_add_fall_back_2(m);
        else
            bot_add_neutral_movement(m);
    } else {
        int rand = rand_r(rng) % 3;
        if (rand == 0)
            bot_add_fall_back_1(m);
        else if (rand == 1)
            bot_add_fall_back_2(m);
        else
            bot_add_neutral_movement(m);
    }
}

static inline void bot_select_attack(BotMem *m, unsigned int *rng,
    const BotFightState *fs)
{
    if (fs->isOpponentDamage || fs->isOpponentGuardBreak
            || fs->isOpponentSpecialAttack) {
        bot_add_two_hit_immediate_attack(m);
    } else if (fs->distanceX > 4.0f) {
        if (rand_r(rng) % 4 <= 3)
            bot_add_no_attack(m);
        else
            bot_add_delay_special_attack(m);
    } else if (fs->distanceX > 3.0f) {
        if (fs->isOpponentNormalAttack) {
            bot_add_two_hit_immediate_attack(m);
            return;
        }
        int rand = rand_r(rng) % 5;
        if (rand <= 1)
            bot_add_no_attack(m);
        else if (rand <= 3)
            bot_add_one_hit_immediate_attack(m);
        else
            bot_add_delay_special_attack(m);
    } else if (fs->distanceX > 2.5f) {
        int rand = rand_r(rng) % 3;
        if (rand == 0)
            bot_add_no_attack(m);
        else if (rand == 1)
            bot_add_one_hit_immediate_attack(m);
        else
            bot_add_two_hit_immediate_attack(m);
    } else if (fs->distanceX > 2.0f) {
        int rand = rand_r(rng) % 6;
        if (rand <= 1)
            bot_add_one_hit_immediate_attack(m);
        else if (rand <= 3)
            bot_add_two_hit_immediate_attack(m);
        else if (rand == 4)
            bot_add_immediate_special_attack(m);
        else
            bot_add_delay_special_attack(m);
    } else {
        int rand = rand_r(rng) % 3;
        if (rand == 0)
            bot_add_one_hit_immediate_attack(m);
        else
            bot_add_two_hit_immediate_attack(m);
    }
}

/* --- per-tick entry point (C# getNextAIInput) --------------------------- */
/* BattleCore only fetches AI input from UpdateIntroState / UpdateFightState,
 * so call this only in ROUND_INTRO / ROUND_FIGHT; the queues then persist
 * verbatim through KO/END exactly like the C# instance, which is created
 * fresh at every Intro (bot_mem_reset on the STOP -> INTRO transition). */

static inline void bot_update_fight_state(Game *g, BotMem *m)
{
    BotFightState cur = { 0 };
    Fighter *opp = &g->fighter1;
    cur.distanceX = fabsf(g->fighter2.posX - g->fighter1.posX);
    cur.isOpponentDamage = opp->currentActionID == ACT_DAMAGE;
    cur.isOpponentGuardBreak = opp->currentActionID == ACT_GUARD_BREAK;
    cur.isOpponentBlocking = opp->currentActionID == ACT_GUARD_CROUCH
        || opp->currentActionID == ACT_GUARD_STAND
        || opp->currentActionID == ACT_GUARD_M;
    cur.isOpponentNormalAttack = opp->currentActionID == ACT_N_ATTACK
        || opp->currentActionID == ACT_B_ATTACK;
    cur.isOpponentSpecialAttack = opp->currentActionID == ACT_N_SPECIAL
        || opp->currentActionID == ACT_B_SPECIAL;

    /* Verbatim from BattleAI.UpdateFightState: the C# shift loop runs
     * ASCENDING (i = 1..n-1), so every slot 1..n-1 ends up holding the
     * previous frame's state, not an n-frame history. The bot therefore
     * reacts with a 1-frame lag, and slot 5 is valid from the 2nd update. */
    for (int i = 1; i < FOOTSIES_BOT_MAX_FIGHT_STATE_RECORD; i++)
        m->fightStates[i] = m->fightStates[i - 1];
    m->fightStates[0] = cur;
    if (m->fightStateLen < FOOTSIES_BOT_MAX_FIGHT_STATE_RECORD)
        m->fightStateLen++;
}

static inline int footsies_bot_input(Game *g, BotMem *m, unsigned int *rng)
{
    bot_update_fight_state(g, m);
    /* C#: fightStates[5] is still null after the first update only (see the
     * ascending-shift note above); AI outputs nothing that tick. */
    if (m->fightStateLen < 2)
        return INPUT_NONE;

    const BotFightState *fs = &m->fightStates[FOOTSIES_BOT_FIGHT_STATE_READ_INDEX];
    int input = INPUT_NONE;
    /* Selection only fires on an empty queue and emits nothing that tick,
     * so a freshly selected pattern starts playing next frame. Same here. */
    if (m->moveLen > 0)
        input |= bot_move_pop(m);
    else
        bot_select_movement(m, rng, fs);

    if (m->attackLen > 0)
        input |= bot_attack_pop(m);
    else
        bot_select_attack(m, rng, fs);

    return input;
}

#endif /* FOOTSIES_BOTS_H */
