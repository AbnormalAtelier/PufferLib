#include "raylib.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Vector2 pos;
    Vector2 vel;
    float w, h;
    int facing;
    int hpMax, hp;
    bool grounded;
    bool attacking;
    float startup_frame, active_frame, recovery_frame;
    float phaseTimer;
    int phase;
    bool hitRegistered;
    bool blocking;
    int tick_since_rewards;
} Agent;

typedef struct {
    Log log;
    Agent *agents;
    float *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    int width;
    int height;
} Fight;

void init(Fight *env) { env->agents = calloc(2, sizeof(Agent)); }

// Helper functions
static Rectangle FighterHitbox(const Agent *f) {
    return (Rectangle){f->pos.x, f->pos.y, f->w, f->h};
}

static Rectangle AttackHitbox(const Agent *f) {
    const float range = 80.0f;
    const float height = f->h * 0.25f;
    float ax = (f->facing >= 0) ? (f->pos.x + f->w) : (f->pos.x - range);
    float ay = f->pos.y + f->h * 0.25f;
    return (Rectangle){ax, ay, range, height};
}

static void PlayerCollision(Agent *a, Agent *b) {
    Rectangle ra = FighterHitbox(a);
    Rectangle rb = FighterHitbox(b);
    if (CheckCollisionRecs(ra, rb)) {
        float overlap = (a->pos.x < b->pos.x) ? (ra.x + ra.width) - rb.x
                                              : (rb.x + rb.width) - ra.x;
        a->pos.x -= overlap * 0.5f;
        b->pos.x += overlap * 0.5f;
        if (a->vel.x < 0)
            a->vel.x = 0;
        if (b->vel.x > 0)
            b->vel.x = 0;
    }
}

static void StartAttack(Agent *f) {
    if (!f->attacking) {
        f->attacking = true;
        f->hitRegistered = false;
        f->phase = 1;
        f->phaseTimer = f->startup_frame;
    }
}

static void FighterUpdateAttack(Agent *f) {
    if (!f->attacking)
        return;
    f->phaseTimer -= 1;
    if (f->phaseTimer <= 0) {
        switch (f->phase) {
        case 1:
            f->phase = 2;
            f->phaseTimer = f->active_frame;
            break;
        case 2:
            f->phase = 3;
            f->phaseTimer = f->recovery_frame;
            break;
        case 3:
            f->attacking = false;
            f->phase = 0;
            break;
        }
    }
}

// RL-compatible observation function
void compute_observations(Fight *env) {
    for (int a = 0; a < 2; a++) {
        Agent *agent = &env->agents[a];
        Agent *opp = &env->agents[1 - a];
        int idx = a * 10;
        env->observations[idx + 0] = agent->pos.x / env->width;
        env->observations[idx + 1] = agent->pos.y / env->height;
        env->observations[idx + 2] = (float)agent->hp / agent->hpMax;
        env->observations[idx + 3] = opp->pos.x / env->width;
        env->observations[idx + 4] = opp->pos.y / env->height;
        env->observations[idx + 5] = (float)opp->hp / opp->hpMax;
        env->observations[idx + 6] = agent->attacking ? 1.0f : 0.0f;
        env->observations[idx + 7] = opp->attacking ? 1.0f : 0.0f;
        env->observations[idx + 8] = env->rewards[a];
    }
}

// RL-compatible reset function
void c_reset(Fight *env) {
    float groundY = env->height - 100.0f;
    env->agents[0] = (Agent){.pos = {env->width * 0.25f - 20, groundY - 80},
                             .vel = {0, 0},
                             .w = 80,
                             .h = 160,
                             .facing = +1,
                             .hpMax = 100,
                             .hp = 100,
                             .startup_frame = 10.0f,
                             .active_frame = 1.0f,
                             .recovery_frame = 8.0f};
    env->agents[1] = (Agent){.pos = {env->width * 0.75f - 20, groundY - 80},
                             .vel = {0, 0},
                             .w = 80,
                             .h = 160,
                             .facing = -1,
                             .hpMax = 100,
                             .hp = 100,
                             .startup_frame = 10.0f,
                             .active_frame = 1.0f,
                             .recovery_frame = 8.0f};
    env->rewards[0] = env->rewards[1] = 0;
    env->terminals[0] = env->terminals[1] = 0;
    env->log = (Log){0};
    compute_observations(env);
}

// RL-compatible step function
void c_step(Fight *env) {
    const float groundY = env->height - 100.0f;
    const float gravity = 1500.0f;
    const float moveSpeed = 300.0f;
    const float jumpVel = -750.0f;
    const float friction = 1800.0f;
    const int dmg = 12;
    const float kbX = 280.0f;
    const float kbY = -220.0f;
    float dt = 1.0f / 60.0f;

    for (int i = 0; i < 2; i++) {
        env->rewards[i] = 0;
        Agent *f = &env->agents[i];
        f->tick_since_rewards += 1;

        // Actions: [move, jump, attack]
        int move = env->actions[3 * i];
        int jump = env->actions[3 * i + 1];
        int attack = env->actions[3 * i + 2];

        float dir = 0.0f;
        if (move == 1)
            dir = -1.0f;
        if (move == 2)
            dir = +1.0f;
        if (dir != 0.0f) {
            f->vel.x = dir * moveSpeed;
            f->facing = (dir > 0) ? +1 : -1;
        } else {
            if (fabsf(f->vel.x) > 0.0f) {
                float s = (f->vel.x > 0) ? 1.0f : -1.0f;
                f->vel.x -= s * friction * dt;
                if (s * f->vel.x < 0)
                    f->vel.x = 0.0f;
            }
        }
        if (jump == 1 && f->grounded) {
            f->vel.y = jumpVel;
            f->grounded = false;
        }
        if (attack == 1)
            StartAttack(f);

        f->vel.y += gravity * dt;
        f->pos.x += f->vel.x * dt;
        f->pos.y += f->vel.y * dt;
    }

    PlayerCollision(&env->agents[0], &env->agents[1]);

    for (int i = 0; i < 2; i++) {
        Agent *f = &env->agents[i];
        if (f->pos.y + f->h >= groundY) {
            f->pos.y = groundY - f->h;
            f->vel.y = 0.0f;
            f->grounded = true;
        } else {
            f->grounded = false;
        }
        if (f->pos.x < 40) {
            f->pos.x = 40;
            if (f->vel.x < 0)
                f->vel.x = 0;
        }
        if (f->pos.x + f->w > env->width - 40) {
            f->pos.x = env->width - 40 - f->w;
            if (f->vel.x > 0)
                f->vel.x = 0;
        }
    }

    // Face each other
    if (env->agents[0].pos.x < env->agents[1].pos.x) {
        env->agents[0].facing = +1;
        env->agents[1].facing = -1;
    } else {
        env->agents[0].facing = -1;
        env->agents[1].facing = +1;
    }

    // Attack logic
    for (int i = 0; i < 2; i++) {
        Agent *attacker = &env->agents[i];
        Agent *defender = &env->agents[1 - i];
        if (attacker->attacking && attacker->phase == 2 &&
            !attacker->hitRegistered) {
            Rectangle hb = AttackHitbox(attacker);
            if (CheckCollisionRecs(hb, FighterHitbox(defender))) {
                defender->hp -= dmg;
                if (defender->hp < 0)
                    defender->hp = 0;
                defender->vel.x = attacker->facing * kbX;
                defender->vel.y = kbY;
                attacker->hitRegistered = true;
                env->rewards[i] += 1.0f;
                env->log.perf += 1.0f;
                env->log.score += 1.0f;
                env->log.episode_return += 1.0f;
                env->log.n++;
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        if (env->agents[i].hp <= 0)
            env->terminals[i] = 1;
        FighterUpdateAttack(&env->agents[i]);
    }

    env->log.episode_length += 1.0f;
    compute_observations(env);
}

void c_render(Fight *env) {
    // Initialize window on first call
    static bool window_initialized = false;
    if (!window_initialized) {
        InitWindow(env->width, env->height, "PufferLib Fight");
        SetTargetFPS(60);
        window_initialized = true;
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }
    float groundY = env->height - 100.0f;
    BeginDrawing();
    ClearBackground((Color){24, 26, 33, 255});
    DrawRectangle(0, (int)groundY, env->width, env->height - (int)groundY,
                  (Color){40, 45, 55, 255});
    // Draw Fighters
    DrawRectangleRec(FighterHitbox(&env->agents[0]),
                     (Color){80, 180, 255, 255});
    DrawRectangleRec(FighterHitbox(&env->agents[1]),
                     (Color){255, 120, 120, 255});
    // Draw Hitbox
    if (env->agents[0].attacking)
        DrawRectangleLinesEx(AttackHitbox(&env->agents[0]), 2,
                             (Color){180, 220, 255, 255});
    if (env->agents[1].attacking)
        DrawRectangleLinesEx(AttackHitbox(&env->agents[1]), 2,
                             (Color){255, 180, 180, 255});
    // Health bars
    const float barW = 360.0f, barH = 18.0f, margin = 20.0f;
    DrawText("P1", margin, margin - 4, 16, RAYWHITE);
    DrawRectangle(margin, margin + 16, barW, barH, (Color){60, 60, 70, 255});
    DrawRectangle(margin, margin + 16,
                  (int)(barW * (float)env->agents[0].hp / env->agents[0].hpMax),
                  barH, (Color){80, 180, 255, 255});
    DrawText("P2", env->width - margin - 24, margin - 4, 16, RAYWHITE);
    DrawRectangle(env->width - margin - barW, margin + 16, barW, barH,
                  (Color){60, 60, 70, 255});
    DrawRectangle(
        env->width - margin -
            (int)(barW * (float)env->agents[1].hp / env->agents[1].hpMax),
        margin + 16,
        (int)(barW * (float)env->agents[1].hp / env->agents[1].hpMax), barH,
        (Color){255, 120, 120, 255});
    EndDrawing();
}

void c_close(Fight *env) {
    free(env->agents);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
