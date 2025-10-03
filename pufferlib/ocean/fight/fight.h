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
    int facing; // Where the figheter is facing
    int hpMax, hp;
    bool grounded;

    bool attacking;
    float startup_frame, active_frame, recovery_frame; // attack frame
    float phaseTimer;
    int phase;          // track attack
    bool hitRegistered; // prevent multi-hit per attack

    bool blocking;          // blocking state
    float plus_minus_frame; // to track so fighter is not doing anything when
                            // still on recovery
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
                                              : -(rb.x + rb.width) + ra.x;
        a->pos.x -= overlap * 0.5f;
        b->pos.x += overlap * 0.5f;
        if (a->vel.x < 0)
            a->vel.x = 0;
        if (b->vel.x > 0)
            b->vel.x = 0;
    }
}

static void StartAttack(Agent *f) {
    if ((f->phase != 0) || (f->recovery_frame > 0.0f))
        return;
    f->attacking = true;
    f->hitRegistered = false;
    f->phase = 1;
    f->phaseTimer = f->startup_frame;
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
            f->plus_minus_frame = f->recovery_frame;
            break;
        case 3:
            f->attacking = false;
            f->phase = 0;
            break;
        }
    }
}

static void CheckAttack(Agent *att, Agent *def, int dmg, int hitStun,
                        int blockStun) {
    if (!att->attacking || att->phase != 2 || att->hitRegistered)
        return; // if not attacking, or it is not attack active frame, or the
                // move alr attacked

    Rectangle hb = AttackHitbox(att);
    Rectangle defBox = FighterHitbox(def);

    if (!CheckCollisionRecs(hb, defBox)) {
        return; // whiff
    }

    att->hitRegistered = true;
    if (def->blocking && def->recovery_frame == 0) {
        def->recovery_frame = blockStun;
        def->vel.x = 0;
        def->vel.y = 0;
    } else { // got hit
        def->hp -= dmg;
        if (def->hp < 0)
            def->hp = 0; // clipping hp to 0 to 100
        def->recovery_frame = hitStun;
        def->vel.x = att->facing * 280.0f;
        def->vel.y = -220.f;
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
        env->observations[idx + 8] = agent->blocking ? 1.0f : 0.0f;
        env->observations[idx + 9] = opp->blocking ? 1.0f : 0.0f;
        env->observations[idx + 10] = env->rewards[a];
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
                             .grounded = false,
                             .attacking = false,
                             .blocking = false,
                             .phase = 0,
                             .startup_frame = 10.0f,
                             .active_frame = 1.0f,
                             .recovery_frame = 18.0f,
                             .plus_minus_frame = 0.0f,
                             .hitRegistered = false,
                             .tick_since_rewards = 0};
    env->agents[1] = (Agent){.pos = {env->width * 0.75f - 20, groundY - 80},
                             .vel = {0, 0},
                             .w = 80,
                             .h = 160,
                             .facing = -1,
                             .hpMax = 100,
                             .hp = 100,
                             .grounded = false,
                             .attacking = false,
                             .blocking = false,
                             .phase = 0,
                             .startup_frame = 10.0f,
                             .active_frame = 1.0f,
                             .recovery_frame = 18.0f,
                             .plus_minus_frame = 0.0f,
                             .hitRegistered = false,
                             .tick_since_rewards = 0};

    compute_observations(env);
}

// RL-compatible step function
void c_step(Fight *env) {
    const float groundY = env->height - 100.0f;
    const float gravity = 1500.0f;
    const float moveSpeed = 300.0f;
    const float jumpVel = -750.0f;
    const float friction = 1800.0f;
    float dt = 1.0f / 60.0f;
    // Face each other
    if (env->agents[0].pos.x < env->agents[1].pos.x) {
        env->agents[0].facing = +1;
        env->agents[1].facing = -1;
    } else {
        env->agents[0].facing = -1;
        env->agents[1].facing = +1;
    }
    for (int i = 0; i < 2; i++) {
        Agent *f = &env->agents[i];

        f->tick_since_rewards += 1;

        // Actions: [move, jump, attack]
        int move = env->actions[3 * i];
        int jump = env->actions[3 * i + 1];
        int attack = env->actions[3 * i + 2];

        bool canAct = (i == 0) ? (env->agents[0].recovery_frame == 0.0f)
                               : (env->agents[1].recovery_frame == 0.0f);
        if (canAct) {
            float dir1 = 0.0f;
            if (move == 1)
                dir1 -= 1.0f;
            if (move == 2)
                dir1 += 1.0f;
            bool pressingBack = // facing right → A is back; facing left  →
                                // D is back
                ((f->facing > 0) && (move == 1)) ||
                (f->facing < 0 && (move == 2));
            bool pressingForward = ((f->facing > 0) && (move == 2)) ||
                                   ((f->facing < 0) && (move == 1));

            if (pressingBack && !pressingForward) // back only → block
                f->blocking = true;
            if (dir1 != 0.0f) {
                f->vel.x = dir1 * moveSpeed;
                f->facing = (dir1 > 0) ? +1 : -1;
            } else {
                // friction
                if (fabsf(f->vel.x) > 0.0f) {
                    float s = (f->vel.x > 0) ? 1.0f : -1.0f;
                    f->vel.x -= s * friction * dt;
                    if (s * f->vel.x < 0)
                        f->vel.x = 0.0f;
                }
            }
            if ((jump == 1) && f->grounded) {
                f->vel.y = jumpVel;
                f->grounded = false;
            }
            if (attack == 1)
                StartAttack(f);
        }

        PlayerCollision(&env->agents[0], &env->agents[1]);

        if (f->recovery_frame > 0) { // moving
            f->recovery_frame--;
            f->vel = (Vector2){0, 0}; // kill momentum
            continue;
        }
        f->vel.y += gravity * dt;
        f->pos.y += f->vel.y * dt;
        f->pos.x += f->vel.x * dt;

        // ground collision
        if (f->pos.y + f->h >= groundY) {
            f->pos.y = groundY - f->h;
            f->vel.y = 0.0f;
            f->grounded = true;
        } else {
            f->grounded = false;
        }

        // stage bounds (x)
        if (f->pos.x < 40) {
            f->pos.x = 40;
            if (f->vel.x < 0)
                f->vel.x = 0;
        }
        if (f->pos.x + f->w > env->width - 40) {
            f->pos.x = env->height - 40 - f->w;
            if (f->vel.x > 0)
                f->vel.x = 0;
        }
    }

    CheckAttack(&env->agents[0], &env->agents[1], 12, 20, 8);
    CheckAttack(&env->agents[1], &env->agents[0], 12, 20, 8);

    for (int i = 0; i < 2; i++) {
        if (env->agents[i].hp <= 0) {
            env->terminals[i] = 1;
            env->rewards[1 - i] += 1.0f; // Example win reward
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += env->agents[1 - i].tick_since_rewards;
            env->log.episode_return += 1.0f;
            env->log.n++;
            c_reset(env);
            return;
        }
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
