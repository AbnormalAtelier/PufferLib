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
    float startup_frame;
    float active_frame;
    float inactive_frame;
    float reach;     // horizontal reach
    float thickness; // how thick the hitbox is
    float hitLevel;  // Percentage of the attack heigh to the heigh of the
                     // attacker 0.0 high 1.0 low
    int dmg;
    int hitStun;
    int blockStun;
} MoveStats;

static const MoveStats MOVE_DATA[5] = {
    // for now use Dragunov Tekken 8 frame
    {0},                                   // dummy
    {10, 2, 19, 70, 20, 0.1, 5, 28, 18},   // 1 Jab
    {12, 2, 23, 70, 20, 0.45, 13, 30, 25}, // df1 poke
    {15, 2, 31, 70, 20, 0.5, 40, 25, 29},  //  df 2 launcher
    {18, 2, 31, 70, 20, 0.8, 14, 31, 18}   // d2 low
};

enum {
    ST_IDLE,
    ST_STARTUP,
    ST_ACTIVE,
    ST_RECOVERY
}; // state for idle, startup, active, recovery

typedef struct {
    Vector2 pos;
    Vector2 vel;
    float width, height;
    int dir; // where the fighter is dir
    int hpMax, hp;
    bool grounded;
    bool normal_blocking;
    bool low_blocking;

    int attackID;
    float phaseTimer; // to track transition between idle, startup, active, and
                      // recovery
    int state;
    bool hitRegistered;   // prevent multiple hit per attack
    float inactive_frame; // attacker frame is locked through the attack data,
                          // defender frame depends on block or hit with the
                          // blockstun, hitstun so there can be plus minus
    int tick_since_rewards;
} Fighter;

typedef struct {
    Log log;
    Fighter *fighters;
    float *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    int width;
    int height;
} Fight;

void init(Fight *env) { env->fighters = calloc(2, sizeof(Fighter)); }

static Rectangle FighterHitbox(const Fighter *f) {
    float y = f->pos.y, h = f->height;
    if (f->low_blocking) {
        y += 60;
        h -= 60;
    } // crouch
    return (Rectangle){f->pos.x, y, f->width, h};
}

static Rectangle AttackHitbox(const Fighter *f) {
    if (f->attackID == 0)
        return (Rectangle){0, 0, 0, 0};

    float reach = MOVE_DATA[f->attackID].reach;
    float thickness = MOVE_DATA[f->attackID].thickness;
    float x0 = (f->dir >= 0) ? f->pos.x + f->width : f->pos.x - reach;
    float y0 = f->pos.y + f->height * MOVE_DATA[f->attackID].hitLevel;
    return (Rectangle){x0, y0, reach, thickness};
}

static void PlayerCollision(Fighter *a, Fighter *b) {
    Rectangle ra = FighterHitbox(a);
    Rectangle rb = FighterHitbox(b);

    if (CheckCollisionRecs(ra, rb)) {
        float overlap = 0.0f;      // how much it is overlaping
        if (a->pos.x < b->pos.x) { // a is left of b
            overlap = (ra.x + ra.width) - rb.x;
            a->pos.x -= overlap * 0.5f;
            b->pos.x += overlap * 0.5f;
        } else { // b is left of a
            overlap = (rb.x + rb.width) - ra.x;
            b->pos.x -= overlap * 0.5f;
            a->pos.x += overlap * 0.5f;
        }

        if (a->vel.x < 0)
            a->vel.x = 0;
        if (b->vel.x > 0)
            b->vel.x = 0;
    }
}

static void StartAttack(Fighter *f, int id) {
    if (f->state != ST_IDLE || f->inactive_frame > 0.0f)
        return;

    f->attackID = id;
    f->hitRegistered = false;
    f->state = ST_STARTUP;
    f->phaseTimer = MOVE_DATA[id].startup_frame;
}

static void FighterUpdateAttack(Fighter *f) {
    if (f->attackID == 0)
        return;

    f->phaseTimer -= 1.0f;

    if (f->phaseTimer <= 0.0f) {
        switch (f->state) {
        case ST_STARTUP: /* startup → active */
            f->state = ST_ACTIVE;
            f->phaseTimer = MOVE_DATA[f->attackID].active_frame;
            break;
        case ST_ACTIVE: /* active → recovery */
            f->state = ST_RECOVERY;
            f->phaseTimer = MOVE_DATA[f->attackID].inactive_frame;
            f->inactive_frame = MOVE_DATA[f->attackID].inactive_frame;
            break;
        case ST_RECOVERY: /* recovery → idle */
            f->attackID = 0;
            f->state = ST_IDLE;
            break;
        }
    }
}

static void CheckAttack(Fighter *att, Fighter *def) {
    if (att->attackID == 0 || att->state != ST_ACTIVE || att->hitRegistered)
        return;

    Rectangle hb = AttackHitbox(att);
    Rectangle defBox = FighterHitbox(def);

    if (!CheckCollisionRecs(hb, defBox))
        return; /* whiff */

    bool isHigh = (att->attackID == 1);
    bool isMid = (att->attackID == 2 || att->attackID == 3);
    bool isLow = (att->attackID == 4);

    att->hitRegistered = true;

    if (def->inactive_frame == 0.0f && def->low_blocking && isLow) {
        def->inactive_frame = MOVE_DATA[att->attackID].blockStun;
        def->vel = (Vector2){0, 0};
        return;
    }

    if (def->normal_blocking && def->inactive_frame == 0.0f &&
        (isHigh || isMid)) { // normal block
        def->inactive_frame = MOVE_DATA[att->attackID].blockStun;
        def->vel = (Vector2){0, 0};
        return;
    }
    def->hp -= MOVE_DATA[att->attackID].dmg;
    if (def->hp < 0)
        def->hp = 0;
    def->inactive_frame = MOVE_DATA[att->attackID].hitStun;
    def->vel.x = att->dir * 280.0f;
    def->vel.y = -220.0f;
    return;
}

// RL-compatible observation function
void compute_observations(Fight *env) {
    for (int a = 0; a < 2; a++) {
        Fighter *agent = &env->fighters[a];
        Fighter *opp = &env->fighters[1 - a];
        int idx = a * 10;
        env->observations[idx + 0] = agent->pos.x / env->width;
        env->observations[idx + 1] = agent->pos.y / env->height;
        env->observations[idx + 2] = (float)agent->hp / agent->hpMax;
        env->observations[idx + 3] = opp->pos.x / env->width;
        env->observations[idx + 4] = opp->pos.y / env->height;
        env->observations[idx + 5] = (float)opp->hp / opp->hpMax;
        env->observations[idx + 6] = env->rewards[a];
    }
}

void c_reset(Fight *env) {
    float groundY = env->height - 100.0f;
    env->fighters[0] = (Fighter){.pos = {env->width * 0.25f - 20, groundY - 80},
                                 .vel = {0, 0},
                                 .width = 80,
                                 .height = 160,
                                 .dir = +1,
                                 .hpMax = 180,
                                 .hp = 180,
                                 .grounded = true,
                                 .normal_blocking = false,
                                 .low_blocking = false,
                                 .attackID = 0,
                                 .state = ST_IDLE,
                                 .phaseTimer = 0.0f,
                                 .hitRegistered = false,
                                 .inactive_frame = 0.0f,
                                 .tick_since_rewards = 0};
    env->fighters[1] = (Fighter){.pos = {env->width * 0.75f - 20, groundY - 80},
                                 .vel = {0, 0},
                                 .width = 80,
                                 .height = 160,
                                 .dir = -1,
                                 .hpMax = 180,
                                 .hp = 180,
                                 .grounded = true,
                                 .normal_blocking = false,
                                 .low_blocking = false,
                                 .attackID = 0,
                                 .state = ST_IDLE,
                                 .phaseTimer = 0.0f,
                                 .hitRegistered = false,
                                 .inactive_frame = 0.0f,
                                 .tick_since_rewards = 0};

    compute_observations(env);
}

void c_step(Fight *env) {
    const float groundY = env->height - 100.0f;
    const float gravity = 1500.0f;
    const float moveSpeed = 300.0f;
    const float jumpVel = -750.0f;
    const float friction = 1800.0f;
    float dt = 1.0f / 60.0f;

    if (env->fighters[0].pos.x < env->fighters[1].pos.x) {
        env->fighters[0].dir = +1;
        env->fighters[1].dir = -1;
    } else {
        env->fighters[0].dir = -1;
        env->fighters[1].dir = +1;
    }

    for (int i = 0; i < 2; i++) {
        Fighter *f = &env->fighters[i];
        int move = env->actions[2 * i]; // 1,2 left right, 3 4 up down
        int attack = env->actions[2 * i + 1];

        f->tick_since_rewards += 1;

        bool canAct = (i == 0) ? (env->fighters[0].inactive_frame == 0.0f)
                               : (env->fighters[1].inactive_frame == 0.0f);
        if (canAct) {
            float dir1 = 0.0f;
            if (move == 1)
                dir1 -= 1.0f;
            if (move == 2)
                dir1 += 1.0f;

            bool pressingBack =
                ((f->dir > 0) && (move == 1)) || (f->dir < 0 && (move == 2));
            bool pressingForward =
                ((f->dir > 0) && (move == 2)) || (f->dir < 0 && (move == 1));
            bool pressingDown = (move == 3);

            if (pressingBack && !pressingForward) { // back only
                f->normal_blocking = true;
                f->low_blocking = false;
            } else if (pressingDown) {
                f->normal_blocking = false;
                f->low_blocking = true;
            } else {
                f->normal_blocking = false;
                f->low_blocking = false;
            }
            if (dir1 != 0.0f) {
                f->vel.x = dir1 * moveSpeed;
            } else {
                // friction
                if (fabsf(f->vel.x) > 0.0f) {
                    float s = (f->vel.x > 0) ? 1.0f : -1.0f;
                    f->vel.x -= s * friction * dt;
                    if (s * f->vel.x < 0)
                        f->vel.x = 0.0f;
                }
                if ((move == 4) && f->grounded) {
                    f->vel.y = jumpVel;
                    f->grounded = false;
                }
                StartAttack(f, attack);
            }
        }
        PlayerCollision(&env->fighters[0], &env->fighters[1]);

        if (f->inactive_frame > 0) { // moving
            f->inactive_frame--;
            f->vel = (Vector2){0, 0}; // kill momentum
            continue;
        }
        f->vel.y += gravity * dt;
        f->pos.y += f->vel.y * dt;
        f->pos.x += f->vel.x * dt;

        // ground collision
        if (f->pos.y + f->height >= groundY) {
            f->pos.y = groundY - f->height;
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
        if (f->pos.x + f->width > env->width - 40) {
            f->pos.x = env->height - 40 - f->width;
            if (f->vel.x > 0)
                f->vel.x = 0;
        }
    }
    CheckAttack(&env->fighters[0], &env->fighters[1]);
    CheckAttack(&env->fighters[1], &env->fighters[0]);

    for (int i = 0; i < 2; i++) {
        if (env->fighters[i].hp <= 0) {
            env->terminals[i] = 1;
            env->rewards[1 - i] += 1.0f; // Example win reward
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += env->fighters[1 - i].tick_since_rewards;
            env->log.episode_return += 1.0f;
            env->log.n++;
            c_reset(env);
            return;
        }
        FighterUpdateAttack(&env->fighters[i]);
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
    DrawRectangleRec(FighterHitbox(&env->fighters[0]),
                     (Color){80, 180, 255, 255});
    DrawRectangleRec(FighterHitbox(&env->fighters[1]),
                     (Color){255, 120, 120, 255});
    // Draw Hitbox
    if (env->fighters[0].attackID != 0)
        DrawRectangleLinesEx(AttackHitbox(&env->fighters[0]), 2,
                             (Color){180, 220, 255, 255});
    if (env->fighters[1].attackID != 0)
        DrawRectangleLinesEx(AttackHitbox(&env->fighters[1]), 2,
                             (Color){255, 180, 180, 255});
    // Health bars
    const float barW = 360.0f, barH = 18.0f, margin = 20.0f;
    DrawText("P1", margin, margin - 4, 16, RAYWHITE);
    DrawRectangle(margin, margin + 16, barW, barH, (Color){60, 60, 70, 255});
    DrawRectangle(
        margin, margin + 16,
        (int)(barW * (float)env->fighters[0].hp / env->fighters[0].hpMax), barH,
        (Color){80, 180, 255, 255});
    DrawText("P2", env->width - margin - 24, margin - 4, 16, RAYWHITE);
    DrawRectangle(env->width - margin - barW, margin + 16, barW, barH,
                  (Color){60, 60, 70, 255});
    DrawRectangle(
        env->width - margin -
            (int)(barW * (float)env->fighters[1].hp / env->fighters[1].hpMax),
        margin + 16,
        (int)(barW * (float)env->fighters[1].hp / env->fighters[1].hpMax), barH,
        (Color){255, 120, 120, 255});
    EndDrawing();
}

void c_close(Fight *env) {
    free(env->fighters);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
