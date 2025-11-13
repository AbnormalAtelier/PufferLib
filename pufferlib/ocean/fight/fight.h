#include "raylib.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STEPS 200

typedef struct {
    float perf;  // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float
        episode_return; // Recommended metric: sum of agent rewards over episode
    float
        episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field
} Log;

typedef struct {
    float startup_frame;  // Startup frames
    float active_frame;   // Active frames
    float recovery_frame; // Recovery frames
    float reach;          // Horizontal reach
    float thickness;      // Hitbox thickness
    float hitLevel;       // Height percentage (0.0=high, 1.0=low)
    int dmg;              // Damage dealt
    int hitStun;          // Hit stun frames
    int blockStun;        // Block stun frames
} MoveStats;

// Attack move definitions - based on Tekken frame data
const unsigned char MOVE_JAB = 1;          // Quick punch
const unsigned char MOVE_DF1_POKE = 2;     // Mid poke
const unsigned char MOVE_DF2_LAUNCHER = 3; // Launcher
const unsigned char MOVE_D2_LOW = 4;       // Low attack

// Move data table - static frame data for all attacks
static const MoveStats MOVE_DATA[5] = {
    {0},                                   // dummy (index 0)
    {10, 2, 19, 70, 20, 0.1, 5, 28, 18},   // 1 Jab
    {12, 2, 23, 70, 20, 0.45, 13, 30, 25}, // df1 poke
    {15, 2, 31, 70, 20, 0.5, 40, 25, 29},  // df2 launcher
    {18, 2, 31, 70, 20, 0.8, 14, 31, 18}   // d2 low
};

// Fighter states
const unsigned char ST_IDLE = 0;
const unsigned char ST_STARTUP = 1;
const unsigned char ST_ACTIVE = 2;
const unsigned char ST_RECOVERY = 3;

typedef struct {
    Vector2 pos;          // Position
    Vector2 vel;          // Velocity
    float width, height;  // Hitbox dimensions
    int dir;              // Facing direction (-1 or 1)
    int hpMax, hp;        // Health
    bool grounded;        // On ground flag
    bool normal_blocking; // High/mid block
    bool low_blocking;    // Low block

    int attackID;         // Current attack (0=none)
    float phaseTimer;     // Attack phase timer
    int state;            // Fighter state
    bool hitRegistered;   // Prevent multiple hits
    float inactive_frame; // Stun/blockstun timer
    int tick_since_rewards;
} Fighter;

// Physics constants
const float GROUND_Y = 720.0f;      // Ground level
const float GRAVITY = 2000.0f;      // Gravity acceleration
const float MOVE_SPEED = 300.0f;    // Horizontal movement speed
const float JUMP_VELOCITY = 800.0f; // Initial jump velocity
const float FRICTION = 800.0f;      // Ground friction

typedef struct {
    Log log;                  // Required logging field
    float *observations;      // Observation buffer
    int *actions;             // Action buffer (discrete)
    float *rewards;           // Reward buffer
    unsigned char *terminals; // Terminal flags

    Fighter *fighters; // Array of fighters (2 players)
    int screen_width;
    int screen_height;

    // Game state
    bool round_over;
    int current_step;
} Fight;

void init(Fight *env) { env->fighters = calloc(2, sizeof(Fighter)); }

void c_close(Fight *env) {
    free(env->fighters);
    if (IsWindowReady()) {
        CloseWindow();
    }
}

Rectangle FighterHitbox(const Fighter *f) {
    float y = f->pos.y, h = f->height;
    if (f->low_blocking) {
        y += 60;
        h -= 60;
    } // crouch
    return (Rectangle){f->pos.x, y, f->width, h};
}

Rectangle AttackHitbox(const Fighter *f) {
    if (f->attackID == 0)
        return (Rectangle){0, 0, 0, 0};

    float reach = MOVE_DATA[f->attackID].reach;
    float thickness = MOVE_DATA[f->attackID].thickness;
    float x0 = (f->dir >= 0) ? f->pos.x + f->width : f->pos.x - reach;
    float y0 = f->pos.y + f->height * MOVE_DATA[f->attackID].hitLevel;
    return (Rectangle){x0, y0, reach, thickness};
}

void PlayerCollision(Fighter *a, Fighter *b) {
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

void StartAttack(Fighter *f, int id) {
    if (f->state != ST_IDLE || f->inactive_frame > 0.0f)
        return;

    f->attackID = id;
    f->hitRegistered = false;
    f->state = ST_STARTUP;
    f->phaseTimer = MOVE_DATA[id].startup_frame;
}

void FighterUpdateAttack(Fighter *f) {
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
            f->phaseTimer = MOVE_DATA[f->attackID].recovery_frame;
            f->inactive_frame = MOVE_DATA[f->attackID].recovery_frame;
            break;
        case ST_RECOVERY: /* recovery → idle */
            f->attackID = 0;
            f->state = ST_IDLE;
            break;
        }
    }
}

void CheckAttack(Fighter *att, Fighter *def, Fight *env) {
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
    if (def->hp < 0) {
        def->hp = 0;
    }
    def->inactive_frame = MOVE_DATA[att->attackID].hitStun;
    def->vel.x = att->dir * 280.0f;
    def->vel.y = -220.0f;

    int attacker_idx = att - env->fighters; // Pointer arithmetic
    int defender_idx = def - env->fighters;
    env->rewards[attacker_idx] += 0.1f;  // Reward for landing hit
    env->rewards[defender_idx] -= 0.05f; // Penalty for getting hit

    return;
}

void compute_observations(Fight *env) {
    for (int a = 0; a < 2; a++) {
        Fighter *agent = &env->fighters[a];
        Fighter *opp = &env->fighters[1 - a];
        int idx = a * 10;
        env->observations[idx + 0] = agent->pos.x / env->screen_width;
        env->observations[idx + 1] = agent->pos.y / env->screen_height;
        env->observations[idx + 2] = agent->vel.x / MOVE_SPEED;
        env->observations[idx + 3] = agent->vel.y / 1000.0f;
        env->observations[idx + 4] = (float)agent->hp / agent->hpMax;
        env->observations[idx + 5] = opp->pos.x / env->screen_width;
        env->observations[idx + 6] = opp->pos.y / env->screen_height;
        env->observations[idx + 7] = (float)opp->hp / opp->hpMax;
        env->observations[idx + 8] = opp->vel.x / MOVE_SPEED;
        env->observations[idx + 9] = opp->vel.y / 1000.0f;
    }
}
void add_log(Fight *env) {
    float p1_health = (float)env->fighters[0].hp / env->fighters[0].hpMax;
    float p2_health = (float)env->fighters[1].hp / env->fighters[1].hpMax;

    env->log.perf += (p1_health > p2_health) ? 1.0f : 0.0f;
    env->log.score += p1_health - p2_health;
    env->log.n++;
}

void c_reset(Fight *env) {
    float groundY = env->screen_height - 100.0f;
    env->current_step = 0;
    env->fighters[0] =
        (Fighter){.pos = {env->screen_width * 0.25f - 20, groundY - 80},
                  .vel = {0, 0},
                  .width = 80,
                  .height = 160,
                  .dir = +1,
                  .hpMax = 80,
                  .hp = 80, // current hp is back to max
                  .grounded = true,
                  .normal_blocking = false,
                  .low_blocking = false,
                  .attackID = 0,
                  .state = ST_IDLE,
                  .phaseTimer = 0.0f,
                  .hitRegistered = false,
                  .inactive_frame = 0.0f,
                  .tick_since_rewards = 0};
    env->fighters[1] =
        (Fighter){.pos = {env->screen_width * 0.75f - 20, groundY - 80},
                  .vel = {0, 0},
                  .width = 80,
                  .height = 160,
                  .dir = -1,
                  .hpMax = 80,
                  .hp = 80,
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

void end_episode(Fight *env) {
    env->round_over = true;
    env->terminals[0] = 1;
    env->terminals[1] = 1;

    // Set rewards based on outcome
    if (env->fighters[0].hp > env->fighters[1].hp) {
        env->rewards[0] = 1.0f;
        env->rewards[1] = -1.0f;
    } else if (env->fighters[1].hp > env->fighters[0].hp) {
        env->rewards[0] = -1.0f;
        env->rewards[1] = 1.0f;
    } else {
        env->rewards[0] = 0.0f;
        env->rewards[1] = 0.0f;
    }

    add_log(env);
    c_reset(env);
}

void c_step(Fight *env) {
    env->current_step += 1;
    if (env->current_step >= MAX_STEPS) {
        end_episode(env);
    }

    float dt = 1.0f / 60.0f;

    if (env->fighters[0].pos.x < env->fighters[1].pos.x) { // facing direction
        env->fighters[0].dir = +1;
        env->fighters[1].dir = -1;
    } else {
        env->fighters[0].dir = -1;
        env->fighters[1].dir = +1;
    }
    for (int i = 0; i < 2; i++) {
        Fighter *f = &env->fighters[i];
        int action = env->actions[i];

        // Skip if stunned
        if (f->inactive_frame > 0) {
            f->inactive_frame -= 1.0f;
            f->vel = (Vector2){0, 0};
            continue;
        }
        // Reset blocking states
        f->normal_blocking = false;
        f->low_blocking = false;

        switch (action) { // action and movement
        case 1:           // Move left
            f->vel.x = -MOVE_SPEED;
            break;
        case 2: // Move right
            f->vel.x = MOVE_SPEED;
            break;
        case 3: // Jump
            if (f->grounded) {
                f->vel.y = JUMP_VELOCITY;
                f->grounded = false;
            }
            break;
        case 4: // Crouch/block
            f->low_blocking = true;
            break;
        case 5: // Jab
            StartAttack(f, MOVE_JAB);
            break;
        case 6: // DF1 poke
            StartAttack(f, MOVE_DF1_POKE);
            break;
        case 7: // DF2 launcher
            StartAttack(f, MOVE_DF2_LAUNCHER);
            break;
        case 8: // D2 low
            StartAttack(f, MOVE_D2_LOW);
            break;
        default: // No-op - apply friction
            if (fabsf(f->vel.x) > 0.0f) {
                float s = (f->vel.x > 0) ? 1.0f : -1.0f;
                f->vel.x -= s * FRICTION * dt;
                if (s * f->vel.x < 0) {
                    f->vel.x = 0.0f;
                }
            }
            break;
        }
    }
    for (int i = 0; i < 2; i++) {
        Fighter *f = &env->fighters[i];

        // Apply gravity
        f->vel.y += GRAVITY * dt;
        f->pos.y += f->vel.y * dt;
        f->pos.x += f->vel.x * dt;

        // Ground collision
        if (f->pos.y + f->height >= GROUND_Y) {
            f->pos.y = GROUND_Y - f->height;
            f->vel.y = 0.0f;
            f->grounded = true;
        } else {
            f->grounded = false;
        }

        // Stage bounds (x)
        if (f->pos.x < 40) {
            f->pos.x = 40;
            if (f->vel.x < 0)
                f->vel.x = 0;
        }
        if (f->pos.x + f->width > env->screen_width - 40) {
            f->pos.x = env->screen_width - 40 - f->width;
            if (f->vel.x > 0)
                f->vel.x = 0;
        }
    }
    // Player collision
    PlayerCollision(&env->fighters[0], &env->fighters[1]);

    // Update attacks
    FighterUpdateAttack(&env->fighters[0]);
    FighterUpdateAttack(&env->fighters[1]);

    // Check attacks
    CheckAttack(&env->fighters[0], &env->fighters[1], env);
    CheckAttack(&env->fighters[1], &env->fighters[0], env);

    // Check for round end
    if (env->fighters[0].hp <= 0 || env->fighters[1].hp <= 0) {
        end_episode(env);
    } else {
        env->rewards[0] = -0.01f;
        env->rewards[1] = -0.01f;
    }
    compute_observations(env);
}

void c_render(Fight *env) {
    if (!IsWindowReady()) {
        InitWindow(env->screen_width, env->screen_height, "PufferLib Fight");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){24, 26, 33, 255});

    // Draw ground
    DrawRectangle(0, (int)GROUND_Y, env->screen_width,
                  env->screen_height - (int)GROUND_Y, (Color){40, 45, 55, 255});

    // Draw fighters
    Fighter *p1 = &env->fighters[0];
    Fighter *p2 = &env->fighters[1];

    Color p1_color = p1->low_blocking ? (Color){150, 150, 150, 255}
                                      : (Color){80, 180, 255, 255};
    Color p2_color = p2->low_blocking ? (Color){150, 150, 150, 255}
                                      : (Color){255, 120, 120, 255};

    DrawRectangleRec(FighterHitbox(p1), p1_color);
    DrawRectangleRec(FighterHitbox(p2), p2_color);

    // Draw attack boxes when active
    if (p1->attackID != 0 && p1->state == ST_ACTIVE)
        DrawRectangleLinesEx(AttackHitbox(p1), 2, (Color){180, 220, 255, 255});
    if (p2->attackID != 0 && p2->state == ST_ACTIVE)
        DrawRectangleLinesEx(AttackHitbox(p2), 2, (Color){255, 180, 180, 255});

    // Draw health bars
    const float barW = 360.0f, barH = 18.0f, margin = 20.0f;
    DrawText("P1", margin, margin - 4, 16, RAYWHITE);
    DrawRectangle(margin, margin + 16, barW, barH, (Color){60, 60, 70, 255});
    DrawRectangle(margin, margin + 16, (int)(barW * (float)p1->hp / p1->hpMax),
                  barH, (Color){80, 180, 255, 255});

    DrawText("P2", env->screen_width - margin - 24, margin - 4, 16, RAYWHITE);
    DrawRectangle(env->screen_width - margin - barW, margin + 16, barW, barH,
                  (Color){60, 60, 70, 255});
    DrawRectangle(env->screen_width - margin -
                      (int)(barW * (float)p2->hp / p2->hpMax),
                  margin + 16, (int)(barW * (float)p2->hp / p2->hpMax), barH,
                  (Color){255, 120, 120, 255});

    EndDrawing();
}
