// Footsies: 2-player battle (port of hifight/Footsies, Unity/C#).
// Faithful port of Assets/Script/Fighter.cs, BattleCore.cs, InputData.cs,
// BattleAI.cs (scripted opponent, in bots.h) + Assets/Fighter/F00 asset
// files (handwritten transcription). Ported once
// already, twice actually: standalone at Footsies_raylib/src/*.{h,c}, then
// as a PufferLib 4.0 ocean env at Footsies_raylib/puffer/footsies/footsies.h
// (ocean/target's contract). This file re-ports that same validated logic
// onto 5.0's ocean/minimal contract (Agent-based per-agent pointers, puf_
// function names, tag/boundary_reached). Keep in sync with the Unity assets
// if F00's data ever changes - same as the other two ports.
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
typedef float obs_t;
#define PUF_HAS_BOT_POLICY
#include "pufferenv.h"

#define AGENTS 2
#define ACT_SIZES {3, 2}   // per agent: move (none/left/right), attack (none/attack)
#define OBS_SIZE 21        // per agent: 10 self + 10 opp + 1 distance
#define NUM_ATNS 2
#define PUF_STEPS_PER_SEC 50 // FIXED_DT's 50 Hz; also puf_render's target FPS

typedef Env Footsies;

/* ============================================================
 * data - frame-data tables, mirrors the earlier 4.0 port
 * All frame counts are 50 Hz fixed-update frames.
 * Fight-space rects: x = center, y = bottom, w/h = size.
 * ============================================================ */

#define FIXED_DT 0.02f

#define INPUT_NONE   0
#define INPUT_LEFT   (1 << 0)
#define INPUT_RIGHT  (1 << 1)
#define INPUT_ATTACK (1 << 2)

enum {
    ACT_STAND = 0,
    ACT_FORWARD = 1,
    ACT_BACKWARD = 2,
    ACT_DASH_FORWARD = 10,
    ACT_DASH_BACKWARD = 11,
    ACT_N_ATTACK = 100,
    ACT_B_ATTACK = 105,
    ACT_N_SPECIAL = 110,
    ACT_B_SPECIAL = 115,
    ACT_DAMAGE = 200,
    ACT_GUARD_M = 301,
    ACT_GUARD_STAND = 305,
    ACT_GUARD_CROUCH = 306,
    ACT_GUARD_BREAK = 310,
    ACT_GUARD_PROXIMITY = 350,
    ACT_DEAD = 500,
    ACT_WIN = 510,
};

enum {
    ATYPE_MOVEMENT = 0,
    ATYPE_ATTACK = 1,
    ATYPE_DAMAGE = 2,
    ATYPE_GUARD = 3,
};

enum {
    DMG_NONE = 0,
    DMG_DAMAGE = 1,
    DMG_GUARD = 2,
    DMG_GUARD_BREAK = 3,
    DMG_COUNTER = 4,
};

enum {
    SND_NONE = -1,
    SND_ATTACK = 0,
    SND_ATTACK2,
    SND_N_SPECIAL,
    SND_B_SPECIAL,
    SND_NORMAL_HIT,
    SND_GUARD,
    SND_GUARD_BREAK,
    SND_SPECIAL_HIT,
    SND_DASH_FORWARD,
    SND_DASH_BACKWARD,
    SND_COUNT
};

typedef struct { int start, end; } FrameRange;
typedef struct { FrameRange fr; int motionID; } MotionFrameData;
typedef struct { FrameRange fr; float x, y, w, h; int attackID; bool proximity; } HitboxData;
typedef struct { FrameRange fr; float x, y, w, h; bool useBaseRect; } HurtboxData;
typedef struct { FrameRange fr; float x, y, w, h; bool useBaseRect; } PushboxData;
typedef struct { FrameRange fr; float velX; } MovementData;
typedef struct { FrameRange fr; bool buffer, execute; int actionID; } CancelData;

typedef struct {
    int actionID;
    const char *name;
    int type; /* ATYPE_* */
    int frameCount;
    bool isLoop;
    int loopFromFrame;
    bool alwaysCancelable;
    int sound; /* SND_* */
    const MotionFrameData *motions;  int motionCount;
    const HitboxData      *hitboxes; int hitboxCount;
    const HurtboxData     *hurtboxes; int hurtboxCount;
    const PushboxData     *pushboxes; int pushboxCount;
    const MovementData    *movements; int movementCount;
    const CancelData      *cancels;  int cancelCount;
} ActionData;

typedef struct {
    int attackID;
    const char *name;
    int damageActionID;
    int guardActionID;
    int numberOfHit;
    int vitalHealthDamage;
    int guardHealthDamage;
    int hitStunFrame;
    int guardStunFrame;
    int guardBreakStunFrame;
} AttackData;

typedef struct { float x, y, w, h; } BoxData;

#define FIGHTER_START_GUARD_HEALTH   3
#define FIGHTER_FORWARD_SPEED        2.2f
#define FIGHTER_BACKWARD_SPEED       1.8f
#define FIGHTER_DASH_ALLOW_FRAME     9
#define FIGHTER_SPECIAL_HOLD_FRAME   60
#define FIGHTER_CAN_CANCEL_ON_WHIFF  false
#define BASE_HURTBOX  ((BoxData){ 0, 0, 1.5f, 1.2f })
#define BASE_PUSHBOX  ((BoxData){ 0, 0, 1.4f, 1.0f })

typedef struct {
    int motionID;
    int texIndex;      /* index into g_sheetFiles */
    float srcX, srcY;  /* top-left in texture pixels, Unity bottom-up y */
} MotionSprite;

#define SHEET_COUNT 19
/* the original data.h defines this as 81, but the table below only ever
 * populates 77 designated entries - the trailing 4 silently zero-init to a
 * duplicate of index 0, which GetMotionSprite's linear search never
 * reaches (harmless there, but this file uses the true count). */
#define MOTION_SPRITE_COUNT 77

static const char *g_sheetFiles[SHEET_COUNT] = {
    "resources/footsies/Idle_0.png",
    "resources/footsies/Idle_1.png",
    "resources/footsies/Idle_2.png",
    "resources/footsies/Idle_3.png",
    "resources/footsies/Idle_4.png",
    "resources/footsies/F00_Forward.png",
    "resources/footsies/F00_Backward.png",
    "resources/footsies/F00_ForwardDash.png",
    "resources/footsies/F00_BackwardDash.png",
    "resources/footsies/F00_Attack_0.png",
    "resources/footsies/F00_Attack_1.png",
    "resources/footsies/F00_Attack_2.png",
    "resources/footsies/F00_Attack_3.png",
    "resources/footsies/F00_Damage.png",
    "resources/footsies/F00_Dead.png",
    "resources/footsies/F00_StandGuard.png",
    "resources/footsies/F00_CrouchGuard.png",
    "resources/footsies/F00_GuardBreak.png",
    "resources/footsies/F00_Win.png",
};

static const int g_sheetHeights[SHEET_COUNT] = {
    50, 50, 50, 50, 50,
    150, 150, 150, 100,
    150, 150, 150, 200,
    100, 150, 100, 100, 100,
    150,
};

enum {
    SH_IDLE0, SH_IDLE1, SH_IDLE2, SH_IDLE3, SH_IDLE4,
    SH_FORWARD, SH_BACKWARD, SH_FORWARD_DASH, SH_BACKWARD_DASH,
    SH_ATTACK0, SH_ATTACK1, SH_ATTACK2, SH_ATTACK3,
    SH_DAMAGE, SH_DEAD, SH_STAND_GUARD, SH_CROUCH_GUARD, SH_GUARD_BREAK,
    SH_WIN
};

/* { motionID, sheet, cellX, unityY } */
static const MotionSprite g_motionSprites[MOTION_SPRITE_COUNT] = {
    { 0, SH_IDLE0, 0, 0 },
    { 1, SH_IDLE1, 0, 0 },
    { 2, SH_IDLE2, 0, 0 },
    { 3, SH_IDLE3, 0, 0 },
    { 4, SH_IDLE4, 0, 0 },
    { 10, SH_FORWARD, 0, 100 }, { 11, SH_FORWARD, 60, 100 },
    { 12, SH_FORWARD, 0, 50 },  { 13, SH_FORWARD, 60, 50 },
    { 14, SH_FORWARD, 0, 0 },   { 15, SH_FORWARD, 60, 0 },
    { 20, SH_BACKWARD, 0, 100 }, { 21, SH_BACKWARD, 60, 100 },
    { 22, SH_BACKWARD, 0, 50 },  { 23, SH_BACKWARD, 60, 50 },
    { 24, SH_BACKWARD, 0, 0 },   { 25, SH_BACKWARD, 60, 0 },
    { 30, SH_FORWARD_DASH, 0, 100 }, { 31, SH_FORWARD_DASH, 60, 100 },
    { 32, SH_FORWARD_DASH, 0, 50 },  { 33, SH_FORWARD_DASH, 60, 50 },
    { 34, SH_FORWARD_DASH, 0, 0 },
    { 40, SH_BACKWARD_DASH, 0, 50 }, { 41, SH_BACKWARD_DASH, 60, 50 },
    { 42, SH_BACKWARD_DASH, 0, 0 },  { 43, SH_BACKWARD_DASH, 60, 0 },
    { 50, SH_ATTACK0, 0, 100 }, { 51, SH_ATTACK0, 60, 100 },
    { 52, SH_ATTACK0, 0, 50 },  { 53, SH_ATTACK0, 60, 50 },
    { 54, SH_ATTACK0, 0, 0 },
    { 60, SH_ATTACK1, 0, 100 }, { 61, SH_ATTACK1, 60, 100 },
    { 62, SH_ATTACK1, 120, 100 }, { 63, SH_ATTACK1, 0, 50 },
    { 64, SH_ATTACK1, 60, 50 }, { 65, SH_ATTACK1, 120, 50 },
    { 66, SH_ATTACK1, 0, 0 },   { 67, SH_ATTACK1, 60, 0 },
    { 70, SH_DAMAGE, 0, 50 }, { 71, SH_DAMAGE, 60, 50 },
    { 72, SH_DAMAGE, 0, 0 },  { 73, SH_DAMAGE, 60, 0 },
    { 80, SH_DEAD, 0, 100 }, { 81, SH_DEAD, 60, 100 },
    { 82, SH_DEAD, 120, 100 }, { 83, SH_DEAD, 0, 50 },
    { 84, SH_DEAD, 60, 50 }, { 85, SH_DEAD, 120, 50 },
    { 86, SH_DEAD, 0, 0 },   { 87, SH_DEAD, 60, 0 },
    { 100, SH_STAND_GUARD, 0, 50 }, { 101, SH_STAND_GUARD, 0, 0 },
    { 110, SH_CROUCH_GUARD, 0, 50 }, { 111, SH_CROUCH_GUARD, 0, 0 },
    { 120, SH_GUARD_BREAK, 0, 50 }, { 121, SH_GUARD_BREAK, 0, 0 },
    { 130, SH_ATTACK2, 0, 100 }, { 131, SH_ATTACK2, 60, 100 },
    { 132, SH_ATTACK2, 0, 50 },  { 133, SH_ATTACK2, 60, 50 },
    { 134, SH_ATTACK2, 0, 0 },
    { 140, SH_ATTACK3, 0, 150 }, { 141, SH_ATTACK3, 60, 150 },
    { 142, SH_ATTACK3, 0, 100 }, { 143, SH_ATTACK3, 60, 100 },
    { 144, SH_ATTACK3, 0, 50 },  { 145, SH_ATTACK3, 60, 50 },
    { 146, SH_ATTACK3, 0, 0 },
    { 200, SH_WIN, 0, 100 }, { 201, SH_WIN, 60, 100 },
    { 202, SH_WIN, 120, 100 }, { 203, SH_WIN, 0, 50 },
    { 204, SH_WIN, 60, 50 }, { 205, SH_WIN, 120, 50 },
    { 206, SH_WIN, 0, 0 },   { 207, SH_WIN, 60, 0 },
};

static const MotionSprite *GetMotionSprite(int motionID)
{
    for (int i = 0; i < MOTION_SPRITE_COUNT; i++)
        if (g_motionSprites[i].motionID == motionID)
            return &g_motionSprites[i];
    return NULL;
}

/* ---- attack data (F00_AttackDataContainer.asset) ---- */
static const AttackData s_attackData[] = {
    { 1,  "N_ATTACK",  ACT_DAMAGE, ACT_GUARD_CROUCH, 1, 0, 1, 12, 12, 30 },
    { 2,  "B_ATTACK",  ACT_DAMAGE, ACT_GUARD_STAND,  1, 0, 1, 12, 12, 30 },
    { 10, "N_SPECIAL", ACT_DEAD,   ACT_GUARD_M,      1, 1, 1, 0,  15, 30 },
    { 11, "B_SPECIAL", ACT_DEAD,   ACT_GUARD_M,      1, 1, 1, 0,  15, 30 },
};

static const AttackData *GetAttackData(int attackID)
{
    for (int i = 0; i < (int)(sizeof(s_attackData) / sizeof(s_attackData[0])); i++)
        if (s_attackData[i].attackID == attackID)
            return &s_attackData[i];
    return NULL;
}

/* ---- actions (Assets/Fighter/F00/Actions assets) ---- */
#define FR(a, b) { a, b }

/* STAND */
static const MotionFrameData s_stand_mot[] = {
    { FR(0, 5), 0 }, { FR(6, 8), 1 }, { FR(9, 14), 2 }, { FR(15, 20), 3 }, { FR(21, 23), 4 },
};
static const HurtboxData s_stand_hurt[] = { { FR(0, 59), 0, 0, 0, 0, true } };
static const PushboxData s_stand_push[] = { { FR(0, 59), 0, 0, 0, 0, true } };

/* FORWARD */
static const MotionFrameData s_forward_mot[] = {
    { FR(0, 3), 10 }, { FR(4, 7), 11 }, { FR(8, 11), 12 },
    { FR(12, 15), 13 }, { FR(16, 19), 14 }, { FR(20, 23), 15 },
};
static const HurtboxData s_forward_hurt[] = { { FR(0, 59), 0, 0, 0, 0, true } };
static const PushboxData s_forward_push[] = { { FR(0, 59), 0, 0, 0, 0, true } };

/* BACKWARD */
static const MotionFrameData s_backward_mot[] = {
    { FR(0, 3), 20 }, { FR(4, 7), 21 }, { FR(8, 11), 22 },
    { FR(12, 15), 23 }, { FR(16, 19), 24 }, { FR(20, 23), 25 },
};
static const HurtboxData s_backward_hurt[] = { { FR(0, 59), 0, 0, 0, 0, true } };
static const PushboxData s_backward_push[] = { { FR(0, 59), 0, 0, 0, 0, true } };

/* DASH_FORWARD */
static const MotionFrameData s_dashf_mot[] = {
    { FR(0, 7), 30 }, { FR(8, 10), 31 }, { FR(11, 12), 32 }, { FR(13, 14), 33 }, { FR(15, 15), 34 },
};
static const HurtboxData s_dashf_hurt[] = { { FR(0, 15), 0, 0, 0, 0, true } };
static const PushboxData s_dashf_push[] = { { FR(0, 15), 0, 0, 0, 0, true } };
static const MovementData s_dashf_mov[] = {
    { FR(0, 2), 5 }, { FR(3, 8), 7 }, { FR(9, 11), 5 },
    { FR(12, 13), 2 }, { FR(14, 14), 1 }, { FR(15, 15), 0 },
};

/* DASH_BACKWARD */
static const MotionFrameData s_dashb_mot[] = {
    { FR(0, 10), 40 }, { FR(11, 16), 41 }, { FR(17, 20), 42 }, { FR(21, 21), 43 },
};
static const HurtboxData s_dashb_hurt[] = { { FR(4, 21), 0, 0, 0.9f, 1.2f, false } };
static const PushboxData s_dashb_push[] = { { FR(0, 21), 0, 0, 0.8f, 1.0f, false } };
static const MovementData s_dashb_mov[] = {
    { FR(0, 2), -10 }, { FR(3, 8), -5 }, { FR(9, 12), -3 },
    { FR(13, 14), -1 }, { FR(15, 15), 0 },
};

/* N_ATTACK */
static const MotionFrameData s_natk_mot[] = {
    { FR(0, 1), 50 }, { FR(2, 3), 51 }, { FR(4, 15), 52 }, { FR(16, 19), 53 }, { FR(20, 21), 54 },
};
static const HitboxData s_natk_hit[] = {
    { FR(0, 5), 1.5f, 0, 3.0f, 0.4f, 1, true },
    { FR(4, 5), 0.9f, 0, 1.8f, 0.3f, 1, false },
};
static const HurtboxData s_natk_hurt[] = {
    { FR(0, 21), 0, 0, 0, 0, true },
    { FR(4, 15), 0.8f, 0, 1.6f, 0.4f, false },
    { FR(16, 18), 0.5f, 0, 1.0f, 0.4f, false },
};
static const PushboxData s_natk_push[] = { { FR(0, 21), 0, 0, 0, 0, true } };
static const CancelData s_natk_cancel[] = {
    { FR(1, 3), true, false, ACT_N_SPECIAL },
    { FR(4, 5), false, true, ACT_N_SPECIAL },
};

/* B_ATTACK */
static const MotionFrameData s_batk_mot[] = {
    { FR(0, 0), 130 }, { FR(1, 2), 131 }, { FR(3, 15), 132 }, { FR(16, 19), 133 }, { FR(20, 20), 134 },
};
static const HitboxData s_batk_hit[] = {
    { FR(0, 5), 1.4f, 0.5f, 2.8f, 0.4f, 2, true },
    { FR(3, 5), 0.8f, 0, 1.6f, 0.9f, 2, false },
};
static const HurtboxData s_batk_hurt[] = {
    { FR(0, 21), 0, 0, 0, 0, true },
    { FR(3, 14), 0.7f, 0, 1.4f, 0.9f, false },
    { FR(15, 17), 0.6f, 0, 1.2f, 0.9f, false },
};
static const PushboxData s_batk_push[] = { { FR(0, 21), 0, 0, 0, 0, true } };
static const CancelData s_batk_cancel[] = {
    { FR(1, 2), true, false, ACT_N_SPECIAL },
    { FR(3, 5), false, true, ACT_N_SPECIAL },
};

/* N_SPECIAL */
static const MotionFrameData s_nsp_mot[] = {
    { FR(0, 2), 60 }, { FR(3, 4), 61 }, { FR(5, 7), 62 }, { FR(8, 9), 63 },
    { FR(10, 10), 64 }, { FR(11, 25), 65 }, { FR(26, 40), 66 }, { FR(41, 43), 67 },
};
static const HitboxData s_nsp_hit[] = {
    { FR(0, 14), 1.5f, 0.8f, 3.0f, 0.2f, 10, true },
    { FR(11, 14), 1.0f, 0.75f, 2.0f, 0.3f, 10, false },
};
static const HurtboxData s_nsp_hurt[] = {
    { FR(0, 43), 0, 0, 0, 0, true },
    { FR(10, 25), 0.8f, 0.7f, 1.6f, 0.4f, false },
    { FR(26, 28), 0.7f, 0.7f, 1.4f, 0.4f, false },
};
static const PushboxData s_nsp_push[] = { { FR(0, 43), 0, 0, 0, 0, true } };
static const MovementData s_nsp_mov[] = {
    { FR(0, 3), 2 }, { FR(4, 15), 5 }, { FR(16, 17), 2 },
    { FR(18, 20), 1 }, { FR(21, 21), 0 },
};

/* B_SPECIAL */
static const MotionFrameData s_bsp_mot[] = {
    { FR(0, 0), 140 }, { FR(1, 1), 141 }, { FR(2, 5), 142 }, { FR(6, 35), 143 },
    { FR(36, 45), 144 }, { FR(46, 52), 145 }, { FR(53, 54), 146 },
};
static const HitboxData s_bsp_hit[] = {
    { FR(0, 5), 1.5f, 0.4f, 3.0f, 0.3f, 11, true },
    { FR(2, 7), 0.6f, 0, 1.2f, 1.0f, 11, false },
};
static const HurtboxData s_bsp_hurt[] = { { FR(6, 54), 0, 0, 0, 0, true } };
static const PushboxData s_bsp_push[] = { { FR(0, 54), 0, 0, 0, 0, true } };
static const MovementData s_bsp_mov[] = {
    { FR(0, 2), 3 }, { FR(0, 10), 2 }, { FR(10, 15), 1 }, { FR(16, 16), 0 },
};

/* DAMAGE */
static const MotionFrameData s_dmg_mot[] = {
    { FR(0, 8), 70 }, { FR(9, 12), 71 }, { FR(13, 15), 72 }, { FR(16, 16), 73 },
};
static const HurtboxData s_dmg_hurt[] = { { FR(0, 16), 0, 0, 0, 0, true } };
static const PushboxData s_dmg_push[] = { { FR(0, 16), 0, 0, 0, 0, true } };
static const MovementData s_dmg_mov[] = {
    { FR(0, 3), -3 }, { FR(4, 8), -2 }, { FR(9, 12), -0.5f }, { FR(13, 13), 0 },
};

/* GUARD_M */
static const MotionFrameData s_guardm_mot[] = { { FR(0, 14), 101 }, { FR(15, 22), 100 } };
static const HurtboxData s_guardm_hurt[] = { { FR(0, 29), 0, 0, 0, 0, true } };
static const PushboxData s_guardm_push[] = { { FR(0, 29), 0, 0, 0, 0, true } };
static const MovementData s_guardm_mov[] = {
    { FR(0, 6), -3 }, { FR(7, 10), -2 }, { FR(11, 15), -0.5f }, { FR(16, 16), 0 },
};

/* GUARD_STAND / GUARD_CROUCH share movement */
static const MotionFrameData s_guards_mot[] = { { FR(0, 10), 101 }, { FR(11, 14), 100 } };
static const MotionFrameData s_guardc_mot[] = { { FR(0, 10), 111 }, { FR(11, 14), 110 } };
static const HurtboxData s_guard_hurt[] = { { FR(0, 14), 0, 0, 0, 0, true } };
static const PushboxData s_guard_push[] = { { FR(0, 14), 0, 0, 0, 0, true } };
static const MovementData s_guard_mov[] = {
    { FR(0, 3), -2 }, { FR(4, 6), -1 }, { FR(7, 8), -0.5f }, { FR(9, 9), 0 },
};

/* GUARD_BREAK */
static const MotionFrameData s_gb_mot[] = { { FR(0, 30), 120 }, { FR(31, 35), 121 } };
static const HurtboxData s_gb_hurt[] = { { FR(0, 35), 0, 0, 1.2f, 1.2f, false } };
static const PushboxData s_gb_push[] = { { FR(0, 35), 0, 0, 0, 0, true } };
static const MovementData s_gb_mov[] = {
    { FR(0, 3), -2 }, { FR(4, 6), -1 }, { FR(7, 10), -0.5f }, { FR(11, 11), 0 },
};

/* GUARD_PROXIMITY */
static const MotionFrameData s_gprox_mot[] = { { FR(0, 0), 100 } };
static const HurtboxData s_gprox_hurt[] = { { FR(0, 0), 0, 0, 0, 0, true } };
static const PushboxData s_gprox_push[] = { { FR(0, 0), 0, 0, 0, 0, true } };

/* DEAD */
static const MotionFrameData s_dead_mot[] = {
    { FR(0, 9), 80 }, { FR(10, 19), 81 }, { FR(20, 29), 82 }, { FR(30, 39), 83 },
    { FR(40, 49), 84 }, { FR(50, 59), 85 }, { FR(60, 69), 86 }, { FR(70, 499), 87 },
};
static const HurtboxData s_dead_hurt[] = {
    { FR(0, 20), 0, 0, 0, 0, true },
    { FR(21, 500), 0, 0, 1.0f, 0.5f, false },
};
static const PushboxData s_dead_push[] = {
    { FR(0, 9), 0, 0, 2.0f, 1.0f, false },
    { FR(10, 20), -0.25f, 0, 2.0f, 1.0f, false },
    { FR(21, 500), -0.5f, 0, 2.5f, 0.5f, false },
};
static const MovementData s_dead_mov[] = {
    { FR(0, 19), -4 }, { FR(20, 39), -3 }, { FR(40, 49), -1 }, { FR(50, 50), 0 },
};

/* WIN */
static const MotionFrameData s_win_mot[] = {
    { FR(0, 4), 200 }, { FR(5, 8), 201 }, { FR(9, 12), 202 }, { FR(13, 16), 203 },
    { FR(17, 20), 204 }, { FR(21, 24), 205 }, { FR(25, 28), 206 }, { FR(29, 32), 207 },
};
static const HurtboxData s_win_hurt[] = { { FR(0, 32), 0, 0, 0, 0, true } };
static const PushboxData s_win_push[] = { { FR(0, 32), 0, 0, 0, 0, true } };

#define CNT(a) (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NONE  NULL, 0

static const ActionData s_actions[] = {
    /* id, name, type, frames, loop, loopFrom, alwaysCancel, sound,
       motions, hitboxes, hurtboxes, pushboxes, movements, cancels */
    { ACT_STAND, "STAND", ATYPE_MOVEMENT, 24, false, 0, true, SND_NONE,
      CNT(s_stand_mot), NONE, CNT(s_stand_hurt), CNT(s_stand_push), NONE, NONE },
    { ACT_FORWARD, "FORWARD", ATYPE_MOVEMENT, 24, false, 0, true, SND_NONE,
      CNT(s_forward_mot), NONE, CNT(s_forward_hurt), CNT(s_forward_push), NONE, NONE },
    { ACT_BACKWARD, "BACKWARD", ATYPE_MOVEMENT, 24, false, 0, true, SND_NONE,
      CNT(s_backward_mot), NONE, CNT(s_backward_hurt), CNT(s_backward_push), NONE, NONE },
    { ACT_DASH_FORWARD, "DASH_FORWARD", ATYPE_MOVEMENT, 16, false, 0, false, SND_DASH_FORWARD,
      CNT(s_dashf_mot), NONE, CNT(s_dashf_hurt), CNT(s_dashf_push), CNT(s_dashf_mov), NONE },
    { ACT_DASH_BACKWARD, "DASH_BACKWARD", ATYPE_MOVEMENT, 22, false, 0, false, SND_DASH_BACKWARD,
      CNT(s_dashb_mot), NONE, CNT(s_dashb_hurt), CNT(s_dashb_push), CNT(s_dashb_mov), NONE },
    { ACT_N_ATTACK, "N_ATTACK", ATYPE_ATTACK, 22, false, 0, false, SND_ATTACK,
      CNT(s_natk_mot), CNT(s_natk_hit), CNT(s_natk_hurt), CNT(s_natk_push), NONE, CNT(s_natk_cancel) },
    { ACT_B_ATTACK, "B_ATTACK", ATYPE_ATTACK, 21, false, 0, false, SND_ATTACK2,
      CNT(s_batk_mot), CNT(s_batk_hit), CNT(s_batk_hurt), CNT(s_batk_push), NONE, CNT(s_batk_cancel) },
    { ACT_N_SPECIAL, "N_SPECIAL", ATYPE_ATTACK, 44, false, 0, false, SND_N_SPECIAL,
      CNT(s_nsp_mot), CNT(s_nsp_hit), CNT(s_nsp_hurt), CNT(s_nsp_push), CNT(s_nsp_mov), NONE },
    { ACT_B_SPECIAL, "B_SPECIAL", ATYPE_ATTACK, 55, false, 0, false, SND_B_SPECIAL,
      CNT(s_bsp_mot), CNT(s_bsp_hit), CNT(s_bsp_hurt), CNT(s_bsp_push), CNT(s_bsp_mov), NONE },
    { ACT_DAMAGE, "DAMAGE", ATYPE_DAMAGE, 17, false, 0, false, SND_NORMAL_HIT,
      CNT(s_dmg_mot), NONE, CNT(s_dmg_hurt), CNT(s_dmg_push), CNT(s_dmg_mov), NONE },
    { ACT_GUARD_M, "GUARD_M", ATYPE_GUARD, 23, false, 0, false, SND_GUARD,
      CNT(s_guardm_mot), NONE, CNT(s_guardm_hurt), CNT(s_guardm_push), CNT(s_guardm_mov), NONE },
    { ACT_GUARD_STAND, "GUARD_STAND", ATYPE_GUARD, 15, false, 0, false, SND_GUARD,
      CNT(s_guards_mot), NONE, CNT(s_guard_hurt), CNT(s_guard_push), CNT(s_guard_mov), NONE },
    { ACT_GUARD_CROUCH, "GUARD_CROUCH", ATYPE_GUARD, 15, false, 0, false, SND_GUARD,
      CNT(s_guardc_mot), NONE, CNT(s_guard_hurt), CNT(s_guard_push), CNT(s_guard_mov), NONE },
    { ACT_GUARD_BREAK, "GUARD_BREAK", ATYPE_DAMAGE, 36, false, 0, false, SND_GUARD_BREAK,
      CNT(s_gb_mot), NONE, CNT(s_gb_hurt), CNT(s_gb_push), CNT(s_gb_mov), NONE },
    { ACT_GUARD_PROXIMITY, "GUARD_PROXIMITY", ATYPE_GUARD, 1, false, 0, true, SND_NONE,
      CNT(s_gprox_mot), NONE, CNT(s_gprox_hurt), CNT(s_gprox_push), NONE, NONE },
    { ACT_DEAD, "DEAD", ATYPE_DAMAGE, 500, false, 0, false, SND_SPECIAL_HIT,
      CNT(s_dead_mot), NONE, CNT(s_dead_hurt), CNT(s_dead_push), CNT(s_dead_mov), NONE },
    { ACT_WIN, "WIN", ATYPE_MOVEMENT, 33, true, 5, false, SND_NONE,
      CNT(s_win_mot), NONE, CNT(s_win_hurt), CNT(s_win_push), NONE, NONE },
};

static const ActionData *GetActionData(int actionID)
{
    for (int i = 0; i < (int)(sizeof(s_actions) / sizeof(s_actions[0])); i++)
        if (s_actions[i].actionID == actionID)
            return &s_actions[i];
    return NULL;
}

/* ============================================================
 * fighter - per-fighter sim state + logic
 * ============================================================ */

#define INPUT_RECORD_FRAME 180
#define MAX_SPRITE_SHAKE_FRAME 6

typedef struct {
    float x, y, w, h; /* fight-space: x center, y bottom */
} Box;

static inline float BoxXMin(Box b) { return b.x - b.w / 2; }
static inline float BoxXMax(Box b) { return b.x + b.w / 2; }
static inline float BoxYMin(Box b) { return b.y; }
static inline float BoxYMax(Box b) { return b.y + b.h; }

static inline bool BoxOverlaps(Box a, Box b)
{
    return BoxXMax(b) >= BoxXMin(a) && BoxXMin(b) <= BoxXMax(a)
        && BoxYMax(b) >= BoxYMin(a) && BoxYMin(b) <= BoxYMax(a);
}

typedef struct {
    Box box;
    bool proximity;
    int attackID;
} Hitbox;

#define MAX_HITBOXES 8
#define MAX_HURTBOXES 8

typedef struct Fighter {
    float posX, posY;
    float velX;
    bool faceRight;

    int vitalHealth;
    int guardHealth;

    int currentActionID;
    int currentActionFrame;
    int currentActionHitCount;

    int currentHitStunFrame;

    int input[INPUT_RECORD_FRAME];
    int inputDown[INPUT_RECORD_FRAME];
    int inputUp[INPUT_RECORD_FRAME];

    bool isInputBackward;
    bool isReserveProximityGuard;

    int bufferActionID;
    int reserveDamageActionID;

    int spriteShakePosition;
    bool hasWon;

    Hitbox hitboxes[MAX_HITBOXES]; int hitboxCount;
    Box    hurtboxes[MAX_HURTBOXES]; int hurtboxCount;
    Box    pushbox;
} Fighter;

static inline bool Fighter_IsDead(Fighter *f) { return f->vitalHealth <= 0; }
static inline bool Fighter_IsInHitStun(Fighter *f) { return f->currentHitStunFrame > 0; }

static inline int Fighter_ActionFrameCount(Fighter *f)
{
    const ActionData *a = GetActionData(f->currentActionID);
    return a ? a->frameCount : 0;
}
static inline bool Fighter_IsActionEnd(Fighter *f)
{
    return f->currentActionFrame >= Fighter_ActionFrameCount(f);
}

/* forward-declared so the definitions below (kept in the original class's
 * method order) can freely forward-reference each other. */
static void Fighter_SetupBattleStart(Fighter *f, float startX, bool isPlayerOne);
static void Fighter_ClearInput(Fighter *f);
static void Fighter_IncrementActionFrame(Fighter *f);
static void Fighter_UpdateInput(Fighter *f, int inputBits);
static void Fighter_UpdateIntroAction(Fighter *f);
static void Fighter_UpdateActionRequest(Fighter *f);
static void Fighter_UpdateMovement(Fighter *f);
static void Fighter_UpdateBoxes(Fighter *f);
static void Fighter_ApplyPositionChange(Fighter *f, float x, float y);
static void Fighter_NotifyAttackHit(Fighter *f);
static int  Fighter_NotifyDamaged(Fighter *f, const AttackData *atk);
static void Fighter_NotifyInProximityGuardRange(Fighter *f);
static bool Fighter_CanAttackHit(Fighter *f, int attackID);
static void Fighter_SetHitStun(Fighter *f, int frames);
static void Fighter_SetSpriteShakeFrame(Fighter *f, int frames);
static int  Fighter_GetHitStunFrame(Fighter *f, int damageResult, int attackID);
static void Fighter_RequestWinAction(Fighter *f);
static bool Fighter_RequestAction(Fighter *f, int actionID);
static int  Fighter_GetCurrentMotionID(Fighter *f);

/* no audio in the puffer env - stub keeps SetCurrentAction unmodified */
static void Sound_PlayFighterSE(int soundID, bool faceRight, float x)
{
    (void)soundID; (void)faceRight; (void)x;
}

static bool inRange(FrameRange fr, int frame) { return frame >= fr.start && frame <= fr.end; }

static const MotionFrameData *GetMotionData(const ActionData *a, int frame)
{
    for (int i = 0; i < a->motionCount; i++)
        if (inRange(a->motions[i].fr, frame)) return &a->motions[i];
    return NULL;
}

static const MovementData *GetMovementData(const ActionData *a, int frame)
{
    for (int i = 0; i < a->movementCount; i++)
        if (inRange(a->movements[i].fr, frame)) return &a->movements[i];
    return NULL;
}

static bool IsAttackInput(Fighter *f, int input) { (void)f; return (input & INPUT_ATTACK) != 0; }

static bool IsForwardInput(Fighter *f, int input)
{
    return f->faceRight ? (input & INPUT_RIGHT) != 0 : (input & INPUT_LEFT) != 0;
}

static bool IsBackwardInput(Fighter *f, int input)
{
    return f->faceRight ? (input & INPUT_LEFT) != 0 : (input & INPUT_RIGHT) != 0;
}

static void SetCurrentAction(Fighter *f, int actionID)
{
    f->currentActionID = actionID;
    f->currentActionFrame = 0;
    f->currentActionHitCount = 0;
    f->bufferActionID = -1;
    f->reserveDamageActionID = -1;
    f->spriteShakePosition = 0;

    const ActionData *a = GetActionData(actionID);
    if (a && a->sound != SND_NONE) {
        /* GUARD_BREAK sound is played from NotifyDamaged instead (as in original) */
        if (actionID != ACT_GUARD_BREAK)
            Sound_PlayFighterSE(a->sound, f->faceRight, f->posX);
    }
}

static void Fighter_SetupBattleStart(Fighter *f, float startX, bool isPlayerOne)
{
    f->posX = startX;
    f->posY = 0;
    f->faceRight = isPlayerOne;
    f->vitalHealth = 1;
    f->guardHealth = FIGHTER_START_GUARD_HEALTH;
    f->hasWon = false;
    f->velX = 0;
    f->currentHitStunFrame = 0;
    f->bufferActionID = -1;
    f->reserveDamageActionID = -1;
    f->isInputBackward = false;
    f->isReserveProximityGuard = false;
    Fighter_ClearInput(f);
    f->currentActionID = ACT_STAND;
    f->currentActionFrame = 0;
    f->currentActionHitCount = 0;
    f->spriteShakePosition = 0;
    f->hitboxCount = 0;
    f->hurtboxCount = 0;
    f->pushbox = (Box){ startX, 0, BASE_PUSHBOX.w, BASE_PUSHBOX.h };
}

static void Fighter_ClearInput(Fighter *f)
{
    for (int i = 0; i < INPUT_RECORD_FRAME; i++) {
        f->input[i] = 0;
        f->inputDown[i] = 0;
        f->inputUp[i] = 0;
    }
}

static void Fighter_IncrementActionFrame(Fighter *f)
{
    /* sprite shake decay: swap sign and move toward 0 */
    if (abs(f->spriteShakePosition) > 0) {
        f->spriteShakePosition *= -1;
        f->spriteShakePosition += (f->spriteShakePosition > 0 ? -1 : 1);
    }

    /* action frame frozen during hit stun */
    if (f->currentHitStunFrame > 0) {
        f->currentHitStunFrame--;
        return;
    }

    f->currentActionFrame++;

    const ActionData *a = GetActionData(f->currentActionID);
    if (a && f->currentActionFrame >= a->frameCount && a->isLoop)
        f->currentActionFrame = a->loopFromFrame;
}

static void Fighter_UpdateInput(Fighter *f, int inputBits)
{
    for (int i = INPUT_RECORD_FRAME - 1; i >= 1; i--) {
        f->input[i] = f->input[i - 1];
        f->inputDown[i] = f->inputDown[i - 1];
        f->inputUp[i] = f->inputUp[i - 1];
    }
    f->input[0] = inputBits;
    f->inputDown[0] = (f->input[0] ^ f->input[1]) & f->input[0];
    f->inputUp[0] = (f->input[0] ^ f->input[1]) & ~f->input[0];
}

static void Fighter_UpdateIntroAction(Fighter *f)
{
    Fighter_RequestAction(f, ACT_STAND);
}

static bool canCancelAttack(Fighter *f)
{
    if (FIGHTER_CAN_CANCEL_ON_WHIFF)
        return true;
    return f->currentActionHitCount > 0;
}

static bool CheckSpecialAttackInput(Fighter *f)
{
    if (!IsAttackInput(f, f->inputUp[0]))
        return false;
    for (int i = 1; i < FIGHTER_SPECIAL_HOLD_FRAME; i++)
        if (!IsAttackInput(f, f->input[i]))
            return false;
    return true;
}

static bool CheckForwardDashInput(Fighter *f)
{
    if (!IsForwardInput(f, f->inputDown[0]))
        return false;
    for (int i = 1; i < FIGHTER_DASH_ALLOW_FRAME; i++) {
        if (IsBackwardInput(f, f->input[i]))
            return false;
        if (IsForwardInput(f, f->input[i])) {
            for (int j = i + 1; j < i + FIGHTER_DASH_ALLOW_FRAME; j++) {
                if (!IsForwardInput(f, f->input[j]) && !IsBackwardInput(f, f->input[j]))
                    return true;
            }
            return false;
        }
    }
    return false;
}

static bool CheckBackwardDashInput(Fighter *f)
{
    if (!IsBackwardInput(f, f->inputDown[0]))
        return false;
    for (int i = 1; i < FIGHTER_DASH_ALLOW_FRAME; i++) {
        if (IsForwardInput(f, f->input[i]))
            return false;
        if (IsBackwardInput(f, f->input[i])) {
            for (int j = i + 1; j < i + FIGHTER_DASH_ALLOW_FRAME; j++) {
                if (!IsForwardInput(f, f->input[j]) && !IsBackwardInput(f, f->input[j]))
                    return true;
            }
            return false;
        }
    }
    return false;
}

static void Fighter_UpdateActionRequest(Fighter *f)
{
    if (f->hasWon) {
        Fighter_RequestAction(f, ACT_WIN);
        return;
    }

    /* reserved damage action (guard break) plays after hit stun ends */
    if (f->reserveDamageActionID != -1 && f->currentHitStunFrame <= 0) {
        SetCurrentAction(f, f->reserveDamageActionID);
        f->reserveDamageActionID = -1;
        return;
    }

    /* buffered cancel (normal -> special on hit) */
    if (f->bufferActionID != -1 && canCancelAttack(f) && f->currentHitStunFrame <= 0) {
        SetCurrentAction(f, f->bufferActionID);
        f->bufferActionID = -1;
        return;
    }

    bool isForward = IsForwardInput(f, f->input[0]);
    bool isBackward = IsBackwardInput(f, f->input[0]);
    bool isAttack = IsAttackInput(f, f->inputDown[0]);

    if (CheckSpecialAttackInput(f)) {
        if (isBackward || isForward)
            Fighter_RequestAction(f, ACT_B_SPECIAL);
        else
            Fighter_RequestAction(f, ACT_N_SPECIAL);
    } else if (isAttack) {
        if ((f->currentActionID == ACT_N_ATTACK || f->currentActionID == ACT_B_ATTACK)
            && !Fighter_IsActionEnd(f)) {
            Fighter_RequestAction(f, ACT_N_SPECIAL);
        } else {
            if (isBackward || isForward)
                Fighter_RequestAction(f, ACT_B_ATTACK);
            else
                Fighter_RequestAction(f, ACT_N_ATTACK);
        }
    }

    if (CheckForwardDashInput(f))
        Fighter_RequestAction(f, ACT_DASH_FORWARD);
    else if (CheckBackwardDashInput(f))
        Fighter_RequestAction(f, ACT_DASH_BACKWARD);

    /* for proximity guard check */
    f->isInputBackward = isBackward;

    if (isForward && isBackward) {
        Fighter_RequestAction(f, ACT_STAND);
    } else if (isForward) {
        Fighter_RequestAction(f, ACT_FORWARD);
    } else if (isBackward) {
        if (f->isReserveProximityGuard)
            Fighter_RequestAction(f, ACT_GUARD_PROXIMITY);
        else
            Fighter_RequestAction(f, ACT_BACKWARD);
    } else {
        Fighter_RequestAction(f, ACT_STAND);
    }

    f->isReserveProximityGuard = false;
}

static void Fighter_UpdateMovement(Fighter *f)
{
    if (Fighter_IsInHitStun(f))
        return;

    float sign = f->faceRight ? 1.0f : -1.0f;

    if (f->currentActionID == ACT_FORWARD) {
        f->posX += FIGHTER_FORWARD_SPEED * sign * FIXED_DT;
        return;
    }
    if (f->currentActionID == ACT_BACKWARD) {
        f->posX -= FIGHTER_BACKWARD_SPEED * sign * FIXED_DT;
        return;
    }

    const ActionData *a = GetActionData(f->currentActionID);
    if (!a) return;
    const MovementData *md = GetMovementData(a, f->currentActionFrame);
    if (md) {
        f->velX = md->velX;
        if (f->velX != 0)
            f->posX += f->velX * sign * FIXED_DT;
    }
}

static Box TransformToFightRect(BoxData d, float baseX, float baseY, bool faceRight)
{
    float sign = faceRight ? 1.0f : -1.0f;
    Box b;
    b.x = baseX + d.x * sign;
    b.y = baseY + d.y;
    b.w = d.w;
    b.h = d.h;
    return b;
}

static void Fighter_UpdateBoxes(Fighter *f)
{
    f->hitboxCount = 0;
    f->hurtboxCount = 0;

    const ActionData *a = GetActionData(f->currentActionID);
    if (!a) return;

    int frame = f->currentActionFrame;

    for (int i = 0; i < a->hitboxCount && f->hitboxCount < MAX_HITBOXES; i++) {
        const HitboxData *hd = &a->hitboxes[i];
        if (!inRange(hd->fr, frame)) continue;
        BoxData d = { hd->x, hd->y, hd->w, hd->h };
        Hitbox *hb = &f->hitboxes[f->hitboxCount++];
        hb->box = TransformToFightRect(d, f->posX, f->posY, f->faceRight);
        hb->proximity = hd->proximity;
        hb->attackID = hd->attackID;
    }

    for (int i = 0; i < a->hurtboxCount && f->hurtboxCount < MAX_HURTBOXES; i++) {
        const HurtboxData *hd = &a->hurtboxes[i];
        if (!inRange(hd->fr, frame)) continue;
        BoxData d = hd->useBaseRect ? BASE_HURTBOX : (BoxData){ hd->x, hd->y, hd->w, hd->h };
        f->hurtboxes[f->hurtboxCount++] = TransformToFightRect(d, f->posX, f->posY, f->faceRight);
    }

    for (int i = 0; i < a->pushboxCount; i++) {
        const PushboxData *pd = &a->pushboxes[i];
        if (!inRange(pd->fr, frame)) continue;
        BoxData d = pd->useBaseRect ? BASE_PUSHBOX : (BoxData){ pd->x, pd->y, pd->w, pd->h };
        f->pushbox = TransformToFightRect(d, f->posX, f->posY, f->faceRight);
        break;
    }
}

static void Fighter_ApplyPositionChange(Fighter *f, float x, float y)
{
    f->posX += x;
    f->posY += y;
    for (int i = 0; i < f->hitboxCount; i++) {
        f->hitboxes[i].box.x += x;
        f->hitboxes[i].box.y += y;
    }
    for (int i = 0; i < f->hurtboxCount; i++) {
        f->hurtboxes[i].x += x;
        f->hurtboxes[i].y += y;
    }
    f->pushbox.x += x;
    f->pushbox.y += y;
}

static void Fighter_NotifyAttackHit(Fighter *f)
{
    f->currentActionHitCount++;
}

static int Fighter_NotifyDamaged(Fighter *f, const AttackData *atk)
{
    bool isGuardBreak = false;
    if (atk->guardHealthDamage > 0) {
        f->guardHealth -= atk->guardHealthDamage;
        if (f->guardHealth < 0) {
            isGuardBreak = true;
            f->guardHealth = 0;
        }
    }

    const ActionData *cur = GetActionData(f->currentActionID);
    bool inGuardMotion = (f->currentActionID == ACT_BACKWARD)
        || (cur && cur->type == ATYPE_GUARD);

    if (inGuardMotion) {
        if (isGuardBreak) {
            SetCurrentAction(f, atk->guardActionID);
            f->reserveDamageActionID = ACT_GUARD_BREAK;
            const ActionData *gb = GetActionData(ACT_GUARD_BREAK);
            if (gb && gb->sound != SND_NONE)
                Sound_PlayFighterSE(gb->sound, f->faceRight, f->posX);
            return DMG_GUARD_BREAK;
        }
        SetCurrentAction(f, atk->guardActionID);
        return DMG_GUARD;
    }

    if (atk->vitalHealthDamage > 0) {
        f->vitalHealth -= atk->vitalHealthDamage;
        if (f->vitalHealth <= 0)
            f->vitalHealth = 0;
    }
    SetCurrentAction(f, atk->damageActionID);
    return DMG_DAMAGE;
}

static void Fighter_NotifyInProximityGuardRange(Fighter *f)
{
    if (f->isInputBackward)
        f->isReserveProximityGuard = true;
}

static bool Fighter_CanAttackHit(Fighter *f, int attackID)
{
    const AttackData *atk = GetAttackData(attackID);
    if (!atk)
        return true;
    return f->currentActionHitCount < atk->numberOfHit;
}

static void Fighter_SetHitStun(Fighter *f, int frames)
{
    f->currentHitStunFrame = frames;
}

static void Fighter_SetSpriteShakeFrame(Fighter *f, int frames)
{
    if (frames > MAX_SPRITE_SHAKE_FRAME)
        frames = MAX_SPRITE_SHAKE_FRAME;
    f->spriteShakePosition = frames * (f->faceRight ? -1 : 1);
}

static int Fighter_GetHitStunFrame(Fighter *f, int damageResult, int attackID)
{
    (void)f;
    const AttackData *atk = GetAttackData(attackID);
    if (!atk) return 0;
    if (damageResult == DMG_GUARD)
        return atk->guardStunFrame;
    if (damageResult == DMG_GUARD_BREAK)
        return atk->guardBreakStunFrame;
    return atk->hitStunFrame;
}

static void Fighter_RequestWinAction(Fighter *f)
{
    f->hasWon = true;
}

static bool Fighter_RequestAction(Fighter *f, int actionID)
{
    if (Fighter_IsActionEnd(f)) {
        SetCurrentAction(f, actionID);
        return true;
    }

    if (f->currentActionID == actionID)
        return false;

    const ActionData *cur = GetActionData(f->currentActionID);
    if (cur && cur->alwaysCancelable) {
        SetCurrentAction(f, actionID);
        return true;
    }

    if (cur) {
        for (int i = 0; i < cur->cancelCount; i++) {
            const CancelData *cd = &cur->cancels[i];
            if (!inRange(cd->fr, f->currentActionFrame)) continue;
            if (cd->actionID != actionID) continue;
            if (cd->execute) {
                f->bufferActionID = actionID;
                return true;
            }
            if (cd->buffer) {
                f->bufferActionID = actionID;
            }
        }
    }

    return false;
}

static int Fighter_GetCurrentMotionID(Fighter *f)
{
    const ActionData *a = GetActionData(f->currentActionID);
    if (!a) return -1;
    const MotionFrameData *md = GetMotionData(a, f->currentActionFrame);
    return md ? md->motionID : -1;
}

/* ============================================================
 * game - round state machine + collision, owns both fighters
 * ============================================================ */

#define BATTLE_AREA_WIDTH 10.0f
#define BATTLE_AREA_MAX_HEIGHT 2.0f
#define MAX_ROUND_WON 3

typedef enum {
    ROUND_STOP = 0,
    ROUND_INTRO,
    ROUND_FIGHT,
    ROUND_KO,
    ROUND_END,
} RoundState;

/* hit event for rendering effects (set from damageHandler) */
typedef struct {
    bool active;
    float x, y;
    int result; /* DMG_* */
    int damagedIndex; /* 0 = fighter1, 1 = fighter2 */
} HitEvent;

typedef struct {
    Fighter fighter1;
    Fighter fighter2;

    RoundState roundState;
    float timer;
    int frameCount;

    int fighter1RoundWon;
    int fighter2RoundWon;

    bool matchOver;
    int matchWinner; /* 0 = P1, 1 = P2 */

    bool isDebugPause;

    HitEvent lastHit;
} Game;

#define INTRO_STATE_TIME 0.0f
#define KO_STATE_TIME 2.0f
#define END_STATE_TIME 3.0f
#define END_STATE_SKIPPABLE_TIME 1.5f

static void Game_Init(Game *g)
{
    g->roundState = ROUND_STOP;
    g->timer = 0;
    g->frameCount = 0;
    g->fighter1RoundWon = 0;
    g->fighter2RoundWon = 0;
    g->matchOver = false;
    g->matchWinner = 0;
    g->isDebugPause = false;
    g->lastHit.active = false;
    Fighter_SetupBattleStart(&g->fighter1, -2.0f, true);
    Fighter_SetupBattleStart(&g->fighter2, 2.0f, false);
}

static void ChangeRoundState(Game *g, RoundState state);

static void UpdatePushCharacterVsCharacter(Game *g)
{
    Box r1 = g->fighter1.pushbox;
    Box r2 = g->fighter2.pushbox;

    if (BoxOverlaps(r1, r2)) {
        if (g->fighter1.posX < g->fighter2.posX) {
            float overlap = BoxXMax(r1) - BoxXMin(r2);
            Fighter_ApplyPositionChange(&g->fighter1, overlap * -0.5f, g->fighter1.posY);
            Fighter_ApplyPositionChange(&g->fighter2, overlap * 0.5f, g->fighter2.posY);
        } else if (g->fighter1.posX > g->fighter2.posX) {
            float overlap = BoxXMax(r2) - BoxXMin(r1);
            Fighter_ApplyPositionChange(&g->fighter1, overlap * 0.5f, g->fighter1.posY);
            Fighter_ApplyPositionChange(&g->fighter2, overlap * -0.5f, g->fighter2.posY);
        }
    }
}

static void UpdatePushCharacterVsBackground(Game *g)
{
    float stageMinX = BATTLE_AREA_WIDTH * -0.5f;
    float stageMaxX = BATTLE_AREA_WIDTH * 0.5f;

    Fighter *fs[2] = { &g->fighter1, &g->fighter2 };
    for (int i = 0; i < 2; i++) {
        Fighter *f = fs[i];
        if (BoxXMin(f->pushbox) < stageMinX)
            Fighter_ApplyPositionChange(f, stageMinX - BoxXMin(f->pushbox), f->posY);
        else if (BoxXMax(f->pushbox) > stageMaxX)
            Fighter_ApplyPositionChange(f, stageMaxX - BoxXMax(f->pushbox), f->posY);
    }
}

static void UpdateHitboxHurtboxCollision(Game *g)
{
    Fighter *fs[2] = { &g->fighter1, &g->fighter2 };

    for (int ai = 0; ai < 2; ai++) {
        Fighter *attacker = fs[ai];
        float dmgX = 0, dmgY = 0;
        bool isHit = false;
        bool isProximity = false;
        int hitAttackID = 0;

        for (int di = 0; di < 2; di++) {
            Fighter *damaged = fs[di];
            if (attacker == damaged) continue;

            for (int hi = 0; hi < attacker->hitboxCount; hi++) {
                Hitbox *hb = &attacker->hitboxes[hi];

                if (!Fighter_CanAttackHit(attacker, hb->attackID))
                    continue;

                for (int vi = 0; vi < damaged->hurtboxCount; vi++) {
                    Box hurt = damaged->hurtboxes[vi];
                    if (BoxOverlaps(hb->box, hurt)) {
                        if (hb->proximity) {
                            isProximity = true;
                        } else {
                            isHit = true;
                            hitAttackID = hb->attackID;
                            float x1 = BoxXMax(hb->box) < BoxXMax(hurt) ? BoxXMax(hb->box) : BoxXMax(hurt);
                            float x2 = BoxXMin(hb->box) > BoxXMin(hurt) ? BoxXMin(hb->box) : BoxXMin(hurt);
                            float y1 = BoxYMax(hb->box) < BoxYMax(hurt) ? BoxYMax(hb->box) : BoxYMax(hurt);
                            float y2 = BoxYMin(hb->box) > BoxYMin(hurt) ? BoxYMin(hb->box) : BoxYMin(hurt);
                            dmgX = (x1 + x2) / 2;
                            dmgY = (y1 + y2) / 2;
                            break;
                        }
                    }
                }
                if (isHit) break;
            }

            if (isHit) {
                Fighter_NotifyAttackHit(attacker);
                const AttackData *atk = GetAttackData(hitAttackID);
                int damageResult = atk ? Fighter_NotifyDamaged(damaged, atk) : DMG_DAMAGE;

                int hitStunFrame = Fighter_GetHitStunFrame(attacker, damageResult, hitAttackID);
                Fighter_SetHitStun(attacker, hitStunFrame);
                Fighter_SetHitStun(damaged, hitStunFrame);
                Fighter_SetSpriteShakeFrame(damaged, hitStunFrame / 3);

                g->lastHit.active = true;
                g->lastHit.x = dmgX;
                g->lastHit.y = dmgY;
                g->lastHit.result = damageResult;
                g->lastHit.damagedIndex = di;
            } else if (isProximity) {
                Fighter_NotifyInProximityGuardRange(damaged);
            }
        }
    }
}

static void UpdateIntroState(Game *g, int p1Input, int p2Input)
{
    Fighter_UpdateInput(&g->fighter1, p1Input);
    Fighter_UpdateInput(&g->fighter2, p2Input);

    Fighter_IncrementActionFrame(&g->fighter1);
    Fighter_IncrementActionFrame(&g->fighter2);

    Fighter_UpdateIntroAction(&g->fighter1);
    Fighter_UpdateIntroAction(&g->fighter2);
    Fighter_UpdateMovement(&g->fighter1);
    Fighter_UpdateMovement(&g->fighter2);
    Fighter_UpdateBoxes(&g->fighter1);
    Fighter_UpdateBoxes(&g->fighter2);

    UpdatePushCharacterVsCharacter(g);
    UpdatePushCharacterVsBackground(g);
}

static void UpdateFightState(Game *g, int p1Input, int p2Input)
{
    Fighter_UpdateInput(&g->fighter1, p1Input);
    Fighter_UpdateInput(&g->fighter2, p2Input);

    Fighter_IncrementActionFrame(&g->fighter1);
    Fighter_IncrementActionFrame(&g->fighter2);

    Fighter_UpdateActionRequest(&g->fighter1);
    Fighter_UpdateActionRequest(&g->fighter2);
    Fighter_UpdateMovement(&g->fighter1);
    Fighter_UpdateMovement(&g->fighter2);
    Fighter_UpdateBoxes(&g->fighter1);
    Fighter_UpdateBoxes(&g->fighter2);

    UpdatePushCharacterVsCharacter(g);
    UpdatePushCharacterVsBackground(g);
    UpdateHitboxHurtboxCollision(g);
}

static void UpdateEndState(Game *g)
{
    Fighter_IncrementActionFrame(&g->fighter1);
    Fighter_IncrementActionFrame(&g->fighter2);

    Fighter_UpdateActionRequest(&g->fighter1);
    Fighter_UpdateActionRequest(&g->fighter2);
    Fighter_UpdateMovement(&g->fighter1);
    Fighter_UpdateMovement(&g->fighter2);
    Fighter_UpdateBoxes(&g->fighter1);
    Fighter_UpdateBoxes(&g->fighter2);

    UpdatePushCharacterVsCharacter(g);
    UpdatePushCharacterVsBackground(g);
}

static void ChangeRoundState(Game *g, RoundState state)
{
    g->roundState = state;
    switch (state) {
    case ROUND_STOP:
        if (g->fighter1RoundWon >= MAX_ROUND_WON || g->fighter2RoundWon >= MAX_ROUND_WON) {
            g->matchOver = true;
            g->matchWinner = (g->fighter1RoundWon >= MAX_ROUND_WON) ? 0 : 1;
        }
        break;
    case ROUND_INTRO:
        Fighter_SetupBattleStart(&g->fighter1, -2.0f, true);
        Fighter_SetupBattleStart(&g->fighter2, 2.0f, false);
        g->timer = INTRO_STATE_TIME;
        break;
    case ROUND_FIGHT:
        g->frameCount = -1;
        break;
    case ROUND_KO:
        g->timer = KO_STATE_TIME;
        Fighter_ClearInput(&g->fighter1);
        Fighter_ClearInput(&g->fighter2);
        break;
    case ROUND_END:
        g->timer = END_STATE_TIME;
        {
            bool dead1 = Fighter_IsDead(&g->fighter1);
            bool dead2 = Fighter_IsDead(&g->fighter2);
            if (dead1 != dead2) {
                if (dead1) {
                    g->fighter2RoundWon++;
                    Fighter_RequestWinAction(&g->fighter2);
                } else {
                    g->fighter1RoundWon++;
                    Fighter_RequestWinAction(&g->fighter1);
                }
            }
        }
        break;
    }
}

static void Game_Update(Game *g, int p1Input, int p2Input)
{
    if (g->matchOver)
        return;

    switch (g->roundState) {
    case ROUND_STOP:
        ChangeRoundState(g, ROUND_INTRO);
        break;
    case ROUND_INTRO:
        UpdateIntroState(g, p1Input, p2Input);
        g->timer -= FIXED_DT;
        if (g->timer <= 0.0f)
            ChangeRoundState(g, ROUND_FIGHT);
        break;
    case ROUND_FIGHT:
        if (g->isDebugPause)
            break;
        g->frameCount++;
        UpdateFightState(g, p1Input, p2Input);
        if (Fighter_IsDead(&g->fighter1) || Fighter_IsDead(&g->fighter2))
            ChangeRoundState(g, ROUND_KO);
        break;
    case ROUND_KO:
        g->timer -= FIXED_DT;
        if (g->timer <= 0.0f)
            ChangeRoundState(g, ROUND_END);
        break;
    case ROUND_END:
        UpdateEndState(g);
        g->timer -= FIXED_DT;
        if (g->timer <= 0.0f
            || (g->timer <= END_STATE_SKIPPABLE_TIME
                && ((p1Input & INPUT_ATTACK) || (p2Input & INPUT_ATTACK)))) {
            ChangeRoundState(g, ROUND_STOP);
        }
        break;
    }
}

/* ============================================================
 * env struct - required by pufferenv.h's contract
 * ============================================================ */

/* CPU opponent lives in bots.h (BattleAI.cs port). Included here, after
 * Game so it can read fighters and before Env so Env can embed BotMem -
 * same layout contract as ocean/robocode/robocode.h's bots.h include. */
#include "bots.h"

// Required struct. Floats only, n last
struct Log {
    float perf;             /* 1 if fighter1 (agent 0) won the round, else 0 */
    float score;            /* same as perf, kept separate for dashboard convention */
    float episode_return;   /* fighter1's reward this round */
    float episode_length;   /* frames elapsed this round */
    float n;
};

/* No Entity/Target arrays like minimal.h: Fighter already carries full
 * per-agent sim state, Game already owns both fighters + the round state
 * machine. Env wraps Game directly. Render resources are embedded fields
 * (not a separate Client*), matching minimal.h's `Texture2D sprites;`
 * pattern - lazy-inited via IsWindowReady() instead of a null check. */
struct Env {
    Log log; int num_agents; unsigned int rng; // Required
    Agent agents[AGENTS]; int tag, boundary_reached; // Required
    Game game;
    int bot_policy;   /* BOT_* from bots.h; BOT_NONE = pure self-play */
    BotMem bot_mem;   /* scratch for the scripted opponent, reset per round */
    Texture2D sheets[SHEET_COUNT];
    Texture2D bg, window, hitEffect, guardPip;
    RenderTexture2D target;
    int hitFxFrame; float hitFxX, hitFxY; int hitFxScale; int lastDamagedIndex;
    bool human; int humanInput; /* latched by puf_render, applied in puf_step */
}; // Required: An env struct

static inline void puf_set_bot_policy(Env *env, int bot_policy) {
    env->bot_policy = bot_policy;
}

/* ============================================================
 * rendering - sprites + always-on hit/hurt/push box overlay
 * (frame-advantage text HUD not ported, wasn't requested)
 * ============================================================ */

#define VIRTUAL_W 160
#define VIRTUAL_H 90
#define SCREEN_SCALE 6

#define BATTLE_X 5.0f
#define BATTLE_Y 5.0f
#define BATTLE_W 149.0f
#define BATTLE_H 60.0f

#define SPRITE_SCALE 1.0f

static inline float FightToScreenX(float x)
{
    return BATTLE_X + (x + BATTLE_AREA_WIDTH * 0.5f) * (BATTLE_W / BATTLE_AREA_WIDTH);
}

static inline float FightToScreenY(float y)
{
    return BATTLE_Y + BATTLE_H - y * (BATTLE_H / BATTLE_AREA_MAX_HEIGHT);
}

static Rectangle MotionSrcRect(const MotionSprite *ms, bool faceRight)
{
    Rectangle src = { ms->srcX, g_sheetHeights[ms->texIndex] - ms->srcY - 50.0f, 60, 50 };
    if (!faceRight)
        src.width = -60; /* mirror */
    return src;
}

static void DrawFighter(Env *env, Fighter *f)
{
    int motionID = Fighter_GetCurrentMotionID(f);
    if (motionID < 0) return;
    const MotionSprite *ms = GetMotionSprite(motionID);
    if (!ms) return;

    float sx = FightToScreenX(f->posX) + f->spriteShakePosition;
    float feetY = FightToScreenY(f->posY);

    Rectangle src = MotionSrcRect(ms, f->faceRight);
    Rectangle dst = {
        sx - 30.0f * SPRITE_SCALE,
        feetY - 50.0f * SPRITE_SCALE,
        60.0f * SPRITE_SCALE,
        50.0f * SPRITE_SCALE,
    };
    DrawTexturePro(env->sheets[ms->texIndex], src, dst, (Vector2){ 0, 0 }, 0, WHITE);
}

/* HitEffect cells in .meta order, Unity bottom-up y, 30x45 sheet */
static const int s_hitFxCells[6][2] = {
    { 15, 30 }, { 0, 15 }, { 15, 15 }, { 0, 0 }, { 15, 0 }, { 0, 30 },
};

static void UpdateHitEffect(Env *env)
{
    Game *g = &env->game;
    if (g->lastHit.active) {
        env->hitFxFrame = 0;
        env->hitFxX = FightToScreenX(g->lastHit.x);
        env->hitFxY = FightToScreenY(g->lastHit.y);
        env->hitFxScale = (g->lastHit.result == DMG_GUARD_BREAK) ? 5
                         : (g->lastHit.result == DMG_DAMAGE) ? 2 : 1;
        env->lastDamagedIndex = g->lastHit.damagedIndex;
        g->lastHit.active = false;
    }
}

static void DrawHitEffect(Env *env)
{
    if (env->hitFxFrame < 0) return;

    int cell = env->hitFxFrame / 2; /* 2 render frames per cell */
    if (cell >= 6) {
        env->hitFxFrame = -1;
        return;
    }
    Rectangle src = { (float)s_hitFxCells[cell][0], 45.0f - s_hitFxCells[cell][1] - 15.0f, 15, 15 };
    float size = 15.0f * env->hitFxScale;
    Rectangle dst = { env->hitFxX - size / 2, env->hitFxY - size / 2, size, size };
    DrawTexturePro(env->hitEffect, src, dst, (Vector2){ 0, 0 }, 0, WHITE);
    env->hitFxFrame++;
}

static void DrawFightBox(Box b, Color color)
{
    float x = FightToScreenX(b.x) - b.w * (BATTLE_W / BATTLE_AREA_WIDTH) / 2.0f;
    float w = b.w * (BATTLE_W / BATTLE_AREA_WIDTH);
    float h = b.h * (BATTLE_H / BATTLE_AREA_MAX_HEIGHT);
    float y = FightToScreenY(b.y) - h;

    Color fill = color;
    fill.a = 64;
    DrawRectangle((int)x, (int)y, (int)w, (int)h, fill);
    DrawRectangleLines((int)x, (int)y, (int)w, (int)h, color);
}

static void DebugDrawBoxes(Game *g)
{
    Fighter *fs[2] = { &g->fighter1, &g->fighter2 };
    for (int i = 0; i < 2; i++) {
        Fighter *f = fs[i];
        for (int v = 0; v < f->hurtboxCount; v++)
            DrawFightBox(f->hurtboxes[v], YELLOW);
        DrawFightBox(f->pushbox, BLUE);
        for (int h = 0; h < f->hitboxCount; h++)
            DrawFightBox(f->hitboxes[h].box, f->hitboxes[h].proximity ? GRAY : RED);
    }
}

static void DrawPips(Env *env)
{
    Game *g = &env->game;
    /* score: 3 squares per side (baked shields painted over) */
    DrawRectangle(9, 9, 32, 12, WHITE);
    DrawRectangle(119, 9, 32, 12, WHITE);
    for (int side = 0; side < 2; side++) {
        int won = side == 0 ? g->fighter1RoundWon : g->fighter2RoundWon;
        for (int i = 0; i < MAX_ROUND_WON; i++) {
            float x = side == 0 ? 11 + i * 10 : 141 - i * 10;
            Rectangle r = { x, 11, 8, 8 };
            DrawRectangleRec(r, i < won ? BLACK : WHITE);
            DrawRectangleLinesEx(r, 1, BLACK);
        }
    }
    /* guard pips over the baked square slots */
    for (int side = 0; side < 2; side++) {
        int guard = side == 0 ? g->fighter1.guardHealth : g->fighter2.guardHealth;
        for (int i = 0; i < FIGHTER_START_GUARD_HEALTH; i++) {
            if (i >= guard) continue; /* baked empty slot stays visible */
            float x = side == 0 ? 9 + i * 9 : 124 + i * 9;
            Rectangle dst = { x, 77, 10, 10 };
            DrawTexturePro(env->guardPip, (Rectangle){ 0, 0, 10, 10 }, dst, (Vector2){ 0, 0 }, 0, WHITE);
        }
    }
}

static void DrawTextShadow(const char *msg, int size, int y, Color color)
{
    int w = VIRTUAL_W * SCREEN_SCALE;
    int x = w / 2 - MeasureText(msg, size) / 2;
    DrawText(msg, x + 2, y + 2, size, BLACK);
    DrawText(msg, x, y, size, color);
}

static void DrawScreenText(Game *g)
{
    int h = VIRTUAL_H * SCREEN_SCALE;

    if (g->roundState == ROUND_KO)
        DrawTextShadow("K.O.!", 40, h / 2 - 40, RED);

    if (g->matchOver) {
        const char *msg = g->matchWinner == 0 ? "P1 WINS THE MATCH" : "P2 WINS THE MATCH";
        DrawTextShadow(msg, 40, h / 2 - 60, YELLOW);
    }
}

static void Render_Init(Env *env)
{
    for (int i = 0; i < SHEET_COUNT; i++)
        env->sheets[i] = LoadTexture(g_sheetFiles[i]);

    env->bg = LoadTexture("resources/footsies/Battle_BG.png");
    env->window = LoadTexture("resources/footsies/Battle_Window.png");
    env->hitEffect = LoadTexture("resources/footsies/HitEffect.png");
    env->guardPip = LoadTexture("resources/footsies/Guard.png");

    SetTextureFilter(env->bg, TEXTURE_FILTER_POINT);
    SetTextureFilter(env->window, TEXTURE_FILTER_POINT);
    SetTextureFilter(env->hitEffect, TEXTURE_FILTER_POINT);

    env->target = LoadRenderTexture(VIRTUAL_W, VIRTUAL_H);
    SetTextureFilter(env->target.texture, TEXTURE_FILTER_POINT);

    env->hitFxFrame = -1;
    env->lastDamagedIndex = 1;
}

static void Render_Shutdown(Env *env)
{
    for (int i = 0; i < SHEET_COUNT; i++)
        UnloadTexture(env->sheets[i]);
    UnloadTexture(env->bg);
    UnloadTexture(env->window);
    UnloadTexture(env->hitEffect);
    UnloadTexture(env->guardPip);
    UnloadRenderTexture(env->target);
}

static void Render_Draw(Env *env)
{
    UpdateHitEffect(env);

    BeginTextureMode(env->target);
    ClearBackground(BLACK);

    DrawTexture(env->bg, 0, 0, WHITE);
    DrawTexture(env->window, 2, 2, WHITE);

    /* attacker (not last damaged) drawn last so it overlaps */
    if (env->lastDamagedIndex == 0) {
        DrawFighter(env, &env->game.fighter1);
        DrawFighter(env, &env->game.fighter2);
    } else {
        DrawFighter(env, &env->game.fighter2);
        DrawFighter(env, &env->game.fighter1);
    }

    DrawHitEffect(env);
    DebugDrawBoxes(&env->game);
    DrawPips(env);

    EndTextureMode();

    BeginDrawing();
    ClearBackground(BLACK);
    Rectangle src = { 0, 0, VIRTUAL_W, -VIRTUAL_H };
    Rectangle dst = { 0, 0, VIRTUAL_W * SCREEN_SCALE, VIRTUAL_H * SCREEN_SCALE };
    DrawTexturePro(env->target.texture, src, dst, (Vector2){ 0, 0 }, 0, WHITE);

    DrawScreenText(&env->game);

    EndDrawing();
}

/* ============================================================
 * required contract functions
 * ============================================================ */

static inline float footsies_get_float(Dict *kwargs, const char *key, float default_value)
{
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, key) == 0)
            return (float)kwargs->items[i].value;
    }
    return default_value;
}

void puf_init(Env *env, Dict *kwargs)
{
    env->num_agents = AGENTS;
    for (int i = 0; i < AGENTS; i++) {
        /* slot 1 is the historical opponent in selfplay-pool envs (tag > 0),
         * same layout as robocode; ignored when num_policies == 1. */
        env->agents[i].policy = i;
        env->agents[i].action_mask = NULL;
    }
    env->bot_policy = (int)footsies_get_float(kwargs, "bot_policy", BOT_NONE);
}

void puf_log(Log *log, Dict *out)
{
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

static inline int decode_input(int move, int attack)
{
    int input = INPUT_NONE;
    if (move == 1) input |= INPUT_LEFT;
    else if (move == 2) input |= INPUT_RIGHT;
    if (attack == 1) input |= INPUT_ATTACK;
    return input;
}

/* [0, 1]: 0 = touching either wall, 1 = dead center. posX alone already
 * says where a fighter is on the stage, but burying "am I near a corner"
 * inside a raw position value makes the network re-derive the stage
 * bounds itself; this hands it over directly since corner proximity is
 * a first-order footsies concept (no backward retreat left = forced
 * block/whiff-punish situations). */
static float DistToNearestWall(float posX)
{
    float stageMinX = BATTLE_AREA_WIDTH * -0.5f;
    float stageMaxX = BATTLE_AREA_WIDTH * 0.5f;
    float distLeft = posX - stageMinX;
    float distRight = stageMaxX - posX;
    return (distLeft < distRight ? distLeft : distRight) / (BATTLE_AREA_WIDTH * 0.5f);
}

static void write_agent_obs(obs_t *obs, Fighter *self, Fighter *opp)
{
    int idx = 0;
    const ActionData *selfAct = GetActionData(self->currentActionID);
    obs[idx++] = self->posX / (BATTLE_AREA_WIDTH / 2.0f);
    obs[idx++] = self->posY / BATTLE_AREA_MAX_HEIGHT;
    obs[idx++] = self->velX / FIGHTER_FORWARD_SPEED;
    obs[idx++] = self->faceRight ? 1.0f : -1.0f;
    obs[idx++] = (float)self->vitalHealth;
    obs[idx++] = self->guardHealth / (float)FIGHTER_START_GUARD_HEALTH;
    obs[idx++] = (float)selfAct->type / (float)ATYPE_GUARD;
    obs[idx++] = (float)self->currentActionFrame / fmaxf(1.0f, (float)Fighter_ActionFrameCount(self));
    obs[idx++] = self->currentHitStunFrame / 30.0f; /* 30 = guardBreakStunFrame max */
    obs[idx++] = DistToNearestWall(self->posX);

    const ActionData *oppAct = GetActionData(opp->currentActionID);
    obs[idx++] = opp->posX / (BATTLE_AREA_WIDTH / 2.0f);
    obs[idx++] = opp->posY / BATTLE_AREA_MAX_HEIGHT;
    obs[idx++] = opp->velX / FIGHTER_FORWARD_SPEED;
    obs[idx++] = opp->faceRight ? 1.0f : -1.0f;
    obs[idx++] = (float)opp->vitalHealth;
    obs[idx++] = opp->guardHealth / (float)FIGHTER_START_GUARD_HEALTH;
    obs[idx++] = (float)oppAct->type / (float)ATYPE_GUARD;
    obs[idx++] = (float)opp->currentActionFrame / fmaxf(1.0f, (float)Fighter_ActionFrameCount(opp));
    obs[idx++] = opp->currentHitStunFrame / 30.0f;
    obs[idx++] = DistToNearestWall(opp->posX);

    obs[idx++] = (opp->posX - self->posX) / BATTLE_AREA_WIDTH;
}

void compute_observations(Env *env)
{
    write_agent_obs(env->agents[0].observations, &env->game.fighter1, &env->game.fighter2);
    write_agent_obs(env->agents[1].observations, &env->game.fighter2, &env->game.fighter1);
}

static void reset_match(Env *env)
{
    Game_Init(&env->game);
    bot_mem_reset(&env->bot_mem);
    env->hitFxFrame = -1;
    env->lastDamagedIndex = 1;
}

void puf_reset(Env *env)
{
    reset_match(env);
    compute_observations(env);
}

/* footsies has no native round timer - Game's `timer` field only covers the
 * INTRO/KO/END presentation delays, ROUND_FIGHT itself runs until someone
 * dies. Fine for two humans, not fine for an untrained policy that never
 * presses attack: without a cap that episode runs forever. Cross-checked
 * against ocean/robocode/robocode.h in the 4.0 port, which hits the same
 * issue and bounds it with max_ticks -> end_episode(env, 0) (a draw). */
#define MAX_EPISODE_FRAMES 3000  /* 60s at 50Hz, well past any real round */

/* Episode = one round, and the game keeps its own flow: after a KO,
 * Game_Update shows K.O. and the win pose, scores the round and starts the
 * next one; first to MAX_ROUND_WON takes the match. Nothing resets the env
 * from outside (5.0's trainer and eval loops call puf_reset only at
 * startup), so:
 * - terminal fires only on the step the fight ends. The loser stays dead
 *   through ROUND_KO/ROUND_END (~250 steps); keying on "is anyone dead"
 *   re-fired terminal + reward on every one of those steps.
 * - a timeout has no KO, so the round is replayed (score kept), which also
 *   restarts frameCount.
 * - a finished match is reset here: Game_Update's `if (matchOver) return;`
 *   would otherwise freeze the sim for good. */
void puf_step(Env *env)
{
    if (env->game.matchOver) {
        reset_match(env);
    }

    float *p1act = env->agents[0].actions;
    float *p2act = env->agents[1].actions;
    int input1 = env->human ? env->humanInput : decode_input((int)p1act[0], (int)p1act[1]);
    int input2 = decode_input((int)p2act[0], (int)p2act[1]);
    if (env->bot_policy >= 0 && (env->game.roundState == ROUND_INTRO
            || env->game.roundState == ROUND_FIGHT)) {
        /* BattleCore.cs fetches AI input only from UpdateIntroState /
         * UpdateFightState; the bot stands frozen (queues intact) through
         * KO/END, and a fresh BattleAI is constructed at every Intro, so
         * reset its memory on the STOP -> INTRO transition below. */
        input2 = footsies_bot_input(&env->game, &env->bot_mem, &env->rng);
    }
    RoundState prevState = env->game.roundState;
    Game_Update(&env->game, input1, input2);
    if (env->bot_policy >= 0 && prevState != ROUND_INTRO
            && env->game.roundState == ROUND_INTRO)
        bot_mem_reset(&env->bot_mem);

    env->agents[0].rewards[0] = 0.0f;
    env->agents[1].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    env->agents[1].terminals[0] = 0.0f;

    bool dead1 = Fighter_IsDead(&env->game.fighter1);
    bool dead2 = Fighter_IsDead(&env->game.fighter2);
    bool ko = prevState == ROUND_FIGHT && env->game.roundState == ROUND_KO;
    bool timedOut = env->game.roundState == ROUND_FIGHT && env->game.frameCount >= MAX_EPISODE_FRAMES;

    if (ko || timedOut) {
        env->agents[0].rewards[0] = dead1 ? -1.0f : (dead2 ? 1.0f : 0.0f);
        env->agents[1].rewards[0] = dead2 ? -1.0f : (dead1 ? 1.0f : 0.0f);
        env->agents[0].terminals[0] = 1.0f;
        env->agents[1].terminals[0] = 1.0f;

        /* perf/score/n all accumulate (matches minimal.h's own +=/++ style)
         * so the generic sum-then-divide-by-n aggregation produces a
         * correct mean win rate even when an instance completes several
         * episodes between drains. */
        env->log.perf += dead2 ? 1.0f : 0.0f;
        env->log.score += dead2 ? 1.0f : 0.0f;
        env->log.episode_return += env->agents[0].rewards[0];
        env->log.episode_length += (float)env->game.frameCount;
        env->log.n++;
        if (timedOut)
            ChangeRoundState(&env->game, ROUND_STOP);
    }

    compute_observations(env);
}

/* Hold Left Shift to play P1 with Footsies_raylib's P1 keys (src/main.c):
 * A/D move, Space attack. P2 stays on the policy. Keys are only latched
 * here: ./puffer eval runs the policy forward between puf_render and
 * puf_step, which overwrites anything written into actions, so puf_step
 * swaps in P1's input instead. Same INPUT_* bits as the standalone game
 * (A+D = LEFT|RIGHT = STAND), so dash and hold-to-special match it too. */
static void footsies_human_controls(Env *env)
{
    env->human = IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT);
    if (!env->human)
        return;
    int input = INPUT_NONE;
    if (IsKeyDown(KEY_A)) input |= INPUT_LEFT;
    if (IsKeyDown(KEY_D)) input |= INPUT_RIGHT;
    if (IsKeyDown(KEY_SPACE)) input |= INPUT_ATTACK;
    env->humanInput = input;
}

void puf_render(Env *env)
{
    if (!IsWindowReady()) {
        InitWindow(VIRTUAL_W * SCREEN_SCALE, VIRTUAL_H * SCREEN_SCALE, "Footsies");
        SetTargetFPS(PUF_STEPS_PER_SEC); /* ./puffer eval: 1 step per frame */
        Render_Init(env);
    }
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    footsies_human_controls(env);
    Render_Draw(env);
    puf_web_vsync();
}

void puf_close(Env *env)
{
    if (IsWindowReady()) {
        Render_Shutdown(env);
        CloseWindow();
    }
}
