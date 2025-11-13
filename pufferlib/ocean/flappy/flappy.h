#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char BLOCK = 2;

const unsigned char UP = 1;
const unsigned char DOWN = 0;

typedef struct {
    float perf;           // Performance metric, normalized between 0 and 1
    float score;          // Unnormalized score metric
    float episode_return; // Total rewards accumulated in the episode
    float episode_length; // Number of steps in the episode
    float n;              // Required as the last field for logging purposes
} Log;

typedef struct {
    Log log;                     // Required field for logging
    unsigned char *observations; // Observations of the environment
    int *actions;                // Actions taken by the agent
    float *rewards;              // Rewards received by the agent
    unsigned char *terminals;    // Terminal states of the environment
    int tick;                    // Current tick or step in the environment
    int pos_bird;  // Bird agent stays at first column and only flips the row
    int pos_block; // Position of the block in the last column
} Bird;

void add_log(Bird *env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void c_reset(Bird *env) {
    memset(env->observations, 0, 10 * sizeof(unsigned char));
    env->observations[0] = AGENT;
    env->pos_bird = 0;
    env->tick = 0;
    env->pos_block =
        (rand() % 2) * 5 +
        4; // the block randomly appears at last column, first or second row
    env->observations[env->pos_block] =
        BLOCK; // Place the block in the last column
}

void c_step(Bird *env) {
    env->tick += 1;
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->observations[env->pos_bird] = EMPTY;
    env->observations[env->pos_block] = EMPTY;

    if (env->pos_block == 0 ||
        env->pos_block == 5) // Check if the block is in the first or second row
    {
        env->pos_block =
            (rand() % 2) * 5 + 4; // Move the block to the last column, randomly
                                  // in first or second row
    } else {
        env->pos_block -= 1; // Move the block one step to the left
    }

    if (action == UP) {
        env->pos_bird = 5;
    } else {
        env->pos_bird = 0;
    }

    env->observations[env->pos_block] = BLOCK;

    if (env->observations[env->pos_bird] == BLOCK) {
        env->terminals[0] = 1;  // Game over if the bird collides with the block
        env->rewards[0] = -1.0; // Negative reward for collision
        add_log(env);
        c_reset(env); // Reset the environment
        return;
    }

    env->observations[env->pos_bird] = AGENT; // Update the bird's position
}

void c_render(Bird *env) {
    if (!IsWindowReady()) {
        InitWindow(400, 200, "Flappy Bird Environment");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);

    // Draw grid
    int cell_size = 80;
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 5; col++) {
            DrawRectangleLines(col * cell_size, row * cell_size, cell_size,
                               cell_size, LIGHTGRAY);
        }
    }

    // Draw bird
    DrawRectangle(10, env->pos_bird * cell_size + 10, cell_size - 20,
                  cell_size - 20, BLUE);

    // Draw block (pipe)
    int block_col = env->pos_block % 5;
    int block_row = env->pos_block >= 5 ? 1 : 0;
    DrawRectangle(block_col * cell_size + 10, block_row * cell_size + 10,
                  cell_size - 20, cell_size - 20, GREEN);

    // Draw score
    char scoreText[32];
    sprintf(scoreText, "Score: %.1f", env->log.score);
    DrawText(scoreText, 10, 170, 20, BLACK);

    EndDrawing();
}

void c_close(Bird *env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
