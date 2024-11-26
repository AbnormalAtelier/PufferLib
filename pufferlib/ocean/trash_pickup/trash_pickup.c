#include <time.h>
#include "trash_pickup.h"

#define INDEX_C(env, x, y) ((y) * (env).grid_size + (x))

// Demo function for visualizing the TrashPickupEnv
void demo(int grid_size, int num_agents, int num_trash, int num_bins, int max_steps) {
    CTrashPickupEnv env = {
        .grid_size = grid_size,
        .num_agents = num_agents,
        .num_trash = num_trash,
        .num_bins = num_bins,
        .max_steps = max_steps,
        .agent_sight_range = 5,
        .do_human_control = true
    };

    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);

    while (!WindowShouldClose()) {
        // Random actions for all agents
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[i] = rand() % 4; // 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
            // printf("action: %d \n", env.actions[i]);
        }

        // Override human control actions
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Handle keyboard input only for selected agent
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                env.actions[0] = ACTION_UP;
            }
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.actions[0] = ACTION_LEFT;
            }
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.actions[0] = ACTION_RIGHT;
            }
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                env.actions[0] = ACTION_DOWN;
            }
        }

        // Step the environment and render the grid
        step(&env);
        render(client, &env);
        usleep(500000);
    }

    free_allocated(&env);
    close_client(client);
}



// Performance test function for benchmarking
void performance_test() {
    long test_time = 10; // Test duration in seconds

    CTrashPickupEnv env = {
        .grid_size = 10,
        .num_agents = 4,
        .num_trash = 15,
        .num_bins = 1,
        .max_steps = 300,
        .agent_sight_range = 5
    };
    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    int inc = env.num_agents;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 4;
        step(&env);
        i += inc;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}


// Main entry point
int main() {
    demo(10, 3, 15, 1, 500); // Visual demo
    // performance_test(); // Uncomment for benchmarking
    return 0;
}