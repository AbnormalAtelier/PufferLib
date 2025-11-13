#include "fight.h"
#include "puffernet.h"
#include <stdlib.h>

int main() {
    int num_obs = 16; // hard coded because only 2 fighters, 8 per fighter
    Weights *weights =
        load_weights("resources/fight/fight_weights.bin", 137743);
    int logit_sizes[3] = {2, 2, 2};
    LinearLSTM *net = make_linearlstm(weights, 2, num_obs, logit_sizes, 2);

    Fight env = {
        .width = 960,
        .height = 670,
    };
    init(&env);

    env.observations = calloc(16, sizeof(float));
    env.actions =
        calloc(6, sizeof(int)); // 3 actions per fighter, move, attack, jump
    env.rewards = calloc(2, sizeof(float));
    env.terminals = calloc(2, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        for (int i = 0; i < 2; i++) {
            env.actions[3 * i] = rand() % 2;
            env.actions[3 * i + 1] = rand() % 2;
            env.actions[3 * i + 2] = rand() % 2;
        }

        forward_linearlstm(net, env.observations, env.actions);
        c_step(&env);
        c_render(&env);
    }

    // Try to clean up after yourself
    free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
