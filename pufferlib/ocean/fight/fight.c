#include "fight.h"
#include "puffernet.h"
#include <stdlib.h>

int main() {
    int num_obs = 18; // hard coded
    Weights *weights =
        load_weights("resources/fight/fight_weights.bin", 137743);
    int logit_sizes[1] = {8};
    LinearLSTM *net = make_linearlstm(weights, 2, num_obs, logit_sizes, 2);

    Fight env = {
        .screen_width = 960,
        .screen_height = 670,
    };
    init(&env);

    env.observations = calloc(12, sizeof(float));
    env.actions = calloc(8, sizeof(int));
    env.rewards = calloc(2, sizeof(float));
    env.terminals = calloc(2, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        for (int i = 0; i < 2; i++) {
            env.actions[i] = rand() % 9;
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
