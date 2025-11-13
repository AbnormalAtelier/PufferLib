#include "flappy.h"

int main() {
  Bird env = {.observations =
                  (unsigned char *)calloc(10, sizeof(unsigned char)),
              .actions = (int *)calloc(1, sizeof(int)),
              .rewards = (float *)calloc(1, sizeof(float)),
              .terminals = (unsigned char *)calloc(1, sizeof(unsigned char))};

  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      env.actions[0] = 0;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.actions[0] = UP;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.actions[0] = DOWN;

    } else {
      env.actions[0] = rand() % 2;
    }
    c_step(&env);
    c_render(&env);
  }
  free(env.observations);
  free(env.actions);
  free(env.rewards);
  free(env.terminals);
  c_close(&env);
}
