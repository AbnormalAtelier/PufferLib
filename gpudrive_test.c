#include "gpudrive_test.h"
#include <time.h>

void demo(){
    long test_time = 10;
    test_struct env = {
        .num_agents = 128,
        .active_agents =10,
    };
    init(&env);
    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        step(&env);
        // return;
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (i * env.active_agents) / (end - start));
    free_initialized(&env);

}

int main() {
    demo();
    return 0;
}