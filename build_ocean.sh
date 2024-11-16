#!/bin/bash

# Usage: ./build_env.sh pong [local|web]

ENV=$1
MODE=${2:-local}

SRC_DIR="pufferlib/environments/ocean/$ENV"
OUTPUT_DIR="."
WEB_OUTPUT_DIR="build_web/$ENV"
RESOURCES_DIR="resources"

# Create build output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WEB_OUTPUT_DIR"

if [ "$MODE" = "local" ]; then
    echo "Building $ENV for local testing..."
    clang -g -O0 -Wall \
        -I./raylib/include \
        -I./pufferlib\
        "$SRC_DIR/$ENV.c" -o "$OUTPUT_DIR/$ENV" \
        ./raylib/lib/libraylib.a -lm -lpthread \
        -fsanitize=address,undefined,bounds,pointer-overflow,leak \
        #-framework Cocoa -framework IOKit -framework CoreVideo
    # gcc -g -O2 -Wall \
    #     -I./raylib/include \
    #     -I./pufferlib\
    #     "$SRC_DIR/$ENV.c" -o "$OUTPUT_DIR/$ENV" \
    #     ./raylib/lib/libraylib.a -lm -lpthread \
    #     -fsanitize=address,undefined,bounds,pointer-overflow,leak \
elif [ "$MODE" = "fast" ]; then
    echo "Building $ENV for local testing..."
    clang -pg -O2 -Wall \
        -I./raylib/include \
        -I./pufferlib\
        "$SRC_DIR/$ENV.c" -o "$OUTPUT_DIR/$ENV" \
        ./raylib/lib/libraylib.a -lm -lpthread \
        #-framework Cocoa -framework IOKit -framework CoreVideo
 
    echo "Built to: $OUTPUT_DIR/$ENV"
elif [ "$MODE" = "web" ]; then
    echo "Building $ENV for web deployment..."

    PRELOAD=""
    if [ -d "$RESOURCES_DIR" ]; then
        PRELOAD="--preload-file $RESOURCES_DIR@resources/"
    fi

    echo "Preloading resources from $RESOURCES_DIR"

    emcc \
        -o "$WEB_OUTPUT_DIR/game.html" \
        "$SRC_DIR/$ENV.c" \
        -Os \
        -Wall \
        ./raylib_wasm/lib/libraylib.a \
        -I./raylib_wasm/include \
        -I./pufferlib\
        -L. \
        -L./raylib_wasm/lib \
        -sASSERTIONS=2 \
        -gsource-map \
        -s USE_GLFW=3 \
        -s USE_WEBGL2=1 \
        -s ASYNCIFY \
        -sFILESYSTEM \
        -s FORCE_FILESYSTEM=1 \
        --shell-file ./minshell.html \
        -sINITIAL_MEMORY=512MB \
        -sSTACK_SIZE=512KB \
        -DPLATFORM_WEB \
        -DGRAPHICS_API_OPENGL_ES3 $PRELOAD
    echo "Web build completed: $WEB_OUTPUT_DIR/game.html"
else
    echo "Invalid mode specified. Use 'local' or 'web'."
    exit 1
fi

