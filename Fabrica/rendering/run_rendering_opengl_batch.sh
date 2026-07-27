#!/bin/bash

HEADLESS=false
POSITIONAL=()
for arg in "$@"; do
    if [ "$arg" == "--headless" ] || [ "$arg" == "headless" ]; then
        HEADLESS=true
    else
        POSITIONAL+=("$arg")
    fi
done

EXP_NAME=${POSITIONAL[0]}
ASSEMBLY_DIR=${POSITIONAL[1]:-fabrica}

if [ "$HEADLESS" == true ]; then
    xvfb-run -s "-screen 0 1920x1080x24" python rendering/render_motion_plan_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --record-dir records/opengl/$EXP_NAME --num-proc 12
else
    python rendering/render_motion_plan_batch.py --assembly-dir assets/$ASSEMBLY_DIR --log-dir logs/$EXP_NAME --record-dir records/opengl/$EXP_NAME --num-proc 12
fi
