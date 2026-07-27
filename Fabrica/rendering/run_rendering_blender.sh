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
ASSEMBLY=${POSITIONAL[1]}
ASSEMBLY_DIR=${POSITIONAL[2]:-fabrica}

if [ "$HEADLESS" == true ]; then
    xvfb-run -s "-screen 0 1920x1080x24" python rendering/render_motion_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY
    xvfb-run -s "-screen 0 1920x1080x24" python rendering/render_traj_blender.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY --record-path records/blender/$EXP_NAME/${ASSEMBLY}.mp4 --verbose
else
    python rendering/render_motion_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY
    python rendering/render_traj_blender.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY --record-path records/blender/$EXP_NAME/${ASSEMBLY}.mp4 --verbose
fi
