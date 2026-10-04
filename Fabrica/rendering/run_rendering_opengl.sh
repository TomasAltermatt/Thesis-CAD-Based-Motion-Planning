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
# If 3 arguments are passed, treat $3 as PIPELINE and default ASSEMBLY_DIR to fabrica
if [ ${#POSITIONAL[@]} -eq 3 ]; then
    ASSEMBLY_DIR="fabrica"
    PIPELINE=${POSITIONAL[2]}
else
    ASSEMBLY_DIR=${POSITIONAL[2]:-fabrica}
    PIPELINE=${POSITIONAL[3]}
fi


if [ "$HEADLESS" == true ]; then
    xvfb-run -s "-screen 0 1920x1080x24" python rendering/render_motion_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/$PIPELINE --record-path records/opengl/$EXP_NAME/${ASSEMBLY}/${PIPELINE}.mp4
else
    python rendering/render_motion_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/$PIPELINE --record-path records/opengl/$EXP_NAME/${ASSEMBLY}/${PIPELINE}.mp4
fi
