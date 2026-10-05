#!/bin/bash

EXP_NAME=$1
ASSEMBLY=$2
SETUP=${3:-panda}
ASSEMBLY_DIR=${4:-fabrica}

ARM=""
GRIPPER=""
FT_SENSOR=""

if [ "$SETUP" == "panda" ]; then
  ARM="panda"
  GRIPPER="panda"
  FT_SENSOR="none"
elif [ "$SETUP" == "panda-robotiq" ]; then
  ARM="panda"
  GRIPPER="robotiq-140"
  FT_SENSOR="none"
elif [ "$SETUP" == "xarm7" ]; then
  ARM="xarm7"
  GRIPPER="robotiq-140"
  FT_SENSOR="move"
elif [ "$SETUP" == "ur5e" ]; then
  ARM="ur5e"
  GRIPPER="robotiq-85"
  FT_SENSOR="none"
else
  echo "Error: Unsupported SETUP value '$SETUP'. Please use 'panda' or 'xarm7' or 'ur5e'."
  exit 1
fi

export OMP_NUM_THREADS=1

echo "Running precedence and path planning..."
python planning_original/run_preced_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/original --num-proc 12 --arm $ARM

echo "Running grasp and arm IK generation..."
python planning_original/run_grasp_arm_gen.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/original --num-proc 50 --max-n-grasp 140 --arm $ARM --gripper $GRIPPER --ft-sensor $FT_SENSOR

echo "Running sequence planning..."
python planning_original/run_seq_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/original --plot

echo "Running sequence optimization..."
python planning_original/run_seq_opt.py --log-dir logs/$EXP_NAME/$ASSEMBLY/original --plot

echo "Running fixture generation..."
python planning_original/run_fixture_gen.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/original --optimized

echo "Running complete motion planning..."
python planning_original/run_motion_plan.py --assembly-dir assets/$ASSEMBLY_DIR/$ASSEMBLY --log-dir logs/$EXP_NAME/$ASSEMBLY/original --optimized

echo "Done."
