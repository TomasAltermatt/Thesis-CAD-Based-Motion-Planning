

# Fabrica RL Environment

Adapted code from [AutoMate](https://bingjietang718.github.io/automate/) to train policies for multi-part assemblies.


## Overview

There are 2 tasks:

 - **FabricaTaskAsset** provides empty simulation environments where the assemblies are loaded without any RL training code. 
 - **FabricaTaskAssemble** trains policies for assembling parts without the plug fixed to the gripper.
 - **FabricaFixPlugTaskAssemble** trains policies for assembling parts with the plug fixed to the gripper (used in the paper for main experiments).


## Running the Examples

 - Enter `isaacgymenvs/isaacgymenvs`
 - Run the example:
	 - **Load assemblies** without any RL training code:
	```bash
	python train.py task=FabricaTaskAsset
	```
	 - **Load a specified list of assemblies** without any RL training code (e.g., beam): 
	```bash
	python train.py task=FabricaTaskAsset task.env.assemblies=["beam"]
	```
	 - **Train an assembly policy** for a given assembly (e.g., beam):
	```bash
	python train.py task=FabricaFixPlugTaskAssemble task.env.assemblies=["beam"]
	```
- **NOTE**: The first time you run these examples, it may take a long time for Gym to generate signed distance field representations (SDFs) for the assets. However, these SDFs will then be cached (i.e., the next time you run the same examples, it will load the previously generated SDFs).
- Other useful command line arguments:
	 - To run the examples **without rendering**, add: `headless=True`
	 - To resume training from a specific **checkpoint**, add: `checkpoint=[path to checkpoint]`
	 - To **test** a trained policy, add: `checkpoint=[path to trained policy checkpoint] test=True`
	 - To change the number of parallelized environments, add: `task.env.numEnvs=[number of environments]` 
	 - To set a random seed for RL training, add: `seed=-1`, to set a specific seed, add `seed=[seed number]`
	 - To set maximum number of iterations for RL training, add: `max_iterations=[number of max RL training iterations]`
	 - To test a policy, add: `test=True task.env.if_eval=True checkpoint=[path to trained policy checkpoint]`


## Core Code Details

The **core task files** are: 
 - For *FabricaTaskAsset* task (no RL training): 
   - Class file: [fabrica_task_asset.py](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/tasks/fabrica/fabrica_task_asset.py)
   - Task configuration file: [FabricaTaskAsset.yaml](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/cfg/task/FabricaTaskAsset.yaml) 
 - For *FabricaTaskAssemble* task: 
   - Class file: [fabrica_task_assemble.py](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py)
   - Task configuration file: [FabricaTaskAssemble.yaml](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/cfg/task/FabricaTaskAssemble.yaml)  
   - Training configuration file: [FabricaTaskAssemblePPO.yaml](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/cfg/train/FabricaTaskAssemblePPO.yaml)
- For *FabricaFixPlugTaskAssemble* task:
   - Class file: [fabrica_fixplug_task_assemble.py](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/tasks/fabrica/fabrica_fixplug_task_assemble.py)
   - Task configuration file: [FabricaFixPlugTaskAssemble.yaml](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/cfg/task/FabricaFixPlugTaskAssemble.yaml)  
   - Training configuration file: [FabricaFixPlugTaskAssemblePPO.yaml](https://github.com/yunshengtian/Fabrica/blob/main/learning/isaacgymenvs/cfg/train/FabricaFixPlugTaskAssemblePPO.yaml)