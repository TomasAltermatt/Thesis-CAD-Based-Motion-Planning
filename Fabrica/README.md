# 🏭 Fabrica

<p>
  <strong>Fabrica: Dual-Arm Assembly of General Multi-Part Objects via Integrated Planning and Learning</strong><br>
  <strong>[CoRL 2025, Best Paper Award]</strong>
</p>

<a href="http://fabrica.csail.mit.edu">
  <img src="https://img.shields.io/badge/project-website-green.svg" alt="Project Page"/>
</a> <a href="https://arxiv.org/abs/2506.05168">
  <img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="arXiv Paper"/>
</a> <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"/>
</a> <a href="https://x.com/YunshengTian/status/1971853081504878748">
  <img src="https://img.shields.io/badge/twitter-YunshengTian-blue.svg" alt="Twitter"/>
</a> <a href="https://fabrica.csail.mit.edu/static/videos/video_main.mp4">
  <img src="https://img.shields.io/badge/video-overview-orange.svg" alt="Intro Video"/>
</a> <a href="https://www.youtube.com/live/rh2oxU1MCb0?t=21118s">
  <img src="https://img.shields.io/badge/video-talk-orange.svg" alt="Talk Video"/>
</a>

<p>
  <img src="images/teaser.gif" alt="Fabrica teaser" width="640">
</p>

**Fabrica** is an autonomous robotic assembly system capable of planning and executing multi-step contact-rich assembly of general objects without human demonstrations.

## 🔧 Installation

### 1. Clone repository

```bash
git clone --recurse-submodules git@github.com:yunshengtian/Fabrica.git
```

### 2. Create Python environment

```bash
conda env create -f environment.yml
conda activate fabrica
```

or

```bash
sudo apt-get install graphviz graphviz-dev # for linux
brew install graphviz # for mac
pip install -r requirements.txt
```

### 3. Install simulation for planning

```bash
pip install ./simulation
```

To test if the installation steps are successful, run:

```bash
python simulation/test/test_simple_sim.py --model box/box_stack
```

[Here](https://github.com/yunshengtian/Assemble-Them-All?tab=readme-ov-file#simulation-viewer) are some tips on interacting with the simulation viewer. 

To visualize the simulation of our beam assembly in the benchmark, run:

```bash
python simulation/test/test_multi_sim.py --dir fabrica --id beam
```

### 4. Install renderer (optional)

```bash
pip install ./rendering
```

Note: This requires Python 3.11, 3.10, or 3.7 due to the Blender Python's compatibility.

If installing `bpy` fails, you can try to install it directly from the [wheels](https://download.blender.org/pypi/bpy/).

### 5. Install Isaac Gym for learning

Note: It's recommended to install this step in a separate conda environment to avoid conflicts with other packages. For example:

```bash
conda create -n isaacgym python=3.10
conda activate isaacgym
```

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. Once Isaac Gym is installed, run:

```bash
pip install ./learning
```

### 6. Install Frankapy for real-robot experiments

Follow the installation instructions in [frankapy](https://github.com/iamlab-cmu/frankapy). Recommended to install in the same conda environment as Isaac Gym.

## 💻 Experiments

### 1. Planning multi-part assembly processes

Planning consists of 6 stages: precedence planning, grasp planning, sequence planning, sequence optimization, fixture generation, and arm motion planning. We provide separate scripts for each one under `planning/` directory, but also a single bash script that automates the whole pipeline:

```bash
bash ./planning/run_planning.sh EXP_NAME ASSEMBLY_NAME
```

For parallel batch planning over multiple assemblies, run:

```bash
bash ./planning/run_planning_batch.sh EXP_NAME
```

Note: To modify the robot setup, refer to `planning/robot/workcell.py` and `planning/robot/geometry.py` to add your custom settings. 

### 2. Learning two-part assembly policies

Prepare the assets from planning output first:

```bash
bash ./learning/preprocessing/prepare_isaac.sh EXP_NAME
```

Next, enter `learning/isaacgymenvs`.

Train a specialist policy for a given assembly (e.g., beam):
```bash
python train.py task=FabricaTaskAssemble task.env.assemblies=["beam"]
```

Train a generalist policy for all assemblies:
```bash
python train.py task=FabricaTaskAssemble
```

Note: The first time you run these examples, it may take a long time for Gym to generate signed distance field representations (SDFs) for the assets. However, these SDFs will then be cached (i.e., the next time you run the same examples, it will load the previously generated SDFs).

Other useful command line arguments:
  - To run the examples **without rendering**, add: `headless=True`
  - To resume training from a specific **checkpoint**, add: `checkpoint=[path to checkpoint]`
  - To **test** a trained policy, add: `checkpoint=[path to trained policy checkpoint] test=True`
  - To change the number of parallelized environments, add: `task.env.numEnvs=[number of environments]` 
  - To set a random seed for RL training, add: `seed=-1`, to set a specific seed, add `seed=[seed number]`
  - To set maximum number of iterations for RL training, add: `max_iterations=[number of max RL training iterations]`
  - To test a policy, add: `test=True task.env.if_eval=True checkpoint=[path to trained policy checkpoint]`

### 3. Running real-robot experiments with integrated planning and learning

To prepare all parts and fixtures for 3D printing, run:

```bash
python real_robot/prepare_parts_printing.py --assembly-dir assets/fabrica --printing-dir printing
python real_robot/prepare_fixture_printing.py --log-dir logs/EXP_NAME --printing-dir printing
```

The exported STL files will be saved under `printing/ASSEMBLY_NAME/`.

Accurately calibrating the robot setup is crucial to the success of real-robot experiments. Please adjust [this pose](https://github.com/yunshengtian/Fabrica/blob/main/real_robot/robot_interface.py#L368) accordingly and ensure that your real-robot setup aligns well with it.

To run the real-robot experiments on Franka Emika Panda, run:

```bash
python real_robot/run.py --residual --fn logs/EXP_NAME/ASSEMBLY_NAME/motion.pkl --checkpoint-path CHECKPOINT_PATH
```

`CHECKPOINT_PATH` is the path to the trained policy checkpoint.

To use VLMs for error recovery, add `--vlm` and `--video-dir VIDEO_DIR` arguments to the command, where `VIDEO_DIR` is the path to store temporary videos for VLMs to analyze.

## 🎥 Rendering

To render the assembly process after planning, we have two renderers: Blender and OpenGL. Blender is of higher quality but slower, while OpenGL is faster but lower quality.

```bash
bash rendering/run_rendering_blender.sh EXP_NAME ASSEMBLY_NAME
bash rendering/run_rendering_opengl.sh EXP_NAME ASSEMBLY_NAME
```

To render in headless mode, add `--headless` argument to the command.

To render multiple assemblies in batch, run:

```bash
bash rendering/run_rendering_blender_batch.sh EXP_NAME
bash rendering/run_rendering_opengl_batch.sh EXP_NAME
```

## 📧 Contact

Please feel free to contact yunsheng@csail.mit.edu or create a GitHub issue for any questions. Due to limited maintenance bandwidth, we do not anticipate significant changes or feature enhancements to this repository; however, we hope it will serve as a useful reference and are happy to engage in discussion.

## 📚 Citation

If you find our paper, code or dataset is useful, please consider citing:

```bibtex
@inproceedings{tian2025fabrica,
  title={Fabrica: Dual-Arm Assembly of General Multi-Part Objects via Integrated Planning and Learning},
  author={Yunsheng Tian and Joshua Jacob and Yijiang Huang and Jialiang Zhao and Edward Li Gu and Pingchuan Ma and Annan Zhang and Farhad Javid and Branden Romero and Sachin Chitta and Shinjiro Sueda and Hui Li and Wojciech Matusik},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025},
  url={https://openreview.net/forum?id=aSUNzvEJIf}
}
```
