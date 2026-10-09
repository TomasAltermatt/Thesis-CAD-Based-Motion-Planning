# Thesis-CAD-Based-Motion-Planning

## 💻 Planning
Planning consists of 6 stages: precedence planning, grasp planning, sequence planning, sequence optimization, fixture generation, and arm motion planning. You can run either the `planning/`/`planning_original/` directory depending if the original pipeline is run or the new

To run the original pipeline:
```bash
bash ./planning_original/run_planning.sh EXP_NAME ASSEMBLY_NAME ARM
```


To run the new pipeline:
```bash
bash ./planning/run_planning.sh EXP_NAME ASSEMBLY_NAME ARM
```


## 💻 Results and Plots

All of these results are saved onto the ```results``` folder

### Single Study Results

This gives csv logs for all the data (joint travel distances, manipulability, move/hold scores), plots for the global manipulabilities for each assembly, and the separate plots for each part along each insertion movements:

```bash
python result_scripts/run_all_results.py EXP_NAME
```

### Ablation Studies
This is to compare joint angle differences, move/hold scores between the original pipeline and various experiments:
```bash
python result_scripts/ablation_study.py
```