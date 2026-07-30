# Recency: Report Generation Notes

When generating the next evaluation report (or running `generate_report.py`), please ensure the script is pointing to the correct, updated directories for the latest iteration of the project.

## Directory Structure Changes to Note:
- **Latest Model Checkpoints**: The most recently trained models (incorporating Task 3 updates) are saved in `assets/checkpoints/latest model/` rather than the root checkpoints folder.
- **Latest Reports & Outputs**: The outputs, training logs, and evaluation reports for this latest model run are located in `results/results_3/` rather than the root results folder.
- **Latest Code Updates**: The recent updates regarding Adversarial Domain Generalization (GRL) and Driver Baseline Normalization are documented in `docs/task/task_4.md`.

## Actions Required in `generate_report.py`:
Before generating the final report, make sure to update the path constants so the scripts pull from the newest directories:
1. Update checkpoint loading paths to target `assets/checkpoints/{wherever saved}/`.
2. Update the results saving/loading paths to target `results/{wherever you want to save}/`.
