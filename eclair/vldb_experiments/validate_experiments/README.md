# Execute: Validate Pre/Post-Conditions

To generate results for all tasks, run the following command:

```bash
bash run_experiments.sh
```

The bash script runs the following command for each task:

* `--is_actuation` : Test actuation validation
* `--is_precondition` : Test integrity constraint validation
* `--is_task_completion` : Test task completion validation
* `--is_task_trajectory` : Test task trajectory (i.e. follows SOP) validation