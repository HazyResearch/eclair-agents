# Execute: Action/Actuation Experiments

To generate results for all tasks, run the following command:

```bash
bash run_experiments.sh ../../../data/vldb_experiments/demos
```

The bash script runs several prompt methods:
* `--is_action`: Test ability of model tom make next action suggestion
* `--is_actuation`: Test ability of model to actuate a given next action suggestion
* `--is_include_sop` : If TRUE, include the SOP in the prompt