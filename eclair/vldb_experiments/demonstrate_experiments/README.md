# Demonstrate: SOP Generation Experiment

To generate SOPs for all experiments, run the following command:

```bash
# Takes ~30 mins to run for 30 tasks
bash run_experiments.sh ../../../data/vldb_experiments/demos
```

The bash script runs several prompt methods:
* `--is_td`: Just use task description to generate SOPs
* `--is_td_kf` : Use task description + key frames to generate SOPs; feed all keyframes into model in one prompt, e.g. (S, S', S'', ...)
* `--is_td_kf_act` : Use task description + key frames + action traces to generate SOPs; interleave all keyframes and action traces into a chat history -- e.g. (S, A, S', A'', ...)
* `--is_td_kf --is_pairwise` : Use task description + key frames to generate SOPs; feed each pair of keyframes independently into model -- e.g. (S,S')
* `--is_td_kf_act --is_pairwise` : Use task description + key frames + action traces to generate SOPs; feed ecah pair of keyframes and the one action between them independently into model -- e.g. (S, A, S')
