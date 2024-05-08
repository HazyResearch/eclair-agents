# ECLAIR

**E**nterprise s**C**a**L**e **AI** for **W**orkflows



https://github.com/HazyResearch/eclair-agents/assets/5491790/580117a0-2afe-4137-97bb-e88fd2a5079e



*Example of ECLAIR running on a real-world nursing workflow in Epic (a popular electronic health record software) after being given only a video recording and natural language description of the task. Please note that this is sped up from actual model execution.*

## 💿 Installation

```bash
# Create virtual env
conda create -n eclair_env python=3.10 -y
conda activate eclair_env

# Install repo
git clone https://github.com/HazyResearch/eclair.git
cd eclair-agents/
pip install -r requirements.txt
pip install -e .
```

## 📊 Paper Experiments

Generate the experimental results in [our paper](TODO) using the dataset + scripts in this section. 

### 💾 Data

* [Link to Data](https://drive.google.com/drive/folders/1WL6pMfoAaar5uDEV-SWLalsAzEPsuzJp?usp=sharing) -- Download this file into the `data/` folder and unzip it.
* You should now have a folder at `data/vldb_experiments` with two subfolders named `demos` and `grounding`.

### 🚀 How to Run

```bash
export OPENAI_API_KEY=<your_openai_api_key>

# Demonstrate
bash eclair/vldb_experiments/demonstrate_experiments/run_experiments.sh data/vldb_experiments/demos

# Execute
bash eclair/vldb_experiments/execute_actions/run_experiments.sh data/vldb_experiments/demos
bash eclair/vldb_experiments/execute_grounding/run_experiments.sh data/vldb_experiments/demos

# Validate
bash eclair/vldb_experiments/validate_experiments/run_experiments.sh data/vldb_experiments/demos
```

## 🏥 Hospital Workflow

This section contains the workflow data and scripts used to automate a real-world nursing workflow in Epic (i.e. the demo video at top of this README).

### 🎥 Demo

* [Link to ECLAIR Demo](https://drive.google.com/drive/folders/1U6fC67mDNlHQ0ikx-OOHx-7Bdv91XJ15?usp=drive_link) -- Visit this folder to view the outputs of ECLAIR executing the nursing workflow in Epic. 
* Please note that there are two versions of the demo video -- the raw recording as well as a 10x sped up version (labeled as `[fast]`).

### 💾 Data

* [Link to Data](https://drive.google.com/drive/folders/1TZp38_0IPf8aXFjh2UJa6AMdZyEyqCBA?usp=drive_link) -- Download this folder into the `data/` folder. 
* You should now have a folder at `data/hospital_data` with 3 subfolders (one for each task in the hospital demo).

### 🚀 How to Run

This will run the end-to-end automation pipeline for the nursing workflow. First, it generates an SOP from a demonstration. Second, it runs ECLAIR on the given workflow. Third, it validates that the workflow was completed successfully.

Note that this assumes you have an instance of [Epic](https://www.epic.com/) running on your computer.

```bash
cd eclair/hospital_data
python3 pipeline.py
```
