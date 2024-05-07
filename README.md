# ECLAIR

**E**nterprise s**C**a**L**e **AI** for **W**orkflows



https://github.com/HazyResearch/eclair-agents/assets/5491790/44f6d0dc-f103-4581-b67b-19e26a225403



*Example of ECLAIR running on a real-world nursing workflow in Epic (a popular electronic health record software) after being given only a video recording and natural language description of the task.*

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

### 💾 Data

* [Link to Data](https://drive.google.com/file/d/1h-sf1WlaIblvxhLNQbDlSiMDp1CkGFx1/view?usp=drive_link) -- Download this file into the `data/` folder and unzip it.
* You should now have a folder at `data/vldb_experiments` with 30 subfolders (one for each WebArena task evaluated in the paper).

### 🚀 How to Run

```bash
# Demonstrate
cd 
bash ./eclair/vldb_experiments/demonstrate_experiments/run_experiments.sh ./data/vldb_experiments

# Execute
bash ./eclair/vldb_experiments/execute_actions/run_experiments.sh ./data/vldb_experiments
bash ./eclair/vldb_experiments/execute_grounding/run_experiments.sh ./data/vldb_experiments

# Validate
bash ./eclair/vldb_experiments/validate_experiments/run_experiments.sh ./data/vldb_experiments
```

## 🏥 Hospital Workflow

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
