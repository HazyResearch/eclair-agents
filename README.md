<div align="center">
    <h1>ECLAIR</h1>
    <p><strong>E</strong>nterprise s<strong>C</strong>a<strong>L</strong>e <strong>AI</strong> for wo<strong>R</strong>kflows</p>
</div>



https://github.com/HazyResearch/eclair-agents/assets/5491790/580117a0-2afe-4137-97bb-e88fd2a5079e



*Example of **ECLAIR** running on a **real-world nursing workflow** in [Epic](https://www.epic.com/) after being given only a video recording and natural language description of the task. Note that this is sped up from actual model execution.*

🤖 Multimodal foundation models (FMs) such as GPT-4 offer a promising approach for end-to-end workflow automation given their generalized reasoning and planning abilities. 

⚙️ To study these capabilities [we propose **ECLAIR**](https://arxiv.org/abs/2405.03710), a system to automate enterprise workflows with minimal human supervision. 

📊 Our initial experiments suggest that **ECLAIR** can overcome the limitations of traditional automation technologies (e.g. RPA) with (1) near-human-level understanding of workflows and (2) instant set-up with minimal technical barrier.

Please note that **ECLAIR** is an ongoing research project and is not production-ready.

# 💿 Installation

```bash
# Create virtual env
conda create -n eclair_env python=3.10 -y
conda activate eclair_env

# Install repo
git clone https://github.com/HazyResearch/eclair-agents.git
cd eclair-agents/
pip install -r requirements.txt
pip install -e .
```

# 📊 Paper Experiments

Generate the experimental results in [our paper](https://arxiv.org/abs/2405.03710) using the dataset + scripts in this section. 

### 💾 Data

* [Link to Data](https://drive.google.com/drive/folders/1WL6pMfoAaar5uDEV-SWLalsAzEPsuzJp?usp=sharing) -- Download this file into the `data/` folder and unzip it.
* You should now have a folder at `data/vldb_experiments`.

### 🚀 How to Run

```bash
export OPENAI_API_KEY=<your_openai_api_key>

# Demonstrate
bash eclair/vldb_experiments/demonstrate_experiments/run_experiments.sh

# Execute
bash eclair/vldb_experiments/execute_actions/run_experiments.sh
bash eclair/vldb_experiments/execute_grounding/run_experiments.sh # [TODO]

# Validate
bash eclair/vldb_experiments/validate_experiments/run_experiments.sh
```

# 🏥 Hospital Workflow

This section contains the workflow data and scripts used to automate a real-world nursing workflow in Epic (i.e. the demo video at top of this README).

### 🎥 Demo

* [Link to ECLAIR Demo](https://drive.google.com/drive/folders/1U6fC67mDNlHQ0ikx-OOHx-7Bdv91XJ15?usp=drive_link) -- Visit this folder to view the outputs of ECLAIR executing the nursing workflow in Epic. 
* Please note that there are two versions of the demo video -- the raw recording as well as a 10x sped up version (labeled as `[fast]`).

### 💾 Data

* [Link to Data](https://drive.google.com/drive/folders/1TZp38_0IPf8aXFjh2UJa6AMdZyEyqCBA?usp=drive_link) -- Download this folder into the `data/` folder. 
* You should now have a folder at `data/hospital_data`.

### 🚀 How to Run

This will run the end-to-end automation pipeline for the nursing workflow. 

First, it generates an SOP from a demonstration. Second, it runs ECLAIR on the given workflow. Third, it validates that the workflow was completed successfully.

Note that this assumes you have a sandboxed instance of [Epic](https://www.epic.com/) running on your computer.

```bash
cd eclair/hospital_data
python3 pipeline.py
```

# Citation

Please consider citing if you found this work or code helpful!

```
@misc{wornow2024automating,
      title={Automating the Enterprise with Foundation Models}, 
      author={Michael Wornow and Avanika Narayan and Krista Opsahl-Ong and Quinn McIntyre and Nigam H. Shah and Christopher Re},
      year={2024},
      eprint={2405.03710},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```
