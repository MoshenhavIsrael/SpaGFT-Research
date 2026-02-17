# SpaGFT-Research

**Experimental workflows and benchmarks for evaluating and optimizing the SpaGFT algorithm.**

This repository serves as a research workspace for investigating the **SpaGFT** (Spatial Graph Fourier Transform) method. The primary focus is on analyzing and improving signal robustness, specifically regarding rotation stability and consistency across biological replicates in high-resolution spatial transcriptomics data (e.g., EXSEQ).

> **Note:** This is a research repository containing analysis code and workflows. The source code for the modified SpaGFT library is maintained in a separate fork.

## Research Objectives

1.  **Rotation Stability:** Analyze the sensitivity of `gft_score` and SVG detection to spatial coordinate rotation and implement mathematical fixes to ensure invariance.
2.  **Consistency:** Improve the overlap of identified Spatially Variable Genes (SVGs) across biological replicates.
3.  **High-Resolution Application:** Optimize parameters and graph construction for single-cell resolution datasets (e.g., Hippocampus EXSEQ).

## Repository Structure

* `code/`: Core scripts for data setup, linear transformation experiments, and analysis pipelines.
    * `stage_0_setup`: Initial data processing and environment setup.
    * `stage_1_lin_trns`: Experiments regarding linear transformations (rotation/translation).
* `show_workflow.ipynb`: Main notebook demonstrating the analysis flow and visualizing results.
* `data/` & `results/`: Local directories for datasets and output figures (excluded from version control).

## Installation & Usage

This project is designed to work in tandem with a modified fork of the SpaGFT library.

### Prerequisites

1.  **Clone this repository** (The analysis workflows):
    ```bash
    git clone https://github.com/MoshenhavIsrael/SpaGFT-Research.git
    cd SpaGFT-Research
    ```

2.  **Clone and Install the SpaGFT Fork** (The library) **and other requirements**:
    The dependencies of SpaGFT wouldn't be solved on all python versions, so the recomendation is to install it in a new environment of python 3.8.  It is recommended to install the library in **editable mode** so changes to the algorithm are immediately reflected in the notebooks.

    ```bash
    conda create -n m_spagft_env python==3.8.0
    conda activate m_spagft_env
    # Go to your workspace folder (outside of SpaGFT-Research)
    cd ..
    # Install SpaGFT fork
    git clone https://github.com/MoshenhavIsrael/SpaGFT.git
    cd SpaGFT
    pip install -e .
    # Install another requirements
    pip install huggingface_hub
    ```   
3. **Install jupyter (optional):**
   Note [jupyter](https://jupyter.org/) is recommend for interactive usage. It can be installed and configured by

   ```bash
   conda install jupyter
   python -m ipykernel install --user --name=m_spagft_env --display-name=m_spagft_env
   ```

### Running the Analysis

Open the main notebook to view the workflow:
```bash
jupyter notebook code/show_workflow.ipynb
