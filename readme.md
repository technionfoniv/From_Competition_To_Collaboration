# Sequential Interaction Framework: GenAI and Q\&A Forums

[](https://arxiv.org/abs/2602.04572)

## Overview

While Generative AI (GenAI) systems draw users away from Q\&A forums, they also depend on the very data those forums produce to improve their performance. Addressing this paradox, this repository proposes a framework of sequential interaction, in which a GenAI system proposes questions to a forum that can publish some of them.

Our framework captures several intricacies of such a collaboration, including non-monetary exchanges, asymmetric information, and incentive misalignment. We bring the framework to life through comprehensive, data-driven simulations using real Stack Exchange data and commonly used LLMs. We demonstrate the incentive misalignment empirically, yet show that players can achieve roughly 50% of the utility in an ideal full-information scenario. Our results highlight the potential for sustainable collaboration that preserves effective knowledge sharing between AI systems and human knowledge platforms.

## Repository Structure

The project relies on a specific sequence of data extraction, evaluation, and simulation. Below is a high-level overview of the primary working directories:

  * `Data/`: Contains the initial queries and extracted raw data.
  * `incentive misalignment/`: Contains plots and analyses demonstrating the misalignment between players.
  * `partial_collaboration_Bootstrap/`: Contains the bootstrapping notebook and strategy execution scripts.
  * `full collaboration/`: Contains the scripts for running heuristics and full collaboration analyses.
  * `EURR/`: The final directory for aggregating evaluation results.

## Pipeline and Usage

Follow these steps in order to reproduce the project results.Pay attention to
change the metrics by your simulations (for example using upvotes instead of views):


### 1\. Data Extraction

Navigate to the `Data` directory. Use the provided query to extract the required dataset from Stack Exchange dump and ensure the resulting data files are stored directly within the `Data` folder.

[]('https://data.stackexchange.com/')
### 2\. LLM Perplexity Evaluation

To calculate the perplexity scores for the models:
Run the Jupyter Notebook `Evaluate_LLMs_By_Perplexity.ipynb`.

*Note: You can view the resulting incentive misalignment graphs by navigating to `incentive misalignment -> Relations_Plots`.*

### 3\. Train Player F Classifiers

To generate the classifiers for Player F (the forum), execute the following notebook:
Run `research_classifier.ipynb`.

### 4\. Create the Bootstrapped Dataset

Navigate to the `partial_collaboration_Bootstrap` directory and run the bootstrapping notebook to prepare the dataset for the strategy simulations:
Run `Create_Bootstrap_Dataset.ipynb`.

### 5\. Execute Strategies (Asymetric Collaboration)

Once the bootstrapped dataset is generated, you can evaluate individual strategies. From your command line, run the desired strategy script:

```bash
python [strategy].py
```

*(Replace `[strategy].py` with the specific strategy file you wish to run).*

### 6\. Heuristics Analysis (Full Information)

To evaluate the heuristics:

1.  Copy your newly generated bootstrapped dataset into the `full collaboration` directory.
2.  Run the full collaboration script from the command line:

<!-- end list -->

```bash
python full_collab_with_classifier.py
```

### 7\. Evaluation

After obtaining results from both the partial and full collaboration steps:

1.  Run the `evaluation.ipynb` notebook located in **both** the partial and full collaboration directories.
2.  Copy the output results from both evaluations into the `EURR` directory.

### 8\. Final Results

Navigate to the `EURR` directory and run the final aggregator notebook to generate the conclusive results and metrics for the study.

## Citation

If you use this code or framework in your research, please cite the original article:

```bibtex
@misc{fono2026competitioncollaborationdesigningsustainable,
      title={From Competition to Collaboration: Designing Sustainable Mechanisms Between LLMs and Online Forums}, 
      author={Niv Fono and Yftah Ziser and Omer Ben-Porat},
      year={2026},
      eprint={2602.04572},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.04572}, 
}
```

