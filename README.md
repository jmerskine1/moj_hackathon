

## Install
```bash
git clone https://github.com/jmerskine1/moj_hackathon.git

cd bias-bench 
python -m pip install -e .
```

## Required Datasets
Below, a list of the external datasets required by this repository is provided:

Dataset | Download Link | Notes | Download Directory
--------|---------------|-------|-------------------
Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing) | English Wikipedia dump used for SentenceDebias and INLP. | `data/text`
Wikipedia-10 | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing) | English Wikipedia dump used for CDA and Dropout. | `data/text`

Each dataset should be downloaded to the specified path, relative to the root directory of the project.

## Experiments
We provide scripts for running all of the experiments presented in the paper.
Generally, each script takes a `--model` argument and a `--model_name_or_path` argument.
We briefly describe the script(s) for each experiment below:

* **CrowS-Pairs**: Two scripts are provided for evaluating models against CrowS-Pairs: `experiments/crows.py` evaluates non-debiased
  models against CrowS-Pairs and `experiments/crows_debias.py` evaluates debiased models against CrowS-Pairs.
* **INLP Projection Matrix**: `experiments/inlp_projection_matrix.py` is used to compute INLP projection matrices.
* **SEAT**: Two scripts are provided for evaluating models against SEAT: `experiments/seat.py` evaluates non-debiased models against SEAT and
  `experiments/seat_debias.py` evaluates debiased models against SEAT.
* **StereoSet**: Two scripts are provided for evaluating models against StereoSet: `experiments/stereoset.py` evaluates non-debiased models against StereoSet and
  `experiments/stereoset_debias.py` evaluates debiased models against StereoSet.
* **SentenceDebias Subspace**: `experiments/sentence_debias_subspace.py` is used to compute SentenceDebias subspaces.
* **GLUE**: `experiments/run_glue.py` is used to run the GLUE benchmark.
* **Perplexity**: `experiments/perplexity.py` is used to compute perplexities on WikiText-2.

For a complete list of options for each experiment, run each experiment script with the `--h` option.
For example usages of these experiment scripts, refer to `batch_jobs`.
The commands used in `batch_jobs` produce the results presented in the paper.

### Notes
* To run SentenceDebias models against any of the benchmarks, you will first need to run `experiments/sentence_debias_subspace.py`.
* To run INLP models against any of the benchmarks, you will first need to run `experiments/inlp_projection_matrix.py`.
* `export` contains a collection of scripts to format the results into the tables presented in the [paper](https://arxiv.org/abs/2110.08527).

* To help get you started you can run the following to test:
#### Stereoset
 the default model (`bert base cased`)
```bash
python experiments/stereoset.py
```
or e.g. for gpt2
```bash
python experiments/stereoset.py --model "GPT2LMHeadModel" --model_name_or_path "gpt2"
```
#### CrowS-Pairs
* You can get the evaluations for your models tested on Stereoset by running 
```bash
python experiments/stereoset_evaluation.py --predictions_dir "results/stereoset"
```

* I've created a rudimentary example results plotter to help demonstrate how to access model outputs, which can be called with 

```bash
python plot_results.py "results/stereoset/stereoset_m-BertForMaskedLM_c-bert-base-uncased.json" "results/stereoset/stereoset_m-GPT2LMHeadModel_c-gpt2.json" --output 'outputs/fig.png'
```



## Running on an HPC Cluster
We provide scripts for running all of the experiments presented in the paper on a SLURM cluster in `batch_jobs`.
If you plan to use these scripts, make sure you customize `python_job.sh` to run the jobs on your cluster.
In addition, you will also need to change both the output (`-o`) and error (`-e`) paths.

## Acknowledgements
This repository is a slightly modified version of the [bias-bench](https://github.com/McGill-NLP/bias-bench). For more info see [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models](https://arxiv.org/abs/2110.08527) presented at ACL 2022.
