# Activation-Guided Neuron Intervention for Alzheimer’s-Related Language Phenotypes

Code and results for the paper **“Activation-Guided Neuron Intervention to Induce Alzheimer’s-Related Computational Language Phenotypes in a Large Language Model.”**

This study identifies feed-forward network units that are more frequently activated by Alzheimer’s disease (AD) transcripts and examines whether controlled modulation of these units induces AD-related changes in the language and cognitive-task performance of Qwen3-8B.

The scripts evaluate immediate and delayed recall, verbal fluency, working memory, procedural discourse, scene construction, and coreference resolution.

## Model checkpoints

The generated intervention checkpoints and intermediate tensors are not included in this repository because they require approximately 336 GB of storage. The repository provides the neuron-intervention code, neuron-selection outputs, analysis scripts, and reported results. The checkpoints can be regenerated locally using the provided intervention scripts.

## Batch execution
SLURM scripts are provided for the original model, AD-guided interventions, attenuation controls, and random-neuron controls:
```
sbatch neuropsy_org.sh
sbatch neuropsy_10000.sh
sbatch neuropsy_allsig.sh
sbatch neuropsy_reverse.sh
sbatch neuropsy_random.sh
```
## Data access
The ADReSS(o) transcripts used to identify AD-associated neurons were accessed through DementiaBank. These transcripts cannot be redistributed in this repository. Researchers must apply for DementiaBank access and comply with its data-use requirements.

## Results and analysis
Generated conversations are stored in:
```
Results/Chat/
Results/Chat_Random/ (the random-neuron control condition in the appendix)
```

Human annotations, automated analyses, summary files, and computational linguistic measures are available in:
```
Results/Chat_analysis/
```

Task-specific analysis scripts are provided in the ```Results/``` directory.














