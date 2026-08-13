# LLM-Check: Investigating Detection of Hallucinations

## Overview
This repository contains code for the implementation of the paper [LLM-Check: Investigating Detection of Hallucinations in Large Language Models](https://openreview.net/forum?id=LYx4w3CAgy).

- It analyzes hallucination detection within a single LLM response using its corresponding internal attention kernel maps, hidden activations and output prediction probabilities. 
- The main evaluation dataset is FAVA-Annotation, a human-annotated hallucination detection benchmark


## LLM-Check Scoring Suite
In this work, the original implementation from the paper extracts internal model information and calculates scores such as: Perplexity, Logit entropy, Window entropy, Hidden-state eigenvalue scores, and Attention eigenvalue scores.

These scores are calculated from a forward pass through the LLM. The implementation supports several pretrained models, but for comparison, we mostly focus on evaluating Llama-2, Llama-3.

<p align="center">
    <img src="images_readme/EigenAnalysis_Pipeline.png" width="700"\>
</p>

## Contextual embeddings
The extension introduced in this project extracts the contextual embeddings of the generated response from the model hidden states. 

For a response consisting of multiple tokens, the hidden representation of each token can be extracted from a selected model layer: $h_1, h_2, ..., h_T$ where each $h_i$ is the contextual representation of token $i$

These token level representations can then be aggregated by mean pooling. The final vector provides a compact representation of the response in the latent space.

<!-- <p align="center">
    <img src="images_readme/Hallucinated_and_truthful_pair_examples.png" width="800"\>
</p> -->
<!-- 
The proposed method: LLM-Check - a suite of simple, effective detection techniques over current LLMs. We propose two distinct lines of analysis, which we collectively term as LLM-Check: 
1. Eigenvalue Analysis of Internal LLM Representations 
2. Output Token Uncertainty Quantification.

We utilize these diversified scoring methods from different model components to potentially maximize the capture of hallucinations amongst its various forms without incurring computational overheads at training or inference time.

- Towards this, the Eigen-analysis of internal LLM representations helps highlight the consistent pattern of modifications to the hidden states and the model attention across different token representations in latent space when hallucinations are present in model responses as compared to truthful, grounded responses. 
- On the other hand, the uncertainty quantification of the output tokens using Perplexity and Logit Entropy, helps analyze hallucinations based on the likelihood assigned by the model on the actual tokens predicted at a specific point in the sequence generated auto-regressively.  -->

<!-- We present qualitative comparisons of the proposed method with the most pertinent baselines in the table below. We present various trade-offs and advantages in the table such as to whether the method requires fine-tuning of an LLM, if it inherently requires multiple model responses to detect hallucinations, if the method is computationally efficient, if it performs detection on per-sample basis or at a population level, and whether the method is inherently dependent on retrieval during inference time.

<p align="center">
    <img src="images_readme/Qualitative_Table.png" width="500" \>
</p> -->

<!-- ## Run-Time Analysis

We compare the overall runtime cost of the proposed detection scores with other baselines using a Llama-2-7b Chat model on the FAVA-Annotation dataset on a single Nvidia A5000 GPU. For the Eigen Analysis based methods, we report the total time needed for all 32 Layers for Attention and Hidden Scores. We observe that the Logit and Attention scores are indeed very efficient, while the Hidden Score is slightly slower since it uses SVD explicitly. We also observe that LLM-Check is considerably faster than most baselines with speedups of up to 45x and 450x, since it only uses model representations with teacher forcing, without additional inference time overheads.


<p align="center">
    <img src="images_readme/runtimes.png" width="400" \>
</p> -->

## Code Setup and Organization
1. Install required packages from `requirements.txt` file with Python 3.10.12
2. Download files the FAVA-Annotation dataset at [[here](https://huggingface.co/datasets/fava-uw/fava-data/blob/main/annotations.json)], and save it at the top-level directory. 
3. Run `run_detection_combined.py` to extract internal representations and detect hallucinations using different scores. This can be executed using the `run.sh` file, where different configurations can be specified.
4. The scores so selected will be saved to disk in the `/data` folder.
5. Finally, run `detection_score.py` file to analyze the scores, plots, train and evaluate classifier on the FAVA-Annotation dataset.
