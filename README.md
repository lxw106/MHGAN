# Masked Heterogeneous Graph Attention Network for Robust Recommendation
[framework (1).pdf](https://github.com/user-attachments/files/16014485/framework.1.pdf)
## Abstract
### Heterogeneous Graph Neural Networks (HGNNs) have gained significant attraction in recommendation due to their proficiency in capturing and utilizing the diverse data types inherent in social network. Nevertheless, HGNNs are susceptible to noise and subtle adversarial attacks, as disturbances from connected nodes can cumulatively impact a target user/item node. To address this challenge, we propose the Masked Heterogeneous Graph Attention Network for Robust Recommendation (MHGAN), which aims to enhance the resilience of recommendation against adversarial attacks. Specifically, we achieve robust recommendation through two primary strategies: de-weighting and pruning. (1) \textbf{De-weighting}: We introduced meta-path based propagation constraint probability that effectively reduces the weights of perturbed edges, thereby enhancing the recommendation's robustness. (2) \textbf{Pruning}: We design an innovative attention-based masking mechanism that selectively prunes malicious neighboring nodes using topology and node features to defend against adversarial attacks. Extensive experiments on multiple heterogeneous graph neural network models across three benchmark datasets demonstrate that MHGAN surpasses state-of-the-art methods in heterogeneous graph recommendation, showcasing its robustness and generalization capabilities despite varying levels of random noise.
## DATASET
### The datasets used in this paper can be accessed via this link: https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding.
## Enviroment Requirement
### pytorch==1.13.1
### pandas==2.0.3
### numpy==1.24.3
### scipy==1.10.1
### scikit-learn==1.3.2
### tqdm==4.65.0
### dgl==1.1.2.cu117
