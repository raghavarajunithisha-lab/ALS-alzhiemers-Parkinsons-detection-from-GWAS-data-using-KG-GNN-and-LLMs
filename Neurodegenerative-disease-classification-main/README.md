# Neurodegenerative-disease-classification

Developed a deep learning model using Graph Neural Networks (GNNs) to predict neurodegenerative diseases such as ALS, Alzheimer’s, and Parkinson’s. The model integrates genetic variant (SNP) connections to identify disease-associated patterns and potential molecular interactions underlying these conditions.

## Project Overview
This project aims to classify neurodegenerative diseases using advanced Graph Neural Network (GNN) architectures. It leverages graph-structured data where nodes represent entities (e.g., genetic variants, patients) and edges represent their relationships or interactions.

## Dataset
The dataset utilized is a PyTorch Geometric `Data` object (`02-alzheimers-parkinson_disease-amyotrophic_lateral_sclerosis_data.pth`) containing:
- 3222 nodes and 3222 edges.
- Node features, edge indices, and edge attributes (relations).
- Class distribution: {0: 2383, 1: 181, 2: 658} corresponding to different neurodegenerative conditions.

## Models Implemented
The project explores and compares three distinct GNN architectures:

1. **Standard GNN Model**: A 3-layer Graph Convolutional Network (GCN) incorporating Batch Normalization, Dropout, and Multi-head Attention mechanisms.
2. **MGPOOL (Multi-Graph Pooling) Model**: A hybrid architecture combining GCN and Graph Isomorphism Network (GIN) branches with an attention mechanism to effectively capture both structural and feature-based information.
3. **PRGNN (Proximity Relational GNN) / GPT2Graph**: An advanced model that encodes node features with GCNs, constructs edge features using proximity blocks (DeepEBlock), and employs multi-head self-attention. This model demonstrates the highest performance.

## Installation
To run this project, you need the following dependencies:
- Python 3.8+
- PyTorch
- PyTorch Geometric (PyG)
- scikit-learn
- matplotlib
- numpy

Clone this repository:
```bash
git clone https://github.com/nikhitha2201/Neurodegenerative-disease-classification.git
cd Neurodegenerative-disease-classification
```

## Training and Evaluation
The training pipeline uses the `AdamW` optimizer and `CrossEntropyLoss` with class weights to manage dataset imbalance. The models are evaluated using metrics such as Accuracy, Macro F1-score, and Weighted F1-score. 

**Performance Summary (Test Set):**
- **GNN Model**: ~96.1% Accuracy, ~0.96 Weighted F1
- **MGPOOL Model**: ~92.5% Accuracy, ~0.92 Weighted F1
- **PRGNN Model**: ~95.0% Accuracy, ~0.95 Weighted F1

Training scripts log the loss and accuracy/F1 metrics every 5 epochs and generate plots visualizing `Epoch vs Accuracy` for both Training and Validation sets.

## How to Run
Open and execute the `Project.ipynb` Jupyter Notebook. The notebook contains sequential cells for:
1. Setting up the environment and installing PyG.
2. Loading and preprocessing the graph dataset.
3. Defining the model architectures.
4. Running the training loops.
5. Evaluating models and plotting results.

## Future Work
- Integration with Large Language Models (LLMs) to enhance relation extraction and node feature representation.
- Expanding the dataset to include broader molecular interaction networks.
- Exploring other advanced GNN architectures and pooling strategies.
