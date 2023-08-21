## NeMo Sentence Transformer

This repository contains the code to refactor a [SentenceTransformer](https://www.sbert.net/) model architecture using [NVIDIA NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/starthere/intro.html). The model aims to embed sentences such that sentences that are semantically similar are closer in the embedded space.

We also use LoRA, from [Hugging Face PEFT](https://huggingface.co/docs/peft/index), to optimize the fine-tuning process.

### Notebooks provided

**00_Data_Preparation**:

   This notebook prepares the datasets for model fine tuning, validation, and testing.
   
   We use the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle, and sample a small set of news from 2 different categories that are difficult for the pre-trained model to separate in the embedding space. During fine-tuning, the model will try to pull together in the embedding space news of the same category, and push apart news of different categories.

**01_NeMo_Sentence_Transformer**:

   This notebook shows how to refactor the pre-trained Sentence Transformer model using NeMo.

   It begins by defining a custom `NewsCategoryDataset` class that tokenizes and label-encodes data for the Sentence Transformer model.
   
   Then it decomposes the original model into distinct [Nemo Neural Modules](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/core/api.html#nemo.core.NeuralModule):
   - `EncoderLayer`: Provides token embeddings from input sequences. It is prepared to be fine-tuned using LoRA. 
   - `PoolingLayer`: Aggregates over sequence length to provide sentence (sequence) embeddings.
   - `NormLayer`: L2 normalization layer.
   - `TripletLoss`: Custom loss function used for training. We inherit from one of the implementations from Sentence Transformers, which provides on-line triplet mining from a dataset with categorical labels associated to sentences.

   We then create the main model class `SentenceTransformer`, using the above Neural Modules, and define the forward flow, optimization configurations, and training/validation/test steps.

   When creating the Neural Modules and the main model class, we also show how to use NeMo's support for semantic and dimensionality checks of inputs and outputs using [Neural Types](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/core/neural_types.html).

   We also show how to use NeMo support for [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) to manage all configurations we need to instantiate, train, and test the model.

   The main model class implements the core base class for all NeMo models, `ModelPT`, which is an interface for [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/).

   Finally, we show how to train, test, save, and load the model.

   The code and dataset were prepared to be easily run on a single small commodity GPU with 16 GB memory, such as the NVIDIA T4, but one can run it with a much larger dataset and in a multi-node, multi-GPU environment with few modifications.