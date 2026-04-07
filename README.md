# LLM-from-scratch
Going to Build a small LLM which  explains the LLM architecture in detail
# 🧠 Building an LLM from Scratch

A complete, step-by-step implementation of a GPT-style Large Language Model built from the ground up using PyTorch. This project covers everything from raw text tokenization all the way to loading pretrained OpenAI GPT-2 weights.

---

## 📌 Project Overview

This project walks through the full pipeline of building and training a GPT-2 style language model (124M parameters) from scratch. The implementation is broken into 7 progressive notebooks, each focusing on a distinct stage of the LLM pipeline.

---

## 🗂️ Repository Structure

```
LLM-from-scratch/
│
├── 01_tokenization.ipynb                  # Text preprocessing & tokenization
├── 02_dataloader_embeddings.ipynb         # DataLoader & token/positional embeddings
├── 03_attention_mechanism.ipynb           # Self-attention & multi-head attention
├── 04_gpt_architecture.ipynb             # Full GPT model architecture
├── 05_training_evaluation.ipynb           # Training loop & loss evaluation
├── 06_text_generation_strategies.ipynb    # Inference: temperature & top-k sampling
├── 07_loading_pretrained_weights.ipynb    # Loading OpenAI GPT-2 pretrained weights
│
├── the-verdict.txt                        # Training corpus (short story)
├── gpt_download3.py                       # Utility to download GPT-2 weights
└── README.md
```

---

## 📒 Notebook Breakdown

### `01_tokenization.ipynb` — Tokenization
- Loading and preprocessing raw text
- Building a vocabulary from scratch
- Implementing `SimpleTokenizerV1` and `SimpleTokenizerV2` (with `<|unk|>` and `<|endoftext|>` special tokens)
- Introduction to **Byte Pair Encoding (BPE)** via `tiktoken` (GPT-2/3/4 encodings)

### `02_dataloader_embeddings.ipynb` — DataLoader & Embeddings
- Creating input-target pairs using a sliding window approach
- Building `GPTDatasetV1` using PyTorch's `Dataset` and `DataLoader`
- Token embeddings and **positional embeddings**
- Combining embeddings as input to the LLM

### `03_attention_mechanism.ipynb` — Attention Mechanism
- From dot-product to softmax attention
- Matrix multiplication for efficient attention computation
- `CausalAttention` with masking and dropout
- `MultiHeadAttentionWrapper` (sequential heads)
- `MultiHeadAttention` (parallel weight-split implementation)

### `04_gpt_architecture.ipynb` — GPT Model Architecture
- `GPT_CONFIG_124M` — model hyperparameters
- `DummyGPTModel` for architecture understanding
- **Layer Normalization** (`LayerNorm`)
- **GELU activation** vs ReLU (with visualization)
- **FeedForward Network** (FFN with expansion layer)
- **Shortcut (residual) connections** — solving vanishing gradients
- `TransformerBlock` — combining attention + FFN + normalization
- Full **`GPTModel`** class (~163M parameters)
- `generate_text_simple()` — greedy text generation

### `05_training_evaluation.ipynb` — Training & Evaluation
- Cross-entropy loss and **perplexity** as evaluation metrics
- Train/validation split on *The Verdict* corpus
- `calc_loss_batch()` and `calc_loss_loader()` utilities
- `train_model_simple()` — full training loop with AdamW optimizer
- Loss curve visualization (training vs validation)
- Overfitting analysis

### `06_text_generation_strategies.ipynb` — Text Generation Strategies
- **Temperature scaling** — controlling output randomness
- **Multinomial sampling** — probabilistic token selection
- **Top-k sampling** — restricting token candidates
- Combined `generate()` function with temperature + top-k + EOS support
- Saving and loading model weights (`state_dict`, checkpointing with optimizer)

### `07_loading_pretrained_weights.ipynb` — Pretrained GPT-2 Weights
- Downloading OpenAI GPT-2 (124M) weights via `gpt_download3.py`
- Mapping OpenAI weight keys to our `GPTModel` architecture
- `load_weights_into_gpt()` — weight assignment with shape validation
- Generating coherent text using the pretrained model

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch tiktoken tensorflow tqdm matplotlib numpy
```

### Run the Notebooks

Clone the repo and open notebooks in order:

```bash
git clone https://github.com/your-username/LLM-from-scratch.git
cd LLM-from-scratch
jupyter notebook
```

Start from `01_tokenization.ipynb` and work through to `07_loading_pretrained_weights.ipynb`.

---

## 🏗️ Model Architecture

| Component         | Specification              |
|-------------------|---------------------------|
| Architecture      | GPT-2 (decoder-only)      |
| Parameters        | ~163M                     |
| Vocabulary Size   | 50,257 (BPE)              |
| Context Length    | 256 (training) / 1024 (pretrained) |
| Embedding Dim     | 768                       |
| Attention Heads   | 12                        |
| Transformer Layers| 12                        |
| Dropout Rate      | 0.1                       |
| Optimizer         | AdamW (lr=0.0004)         |

---

## 📊 Training Results

The model was trained for 10 epochs on *The Verdict* (a short story, ~5,000 tokens) as a minimal educational dataset:

- **Initial training loss:** ~9.78
- **Final training loss:** ~0.39
- The model learns to generate coherent sentences from the training text within a few epochs.

> ⚠️ Note: Overfitting is expected on this small dataset. This is intentional — the goal is to demonstrate the mechanics of LLM training, not to build a production model.

---

## 📚 Key Concepts Covered

- Tokenization & Byte Pair Encoding (BPE)
- Token & Positional Embeddings
- Scaled Dot-Product Self-Attention
- Causal (Masked) Multi-Head Attention
- Layer Normalization & GELU Activation
- Feed-Forward Networks with Residual Connections
- Transformer Blocks & Full GPT Architecture
- Cross-Entropy Loss & Perplexity
- Temperature Scaling & Top-k Sampling
- Transfer Learning from OpenAI GPT-2

---

## 🙏 Acknowledgements

- Architecture based on **GPT-2** by OpenAI
- Tokenization via [`tiktoken`](https://github.com/openai/tiktoken)
- Inspired by *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka

---

## 📄 License

This project is for educational purposes. GPT-2 model weights are subject to [OpenAI's model card and usage policy](https://github.com/openai/gpt-2/blob/master/model_card.md).
