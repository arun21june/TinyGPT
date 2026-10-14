# TinyGPT

A hands-on implementation of a small GPT-style language model using PyTorch.

This project explores transformer architecture, self-attention, token embeddings,
training mechanics, validation, checkpointing, and autoregressive text generation.

## Project Structure

```text
TinyGPT/
├── tinygpt.py
├── transformerBlock.py
├── tinydataset.py
├── train.py
├── generate.py
├── train_tokenizer.py
├── data/
│   └── tinyshakespeare.txt
├── tokenizer/
│   └── tokenizer.json
└── checkpoints/          # Local model checkpoints; not tracked by Git
Components
- tinygpt.py: GPT model architecture.
- transformerBlock.py: Multi-head self-attention, feed-forward network,
  residual connections, and layer normalization.
- tinydataset.py: Creates input and target sequences for language modeling.
- train.py: Trains the model, evaluates validation loss, and saves checkpoints.
- generate.py: Loads the best checkpoint and generates text.
- train_tokenizer.py: Trains a byte-level BPE tokenizer as a separate experiment.
Tokenization
The current training pipeline builds a character-level vocabulary directly
from data/tinyshakespeare.txt.
The BPE tokenizer experiment is separate and is not yet integrated into the
main training pipeline.
Requirements
- Python
- PyTorch
- tokenizers
Install the dependencies in your Python environment:
python -m pip install torch tokenizers
raining
Run the training script from the project root:
python train.py
Text Generation
After a suitable checkpoint has been created:
python generate.py
Model Checkpoints
Model checkpoints are excluded from Git to avoid storing large binary files
in the repository. They remain available locally after training.
Learning Goals
- Understand GPT and transformer architecture.
- Explore causal self-attention and embeddings.
- Understand forward passes, loss calculation, backpropagation, and optimization.
- Study training and validation behavior.
- Experiment with tokenizer design.
- Learn reproducible development and version control.
Status
This is an educational project under active development.
