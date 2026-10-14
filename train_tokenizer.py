from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


# 1. Define project path
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "tinyshakespeare.txt"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"


# 2. Create Output Directory
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)                         


# 3. Initialize the BPE Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))


# 4. Configure byte-level text processing
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer.decoder = decoders.ByteLevel()

# 5. Configure the BPE training algorithm
trainer = trainers.BpeTrainer(
    vocab_size=2000,
    min_frequency=2,
    special_tokens=["[UNK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# 6. Train the tokenizer on the corpus
tokenizer.train([str(CORPUS_PATH)],
                trainer=trainer,)

# 7. Save the train tokenizer to disk
tokenizer.save(str(TOKENIZER_PATH))

# 8. Display basic information
print(f"Corpus: {CORPUS_PATH}")
print(f"Tokenizer saved to: {TOKENIZER_PATH}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# 9. Inspect Tokenization
sample = " To be, or not to be."

encoded = tokenizer.encode(sample)

print("\n Original Text:", repr(sample))
print("Tokens:",encoded.tokens)
print("Token IDs:",encoded.ids)
print("Token count:",len(encoded.ids))

decoded = tokenizer.decode(encoded.ids)

print("Decoded test:", repr(sample))
print("Round trip successful:", decoded==sample)