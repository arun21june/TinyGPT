import torch
from tinygpt import TinyGPT
from pathlib import Path

checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "tinygpt_best.pt"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
    )

#1. Load checkpoints from disk
checkpoint = torch.load(
    checkpoint_path, 
    map_location=device,
    weights_only=False,
)

config = checkpoint["config"]
vocab_size = checkpoint["vocab_size"]

# print("Checkpoint path:", checkpoint_path.resolve())
# print("Checkpoint config block_size:", config["block_size"])
# print(
#     "Checkpoint tensor shape:",
#     checkpoint["model_state_dict"]["position_embedding.weight"].shape
# )

# 2. construct the Model Architecture
model = TinyGPT(
    vocab_size=vocab_size,
    d_model=config["d_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    block_size=config["block_size"],

)
# 3. Restore the parameters
model.load_state_dict(checkpoint["model_state_dict"])

# position_weights = model.position_embedding.weight

# print("Position embedding shape:",position_weights.shape)
# print("Position embedding standard deviation:", position_weights.std().item())

model = model.to(device)
model.eval()

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]


def generate(prompt, mx_new_tokens=50):
    # Convert prompt characters into token IDs
    tokens = [stoi[char] for char in prompt]

    # Convert to PyTorch tensor with batch dimension
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0)

    # Generate one token at a time

    for _ in range(mx_new_tokens):
        context = tokens[:,-128:]
        with torch.no_grad():
            logits = model(context)

        # Get logits for the final sequence position
        next_token_logits = logits[:, -1, :]

        #Introducing temperature to control the probability ditribution 
        temperature = 0.1

        # Probablistic sampling
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        #next_token = torch.argmax(probs, dim=-1)

        # Append predicted token to sequence 
        tokens = torch.cat((tokens, next_token.unsqueeze(1)), dim=1)

    # Convert token IDs back to characters
    generated_ids = tokens[0].tolist()
    return "".join(itos[token_id] for token_id in generated_ids)

torch.manual_seed(42)
print(generate("Language", mx_new_tokens=200))