import torch
from tinygpt import TinyGPT
from tinydataset import TinyDataset
import copy
from pathlib import Path

# Define checkpoints paths
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True,exist_ok=True)

best_checkpoint_path = checkpoint_dir / "tinygpt_best.pt"
latest_checkpoint_path = checkpoint_dir / "tinygpt_latest.pt"

# Define the text and the vocabulary
data_path = Path("data/tinyshakespeare.txt")
text = data_path.read_text(encoding="utf-8")

print(f"Corpus characters: {len(text):,}")
print(f"Unique characters: {len(set(text))}")

# Create the vocabulary and the mappings
vocab = sorted(set(text))

vocab_size = len(vocab)
block_size = 128 # how many elements in a block, this is the context
batch_size = 16 # how many blocks in a batch

# Create the mappings from characters to integers and vice versa
stoi = {ch:i for i, ch in enumerate(vocab)}

# Create the mappings from integers to characters
itos = {i:ch for i, ch in enumerate(vocab)}

# Encode the text as a list of integers
encoded = [stoi[ch] for ch in text]

#decoded = ''.join(itos[i] for i in encoded)

# Convert the encoded text to a PyTorch tensor
data = torch.tensor(encoded, dtype=torch.long)

# split 
split = int (0.9 * len(data))

train_data = data[:split]
val_data = data[split:]

train_dataset = TinyDataset (train_data, block_size)
val_dataset = TinyDataset(val_data, block_size)

# Print number of tokens for train and validation
print("Number of tokens for train: ", len(train_data ))
print("Number of tokens for validation: ", len(val_data)) 

# Data loader
loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size,
    shuffle=True
)

# Validation loader

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size,
    shuffle=False
)

# Training  starts from here
torch.manual_seed(42)


# Model definition 
d_model = 512
num_layers = 4
num_heads = 8
d_ff = 2048


model = TinyGPT(vocab_size, d_model, num_layers, num_heads, d_ff, block_size)


# Diagnostic: inspect initial model parameters
first_param = next(model.parameters())

print("Initial parameter sample:", first_param.flatten()[:5])
print("Initial parameter mean:", first_param.mean().item())
print("Initial parameter std:", first_param.std().item())

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")

# Port model to device
model = model.to(device)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Set Training configuration
best_val_loss = float('inf')
best_model_state = None
patience = 3
epoch_without_improvement = 0
start_epoch = 0
epoch_len = 20
resume_training = False

# Load the saved training checkpoint to resume training 

if resume_training and latest_checkpoint_path.exists():
    checkpoint = torch.load(
        latest_checkpoint_path,
        map_location=device,
        weights_only= False,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_model_state = checkpoint['best_model_state']
    best_epoch = checkpoint["best_epoch"]
    epoch_without_improvement = checkpoint["epoch_without_improvement"]

    print(f"Resuming training from epoch {start_epoch + 1}")


# Training epoch (Training + Validation)
for epoch in range(start_epoch, epoch_len):

    total_train_loss = 0.0
    total_train_token = 0
    #Training loop
    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
    
        loss = loss_fn(logits.reshape(-1, vocab_size), batch_y.reshape(-1))

        loss.backward()

        optimizer.step()

        # Total avg based on total number of token  * avg loss per token
        total_train_loss += loss.item() * batch_y.numel()
        total_train_token += batch_y.numel()

    average_loss = total_train_loss / total_train_token

    # Diagnostic: inspect model parameters after epoch 1
    if epoch == 0:
        first_param = next(model.parameters())

        print(
            "Epoch 1 parameter sample:",
            first_param.detach().flatten()[:5]
        )

    # Eval starts 
    model.eval()

    with torch.no_grad():
        total_val_loss = 0.0
        total_val_token = 0

        # Eval loop 
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)

            val_loss = loss_fn(logits.reshape(-1, vocab_size), batch_y.reshape(-1))

            total_val_loss += val_loss.item() * batch_y.numel()
            total_val_token += batch_y.numel()

        average_val_loss = total_val_loss / total_val_token

        #Early stop & saving the model parameter state 
        if (average_val_loss < best_val_loss):
            best_val_loss = average_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epoch_without_improvement = 0
            # Save the best training state
            torch.save(
                {
                "epoch": best_epoch,
                "model_state_dict": best_model_state,
                "best_val_loss": best_val_loss,
                "vocab_size": vocab_size,
                "stoi":stoi,
                "itos":itos,
                "config": {
                    "d_model": d_model,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "block_size":block_size,
                },
                },
                best_checkpoint_path,
            )
        else:
            epoch_without_improvement += 1
        
    print(f"Epoch : {epoch + 1:2d} | loss : {average_loss:.4f} | Validation loss : {average_val_loss:.4f}")

    # Validation loops ends here
    # Resume training 
    model.train()

    # Save the latest checkpoint 
    torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss":best_val_loss,
                "best_model_state":best_model_state,
                "best_epoch":best_epoch,
                "epoch_without_improvement":epoch_without_improvement,
                "vocab_size":vocab_size,
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "d_model": d_model,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "block_size":block_size,
                    },

            },
           latest_checkpoint_path,
        )

    # Early stopping
    if (epoch_without_improvement >= patience):
        print(f"Early stopping at epoch {epoch + 1}")
        break

if best_model_state != None:
    model.load_state_dict(best_model_state)
else:
    raise RuntimeError("Training completed without a valid best model.")

print(f"Best Val loss: {best_val_loss:.4f} | Best Epoch: {best_epoch}")

# Training ends here