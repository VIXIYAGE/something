import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# CONVERT ENCODED DATA TO TENSOR
# =========================================================

def make_tensor(encoded_data):

    tensor = torch.tensor(
        encoded_data,
        dtype=torch.float32
    )

    return tensor


# =========================================================
# AUTOENCODER
# =========================================================

class AutoEncoder(nn.Module):

    def __init__(self, input_feature):

        super().__init__()

        self.input_feature = input_feature

        self.encoder = nn.Sequential(

            nn.Linear(input_feature, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, input_feature)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# =========================================================
# LOAD TRAINED AUTOENCODER
# =========================================================

def load_autoencoder(model_path, input_feature):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AutoEncoder(input_feature).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model, device


# =========================================================
# GENERATE EMBEDDING
# =========================================================

def make_embedding(tensor_data, model, device):

    tensor_data = tensor_data.to(device)

    with torch.no_grad():

        embedding = model.encoder(tensor_data)

    return embedding.cpu()


# =========================================================
# TRAINING
# =========================================================

if __name__ == "__main__":

    df = pd.read_json(
        "data/ytr2.json"
    )

    encoded_data = np.loadtxt(
        "data/ecnext2.txt",
        dtype=np.float32
    )

    # -----------------------------------------------------
    # CREATE TENSOR
    # -----------------------------------------------------

    X = make_tensor(encoded_data)

    # -----------------------------------------------------
    # GET SHAPE FROM TENSOR
    # -----------------------------------------------------

    rows = X.shape[0]

    input_feature = X.shape[1]

    print(
        "Tensor shape:",
        X.shape
    )

    print(
        "Tensor dtype:",
        X.dtype
    )

    print(
        "Features:",
        input_feature
    )

    print(
        "Rows:",
        rows
    )

    # -----------------------------------------------------
    # CHECK DATA
    # -----------------------------------------------------

    print(
        "NaN:",
        torch.isnan(X).any().item()
    )

    print(
        "Inf:",
        torch.isinf(X).any().item()
    )

    print(
        "Min:",
        torch.min(X).item()
    )

    print(
        "Max:",
        torch.max(X).item()
    )

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    dataset = TensorDataset(X)

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True
    )

    # -----------------------------------------------------
    # DEVICE
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -----------------------------------------------------
    # MODEL USES TENSOR FEATURE COUNT
    # -----------------------------------------------------

    model = AutoEncoder(
        input_feature
    ).to(device)

    # -----------------------------------------------------
    # LOSS + OPTIMIZER
    # -----------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for (batch,) in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed = model(batch)

            loss = criterion(
                reconstructed,
                batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1:2}/{epochs} "
            f"| Loss: {average_loss:.6f}"
        )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    model.eval()

    with torch.no_grad():

        X_device = X.to(device)

        embeddings = model.encoder(
            X_device
        )

    embeddings = embeddings.cpu().numpy()

    print(
        "\nEmbedding shape:",
        embeddings.shape
    )

    # -----------------------------------------------------
    # SAVE EMBEDDINGS
    # -----------------------------------------------------

    np.savetxt(
        "data/embeddings2.txt",
        embeddings,
        fmt="%.6f"
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    torch.save(
        model.state_dict(),
        "data/autoencoder.pth"
    )

    print(
        "Embeddings saved."
    )

    print(
        "Model saved."
    )