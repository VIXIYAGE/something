from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from backend.pipeline.ENCOD import SchemaEx

from backend.pipeline.TENSORING import (
    make_tensor,
    load_autoencoder,
    make_embedding
)

from backend.pipeline.slm_inference import (
    generate_text
)

# =============================================================
# MODEL
# =============================================================

MODEL_PATH = "backend/data/autoencoder.pth"

autoencoder, device = load_autoencoder(
    MODEL_PATH,
    27
)


# =============================================================
# FASTAPI
# =============================================================

app = FastAPI(
    title="Sameer PRODigy 🧨🧨",
    description="Backend for JSON encoding, tensoring, RAG and SLM",
    version="0.1.0"
)


# =============================================================
# REQUEST
# =============================================================

class AskRequest(BaseModel):
    data: list[dict]
    question: str
# =============================================================
# ROOT
# =============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "SchemaX backend is alive"
    }


# =============================================================
# HEALTH
# =============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =============================================================
# ENCODE
# =============================================================

@app.post("/encode")
def encode(request: EncodeRequest):

    df = pd.DataFrame(
        request.data
    )

    df = df.dropna()

    encoder = SchemaEx()

    encoded_data = (
        encoder.detect_and_encode(df)
    )

    print(
        "ENCODE:",
        encoded_data.shape
    )

    return {

        "rows": encoded_data.shape[0],

        "features": encoded_data.shape[1],

        "encoded_data":
            encoded_data.tolist()
    }


# =============================================================
# TENSOR
# =============================================================

@app.post("/tensor")
def tensor(request: EncodeRequest):

    df = pd.DataFrame(
        request.data
    )

    df = df.dropna()

    encoder = SchemaEx()

    encoded_data = (
        encoder.detect_and_encode(df)
    )

    tensor_data = make_tensor(
        encoded_data
    )

    print(
        "TENSOR:",
        tensor_data.shape
    )

    return {

        "rows":
            tensor_data.shape[0],

        "features":
            tensor_data.shape[1],

        "dtype":
            str(tensor_data.dtype),

        "tensor":
            tensor_data.tolist()
    }


# =============================================================
# EMBED
# =============================================================

@app.post("/embed")
def embed(request: EncodeRequest):

    df = pd.DataFrame(
        request.data
    )

    df = df.dropna()

    if df.empty:

        return {
            "error":
                "No valid rows remain after removing missing values."
        }

    # ---------------------------------------------------------
    # ENCODE
    # ---------------------------------------------------------

    encoder = SchemaEx()

    encoded_data = (
        encoder.detect_and_encode(df)
    )

    print(
        "EMBED encoded shape:",
        encoded_data.shape
    )

    # ---------------------------------------------------------
    # CHECK FEATURES
    # ---------------------------------------------------------

    if encoded_data.shape[1] != 27:

        return {

            "error":
                "Encoded feature count is incompatible with AutoEncoder.",

            "input_encoded_shape":
                list(encoded_data.shape),

            "expected_features":
                27,

            "message":
                "This JSON was encoded successfully, "
                "but the resulting feature count is not 27."
        }

    # ---------------------------------------------------------
    # TENSOR
    # ---------------------------------------------------------

    tensor_data = make_tensor(
        encoded_data
    )

    print(
        "EMBED tensor shape:",
        tensor_data.shape
    )

    # ---------------------------------------------------------
    # TENSOR SAFETY CHECK
    # ---------------------------------------------------------

    if tensor_data.ndim != 2:

        return {
            "error":
                "Tensor must be 2-dimensional.",
            "tensor_shape":
                list(tensor_data.shape)
        }

    if tensor_data.shape[1] != 27:

        return {

            "error":
                "Tensor feature count is incompatible with AutoEncoder.",

            "tensor_shape":
                list(tensor_data.shape),

            "expected_features":
                27
        }

    # ---------------------------------------------------------
    # EMBEDDING
    # ---------------------------------------------------------

    embedding = make_embedding(
        tensor_data,
        autoencoder,
        device
    )

    print(
        "EMBEDDING shape:",
        embedding.shape
    )

    # ---------------------------------------------------------
    # RESPONSE
    # ---------------------------------------------------------

    return {

        "rows":
            embedding.shape[0],

        "embedding_features":
            embedding.shape[1],

        "embedding":
            embedding.tolist()
    }

@app.post("/ask")
def ask(request: AskRequest):

    # =========================
    # JSON → DATAFRAME
    # =========================

    df = pd.DataFrame(request.data)

    df = df.dropna()

    if df.empty:
        return {
            "error": "No valid data remains after removing missing values."
        }

    # =========================
    # ENCODE
    # =========================

    encoder = SchemaEx()

    encoded_data = encoder.detect_and_encode(df)

    print(
        "ASK encoded shape:",
        encoded_data.shape
    )

    # =========================
    # 27 FEATURE CHECK
    # =========================

    if encoded_data.shape[1] != 27:

        return {
            "error": "Input must produce exactly 27 encoded features.",
            "features": encoded_data.shape[1],
            "expected": 27
        }

    # =========================
    # TENSOR
    # =========================

    tensor_data = make_tensor(
        encoded_data
    )

    print(
        "ASK tensor shape:",
        tensor_data.shape
    )

    # =========================
    # EMBEDDING
    # =========================

    embedding = make_embedding(
        tensor_data,
        autoencoder,
        device
    )

    print(
        "ASK embedding shape:",
        embedding.shape
    )

    # =========================
    # SLM
    # =========================

    prompt = (
        request.question
        + "\n\n"
        + "Data embedding: "
        + ", ".join(
            f"{float(x):.3f}"
            for x in embedding[0]
        )
        + "\nAnswer:"
    )

    answer = generate_text(
        prompt
    )

    # =========================
    # RESPONSE
    # =========================

    return {

        "question":
            request.question,

        "embedding_features":
            embedding.shape[1],

        "answer":
            answer
    }