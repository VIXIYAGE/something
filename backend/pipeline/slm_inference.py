import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


MODEL_PATH = "backend/models/slm"


tokenizer = GPT2TokenizerFast.from_pretrained(
    MODEL_PATH
)

tokenizer.pad_token = tokenizer.eos_token


model = GPT2LMHeadModel.from_pretrained(
    MODEL_PATH
)

model.eval()


def generate_text(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=60
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )