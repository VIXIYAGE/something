from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# LOAD MODEL
model = GPT2LMHeadModel.from_pretrained("./mini_gpt")
tokenizer = GPT2TokenizerFast.from_pretrained("./mini_gpt")

model.eval()

# PROMPT
prompt = "Sports videos on mobile users"

inputs = tokenizer(prompt, return_tensors="pt")

# GENERATE
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.7,
        do_sample=True,
        top_k=50
    )

# DECODE
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n")
print(text)