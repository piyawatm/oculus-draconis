from datasets import load_dataset
import os

print("Downloading FFHQ-64...")
dataset = load_dataset("Dmini/FFHQ-64x64", split="train") # No streaming=True
os.makedirs("data/ffhq64_local", exist_ok=True)
dataset.save_to_disk("data/ffhq64_local", num_proc=10)
print("Done! Saved to data/ffhq64_local")
