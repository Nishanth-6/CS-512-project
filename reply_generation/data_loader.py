from datasets import load_dataset
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("wick1d/Personalized_Safety_Data")
ds["train"].to_csv("data.csv")