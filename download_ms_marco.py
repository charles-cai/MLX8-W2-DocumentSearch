from datasets import load_dataset

# Download MS MARCO version 1.1, train split
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

# For validation split:
# val_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

# For test split:
# test_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="test")

# Save to disk (optional)
dataset.save_to_disk("ms_marco_v1.1_train")
