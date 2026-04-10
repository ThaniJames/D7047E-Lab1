# Shared Configuration for D7047E Lab 1 - Sentiment Analysis
# All team members must use these constants.

# Reproducibility
SEED = 42

# Data splits
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# Labels
NUM_LABELS = 2
LABEL_MAP = {0: "negative", 1: "positive"}

# Hugging Face large dataset (~1GB, 3.6M rows total)
HF_DATASET_NAME = "amazon_polarity"

# Transformer defaults
MAX_LENGTH_TRANSFORMER = 128  # token limit for BERT / DistilBERT

# Logging (wandb only)
WANDB_PROJECT = "d7047e-lab1-sentiment"
REPORT_TO = "wandb"
