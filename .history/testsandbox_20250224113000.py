import torch
from torch.utils.data import DataLoader

# Assume the TransformerDataset class has already been defined as provided

# Create a simple sequential dataset.
# Here, we create a 1D tensor of values from 1 to 20 and then unsqueeze to make it 2D (num_samples x num_features).
data = torch.arange(1, 21, dtype=torch.float32).unsqueeze(1)  # shape: [20, 1]

# Define indices for sub-sequences. For example, each sub-sequence will have a length of 10.
# Each tuple is (start_index, end_index) so that sequence length = end_index - start_index.
indices = [
    (0, 10),  # sub-sequence from data[0] to data[9]
    (1, 11),  # sub-sequence from data[1] to data[10]
    (2, 12)   # sub-sequence from data[2] to data[11]
]

# Set the lengths for encoder and target sequences.
# Note: The code assumes that each sub-sequence length equals enc_seq_len + target_seq_len.
enc_seq_len = 6
target_seq_len = 4  # Therefore, total sequence length = 6 + 4 = 10

# In this example, dec_seq_len is not directly used in the slicing within get_src_trg,
# but it might be useful if you need further processing.
dec_seq_len = 4

# Create an instance of the dataset.
dataset = TransformerDataset(
    data=data, 
    indices=indices, 
    enc_seq_len=enc_seq_len, 
    dec_seq_len=dec_seq_len, 
    target_seq_len=target_seq_len
)

# Create a DataLoader to iterate over the dataset.
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Iterate over the DataLoader and print out the source, decoder input, and target sequences.
for src, trg, trg_y in loader:
    print("Source (src):")
    print(src)
    print("\nDecoder Input (trg):")
    print(trg)
    print("\nTarget (trg_y):")
    print(trg_y)
    print("-" * 40)
