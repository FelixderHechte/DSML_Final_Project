{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "# Assume the TransformerDataset class has already been defined as provided\n",
    "\n",
    "# Create a simple sequential dataset.\n",
    "# Here, we create a 1D tensor of values from 1 to 20 and then unsqueeze to make it 2D (num_samples x num_features).\n",
    "data = torch.arange(1, 21, dtype=torch.float32).unsqueeze(1)  # shape: [20, 1]\n",
    "\n",
    "# Define indices for sub-sequences. For example, each sub-sequence will have a length of 10.\n",
    "# Each tuple is (start_index, end_index) so that sequence length = end_index - start_index.\n",
    "indices = [\n",
    "    (0, 10),  # sub-sequence from data[0] to data[9]\n",
    "    (1, 11),  # sub-sequence from data[1] to data[10]\n",
    "    (2, 12)   # sub-sequence from data[2] to data[11]\n",
    "]\n",
    "\n",
    "# Set the lengths for encoder and target sequences.\n",
    "# Note: The code assumes that each sub-sequence length equals enc_seq_len + target_seq_len.\n",
    "enc_seq_len = 6\n",
    "target_seq_len = 4  # Therefore, total sequence length = 6 + 4 = 10\n",
    "\n",
    "# In this example, dec_seq_len is not directly used in the slicing within get_src_trg,\n",
    "# but it might be useful if you need further processing.\n",
    "dec_seq_len = 4\n",
    "\n",
    "# Create an instance of the dataset.\n",
    "dataset = TransformerDataset(\n",
    "    data=data, \n",
    "    indices=indices, \n",
    "    enc_seq_len=enc_seq_len, \n",
    "    dec_seq_len=dec_seq_len, \n",
    "    target_seq_len=target_seq_len\n",
    ")\n",
    "\n",
    "# Create a DataLoader to iterate over the dataset.\n",
    "loader = DataLoader(dataset, batch_size=2, shuffle=False)\n",
    "\n",
    "# Iterate over the DataLoader and print out the source, decoder input, and target sequences.\n",
    "for src, trg, trg_y in loader:\n",
    "    print(\"Source (src):\")\n",
    "    print(src)\n",
    "    print(\"\\nDecoder Input (trg):\")\n",
    "    print(trg)\n",
    "    print(\"\\nTarget (trg_y):\")\n",
    "    print(trg_y)\n",
    "    print(\"-\" * 40)\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
