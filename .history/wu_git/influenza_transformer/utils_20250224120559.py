import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
import dataset as ds
import utils
import transformer_timeseries as tst
import inference
from pathlib import Path

# Hyperparameters
test_size = 0.1
batch_size = 128
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
cutoff_date = datetime.datetime(2017, 1, 1)

# Model parameters
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92
enc_seq_len = 168
output_sequence_length = 48
window_size = enc_seq_len + output_sequence_length
step_size = 1
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False
epochs = 10

# Define input variables
exogenous_vars = []
input_variables = [target_col_name] + exogenous_vars
target_idx = 0
input_size = len(input_variables)

# Ensure data directory exists and contains a CSV file
data_dir = Path("C:\Users\hecht\OneDrive - Universität Heidelberg\Dokumente\_Studium\_Master\_2. Semester\DSML\Final Project\DSML_Final_Project\wu_git\influenza_transformer\data")
csv_files = list(data_dir.glob("*.csv"))
if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {data_dir}. Ensure the data directory contains at least one CSV file.")
data_path = csv_files[0]
print(f"Reading file from {data_path}")

# Read data
data = utils.read_data(data_dir=data_dir, timestamp_col_name=timestamp_col)
training_data = data[:-(round(len(data) * test_size))]

# Get training indices
training_indices = utils.get_indices_entire_sequence(
    data=training_data,
    window_size=window_size,
    step_size=step_size)

# Prepare dataset and dataloader
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data[input_variables].values).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
)
training_dataloader = DataLoader(training_data, batch_size)

# Initialize model
model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    for i, (src, tgt, tgt_y) in enumerate(training_dataloader):
        optimizer.zero_grad()

        if not batch_first:
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=output_sequence_length
        )
        src_mask = utils.generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len
        )

        prediction = model(src, tgt, src_mask, tgt_mask)
        loss = criterion(tgt_y, prediction)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for i, (src, _, tgt_y) in enumerate(training_dataloader):
            prediction = inference.run_encoder_decoder_inference(
                model=model,
                src=src,
                forecast_window=output_sequence_length,
                batch_size=src.shape[1]
            )
            loss = criterion(tgt_y, prediction)

# Ensure consistent indentation in utils.py
with open("utils.py", "r") as file:
    lines = file.readlines()

with open("utils.py", "w") as file:
    for line in lines:
        file.write(line.expandtabs(4))  # Convert tabs to spaces for consistency
