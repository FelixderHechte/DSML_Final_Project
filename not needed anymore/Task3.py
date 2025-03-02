import os 
import numpy as np
import math
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Optional, Union, Tuple
from pathlib import Path
import positional_encoder as pe
import matplotlib.pyplot as plt


def generate_square_subsequent_mask(dim1: int, dim2: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
        device: (optional) torch.device on which to create the mask

    Returns:
        A Tensor of shape [dim1, dim2]
    """
    mask = torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask



def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
        """
        Produce all the start and end index positions of all sub-sequences.
        The indices will be used to split the data into sub-sequences on which 
        the models will be trained. 

        Returns a tuple with four elements:
        1) The index position of the first element to be included in the input sequence
        2) The index position of the last element to be included in the input sequence
        3) The index position of the first element to be included in the target sequence
        4) The index position of the last element to be included in the target sequence

        
        Args:
            num_obs (int): Number of observations in the entire dataset for which
                            indices must be generated.

            input_len (int): Length of the input sequence (a sub-sequence of 
                             of the entire data sequence)

            step_size (int): Size of each step as the data sequence is traversed.
                             If 1, the first sub-sequence will be indices 0-input_len, 
                             and the next will be 1-input_len.

            forecast_horizon (int): How many index positions is the target away from
                                    the last index position of the input sequence?
                                    If forecast_horizon=1, and the input sequence
                                    is data[0:10], the target will be data[11:taget_len].

            target_len (int): Length of the target / output sequence.
        """

        input_len = round(input_len) # just a precaution
        start_position = 0
        stop_position = num_obs-1 # because of 0 indexing
        
        subseq_first_idx = start_position
        subseq_last_idx = start_position + input_len
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len 
        print("target_last_idx is {}".format(target_last_idx))
        print("stop_position is {}".format(stop_position))
        indices = []
        while target_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size
            target_first_idx = subseq_last_idx + forecast_horizon
            target_last_idx = target_first_idx + target_len

        return indices

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences. 

        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences. 
        
        Args:
            num_obs (int): Number of observations (time steps) in the entire 
                           dataset for which indices must be generated, e.g. 
                           len(data)

            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50 
                               time steps, window_size = 100+50 = 150

            step_size (int): Size of each step as the data sequence is traversed 
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size], 
                             and the next will be [1:window_size].

        Return:
            indices: a list of tuples
        """

        stop_position = len(data)-1 # 1- because of 0 indexing
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        
        subseq_last_idx = window_size
        
        indices = []
        
        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            
            subseq_last_idx += step_size

        return indices


def read_data(data_dir: Union[str, Path] = "data",  
    timestamp_col_name: str="timestamp") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        data_dir: str or Path object specifying the path to the directory 
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index 
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)
    
    # Read csv file
    npy_files = list(data_dir.glob("*.npy"))
    
    if len(npy_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(npy_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = npy_files[0]

    print("Reading file in {}".format(data_path))

    data = np.load(data_path, allow_pickle=True).item()
    data = pd.DataFrame(data)

    # Make sure all "n/e" values have been removed from df. 
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df


def generate_square_subsequent_mask(dim1: int, dim2: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
        device: (optional) torch.device on which to create the mask

    Returns:
        A Tensor of shape [dim1, dim2]
    """
    mask = torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask



def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
        """
        Produce all the start and end index positions of all sub-sequences.
        The indices will be used to split the data into sub-sequences on which 
        the models will be trained. 

        Returns a tuple with four elements:
        1) The index position of the first element to be included in the input sequence
        2) The index position of the last element to be included in the input sequence
        3) The index position of the first element to be included in the target sequence
        4) The index position of the last element to be included in the target sequence

        
        Args:
            num_obs (int): Number of observations in the entire dataset for which
                            indices must be generated.

            input_len (int): Length of the input sequence (a sub-sequence of 
                             of the entire data sequence)

            step_size (int): Size of each step as the data sequence is traversed.
                             If 1, the first sub-sequence will be indices 0-input_len, 
                             and the next will be 1-input_len.

            forecast_horizon (int): How many index positions is the target away from
                                    the last index position of the input sequence?
                                    If forecast_horizon=1, and the input sequence
                                    is data[0:10], the target will be data[11:taget_len].

            target_len (int): Length of the target / output sequence.
        """

        input_len = round(input_len) # just a precaution
        start_position = 0
        stop_position = num_obs-1 # because of 0 indexing
        
        subseq_first_idx = start_position
        subseq_last_idx = start_position + input_len
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len 
        print("target_last_idx is {}".format(target_last_idx))
        print("stop_position is {}".format(stop_position))
        indices = []
        while target_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size
            target_first_idx = subseq_last_idx + forecast_horizon
            target_last_idx = target_first_idx + target_len

        return indices

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences. 

        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences. 
        
        Args:
            num_obs (int): Number of observations (time steps) in the entire 
                           dataset for which indices must be generated, e.g. 
                           len(data)

            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50 
                               time steps, window_size = 100+50 = 150

            step_size (int): Size of each step as the data sequence is traversed 
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size], 
                             and the next will be [1:window_size].

        Return:
            indices: a list of tuples
        """

        stop_position = len(data)-1 # 1- because of 0 indexing
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        
        subseq_last_idx = window_size
        
        indices = []
        
        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            
            subseq_last_idx += step_size

        return indices


def read_data(data_dir: Union[str, Path] = "data",  
    timestamp_col_name: str="timestamp") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        data_dir: str or Path object specifying the path to the directory 
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index 
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)
    
    # Read csv file
    npy_files = list(data_dir.glob("*.npy"))
    
    if len(npy_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(npy_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = npy_files[0]

    print("Reading file in {}".format(data_path))

    data = np.load(data_path, allow_pickle=True).item()
    data = pd.DataFrame(data)

    # Make sure all "n/e" values have been removed from df. 
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df




class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)



def run_encoder_decoder_inference(
    model: nn.Module, 
    src: torch.Tensor, 
    forecast_window: int,
    batch_size: int,
    device,
    batch_first: bool=False
    ) -> torch.Tensor:

    """
    NB! This function is currently only tested on models that work with 
    batch_first = False
    
    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.  
    
    This function returns a prediction of length forecast_window for each batch in src
    
    NB! If you want the inference to be done without gradient calculation, 
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()
        
    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be 
    backpropagated during training and gradient calculation hence is required.
    
    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that 
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.
    
    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before 
               passed to this function.
               
        src: The input to the model
        
        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.
                           
        batch_size: batch size
        
        batch_first: If true, the shape of the model input should be 
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]
    
    """

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    # Take the last value of thetarget variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, 0] if batch_first == False else src[:, -1, 0] # shape [1, batch_size, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0).unsqueeze(0) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0).unsqueeze(-1)

    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window-1):

        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        dim_b = src.shape[1] if batch_first == True else src.shape[0]

        tgt_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
            device=device
            )

        src_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b,
            device=device
            )

        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask) 

        # If statement simply makes sure that the predicted value is 
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step 
            # represented in tgt
            last_predicted_value = prediction[-1, :, :] 

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with 
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)
    
    # Create masks
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        device=device
        )

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b,
        device=device
        )

    # Make final prediction
    final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction

    class TimeSeriesTransformer(nn.Module):
            def __init__(self, 
                        input_size: int,
                        dec_seq_len: int,
                        batch_first: bool = True,
                        out_seq_len: int = 58,
                        dim_val: int = 512,  
                        n_encoder_layers: int = 4,
                        n_decoder_layers: int = 4,
                        n_heads: int = 8,
                        dropout_encoder: float = 0.2, 
                        dropout_decoder: float = 0.2,
                        dropout_pos_enc: float = 0.1,
                        dim_feedforward_encoder: int = 2048,
                        dim_feedforward_decoder: int = 2048,
                        num_predicted_features: int = 1,
                        verbose: bool = False):

                super().__init__()
                self.dec_seq_len = dec_seq_len
                self.verbose = verbose  # Enable debug printing

                if verbose:
                    print(f"Initializing Transformer: input_size={input_size}, num_predicted_features={num_predicted_features}")

                # Input layers
                self.encoder_input_layer = nn.Linear(input_size, dim_val)
                self.decoder_input_layer = nn.Linear(num_predicted_features, dim_val)

                # Positional Encoding
                self.positional_encoding_layer = pe.PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc)

                # Transformer Encoder
                encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads,
                                                        dim_feedforward=dim_feedforward_encoder,
                                                        dropout=dropout_encoder, batch_first=batch_first)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

                # Transformer Decoder
                decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val, nhead=n_heads,
                                                        dim_feedforward=dim_feedforward_decoder,
                                                        dropout=dropout_decoder, batch_first=batch_first)
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

                # Output layer
                self.linear_mapping = nn.Linear(dim_val, num_predicted_features)

            def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, 
                        tgt_mask: torch.Tensor = None) -> torch.Tensor:
                
                # Debug: Print shapes if verbose mode is enabled
                if self.verbose:
                    print(f"src shape: {src.shape}, expected last dim: {self.encoder_input_layer.in_features}")
                    print(f"tgt shape: {tgt.shape}, expected last dim: {self.decoder_input_layer.in_features}")

                # Validate input dimensions
                assert src.shape[-1] == self.encoder_input_layer.in_features, \
                    f"src last dim ({src.shape[-1]}) must match input_size ({self.encoder_input_layer.in_features})"
                assert tgt.shape[-1] == self.decoder_input_layer.in_features, \
                    f"tgt last dim ({tgt.shape[-1]}) must match num_predicted_features ({self.decoder_input_layer.in_features})"

                # Encoding
                src = self.encoder_input_layer(src)
                src = self.positional_encoding_layer(src)
                src = self.encoder(src)

                # Decoding
                decoder_output = self.decoder_input_layer(tgt)
                decoder_output = self.decoder(decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)

                # Output mapping
                decoder_output = self.linear_mapping(decoder_output)

                if self.verbose:
                    print(f"Output shape: {decoder_output.shape}, expected last dim: {self.linear_mapping.out_features}")

                return decoder_output

# Hyperparameters
batch_size = 128

target_col_name = 'x' # first of three features
timestamp_col = "t"
# Define input variables 
exogenous_vars = ['y', 'z'] # should contain strings. 

#Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

# Data parameters

input_size = len(input_variables)

step_size=10
forecast_window = 2 # we start with trying 2 values
enc_seq_len = 4 # previous 4 data points


window_size = 6
batch_first = True



epochs = 2  # Increase the number of epochs

# Define training data directory
data_dir = r"C:\Users\hecht\OneDrive - Universität Heidelberg\Dokumente\_Studium\_Master\_2. Semester\DSML\Final Project\DSML_Final_Project\Traindata"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Process each npy file separately
for file_name in os.listdir(data_dir):
    if file_name.endswith('.npy'):
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path)
        print(f"Processing file: {file_name}")
        
        z_size = data.shape[1]  # Update z_size for this file
        print(f"New z_size: {z_size}")

        # Remove old model from memory
        if 'model' in locals():
            del model
            torch.cuda.empty_cache()

        # Create DataLoader
        training_indices = get_indices_entire_sequence(data, window_size, step_size)
        if len(training_indices) == 0:
            print("No valid training indices found. Skipping this file.")
            continue

        training_data = TransformerDataset(
            data=torch.tensor(data).float(),
            indices=training_indices,
            enc_seq_len=enc_seq_len,
            dec_seq_len=forecast_window,
            target_seq_len=forecast_window
        )
        training_dataloader = DataLoader(training_data, batch_size, drop_last=True)

        print(f"New DataLoader size: {len(training_dataloader)}")

        # Initialize new model
        model = TimeSeriesTransformer(
            input_size=z_size,  # Update input_size to match the last dimension of src
            dec_seq_len=forecast_window,
            batch_first=batch_first,
            num_predicted_features=z_size,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters())  # Reinitialize optimizer
        criterion = torch.nn.MSELoss()

        print(f"Model initialized on device: {next(model.parameters()).device}")

        # Store loss for plotting
        batch_losses = []

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            for i, (src, tgt, tgt_y) in enumerate(training_dataloader):
                src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

                optimizer.zero_grad()
                prediction = model(src, tgt, 
                                   generate_square_subsequent_mask(forecast_window, enc_seq_len).to(device), 
                                   generate_square_subsequent_mask(forecast_window, forecast_window).to(device))

                loss = criterion(tgt_y, prediction)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())  # Store loss
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(training_dataloader)}, Loss: {loss.item():.4f}")

        # Plot loss curve for this file
        plt.figure(figsize=(8, 5))
        plt.plot(batch_losses, label='Training Loss', color='blue')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for {file_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
