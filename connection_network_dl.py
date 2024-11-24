#connection_network_DL
import numpy as np

# Define the expected number of features
expected_num_features = 10

# Custom function to check dimensions
def check_dimensions(input_data):
    num_rows, num_columns = input_data.shape
    if num_columns != expected_num_features:
        balance = expected_num_features - num_columns
        print(f"Dimension mismatch: Expected {expected_num_features} features, but got {num_columns}.")
        print(f"Balance: {balance} features missing.")
        
        # Return the remaining data (if any)
        remaining_data = input_data[:, :expected_num_features] if num_columns > expected_num_features else input_data
        return balance, remaining_data
    return 0, input_data  # No mismatch

# Function to process data through a communication node
def process_node(input_data, input_size, output_size):
    # Initialize weights for the node
    weights = np.random.rand(input_size, output_size)
    # Process the input data
    return np.dot(input_data, weights)

# Function to transmit data through the communication network
def transmit_data(input_data):
    # First processing node
    hidden_size = 5
    processed_data = process_node(input_data, input_data.shape[1], hidden_size)
    
    # Second processing node
    output_data = process_node(processed_data, hidden_size, 1)
    
    return output_data

# Example input data
input_data = np.random.rand(5, 8)  # 5 samples, 8 features (mismatch)

# Check dimensions before processing
balance, processed_data = check_dimensions(input_data)

# If there's a mismatch, you can decide how to handle it
if balance != 0:
    # Handle the remaining data as needed
    print("Processing remaining data...")
    # You might want to pad or truncate the data here
else:
    # Proceed with data transmission through the network
    output = transmit_data(processed_data)
    print("Output from the communication network:", output)