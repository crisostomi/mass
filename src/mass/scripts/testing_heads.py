import glob
import torch
import os

def load_and_print_parameters():
    # Find all files matching the pattern heads_*.pt in the current directory.
    path = os.getcwd()
    file_list = glob.glob(path + "/checkpoints/ViT-B-32/head_*.pt")

    if not file_list:
        print("No heads_*.pt files found in the current directory.")
        return

    # Open a file to write the output
    with open("parameters_output.txt", "w") as output_file:
        # Process each file
        for file_path in file_list:
            output_file.write(f"\nProcessing file: {file_path}\n")

            try:
                # Load the file; map_location ensures the checkpoint is loaded on the CPU.
                checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            except Exception as e:
                output_file.write(f"Error loading {file_path}: {e}\n")
                continue

            # If the checkpoint is a dictionary, iterate and write its items.
            if isinstance(checkpoint, dict):
                for param_name, param_value in checkpoint.items():
                    output_file.write(f"\nParameter: {param_name}\n")
                    output_file.write(f"{param_value}\n")
            else:
                # If the checkpoint is not a dict, just write it.
                output_file.write(f"{checkpoint.weight.shape}\n")
                output_file.write(f"{checkpoint.weight}\n")

load_and_print_parameters()