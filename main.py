import argparse
from src.api import update_data


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()
    # Set a directory to save the data
    parser.add_argument('--data_path', '-o', default='dataset', type=str)
    args = parser.parse_args()
    # Check for new data
    update_data(args.data_path)
