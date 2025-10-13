import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run a list of commands from a text file sequentially.")
    parser.add_argument("-f", "--file", required=True, help="Path to the command file (e.g., Commands.txt)")
    args = parser.parse_args()

    command_file = args.file

    if not os.path.exists(command_file):
        print(f"Error: '{command_file}' not found.")
        return

    try:
        with open(command_file, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading '{command_file}': {e}")
        return

    for line in lines:
        cmd = line.strip()

        # Skip comments and empty lines
        if not cmd or cmd.startswith("#"):
            continue

        print(f"\n>>> Running: {cmd}\n{'-'*80}")
        try:
            # Run the command and stream output live
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
        print('-' * 80)

if __name__ == "__main__":
    main()

