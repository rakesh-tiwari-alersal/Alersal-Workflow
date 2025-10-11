import subprocess

COMMAND_FILE = "Commands.txt"

def main():
    try:
        with open(COMMAND_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: '{COMMAND_FILE}' not found.")
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
        print('-'*80)

if __name__ == "__main__":
    main()
