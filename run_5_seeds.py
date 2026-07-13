import subprocess
import sys

seeds = [2022, 2023, 42, 24, 1]  

def main():
    print("=====================================================")
    print("Starting Multi-Seed Evaluation for DGCDR_privata")
    print(f"Seeds to evaluate: {seeds}")
    print("=====================================================\n")

    for i, seed in enumerate(seeds):
        print(f">>> [{i+1}/{len(seeds)}] Running DGCDR with seed: {seed}")
        
        # Build the command to execute run_recbole_cdr.py with the specific seed
        cmd = [sys.executable, "run_recbole_cdr.py", "--seed", str(seed)]
        
        try:
            # Run the command. Output will be streamed directly to your terminal.
            subprocess.run(cmd, check=True)
            print(f">>> Finished run for seed {seed} successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error occurred while running seed {seed}. Exiting early.")
            sys.exit(1)

    print("=====================================================")
    print("All 5 runs completed successfully!")
    print("=====================================================")

if __name__ == '__main__':
    main()
