# compute_yw_DT_wrapper.py
import argparse
import subprocess
import sys
import re
import os
import csv

def main():
    """
    Wrapper script that runs compute_yw_DT.py for all combinations of short and long lags,
    and outputs Median and Stdev matrices separately (with an empty row between them).
    """
    parser = argparse.ArgumentParser(description='Run compute_yw_DT.py with various lag combinations and output results in matrix format.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) for compute_yw_DT.py')
    args = parser.parse_args()

    short_term_lags = [17, 20, 23, 27, 31, 36, 41, 47, 54]
    
    long_term_lags = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    # Create results directory if it doesn't exist
    os.makedirs('DT_results', exist_ok=True)
    
    # Extract base filename without extension for output file
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    output_file = f'DT_results/DT_{base_filename}.csv'
    
    # Prepare two results matrices
    median_matrix = [['Short Cycle/Long Cycle'] + long_term_lags]
    stdev_matrix = [['Short Cycle/Long Cycle'] + long_term_lags]
    
    # Process each short-term lag
    for st_lag in short_term_lags:
        print(f"Processing short-term lag {st_lag}...")
        median_row = [st_lag]
        stdev_row = [st_lag]
        
        # Process each long-term lag
        for lt_lag in long_term_lags:
            lags = f"{st_lag},{lt_lag}"
            
            command = [sys.executable, 'compute_yw_DT.py', '-f', args.file, '-l', lags]
            
            try:
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                output = process.stdout
                
                # Extract Median and Stdev using regex
                median_match = re.search(r"Median\(Ydt\): (-?\d+\.\d+)", output)
                stdev_match = re.search(r"Stdev\(Ydt\): (-?\d+\.\d+)", output)
                
                if median_match:
                    median_val = float(median_match.group(1))
                else:
                    median_val = None
                
                if stdev_match:
                    stdev_val = float(stdev_match.group(1))
                else:
                    stdev_val = None
                
                median_row.append(median_val)
                stdev_row.append(stdev_val)
                    
            except subprocess.CalledProcessError as e:
                print(f"Error running compute_yw_DT.py for lags {lags}: {e.stderr}", file=sys.stderr)
                median_row.append(None)
                stdev_row.append(None)
            except Exception as e:
                print(f"An unexpected error occurred: {e}", file=sys.stderr)
                median_row.append(None)
                stdev_row.append(None)
        
        median_matrix.append(median_row)
        stdev_matrix.append(stdev_row)
    
    # Write results to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(median_matrix)
        writer.writerow([])  # empty row separator
        writer.writerows(stdev_matrix)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
