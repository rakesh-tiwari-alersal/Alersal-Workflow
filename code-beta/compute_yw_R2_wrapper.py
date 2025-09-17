import argparse
import subprocess
import sys
import re
import os
import csv

def main():
    """
    Wrapper script that runs compute_yw_R2.py for all combinations of short and long lags,
    and outputs results in a matrix format suitable for contour plotting.
    """
    parser = argparse.ArgumentParser(description='Run compute_yw_R2.py with various lag combinations and output results in matrix format.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) for compute_yw_R2.py')
    args = parser.parse_args()

    short_term_lags = [17, 20, 23, 27, 31, 36, 41, 47, 54]
    
    # Original Long-term lags from Table 2 of the attached PDF
    long_term_lags = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    # Create results directory if it doesn't exist
    os.makedirs('R2_results', exist_ok=True)
    
    # Extract base filename without extension for output file
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    output_file = f'R2_results/R2_{base_filename}.csv'
    
    # Prepare data matrix
    results_matrix = []
    
    # Add header row (first cell is empty, then long term lags)
    header_row = ['Short Cycle/Long Cycle'] + long_term_lags
    results_matrix.append(header_row)
    
    # Process each short-term lag
    for st_lag in short_term_lags:
        print(f"Processing short-term lag {st_lag}...")
        row_data = [st_lag]  # First column is the short-term lag
        
        # Process each long-term lag
        for lt_lag in long_term_lags:
            lags = f"{st_lag},{lt_lag}"
            
            command = [sys.executable, 'compute_yw_R2.py', '-f', args.file, '-l', lags]
            
            try:
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                output = process.stdout
                
                # Regex to find the R-squared value for linear regression
                match = re.search(r"Linear OLS \(Polynomial Degree 1\): (-?\d+\.\d+)", output)
                if match:
                    current_r_squared = float(match.group(1))
                    row_data.append(current_r_squared)
                else:
                    row_data.append(None)  # Or some indicator of missing data
                    
            except subprocess.CalledProcessError as e:
                print(f"Error running compute_yw_R2.py for lags {lags}: {e.stderr}", file=sys.stderr)
                row_data.append(None)
            except Exception as e:
                print(f"An unexpected error occurred: {e}", file=sys.stderr)
                row_data.append(None)
        
        # Add completed row to matrix
        results_matrix.append(row_data)
    
    # Write results to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results_matrix)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()