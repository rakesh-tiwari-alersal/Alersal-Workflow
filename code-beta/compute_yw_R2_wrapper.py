import argparse
import subprocess
import sys
import re

def main():
    """
    Wrapper script that finds the first three "dips" in R-squared values for each
    short-term lag, iterating through long-term lags in descending order, with a 0.5% threshold.
    """
    parser = argparse.ArgumentParser(description='Run compute_yw_R2.py with various lag combinations and find the first three dips in R^2.')
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
    # Reverse the long-term lags to iterate from largest to smallest
    long_term_lags.reverse()

    all_dips = []

    for st_lag in short_term_lags:
        print(f"\nAnalyzing short-term lag {st_lag}...")
        
        dips_found = 0
        previous_r_squared = None
        
        # Iterate through the reversed long-term lags to find dips
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
                    
                    if previous_r_squared is not None and (previous_r_squared - current_r_squared) / previous_r_squared > 0.005:
                        # Found a dip with more than a 0.5% decrease
                        dips_found += 1
                        all_dips.append({
                            'r_squared': previous_r_squared,
                            'short_lag': st_lag,
                            'long_lag': long_term_lags[long_term_lags.index(lt_lag) - 1]
                        })
                        print(f"  Dip #{dips_found} found at long-term lag {long_term_lags[long_term_lags.index(lt_lag) - 1]} with R^2: {previous_r_squared:.4f}")
                        
                        if dips_found >= 3:
                            break # Stop after finding 3 dips
                    
                    previous_r_squared = current_r_squared
            
            except subprocess.CalledProcessError as e:
                print(f"Error running compute_yw_R2.py for lags {lags}: {e.stderr}", file=sys.stderr)
            except Exception as e:
                print(f"An unexpected error occurred: {e}", file=sys.stderr)

    # Sort the collected results by R-squared in descending order
    all_dips.sort(key=lambda x: x['r_squared'], reverse=True)
    
    # Print the top 10 combinations
    print("\n--- Top 10 Dip-Point Combinations with Highest R^2 ---")
    if not all_dips:
        print("No results were generated. Please check the input file and script paths.")
    else:
        for i, result in enumerate(all_dips[:10]):
            print(f"{i+1}. R^2: {result['r_squared']:.4f}, Lags: ({result['short_lag']}, {result['long_lag']})")

if __name__ == "__main__":
    main()