#!/usr/bin/env python3
"""
Example: How to modify model parameters
======================================

This script demonstrates how to modify parameters in model_parameters.py
and see the changes reflected in the Jupyter notebook.

Run this script to make some example parameter changes, then run the 
parameter reload cell in the notebook to see the updates.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, '/Users/rodrigoc/Documents/AGEL/Duncan_examples')

def modify_parameters():
    """
    Example function showing how to modify model_parameters.py programmatically
    """
    
    # Read the current parameter file
    param_file = '/Users/rodrigoc/Documents/AGEL/Duncan_examples/model_parameters.py'
    
    with open(param_file, 'r') as f:
        content = f.read()
    
    # Example 1: Change Einstein radius
    new_theta_E = 1.85  # Increased from ~1.79
    content = content.replace(
        "'theta_E': 1.7924038004570043,",
        f"'theta_E': {new_theta_E},"
    )
    
    # Example 2: Modify redshifts
    new_z_s1 = 1.55  # Changed from 1.52
    content = content.replace(
        "z_s1 = 1.52  # source 1 redshift",
        f"z_s1 = {new_z_s1}  # source 1 redshift (modified)"
    )
    
    # Example 3: Change shear values
    new_gamma1 = 0.08  # Changed from ~0.079
    content = content.replace(
        "'gamma1': 0.07896648881999048,",
        f"'gamma1': {new_gamma1},"
    )
    
    # Write the modified content back
    with open(param_file, 'w') as f:
        f.write(content)
    
    print("✅ Modified parameters:")
    print(f"   - Einstein radius: {new_theta_E}")
    print(f"   - Source 1 redshift: {new_z_s1}")
    print(f"   - Shear γ1: {new_gamma1}")
    print("\n🔄 Now run the parameter reload cell in the notebook to see changes!")

def reset_parameters():
    """
    Reset parameters to original values
    """
    
    param_file = '/Users/rodrigoc/Documents/AGEL/Duncan_examples/model_parameters.py'
    
    with open(param_file, 'r') as f:
        content = f.read()
    
    # Reset to original values
    content = content.replace(
        "'theta_E': 1.85,",
        "'theta_E': 1.7924038004570043,"
    )
    content = content.replace(
        "z_s1 = 1.55  # source 1 redshift (modified)",
        "z_s1 = 1.52  # source 1 redshift"
    )
    content = content.replace(
        "'gamma1': 0.08,",
        "'gamma1': 0.07896648881999048,"
    )
    
    with open(param_file, 'w') as f:
        f.write(content)
    
    print("✅ Parameters reset to original values")
    print("🔄 Run the parameter reload cell in the notebook to see the reset!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Modify model parameters")
    parser.add_argument("--modify", action="store_true", help="Modify parameters")
    parser.add_argument("--reset", action="store_true", help="Reset parameters")
    
    args = parser.parse_args()
    
    if args.modify:
        modify_parameters()
    elif args.reset:
        reset_parameters()
    else:
        print("Parameter Modification Example")
        print("=" * 30)
        print("Usage:")
        print("  python parameter_example.py --modify   # Modify some parameters")
        print("  python parameter_example.py --reset    # Reset to original values")
        print()
        print("Or import this module and call functions directly:")
        print("  from parameter_example import modify_parameters, reset_parameters")
        print("  modify_parameters()  # or reset_parameters()")
