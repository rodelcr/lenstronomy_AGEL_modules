# Model Parameters System

This directory now includes an external parameter management system for the Jupyter notebook analysis.

## 📁 Files

- **`model_parameters.py`** - Main parameter configuration file
- **`parameter_example.py`** - Example script showing how to modify parameters
- **`20250903_plotting_model_simulation.ipynb`** - Updated notebook with parameter reloading

## 🚀 Quick Start

1. **Open the Jupyter notebook** and run the first cell (parameter reload cell)
2. **Modify parameters** in `model_parameters.py`
3. **Reload parameters** by running the first cell again
4. **Continue analysis** with updated parameters

## 📋 Parameter Categories

### Cosmology & Redshifts
```python
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
z_l = 0.68      # lens redshift
z_s1 = 1.52     # source 1 redshift  
z_s2 = 2.11     # source 2 redshift
```

### Lens Model (SIE + Shear)
```python
kwargs_sie_init = {
    'theta_E': 1.7924,
    'e1': 0.0416,
    'e2': -0.0269,
    'center_x': 0.0979,
    'center_y': 0.0204
}

kwargs_shear_init = {
    'gamma1': 0.0790,
    'gamma2': -0.0086
}
```

### Source Models (3 Sersic profiles)
```python
kwargs_source_init = [
    {'R_sersic': 0.204, 'n_sersic': 1.879, ...},  # Source 1
    {'R_sersic': 0.111, 'n_sersic': 0.525, ...},  # Source 2a  
    {'R_sersic': 0.194, 'n_sersic': 0.995, ...}   # Source 2b
]
```

### Lens Light Model (Sersic)
```python
kwargs_lens_light_init = [{
    'R_sersic': 2.707,
    'n_sersic': 6.004,
    'e1': 0.0727,
    'e2': -0.0502,
    'center_x': 0.0924,
    'center_y': 0.0146
}]
```

## 🔄 Workflow

### Method 1: Direct Editing
1. Open `model_parameters.py` in your editor
2. Modify any parameter values
3. Save the file
4. Run the parameter reload cell in the notebook

### Method 2: Programmatic Modification
```python
# In a separate script or notebook cell
import parameter_example
parameter_example.modify_parameters()  # Make some changes
# Then reload in the main notebook
```

### Method 3: Interactive Testing
```python
# In the notebook, after reloading parameters
print(f"Current Einstein radius: {kwargs_lens_init[0]['theta_E']}")

# Temporarily override for testing
kwargs_lens_init[0]['theta_E'] = 2.0  # Test value
```

## 🎯 Benefits

✅ **Clean separation** - Parameters separate from analysis code  
✅ **Version control friendly** - Easy to track parameter changes  
✅ **Reusable** - Same parameters across multiple notebooks  
✅ **Live reloading** - Update parameters without restarting kernel  
✅ **Organized** - Parameters grouped by category with documentation  

## 🔧 Utility Functions

The parameter file includes helper functions:

```python
import model_parameters as mp

# Get all parameters at once
kwargs_params = mp.get_all_model_params()
kwargs_model = mp.get_model_config()
kwargs_constraints = mp.get_constraints()
kwargs_likelihood = mp.get_likelihood_config()

# Get numerical configuration
kwargs_numerics = mp.get_numerical_config(mask_array, supersampling_mask)

# Debug current values
mp.print_current_params()
```

## 💡 Tips

- **Always run the parameter reload cell first** when opening the notebook
- **Use version control** to track parameter changes over time
- **Test parameter changes incrementally** - make small changes and check results
- **Document important parameter sets** by creating copies of `model_parameters.py`
- **Use the print functions** to verify parameters are loaded correctly

## 🚨 Troubleshooting

**Parameters not loading?**
- Make sure you're in the correct directory
- Check that `model_parameters.py` exists and has no syntax errors
- Run the parameter reload cell (first cell in notebook)

**Import errors?**
- Verify Python path is set correctly in the reload cell
- Check file permissions on `model_parameters.py`

**Values not updating?**
- Make sure you saved `model_parameters.py` after editing
- Run the reload cell again (it forces a module reload)
- Restart the notebook kernel if problems persist

## 📖 Example Workflows

### Systematic Parameter Study
1. Create backup: `cp model_parameters.py model_parameters_backup.py`
2. Modify one parameter at a time in `model_parameters.py`
3. Reload and run analysis in notebook
4. Save results before next parameter change
5. Restore from backup when needed

### Collaborative Work
1. Share `model_parameters.py` with team members
2. Each person can modify parameters locally
3. Use git to merge parameter changes
4. Everyone reloads to get latest parameter set

### A/B Testing
1. Create `model_parameters_v1.py` and `model_parameters_v2.py`
2. Copy desired version to `model_parameters.py`
3. Reload and compare results
4. Keep best performing parameter set
