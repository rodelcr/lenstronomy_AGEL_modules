"""
Model Parameters Configuration
=============================

This script contains all the model parameters for the lensing analysis.
Modify values here and reload in the Jupyter notebook to update parameters.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM, wCDM
import astropy.units as u

# =============================================================================
# COSMOLOGY PARAMETERS
# =============================================================================
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, Ob0=0.05)

# REDSHIFT PARAMETERS # Get from mock data files in /lenstronomy_AGEL_modules/tutorials_DB_2025_09/mocks/redshifts
# ADD THESE REDSHIFTS 
z_l = 
z_s1 = 
z_s2 = 

lens_redshift_list = [z_l, z_l, z_s1]
source_redshift_list = [z_s1, z_s2]

D_s1 = cosmo.angular_diameter_distance(z_s1).value
D_s2 = cosmo.angular_diameter_distance(z_s2).value
D_s1s2 = cosmo.angular_diameter_distance_z1z2(z_s1, z_s2).value
D_ls1 = cosmo.angular_diameter_distance_z1z2(z_l, z_s1).value
D_ls2 = cosmo.angular_diameter_distance_z1z2(z_l, z_s2).value

#cosmological scaling factor=deflection_scaling= 1/β
deflection_scaling = D_s1 / D_ls1 * D_ls2 / D_s2  
beta=1/deflection_scaling

# =============================================================================
# Image multiplicity for Position Modeling ## NEEDS TO BE SET UP MANUALLY
# =============================================================================
# HOW MANY IMAGES DO YOU SEE FROM EACH SOURCE?

s1_image_num = 
s2_image_num = 


# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Basic data parameters
background_rms = 0.001
exp_time = 5000.0
numPix = 80
pixel_scale = 0.05
fwhm = 0.02
psf_type = 'GAUSSIAN'

# Coordinate parameters (will be updated from real data)
lens_center_ra = 0.03
lens_center_dec = -0.140403937171283193
lens_bound = 0.05
source_bound = 1.0

# =============================================================================
# LENS MODEL PARAMETERS
# =============================================================================
lens_model_list = ['SIE', 'SHEAR_GAMMA_PSI', 'SIE']

# SIE (Singular Isothermal Ellipsoid) parameters
# Update these parameters based on your lens model from the position modeling results

kwargs_sie_init = {
    'theta_E': 1.2,
    'e1': -0.14158633009318525,
    'e2': -0.06,
    'center_x': 0.0,
    'center_y': -0.00
}

kwargs_sie_sigma = {
    'theta_E': 0.01,
    'e1': 0.005,
    'e2': 0.005,
    'center_x': 0.005,
    'center_y': 0.005
}

kwargs_sie_lower = {
    'theta_E': 1.0,
    'e1': -0.5,
    'e2': -0.5,
    'center_x': lens_center_ra - lens_bound,
    'center_y': lens_center_dec - lens_bound
}

kwargs_sie_upper = {
    'theta_E': 3.0,
    'e1': 0.5,
    'e2': 0.5,
    'center_x': lens_center_ra + lens_bound,
    'center_y': lens_center_dec + lens_bound
}

kwargs_sie_fixed = {}

# Shear parameters
kwargs_shear_init =   {'gamma_ext': 0.11915106152231696,
   'psi_ext': 0.5038232369170116,
   'ra_0': 0,
   'dec_0': 0}

kwargs_shear_sigma = {
    'gamma_ext': 0.01,
    'psi_ext': 0.01
}

kwargs_shear_lower = {
    'gamma_ext': 0.0,
    'psi_ext': 0.0
}

kwargs_shear_upper = {
    'gamma_ext': 0.2,
    'psi_ext': np.pi
}

kwargs_shear_fixed ={
    'ra_0': 0,
   'dec_0': 0
}


kwargs_sie_init_src = { # Update these parameters based on your lens model from the position modeling results

    'theta_E': 0.2,
    'e1': 0.04158633009318525,
    'e2': -0.02686739316761842,
    'center_x': 0.0,
    'center_y': 0.0
}

kwargs_sie_sigma_src = {
    'theta_E': 0.01,
    'e1': 0.05,
    'e2': 0.05,
    'center_x': 0.005,
    'center_y': 0.005
}

kwargs_sie_lower_src = {
    'theta_E': 0.00001,
    'e1': -0.5,
    'e2': -0.5,
    'center_x': lens_center_ra - lens_bound,
    'center_y': lens_center_dec - lens_bound
}

kwargs_sie_upper_src = {
    'theta_E': 0.6,
    'e1': 0.5,
    'e2': 0.5,
    'center_x': lens_center_ra + lens_bound,
    'center_y': lens_center_dec + lens_bound
}

kwargs_sie_fixed_src = {}

# Combined lens parameters
kwargs_lens_init = [kwargs_sie_init, kwargs_shear_init, kwargs_sie_init_src]
kwargs_lens_sigma = [kwargs_sie_sigma, kwargs_shear_sigma, kwargs_sie_sigma_src]
kwargs_lens_fixed = [kwargs_sie_fixed, kwargs_shear_fixed, kwargs_sie_fixed_src]
kwargs_lower_lens = [kwargs_sie_lower, kwargs_shear_lower, kwargs_sie_lower_src]
kwargs_upper_lens = [kwargs_sie_upper, kwargs_shear_upper, kwargs_sie_upper_src]

lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lower_lens, kwargs_upper_lens]

# =============================================================================
# LENS LIGHT MODEL PARAMETERS
# =============================================================================
lens_light_model_list = ['SERSIC_ELLIPSE']

kwargs_lens_light_init = [{
    'R_sersic': 2.706534933472987,
    'n_sersic': 6.004395416781097,
    'e1': 0.0,
    'e2': 0.0,
    'center_x': kwargs_sie_init['center_x'],
    'center_y': kwargs_sie_init['center_y']
}]

kwargs_lens_light_sigma = [{
    'R_sersic': 0.1,
    'n_sersic': 0.1,
    'e1': 0.01,
    'e2': 0.01,
    'center_x': 0.005,
    'center_y': 0.005
}]

kwargs_lower_lens_light = [{
    'R_sersic': 0.5,
    'n_sersic': 0.5,
    'e1': -0.5,
    'e2': -0.5,
    'center_x': lens_center_ra - lens_bound,
    'center_y': lens_center_dec - lens_bound
}]

kwargs_upper_lens_light = [{
    'R_sersic': 10,
    'n_sersic': 10,
    'e1': 0.5,
    'e2': 0.5,
    'center_x': lens_center_ra + lens_bound,
    'center_y': lens_center_dec + lens_bound
}]

kwargs_lens_light_fixed = [{}]

lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lower_lens_light, kwargs_upper_lens_light]

# =============================================================================
# SOURCE LIGHT MODEL PARAMETERS
# =============================================================================
source_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
image_plane_source_list = [False, False]

kwargs_source_init = [
    {
        'R_sersic': 0.2035989518818442,
        'n_sersic': 1.8792552215500815,
        'e1': 0.4371870389197965,
        'e2': -0.10569696047809995,
        'center_x': 0.,
        'center_y': 0.
    },
    {
        'R_sersic': 0.11098111524947607,
        'n_sersic': 0.5251575461491514,
        'e1': 0.497834080242937,
        'e2': -0.11661295352147125,
        'center_x': 0.,
        'center_y': 0.
    },
    # {
    #     'R_sersic': 0.19362063802763319,
    #     'n_sersic': 0.9945376617604998,
    #     'e1': 0.3888323276915802,
    #     'e2': -0.339222236571743,
    #     'center_x': 0.3773625720089029,
    #     'center_y': -0.41000035597773105
    # }
]

kwargs_source_sigma = [## ALSO CONSIDER TIGHTENING THESE UNCERTAINTIES TO AVOID UNPHYSICAL MODELS/FORCE IT TO KEEP THE SOURCES IN THE REGION OF INTEREST
    {'R_sersic': 0.1, 'n_sersic': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.01, 'center_y': 0.01}, 
    {'R_sersic': 0.1, 'n_sersic': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.01, 'center_y': 0.01},
    #{'R_sersic': 0.1, 'n_sersic': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}
]

kwargs_lower_source = [
    {'R_sersic': 0.005, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10},
    {'R_sersic': 0.005, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10},
    #{'R_sersic': 0.005, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -10, 'center_y': -10}
]

kwargs_upper_source = [
    {'R_sersic': 0.3, 'n_sersic': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
    {'R_sersic': 0.3, 'n_sersic': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10},
    #{'R_sersic': 10, 'n_sersic': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 10, 'center_y': 10}
]

kwargs_source_fixed = [{}, {}]#[{}, {}, {}]

source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_lower_source, kwargs_upper_source]

# =============================================================================
# POINT SOURCE MODEL PARAMETERS (EMPTY)
# =============================================================================
point_source_list = []
kwargs_ps_init = []
kwargs_ps_sigma = []
kwargs_lower_ps = []
kwargs_upper_ps = []
kwargs_ps_fixed = []

ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed, kwargs_lower_ps, kwargs_upper_ps]

# =============================================================================
# MASKING PARAMETERS
# =============================================================================
central_mask = False
central_mask_r = 1
r_mask_list = [[3]]  # list of mask radii for each mask band and filters
threshold_list = [10**(3.9/2.5)]  # supersampling threshold

# =============================================================================
# NUMERICAL PARAMETERS
# =============================================================================
supersampling_factor = 3
supersampling_convolution = True
supersampling_kernel_size = 7
point_source_supersampling_factor = 3
compute_mode = 'adaptive'

# =============================================================================
# CONSTRAINTS PARAMETERS
# =============================================================================
def get_constraints():
    """Return constraints dictionary"""
    return {
        'image_plane_source_list': image_plane_source_list,
        'joint_lens_with_source_light': [[0, 2, ['center_x', 'center_y', 'e1', 'e2']]] # first number is the source , second is index mass
    }

# =============================================================================
# CUSTOM LIKELIHOOD FUNCTION
# =============================================================================
def custom_logL_function(kwargs_lens=None, kwargs_source=None, 
                        kwargs_lens_light=None, kwargs_ps=None, 
                        kwargs_special=None, kwargs_extinction=None,
                        kwargs_tracer_source=None):
    """
    Custom likelihood function with priors
    """
    from lenstronomy.Util.param_util import ellipticity2phi_q
    
    logL = 0.
    bound = 30. / 180. * np.pi

    # PA constraint
    # find e1 & e2 param., convert to PA (phi)
    mass_phi, mass_q = ellipticity2phi_q(kwargs_lens[0]['e1'], kwargs_lens[0]['e2'])
    light_phi, light_q = ellipticity2phi_q(kwargs_lens_light[0]['e1'], kwargs_lens_light[0]['e2'])

    # ellipticity prior to reduce disparity between mass and light
    if mass_q < light_q:  
        logL += -0.5 * (mass_q - light_q)**2 / 0.01**2
    logL += -0.5 * (mass_phi - light_phi)**2 / bound**2

    # lens centre prior
    logL -= ((kwargs_lens[0]['center_x'] - lens_center_ra)**2 + (kwargs_lens[0]['center_y'] - lens_center_dec)**2)*10

    return logL

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_all_model_params():
    """
    Return all model parameters in the format expected by lenstronomy
    """
    return {
        'lens_model': lens_params,
        'source_model': source_params,
        'lens_light_model': lens_light_params,
        'point_source_model': ps_params
    }

def get_model_config():
    """
    Return model configuration dictionary
    """
    return {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list,
        'point_source_model_list': point_source_list,
        'additional_images_list': None,
        'fixed_magnification_list': None,
        'multi_plane': True, ## Key for DSPL
        'lens_redshift_list': lens_redshift_list,
        'cosmo': cosmo,
        'z_source': z_s2,
        'source_redshift_list': source_redshift_list
    }

def get_numerical_config(supersampling_mask_array=None, mask_array=None):
    """
    Return numerical configuration dictionary
    """
    config = {
        'supersampling_factor': supersampling_factor,
        'supersampling_convolution': supersampling_convolution,
        'supersampling_kernel_size': supersampling_kernel_size,
        'flux_evaluate_indexes': None,
        'point_source_supersampling_factor': point_source_supersampling_factor,
        'compute_mode': compute_mode
    }
    
    if supersampling_mask_array is not None and mask_array is not None:
        config['supersampled_indexes'] = np.array(supersampling_mask_array * mask_array, dtype=bool)
    
    return config

def get_likelihood_config():
    """
    Return likelihood configuration dictionary
    """
    return {
        'check_bounds': True,
        'force_no_add_image': False,
        'source_marg': True,
        'image_position_uncertainty': 0.001,
        'custom_logL_addition': custom_logL_function
    }

def print_current_params():
    """
    Print current parameter values for debugging
    """
    print("=" * 50)
    print("CURRENT MODEL PARAMETERS")
    print("=" * 50)
    print(f"Lens center: ({lens_center_ra:.6f}, {lens_center_dec:.6f})")
    print(f"Redshifts - Lens: {z_l}, Source 1: {z_s1}, Source 2: {z_s2}")
    print(f"Einstein radius: {kwargs_lens_init[0]['theta_E']:.4f}")
    print(f"Shear: γ1={kwargs_lens_init[1]['gamma1']:.4f}, γ2={kwargs_lens_init[1]['gamma2']:.4f}")
    print("=" * 50)

def update_init(old, new):  # from old to new
    for i, (o, n) in enumerate(zip(old, new)):
        for key, value in o.items():
            if key in n:
                n[key] = value

    return new


if __name__ == "__main__":
    print_current_params()
