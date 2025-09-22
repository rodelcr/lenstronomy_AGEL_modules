# Developing this as a modular script to load initial modules and parameters for AGEL's DSPL modeling tutorials.
# 
#   
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
import time
import pickle
import h5py
import joblib
import lenstronomy.Util.util as util
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Util import mask_util
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Util.util import array2image
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Plots import  plot_util
import paramiko
from astropy.io import fits
from lenstronomy.Plots.model_plot import ModelPlot
#from chainconsumer import Chain, ChainConsumer, PlotConfig
# import main simulation class of lenstronomy
from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel

from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.Solver.solver2point import Solver2Point
from lenstronomy.LensModel.Solver.solver import Solver
from lenstronomy.Plots import lens_plot


cwd = os.getcwd()
base_path = cwd
print('Base directory path:', base_path)

# =============================================================================
# PARAMETER RELOADING SYSTEM
# =============================================================================
# This cell reloads model parameters from the external script
# Run this cell whenever you want to update parameters from model_parameters.py

import importlib
import sys
import os



# Add the current directory to Python path

def model_update(param_module_name='model_parameters_base_DSPL_mock1'):
    """
    Import or reload the model parameters from a user-defined module.
    param_module_name: str, name of the parameter module (without .py extension)
    """
    try:
        # Dynamically import the module
        mp = importlib.import_module(param_module_name)
        importlib.reload(mp)
        print(f"✅ Model parameters reloaded successfully from {param_module_name}!")
        
        # Load all parameters into local namespace for easy access
        globals().update({
            #models
            'mp': mp,
            # Cosmology
            'cosmo': mp.cosmo,
            'z_l': mp.z_l,
            'z_s1': mp.z_s1,
            'z_s2': mp.z_s2,
            'D_s1': mp.D_s1,
            'D_s2': mp.D_s2,
            'D_s1s2': mp.D_s1s2,
            'D_ls1': mp.D_ls1,
            'D_ls2': mp.D_ls2,
            'deflection_scaling': mp.deflection_scaling,
            'beta': mp.beta,
            'lens_redshift_list': mp.lens_redshift_list,
            'source_redshift_list': mp.source_redshift_list,
            
            #Image multiplicity for Position Modeling
            's1_image_num': mp.s1_image_num,
            's2_image_num': mp.s2_image_num,

            # Coordinate parameters
            'lens_center_ra': mp.lens_center_ra,
            'lens_center_dec': mp.lens_center_dec,
            'lens_bound': mp.lens_bound,
            'source_bound': mp.source_bound,
            
            # Model lists
            'lens_model_list': mp.lens_model_list,
            'lens_light_model_list': mp.lens_light_model_list,
            'source_model_list': mp.source_model_list,
            'point_source_list': mp.point_source_list,
            'image_plane_source_list': mp.image_plane_source_list,
            
            # Parameter sets
            'lens_params': mp.lens_params,
            'lens_light_params': mp.lens_light_params,
            'source_params': mp.source_params,
            'ps_params': mp.ps_params,
            
            # Initial values for easy access
            'kwargs_lens_init': mp.kwargs_lens_init,
            'kwargs_lens_sigma': mp.kwargs_lens_sigma,
            'kwargs_lens_fixed': mp.kwargs_lens_fixed,
            'kwargs_lower_lens': mp.kwargs_lower_lens,
            'kwargs_upper_lens': mp.kwargs_upper_lens,
            
            'kwargs_lens_light_init': mp.kwargs_lens_light_init,
            'kwargs_lens_light_sigma': mp.kwargs_lens_light_sigma,
            'kwargs_lens_light_fixed': mp.kwargs_lens_light_fixed,
            'kwargs_lower_lens_light': mp.kwargs_lower_lens_light,
            'kwargs_upper_lens_light': mp.kwargs_upper_lens_light,
            
            'kwargs_source_init': mp.kwargs_source_init,
            'kwargs_source_sigma': mp.kwargs_source_sigma,
            'kwargs_source_fixed': mp.kwargs_source_fixed,
            'kwargs_lower_source': mp.kwargs_lower_source,
            'kwargs_upper_source': mp.kwargs_upper_source,
            
            # Masking parameters
            'central_mask': mp.central_mask,
            'central_mask_r': mp.central_mask_r,
            'r_mask_list': mp.r_mask_list,
            'threshold_list': mp.threshold_list,
            
            # Functions
            'custom_logL_function': mp.custom_logL_function,
            
            # Build the main model configuration
            'kwargs_model' : mp.get_model_config(),

            # Build parameter configuration
            'kwargs_params': mp.get_all_model_params(),

            # Build constraints (this will be updated later with mask information)
            'kwargs_constraints': mp.get_constraints(),

            # Build likelihood configuration
            'kwargs_likelihood': mp.get_likelihood_config(),

            # Build initial model state for fitting
            'kwargs_model_init': {
                'kwargs_lens': mp.kwargs_lens_init,
                'kwargs_source': mp.kwargs_source_init,
                'kwargs_lens_light': mp.kwargs_lens_light_init,
                'kwargs_ps': mp.kwargs_ps_init,
                'kwargs_special': {}, 
                'kwargs_extinction': {}, 
                'kwargs_tracer_source': {}
            }
        })

        # Print current parameters for verification
        mp.print_current_params()
    
    except ImportError as e:
        print(f"❌ Failed to import model_parameters: {e}")
        print("Make sure the parameter module exists and is in your PYTHONPATH.")
    except Exception as e:
        print(f"❌ Error reloading parameters: {e}")

    print("✅ Model configurations built successfully!")
    print(f"📊 Model includes:")
    print(f"   - Lens models: {len(lens_model_list)} ({', '.join(lens_model_list)})")
    print(f"   - Source models: {len(source_model_list)} ({', '.join(source_model_list)})")
    print(f"   - Lens light models: {len(lens_light_model_list)} ({', '.join(lens_light_model_list)})")
    print(f"   - Multi-plane: {kwargs_model.get('multi_plane', False)}")
    print(f"   - Redshifts: z_l={z_l}, z_s1={z_s1}, z_s2={z_s2}")

    # Verify parameter access
    print(f"\n🔍 Quick parameter check:")
    print(f"   - Einstein radius: {kwargs_lens_init[0]['theta_E']:.4f}")
    print(f"   - Lens center: ({kwargs_lens_init[0]['center_x']:.6f}, {kwargs_lens_init[0]['center_y']:.6f})")
    print(f"   - Shear: gamma_ext={kwargs_lens_init[1]['gamma_ext']:.4f}, psi_ext={kwargs_lens_init[1]['psi_ext']:.4f}")

    print("\n🔄 Run this cell/function anytime to reload parameters from your parameter module")




def import_data_mock(file_path, PSF_path):

    kwargs_data = {}
    
    data_file = os.path.join(file_path)
    f = fits.open(data_file)
    # storing data under correct keys
    kwargs_data.update({'image_data': f[0].data,
                        'background_rms': 0.004,
                        'exposure_time': 500.0*np.ones(shape=f[0].data.shape),
                        'ra_at_xy_0': -2.725,
                        'dec_at_xy_0': -2.725,
                        'transform_pix2angle': np.array([[0.05, 0.  ],
                                                        [0.  , 0.05]]),
                        'ra_shift': 0,
                        'dec_shift': 0})
    f.close()

    # get PSF estimate from fits
    psf_file = os.path.join(PSF_path)
    f = fits.open(psf_file)
    kernel_point_source = f[0].data
    f.close()

    # format psf kwargs
    kwargs_psf = {'psf_type': "PIXEL", 
                'kernel_point_source': kernel_point_source ,
                'kernel_point_source_init': kernel_point_source ,
                }
    return kwargs_data, kwargs_psf




def mock_observable_params_read(kwargs_data, kwargs_psf):
    globals().update({
        'ra_at_xy_0': -2.725,
        'dec_at_xy_0': -2.725,
        'transform_pix2angle': np.array([[0.05, 0.  ],
                                           [0.  , 0.05]]),})
    globals().update({'coords_F200LP': Coordinates(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)})
    globals().update({
                'image_data' : kwargs_data['image_data'],
                'numPix' : len(kwargs_data['image_data']),
                'deltaPix' : coords_F200LP.pixel_width
            })





def mock_observable_params_plot(kwargs_data, kwargs_psf, x = 55, y = 53):
    mock_observable_params_read(kwargs_data, kwargs_psf)
    data = kwargs_data['image_data']
    cmap = sns.cubehelix_palette(start=0.6, rot=-1.7, gamma=1, hue=1, light=-.7, dark=0.7, as_cmap=True)
    cmap.set_bad(color='k')
    ax = plt.figure(figsize=(3,3), dpi=200).gca()
    ax.matshow(np.log10(data), origin='lower', cmap=cmap, vmin=-3.8, vmax=0.5)

    globals().update({'kwargs_pixel_F200LP' :{'nx': numPix, 'ny': numPix,  # number of pixels per axis
                    'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                    'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                    'transform_pix2angle': transform_pix2angle}
                    }) 

    globals().update({'pixel_grid_F200LP': PixelGrid(**kwargs_pixel_F200LP)})
    # plot_util.coordinate_arrows(ax, d=200, coords=pixel_grid, color='red', font_size=18, arrow_size=0.035)


    lens_x_F200LP, lens_y_F200LP = x, y
    lens_ra_F200LP, lens_dec_F200LP = coords_F200LP.map_pix2coord(lens_x_F200LP, lens_y_F200LP)
    ax.plot(lens_x_F200LP, lens_y_F200LP, 'x', color='red', ms = 2, label='Lens') 
    ax.legend(loc='lower right', fontsize='small')
    plt.show()


def position_modeling_plot(kwargs_data, kwargs_psf, x = 55, y = 53, 
                           s1_x_F200LP = [10,20,30,40], s1_y_F200LP = [10,20,30,40], 
                           source2 = False, 
                           s2_x_F200LP = [30,40], s2_y_F200LP = [30,40],
                           positions_found = False):
    mock_observable_params_read(kwargs_data, kwargs_psf)
    image_data = kwargs_data['image_data']

    
    cmap = sns.cubehelix_palette(start=0.6, rot=-1.7, gamma=1, hue=1, light=-.7, dark=0.7, as_cmap=True)
    ax = plt.figure(figsize=(5,5), dpi=200).gca()
    ax.matshow(np.log10(image_data), origin='lower', cmap='gray_r', vmin=-1.8, vmax=0.5)

    # plot_util.coordinate_arrows(ax, d=110, coords=pixel_grid, color='red', font_size=18, arrow_size=0.035)

    # estimate lens position in pixel
    lens_x_F200LP, lens_y_F200LP = x, y
    lens_ra_F200LP, lens_dec_F200LP = coords_F200LP.map_pix2coord(lens_x_F200LP, lens_y_F200LP)
    ax.plot(lens_x_F200LP, lens_y_F200LP, 'x', color='red', ms = 2, label='Lens') 

    # estimate source 1 positions in pixel
    ra_image_s1_F200LP, dec_image_s1_F200LP = coords_F200LP.map_pix2coord(s1_x_F200LP, s1_y_F200LP)
    ax.plot(s1_x_F200LP, s1_y_F200LP, 'o', color='lime', ms = 2, label='Source 1 images')

    # estimate source 2 positions in pixel
    if source2:
        ra_image_s2_F200LP, dec_image_s2_F200LP = coords_F200LP.map_pix2coord(s2_x_F200LP, s2_y_F200LP)
        ax.plot(s2_x_F200LP, s2_y_F200LP, 'o', color='magenta', ms = 2, label='Source 2 images')
        

    for i in range(len(s1_x_F200LP)):
        ax.text(s1_x_F200LP[i], s1_y_F200LP[i], i, fontsize=10, color='limegreen')

    if source2:
        for i in range(len(s2_x_F200LP)):
            ax.text(s2_x_F200LP[i], s2_y_F200LP[i], i, fontsize=10, color='magenta')

    if positions_found:
        globals().update({'s1_image_num': len(s1_x_F200LP),
                          's2_image_num': len(s2_x_F200LP) if source2 else 0,
                          'lens_ra_F200LP': lens_ra_F200LP,
                          'lens_dec_F200LP': lens_dec_F200LP,
                          'ra_image_s1_F200LP': ra_image_s1_F200LP,
                          'dec_image_s1_F200LP': dec_image_s1_F200LP,
                          'ra_image_s2_F200LP': ra_image_s2_F200LP if source2 else [],
                          'dec_image_s2_F200LP': dec_image_s2_F200LP if source2 else [],
                          'source2':  source2})
    plt.legend(loc='lower right', fontsize='small')
    plt.minorticks_on()
    plt.grid(alpha = 0.5)
    plt.show()



def position_modeling_calculation(plot_model_prediction = False, plot_convergence = False):
    print('The number of source 1 images is:', s1_image_num)
    print('The number of source 2 images is:', s2_image_num)
    lens_model_list_posmodel = ['SIE', 'SHEAR']  
    kwargs_lens_init_posmodel =[{'theta_E': 1.65,
    'e1': 0.0,
    'e2': 0.0,
    'center_x': lens_ra_F200LP,
    'center_y': lens_dec_F200LP}, 
    {'gamma1': 0.016, 
    'gamma2': 0.0, 
    'ra_0': 0, 
    'dec_0': 0}]

    # initialisation of the lens model class and the lens equation solver
    lensModel_s1 = LensModel(lens_model_list=lens_model_list_posmodel)

    if s1_image_num==4:
        # 4 image solver
        solver4Point_s1 = Solver4Point(lensModel=lensModel_s1, solver_type='PROFILE_SHEAR')
        # new lens model params based on source image pos
        kwargs_fit_s1, precision = solver4Point_s1.constraint_lensmodel(x_pos=ra_image_s1_F200LP, y_pos=dec_image_s1_F200LP, kwargs_list=kwargs_lens_init_posmodel, xtol=1.49012e-12)

    else:
        # 2 image solver
        solver2Point_s1 = Solver2Point(lensModel=lensModel_s1, solver_type='THETA_E_PHI')
        # new lens model params based on source image pos
        kwargs_fit_s1, precision = solver2Point_s1.constraint_lensmodel(x_pos=ra_image_s1_F200LP, y_pos=dec_image_s1_F200LP, kwargs_list=kwargs_lens_init_posmodel, xtol=1.49012e-12)

    print("\n the fitted model parameters are: ", kwargs_fit_s1)

    # estimating source plane position based on lens model and each image
    s1_x_list, s1_y_list = lensModel_s1.ray_shooting(ra_image_s1_F200LP, dec_image_s1_F200LP, kwargs_fit_s1)
    print("\n The relative x position in the source plane (should match) is: ", s1_x_list)
    print("\n The relative y position in the source plane (should match) is: ", s1_y_list)

    # we can now set a new estimate of the source position
    s1_x = np.mean(s1_x_list)
    s1_y = np.mean(s1_y_list)
    print("\n mean source pos",s1_x,s1_y)

    lensEquationSolver_new = LensEquationSolver(lensModel=lensModel_s1) #gives img positions given lens model and source
    ra_image_s1_pred, dec_image_s1_pred = lensEquationSolver_new.image_position_from_source(kwargs_lens=kwargs_fit_s1, sourcePos_x=s1_x, sourcePos_y=s1_y, min_distance=0.05, 
                                                                                            search_window=10, precision_limit=10**(-12), num_iter_max=100, verbose=True)
    print("\n predicted imag pos based on lens init para",ra_image_s1_pred, dec_image_s1_pred) 

    if plot_model_prediction:
        cmap = sns.cubehelix_palette(start=0.6, rot=-1.7, gamma=1, hue=1, light=-.7, dark=0.7, as_cmap=True)
        ax = plt.figure(1, figsize=(5,5), dpi=200).gca()
        ax.matshow(np.log10(image_data), origin='lower', cmap='gray_r', vmin=-1.9, vmax=0.5)

        pred_lens_x, pred_lens_y = coords_F200LP.map_coord2pix(kwargs_fit_s1[0]['center_x'], kwargs_fit_s1[0]['center_y'])
        ax.plot(pred_lens_x, pred_lens_y, 'o', color='red', ms = 4, label='Lens center') 

        s1_x_F200LP, s1_y_F200LP = coords_F200LP.map_coord2pix(ra_image_s1_pred, dec_image_s1_pred)
        ax.plot(s1_x_F200LP, s1_y_F200LP, 'o', color='lime', ms = 4, label='Lensed source 1')

        mean_beta_x, mean_beta_y = coords_F200LP.map_coord2pix(s1_x_list, s1_y_list)
        ax.plot(mean_beta_x, mean_beta_y, '^', color='lime', ms = 4, label='Source 1 centre')

        plt.legend(loc='lower right', fontsize='small')
        plt.show()

    if plot_convergence:
        f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=False, sharey=False)
        lens_plot.lens_model_plot(ax, lensModel=lensModel_s1, kwargs_lens=kwargs_fit_s1, numPix=110,deltaPix=0.05, sourcePos_x=s1_x, sourcePos_y=s1_y, point_source=True, with_caustics=True, fast_caustic=True, coord_inverse=False)
        plt.show()

    if source2:
        lens_model_list_mp = ['SIE','SHEAR','SIE'] 

        # specifying redshifts of deflectors
        redshift_list = [z_l, z_l, z_s1]  

        # specify source redshift 
        z_source = z_s2

        kwargs_mass_s1 = {'theta_E': 0.0,
        'e1': 0.0,
        'e2': 0.0,
        'center_x': s1_x,
        'center_y': s1_y}
        
        kwargs_lens_init = kwargs_fit_s1   
        kwargs_lens_init.append(kwargs_mass_s1)

        # initialisation of the lens model class and the lens equation solver
        lensModel_mp = LensModel(lens_model_list=lens_model_list_mp, z_source=z_source, lens_redshift_list=redshift_list, cosmo= cosmo, multi_plane=True) #lens_model_list_simple[:2] for slicing

        if s2_image_num==4:
            # 4 image solver
            solver4Point_mp = Solver4Point(lensModel=lensModel_mp, solver_type='PROFILE_SHEAR')
            # new lens model params based on source image pos
            kwargs_fit_mp, precision = solver4Point_mp.constraint_lensmodel(x_pos=ra_image_s2_F200LP, y_pos=dec_image_s2_F200LP, kwargs_list=kwargs_lens_init, xtol=1.49012e-12)

        else:
            # 2 image solver
            solver2Point_mp = Solver2Point(lensModel=lensModel_mp, solver_type='THETA_E_PHI')
            # new lens model params based on source image pos
            kwargs_fit_mp, precision = solver2Point_mp.constraint_lensmodel(x_pos=ra_image_s2_F200LP, y_pos=dec_image_s2_F200LP, kwargs_list=kwargs_lens_init, xtol=1.49012e-12)

        print("\n the re-fitted macro-model parameters are: ", kwargs_fit_mp)

        kwargs_fit_init_use_last_best_mp= kwargs_fit_mp

        s2_x_list, s2_y_list = lensModel_mp.ray_shooting(ra_image_s2_F200LP, dec_image_s2_F200LP, kwargs_fit_mp)
        print(s2_x_list)
        print(s2_y_list)
        print("\n The relative x position in the source plane (should match) is: ", s2_x_list)
        print("\n The relative y position in the source plane (should match) is: ", s2_y_list)

        # we can now set a new estimate of the source position
        s2_x = np.mean(s2_x_list)
        s2_y = np.mean(s2_y_list)
        print("\n new mean source pos",s2_x,s2_y)

        lensEquationSolver_new = LensEquationSolver(lensModel=lensModel_mp) #gives img positions given lens model and source
        ra_image_s2_pred, dec_image_s2_pred = lensEquationSolver_new.image_position_from_source(kwargs_lens=kwargs_fit_mp, sourcePos_x=s2_x, sourcePos_y=s2_y, min_distance=0.04, search_window=100, precision_limit=10**(-10), num_iter_max=500)
        print("\n predicted imag pos based on lens init para",ra_image_s2_pred, dec_image_s2_pred) 


        if plot_model_prediction:
            cmap = sns.cubehelix_palette(start=0.6, rot=-1.7, gamma=1, hue=1, light=-.7, dark=0.7, as_cmap=True)
            ax = plt.figure(figsize=(5,5), dpi=200).gca()
            ax.matshow(np.log10(image_data), origin='lower', cmap='gray_r', vmin=-1.9, vmax=0.5)

            pred_lens_x, pred_lens_y = coords_F200LP.map_coord2pix(kwargs_fit_s1[0]['center_x'], kwargs_fit_s1[0]['center_y'])
            ax.plot(pred_lens_x, pred_lens_y, 'o', color='cyan', ms = 4, label='Lens center') 

            s1_x_F200LP, s1_y_F200LP = coords_F200LP.map_coord2pix(ra_image_s1_pred, dec_image_s1_pred)
            ax.plot(s1_x_F200LP, s1_y_F200LP, 'o', color='lime', ms = 4, label='Source 1  images')

            mean_beta_x, mean_beta_y = coords_F200LP.map_coord2pix(s1_x_list, s1_y_list)
            ax.plot(mean_beta_x, mean_beta_y, '^', color='lime', ms = 4, label='Source 1 centre')

            s2_x_F200LP, s2_y_F200LP = coords_F200LP.map_coord2pix(ra_image_s2_pred, dec_image_s2_pred)
            ax.plot(s2_x_F200LP, s2_y_F200LP, 'o', color='magenta', ms = 4, label='Source 2 images')

            mean_beta_x_s2, mean_beta_y_s2 = coords_F200LP.map_coord2pix(s2_x_list, s2_y_list)
            ax.plot(mean_beta_x_s2, mean_beta_y_s2, '^', color='magenta', ms = 4, label='Source 2 centre')

            plt.legend(loc='lower right', fontsize='small')
            plt.show()

        if plot_convergence:
            f, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=False, sharey=False)
            lens_plot.lens_model_plot(ax, lensModel=lensModel_mp, kwargs_lens=kwargs_fit_mp, numPix=110,deltaPix=0.05, sourcePos_x=s2_x, sourcePos_y=s2_y, point_source=True, with_caustics=True, fast_caustic=True, coord_inverse=False)
            plt.show()

        out = {
            'ra_image_s1_pred': ra_image_s1_pred,
            'dec_image_s1_pred': dec_image_s1_pred,
            'ra_image_s2_pred': ra_image_s2_pred,
            'dec_image_s2_pred': dec_image_s2_pred,
            'center_x_pred': kwargs_fit_mp[0]['center_x'],
            'center_y_pred': kwargs_fit_mp[0]['center_y'],
            's1_x_pred': s1_x,
            's1_y_pred': s1_y,
            's2_x_pred': s2_x,
            's2_y_pred': s2_y,
            'theta_E_lens_pred': kwargs_fit_mp[0]['theta_E'],
            'e1_lens_pred': kwargs_fit_mp[0]['e1'],
            'e2_lens_pred': kwargs_fit_mp[0]['e2'],
            'gamma1_ext_pred': kwargs_fit_mp[1]['gamma1'],
            'gamma2_ext_pred': kwargs_fit_mp[1]['gamma2'],
        }

        return out
    

def supersampling_masking(kwargs_data_, threshold=3.8, plot_mask = False):
    # pixels with flux > max_flux/threshold will not be supersampled
    # here the threshold accepts pixels within 5 magnitude from the pixel with max_flux
    threshold_list = [10**(threshold/2.5)]

    supersampling_mask = []

    for j, kwargs_data in enumerate([kwargs_data_]):
        s_mask = np.ones_like(kwargs_data['image_data'])
        
        max_flux = np.max(kwargs_data['image_data'])
        s_mask[kwargs_data['image_data'] < max_flux/threshold_list[j]] = 0
        
        supersampling_mask.append(s_mask)

    filters = ['F200LP']

    globals().update({'supersampling_mask': supersampling_mask})

    if plot_mask:
        for j, kwargs_data in enumerate([kwargs_data_]):
            image = kwargs_data['image_data']

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)

        mask = supersampling_mask[j]#*mask_list[0][j]
        im = ax.matshow(np.log10(image*mask), origin='lower', cmap='magma')
        ax.text(1, 1, 'mask for chi^2 calculation')
        ax.autoscale(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title("Supersampling mask, "+filters[j])
        plt.colorbar(im, cax=cax)

        plt.show()


def setup_prior_to_sampling(kwargs_data, kwargs_psf):
    #model_update()

    lens_center_ra, lens_center_dec = lens_ra_F200LP, lens_dec_F200LP

    shapelet_beta = 0.15 # initial guess of the shapelet scale (in arcsec)

    source_bound = 0.15 

    kwargs_numerics = {'supersampling_factor':3,  # use 3 for final runs                              
                    'supersampling_convolution':True, 
                    'supersampling_kernel_size': 7, # This will change for real data vs mock data
                    'flux_evaluate_indexes': None,
                    'point_source_supersampling_factor': 3,  # use 3 for final runs
                    'compute_mode': 'adaptive',
                    'supersampled_indexes': np.array(supersampling_mask[0], dtype=bool)
                    }

    num_source_model = len(source_model_list)
    print("Number of source models:", num_source_model)

    def custom_logL_function(kwargs_lens=None, kwargs_source=None, 
                                    kwargs_lens_light=None, kwargs_ps=None, 
                                    kwargs_special=None, kwargs_extinction=None,
                                    kwargs_tracer_source=None):
        """
        Addition custom term to add to log likelihood function.
        """
        
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
        lens_center_x, lens_center_y = 0.09789233337522507, 0.020403937171283193  # visually measured centre
        logL -= ((kwargs_lens[0]['center_x'] - lens_center_x)**2 + (kwargs_lens[0]['center_y'] - lens_center_y)**2)*10

        return logL


    kwargs_likelihood = {'check_bounds': True,
                        'force_no_add_image': False,
                        'source_marg': True,
                        'image_position_uncertainty': 0.001,
                        'image_position_likelihood':True,  # turn on for double source
                        'source_position_tolerance': 0.001,
                        'source_position_sigma': 0.001,
                        'bands_compute':[True],
                        'check_positive_flux':True,
                        #'image_likelihood_mask_list':  mask_list[0],
                        'custom_logL_addition': custom_logL_function,
                        'astrometric_likelihood':True
                        }

    F200LP_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [F200LP_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    fitting_kwargs_list = [#['update_settings', {'kwargs_likelihood': {'bands_compute': [True, True]}}],
                       #['PSO', {'sigma_scale': 10, 'n_particles': 220, 'n_iterations': 3000}],
                       ['PSO', {'sigma_scale': 5, 'n_particles': 10, 'n_iterations': 20}],
                       #['PSO', {'sigma_scale': 0.5, 'n_particles': 220, 'n_iterations': 2000}],
                       #['PSO', {'sigma_scale': 0.1, 'n_particles': 220, 'n_iterations': 2000}],
                       #['MCMC', {'n_burn': 100, 'n_run': 3000, 'walkerRatio': 6, 'sigma_scale': 0.5}]
                       ]
    
    sampling_inputs ={
        'kwargs_numerics': kwargs_numerics,
        'kwargs_likelihood': kwargs_likelihood,
        'multi_band_list': multi_band_list,
        'kwargs_data_joint': kwargs_data_joint,
        'fitting_kwargs_list': fitting_kwargs_list,
        'lens_center_ra': lens_center_ra,
        'lens_center_dec': lens_center_dec,
        'shapelet_beta': shapelet_beta,
        'source_bound': source_bound,
        'custom_logL_function': custom_logL_function,
    }
    return sampling_inputs

def read_sampling_inputs(sampling_inputs):
    kwargs_numerics = sampling_inputs['kwargs_numerics']
    kwargs_likelihood = sampling_inputs['kwargs_likelihood']
    multi_band_list = sampling_inputs['multi_band_list']
    kwargs_data_joint = sampling_inputs['kwargs_data_joint']
    fitting_kwargs_list = sampling_inputs['fitting_kwargs_list']
    lens_center_ra = sampling_inputs['lens_center_ra']
    lens_center_dec = sampling_inputs['lens_center_dec']
    shapelet_beta = sampling_inputs['shapelet_beta']
    source_bound = sampling_inputs['source_bound']
    custom_logL_function = sampling_inputs['custom_logL_function']
    return [kwargs_numerics, kwargs_likelihood, multi_band_list, kwargs_data_joint, fitting_kwargs_list, lens_center_ra, lens_center_dec, shapelet_beta, source_bound, custom_logL_function]



def configure_model_and_run(job_name, sampling_inputs, fitting_kwargs_list=None, 
                    kwargs_params=None, cluster_compute=False, 
                    use_good_start=False, prev_job_name=None, 
                    reuse_samples=False, prev_file_dir=None, verbose=False, pass_new_logL=False):  # TODO: Make functionality to pass a new custom_logL_function

    kwargs_numerics, kwargs_likelihood_read, multi_band_list, kwargs_data_joint, fitting_kwargs_list_default, lens_center_ra, lens_center_dec, shapelet_beta, source_bound, custom_logL_function = read_sampling_inputs(sampling_inputs)
    init_samples = None
    input_temp = job_name +'.txt'
    output_temp = job_name +'_out.txt'

    # load previous job result as starting point
    if use_good_start:

        if prev_job_name is None:
            raise ValueError
        job_name_old =  prev_job_name
        
        if prev_file_dir is None:
            raise ValueError
        
        output_temp_old = os.path.join(base_path, prev_file_dir, job_name_old +'_out.txt') #'local_temp'

        f = open(output_temp_old, 'rb')
        [input_, output_] = joblib.load(f)
        f.close()

        old_fitting_kwargs_list, kwargs_data_joint_out, _, _, _, _, _ = input_
        kwargs_result, multi_band_list_out, fit_output, _ = output_
        lens_result = kwargs_result['kwargs_lens']
        source_result = kwargs_result['kwargs_source']
        lens_light_result = kwargs_result['kwargs_lens_light']
        ps_result = kwargs_result['kwargs_ps']
        #special_result= kwargs_result['kwargs_special']

        # updating init kwargs
        lens_params[0] = update_init(lens_result, lens_params[0])       
        source_params[0] = update_init(source_result, source_params[0]) 
        lens_light_params[0] = update_init(lens_light_result, lens_light_params[0])
        ps_params[0] = update_init(ps_result, ps_params[0])
        #special_params[0] =special_result 
    
        # updating kwargs that will be input into fitting sequence
        kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params,
                    'lens_light_model': lens_light_params,
                    'point_source_model': ps_params,
                    #'special': special_params
                    }
        
        if verbose:
            print('updated lens_params',lens_params)
            print('\n updated source_params',source_params)
            print('\n updated lens_light_params',lens_light_params)
            print('\n updated ps_params',ps_params)
            #print('\n updated special_params',special_params)
        
        # using prev mcmc sapmples for next mcmc
        if reuse_samples:
            samples_mcmc = fit_output[-1][1] #mcmc chain from prev_job_name
            
            n_params = samples_mcmc.shape[1]

            n_walkers = old_fitting_kwargs_list[-1][1]['walkerRatio'] * n_params
            n_step = int(samples_mcmc.shape[0] / n_walkers)

            print('MCMC settings from last chain: ', n_step, n_walkers, n_params)

            chain = np.empty((n_walkers, n_step, n_params))
            
            for i in np.arange(n_params):
                samples = samples_mcmc[:, i].T
                chain[:,:,i] = samples.reshape((n_step, n_walkers)).T
            
            init_samples = chain[:, -1, :]
            #print(np.shape(init_samples))
            #print(init_samples)
            print('Init MCMC samples from: ', prev_job_name)
            print('Init samples shape: ', init_samples.shape)
            
        else:
            init_samples = None

        # updating kwargs that will be input into fitting sequence  
        fitting_kwargs_list=[['update_settings', {'kwargs_likelihood': {'bands_compute': [True]}}],
                             #['PSO', {'sigma_scale': 0.1, 'n_particles': 250, 'n_iterations': 500}],
                        #['MCMC', {'n_burn': 100, 'n_run': 4000, 'walkerRatio': 8, 'sigma_scale': .1}]
                ['MCMC', {'n_burn': 0, #mcmc_n_burn, 
                          'n_run':3000, #mcmc_n_run, 
                          'walkerRatio': 6, 
                          'sigma_scale': 0.5,
                          're_use_samples': True, 
                          'init_samples': init_samples}]
                        ]
        if verbose:
            print('updated fitting_kwargs_list', fitting_kwargs_list)
    
    if cluster_compute is True:
        path2input_temp = os.path.join(base_path, 'midway_temp', input_temp)
        dir_path_cluster = '/pool/public/sao/dbowden/Compound/DCLS0353' #### NEED TO GENERALIZE. 
        path2input_cluster = os.path.join(dir_path_cluster, 'local_temp', input_temp)

        f = open(path2input_temp,'wb')
        pickle.dump([fitting_kwargs_list, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples], f)
        f.close()
        time.sleep(2)

        # copying .txt to remote cluster with model and fitting info
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=None, username=None, password=None) 
        ftp_client = ssh_client.open_sftp()
        ftp_client.put(path2input_temp, path2input_cluster)
        ftp_client.put(base_path+'/jobs/DCLS0353_double_source_F200LP_V18.job', dir_path_cluster+'/DCLS0353_double_source_F200LP_V18.job')  # update job file before executing this
        ftp_client.close()
        ssh_client.close()

        print('File %s uploaded to cluster' %path2input_cluster)
        print('Must run job on cluster with jobname {} and job file {}'.format(job_name, job_name[:-7]+'.job'))
    else:
        path2input_temp = os.path.join(base_path, 'local_temp', input_temp)

        f = open(path2input_temp,'wb')
        #pickle.dump([fitting_kwargs_list, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood_read, kwargs_params, init_samples], f)
        f.close()
        time.sleep(2)

        start_time = time.time()
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=False, verbose=True)
    
        fit_output = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit(bijective=False)
        multi_band_list_out = fitting_seq.multi_band_list
        
        kwargs_fixed_out = fitting_seq._updateManager.fixed_kwargs  
        #kwargs_fixed_out = fitting_seq.kwargs_fixed
        #param_class = fitting_seq.param_class
        output_ = [kwargs_result, multi_band_list_out, fit_output, kwargs_fixed_out]

        input_ = [fitting_kwargs_list, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples]
        path2dump = os.path.join(base_path, 'local_temp', output_temp)
        f = open(path2dump, 'wb')
        joblib.dump([input_, output_], f)
        f.close()
        end_time = time.time()
        print(end_time - start_time, 'total time needed for computation')
        print('Result saved in: %s' % path2dump)
        print('============ CONGRATULATIONS, YOUR JOB WAS SUCCESSFUL ================ ')

        return fitting_seq 



def arraytosubplot(ax, data, pixel_grid, pix2arcsec, imtype, v_min=-2, v_max=1, c_map='gray_r'):
    plot = ax.imshow(np.log10(data), origin='lower', cmap=c_map,vmin=v_min,vmax=v_max)
    ax.set_xticks([])
    ax.set_yticks([])
    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cbar1 = plt.colorbar(plot, cax=cax1)
    cbar1.set_ticks(np.linspace(v_min,v_max,6))
    cbar1.ax.set_ylabel('log$_{10}$ flux', fontsize=12)
    plot_util.coordinate_arrows(ax, pixel_grid._nx, pixel_grid, color="k", arrow_size=0.01, font_size=12)
    plot_util.scale_bar(ax, pixel_grid._nx, dist=pix2arcsec, text='1"', font_size=12, color='k')
    plot_util.text_description(ax, pixel_grid._nx,text=imtype,color="k",backgroundcolor="w",font_size=12)
    return


def output_plot_model_fit(kwargs_data, multi_band_list_out, kwargs_model, kwargs_result, save = False, job_name_out='test_out.txt'):

    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                        arrow_size=0.01, cmap_string='gray_r',)
                        #image_likelihood_mask_list=kwargs_likelihood['image_likelihood_mask_list']
                        #)

    #print(kwargs_model)
    #print(kwargs_result)

    f, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)

    # data for custom plotting
    image_data = kwargs_data['image_data']
    recon_F200LP = modelPlot._band_plot_list[0]._model
    #source_recon = modelPlot.source(deltaPix=0.02, numPix=160, center=(kwargs_result['kwargs_lens'][0]['center_x'], kwargs_result['kwargs_lens'][0]['center_y']))

    arraytosubplot(axes[0,0], image_data, pixel_grid_F200LP, 20, 'Observed', v_min=-1.5, v_max=-0.2)
    arraytosubplot(axes[0,1], recon_F200LP, pixel_grid_F200LP, 20, 'Reconstructed', v_min=-1.5, v_max=-0.2)

    # source = axes[1, 0].imshow(np.log10(source_recon[0]), origin='lower', cmap='gray_r',vmin=-2.6,vmax= 0)
    # axes[1, 0].set_xticks([])
    # axes[1, 0].set_yticks([])
    # divider3 = make_axes_locatable(axes[1, 0])
    # cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    # cbar3 = plt.colorbar(source, cax=cax3)
    # cbar3.ax.set_ylabel('log$_{10}$ flux', fontsize=12)
    # plot_util.coordinate_arrows(axes[1,0], 160, source_recon[1], color="k", arrow_size=0.01, font_size=12)
    # plot_util.scale_bar(axes[1,0], 160, dist=50, text='1"', font_size=12, color='k')
    # plot_util.text_description(axes[1,0], 160,text='Reconstructed sources',color="k",backgroundcolor="w",font_size=12)


    #modelPlot.model_plot(ax=axes[0,1], v_min=-1.5, v_max=0.5, font_size=12)
    modelPlot.normalized_residual_plot(ax=axes[0,2],font_size=12, text='')
    plot_util.text_description(axes[0,2], 0.05*110,text='Normalised Residuals',color="k",backgroundcolor="None",font_size=12)

    dPs = 0.02
    nP = 160
    modelPlot.source_plot(ax=axes[1, 0],font_size=12, deltaPix_source=dPs, numPix=nP, with_caustics=True,v_min=-2.6,v_max= -1,caustic_color="limegreen", center=(kwargs_result['kwargs_source'][0]['center_x'], kwargs_result['kwargs_source'][0]['center_y']), text='')
    plot_util.text_description(axes[1,0], dPs*nP,text='Reconstructed sources',color="k",backgroundcolor="w",font_size=12)
    plot_util.scale_bar(axes[1,0], dPs*nP, dist=1, text='1"', font_size=12, color='k')
    plot_util.coordinate_arrows(axes[1,0], dPs*nP, pixel_grid_F200LP, color="k", arrow_size=0.01, font_size=12)

    modelPlot.convergence_plot(ax=axes[1, 1],font_size=12, v_max=1, text='')
    plot_util.text_description(axes[1,1], 0.05*110,text='Convergence',color="k",backgroundcolor="None",font_size=12)
    plot_util.scale_bar(axes[1,1], 0.05*110, dist=1, text='1"', font_size=12, color='k')
    plot_util.coordinate_arrows(axes[1,1], 0.05*110, pixel_grid_F200LP, color="k", arrow_size=0.01, font_size=12)

    modelPlot.magnification_plot(ax=axes[1, 2],font_size=12, text='')
    plot_util.text_description(axes[1,2], 0.05*110,text='Magnification',color="k",backgroundcolor="None",font_size=12)

    plt.tight_layout()
    if save:
        plt.savefig(fname='results/'+job_name_out[:-8]+'.pdf')
    plt.show()

    kwargs_result