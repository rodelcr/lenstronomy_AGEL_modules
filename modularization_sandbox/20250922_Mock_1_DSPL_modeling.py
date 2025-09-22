from Initial_module_load import *

kwargs_data_F200LP, kwargs_psf_F200LP = import_data_mock('../tutorials_DB_2025_09/mocks/images/mock_1_image.fits',
                                                    '../tutorials_DB_2025_09/mocks/PSF/mock_psf.fits')

mock_observable_params_plot(kwargs_data_F200LP, kwargs_psf_F200LP)

position_modeling_plot(kwargs_data_F200LP, kwargs_psf_F200LP,
                       source2 = True,
                       s1_x_F200LP = np.array([35,55,62,60]),
                       s1_y_F200LP = np.array([60,73,72,37]),
                       s2_x_F200LP = np.array([92,40]),
                       s2_y_F200LP = np.array([75,30]),
                       positions_found = True)

key_initial_guesses = position_modeling_calculation(plot_model_prediction = False, plot_convergence = True)

# You will use these initial guesses in the next, more in-depth modeling steps.
# Using the key_initial_guesses dictionary, you can set up the priors/initial parameters
# for the lens and source model parameters.
#
# Best to update the initial guesses directly in the model_parameters_base*.py file that is being used.
#


# Now, let's create a supersampling mask to speed up the modeling,
# by avoiding supersampling in regions of low flux.
# You can adjust the threshold parameter to change the mask.
# A threshold of 3.8 means that pixels with flux > max_flux/10^(3.8/2.5) will not be supersampled.

supersampling_masking(kwargs_data_F200LP, threshold=3.8, plot_mask = True)


# Cosmology calculations check

print('Angular diameter distance between s1 and s2: {} Mpc'.format(D_s1s2))
print('Deflection angle rescale factor: {:.6f}'.format(deflection_scaling))
print('Beta:{:.6f}'.format(beta))

model_update()

job_name = 'PSO_double_source_mock1_example'

cluster_comp = False # True if running on cluster, False if local

sampling_inputs = setup_prior_to_sampling(kwargs_data_F200LP, kwargs_psf_F200LP)


fitting_kwargs_list = [#['update_settings', {'kwargs_likelihood': {'bands_compute': [True, True]}}],
                       #['PSO', {'sigma_scale': 10, 'n_particles': 220, 'n_iterations': 3000}],
                       ['PSO', {'sigma_scale': 5, 'n_particles': 100, 'n_iterations': 200}],
                       #['PSO', {'sigma_scale': 0.5, 'n_particles': 220, 'n_iterations': 2000}],
                       #['PSO', {'sigma_scale': 0.1, 'n_particles': 220, 'n_iterations': 2000}],
                       #['MCMC', {'n_burn': 100, 'n_run': 3000, 'walkerRatio': 6, 'sigma_scale': 0.5}]
                       ]

configure_model_and_run(job_name, sampling_inputs, cluster_compute=cluster_comp, 
                fitting_kwargs_list=fitting_kwargs_list, kwargs_params=kwargs_params) #kwargs_params is a global file defined inside the setup_prior_to_sampling() function. 


job_name_out = job_name+'_out.txt'  # 'DCLS0353_double_source_F200LP_V18_run_07_out.txt'

output_temp = os.path.join(base_path, 'local_temp', job_name_out)

path2dump = os.path.join(base_path, 'midway_temp', job_name_out)

if cluster_comp:

    if not os.path.exists(path2dump):  # If file does not exist, copy from cluster
        dir_path_cluster = '/pool/public/sao/dbowden/Compound/DCLS0353'
        path2dump_cluster = os.path.join(dir_path_cluster, 'local_temp', job_name_out)

        # copying results _out.txt from remote cluster
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=None, username=None, password=None) 
        ftp_client = ssh_client.open_sftp()
        ftp_client.get(path2dump_cluster, path2dump)
        ftp_client.close()
        ssh_client.close()

    f = open(path2dump, 'rb')
    [input_, output_] = joblib.load(f)
    f.close()
    
else:    
    f = open(output_temp, 'rb')
    [input_, output_] = joblib.load(f)
    f.close()

fitting_kwargs_list, multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples = input_

kwargs_result, multi_band_list_out, fit_output, _ = output_


output_plot_model_fit(kwargs_data_F200LP, multi_band_list_out, kwargs_model, kwargs_result)