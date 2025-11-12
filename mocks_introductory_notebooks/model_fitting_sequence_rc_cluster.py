#  to be run remote on cluster
# Need to install and implement the full reqs2.txt packages on the cluster as an environment and run this script there via a slurm job.


import numpy as np
import os
import time
import pickle
import joblib
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from argparse import ArgumentParser
from schwimmbad.mpi import MPIPool

def custom_logL_function(kwargs_lens=None, kwargs_source=None, 
                                 kwargs_lens_light=None, kwargs_ps=None, 
                                 kwargs_special=None, kwargs_extinction=None,
                                 kwargs_tracer_source=None):
    
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

#command line parameters
parser = ArgumentParser(prog='Model fitting sequence')
parser.add_argument('input_job', nargs=1)
args = parser.parse_args()
job_name = args.input_job[0]

pool = MPIPool(use_dill=True)

if pool.is_master():
    print("job %s loaded" %job_name)

input_temp = job_name +'.txt'
output_temp = job_name +'_out.txt'

dir_path_cluster = os.getcwd()
path2input_cluster = os.path.join(dir_path_cluster, 'local_temp', input_temp)
path2dump = os.path.join(dir_path_cluster, 'local_temp', output_temp)

# reading .txt containing model and fitting info
f = open(path2input_cluster,'rb')
fitting_kwargs_list, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples = pickle.load(f)
f.close()

# performing the PSOs and MCMC
start_time = time.time()
fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=True, verbose=True)

fit_output = fitting_seq.fit_sequence(fitting_kwargs_list) #problem
kwargs_result = fitting_seq.best_fit(bijective=False)
multi_band_list_out = fitting_seq.multi_band_list
kwargs_fixed_out = fitting_seq._updateManager.fixed_kwargs 

# only to be performed by one instance in pool
if pool.is_master():
    output_ = [kwargs_result, multi_band_list_out, fit_output, kwargs_fixed_out]
    input_ = [fitting_kwargs_list, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, init_samples]

    # writing results to _out.txt
    f = open(path2dump, 'wb')
    joblib.dump([input_, output_], f)
    f.close()
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('Result saved in: %s' % path2dump)
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')    