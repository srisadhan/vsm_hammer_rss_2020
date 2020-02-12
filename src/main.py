from pathlib import Path

import numpy as np
import pandas as pd
import collections
import deepdish as dd
import matplotlib.pyplot as plt
import seaborn as sns

import pyomo.environ as pyo
import pyomo.dae as pyodae
from casadi import *

from models.fixed_magnet_model import one_fixed_magnet
from models.two_moving_magnets_model import trajectory_optimization, optimize_tf_and_vel, user_provided_magnet_separation
from models.vsm_plots import plot_disp_vel, plot_settings_together, plot_pareto_front
from models.linear_robot_model import (robot_linear_dynamics, robot_linear_dynamics_optimize_tf, robot_linear_dynamics_optimize_tf_and_alpha)
from models.pyomo_optimizer import pyomo_classical_solver, pyomo_neos_solver
from data.pyomo_parse_data import extract_results, extract_results_for_optimal_tf

from utils import skip_run
import yaml
import io
import sys
import csv

config = yaml.safe_load(io.open('src/config.yml'))


with skip_run('run', 'optimization_old') as check, check():    
    stiff_config    = ['high_stiffness', 'low_stiffness','variable_stiffness']
    Tf              = [0.5, 1.5]
    Tf_folder       = ['tf_05', 'tf_15']

    counter = 0

    for tf in Tf: 
        for setting in stiff_config:
            if (setting == 'high_stiffness'):
                pyo_model = robot_linear_dynamics(tf, config)
            else:
                pyo_model = trajectory_optimization(tf, setting, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_classical_solver(pyo_model, 'ipopt', config)
            # solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)

            data, _ , tf = extract_results(solved_model, setting, config)
            
            # save the csv files         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ Tf_folder[counter] / setting) + '.csv'
            data.to_csv(filepath, index=False)

            plot_disp_vel(data, setting, config)

        counter += 1
                
    plt.show()

# Time sweep optimal solution
with skip_run('skip', 'trajectory_optimization_time_sweep') as check, check():    
    Tf              = np.around(np.arange(0.25, 5.0, 0.25), decimals=2)

    sweep_data      = collections.defaultdict()
    max_velocity    = collections.defaultdict()

    fig  = plt.plot()

    for tf in Tf:
        Data = collections.defaultdict()
        list_vel  = []
        for setting in config['stiff_config']:
            if (setting == 'high_stiffness'):
                pyo_model = robot_linear_dynamics(tf, config)
            else:
                pyo_model = trajectory_optimization(tf, setting, config)
                
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
            data, _ , tf    = extract_results(solved_model, setting, config)

            Data[setting]   = data.to_dict()
            
            # optimal velocity value at alpha * tf
            time            = data['time']
            vel             = data['hv'] + data['bv']
            optimal_time    = time[-1:]
            optimal_vel     = max(vel)

            if setting == 'low_stiffness':
                plt.plot(tf, optimal_vel, 'rD')
            elif setting == 'high_stiffness':
                plt.plot(tf, optimal_vel, 'bD')
            else:
                plt.plot(tf, optimal_vel, 'kD')

            list_vel.append(optimal_vel)

        sweep_data[tf] = Data             
        max_velocity[tf] = vel

    # save the complete dictionary 
    filepath = str(Path(__file__).parents[1] / config['two_moving_mags_time_sweep_data'])       
    dd.io.save(filepath, sweep_data)

    # save the maximum velocity for each tf
    filepath = str(Path(__file__).parents[1] / config['max_vel_time_sweep'])
    dd.io.save(filepath, max_velocity)

    plt.show()

with skip_run('skip', 'load_the_saved_tf_sweep_data') as check, check():
    # load the data
    filepath    = str(Path(__file__).parents[1] / config['two_moving_mags_time_sweep_data'])       
    sweep_data  = dd.io.load(filepath)

    Tf          = np.around(np.arange(0.5, 5.0, 0.25), decimals=2)

    # sns.set()

    _,ax1 = plt.subplots(2,1)
    _,ax2 = plt.subplots(2,1)
    _,ax3 = plt.subplots(2,1)

    ax1[0].set_xlim([0, 5.0])
    ax1[1].set_xlim([0, 5.0])
    ax2[0].set_xlim([0, 5.0])
    ax2[1].set_xlim([0, 5.0])

    for tf in Tf:
        data = sweep_data[tf]
        time_vs = [i for i in data['variable_stiffness']['time'].values()]
        base_vs = [i for i in data['variable_stiffness']['bv'].values()]
        vel_vs  = [i+j for i,j in zip(data['variable_stiffness']['hv'].values(), data['variable_stiffness']['bv'].values())]
        disp_vs = [i+j for i,j in zip(data['variable_stiffness']['hd'].values(), data['variable_stiffness']['bd'].values())]

        time_ls = [i for i in data['low_stiffness']['time'].values()]
        base_ls = [i for i in data['low_stiffness']['bv'].values()]
        vel_ls  = [i+j for i,j in zip(data['low_stiffness']['hv'].values(), data['low_stiffness']['bv'].values())]
        disp_ls = [i+j for i,j in zip(data['low_stiffness']['hd'].values(), data['low_stiffness']['bd'].values())]

        # data_vs = np.array([time_vs, vel_vs]).reshape(time_vs.shape[0],2)
        # data_ls = np.array([time_vs, vel_vs]).reshape(time_vs.shape[0],2)

        ax1[0].plot(time_ls, vel_ls)
        ax1[1].plot(time_vs, vel_vs)

        ax2[0].plot(time_ls, disp_ls)
        ax2[1].plot(time_vs, disp_vs)

        ax3[0].plot(tf, max(base_ls), 'ko')
        ax3[0].plot(tf, max(vel_ls), 'rd')
        ax3[1].plot(tf, max(base_vs), 'ko')
        ax3[1].plot(tf, max(vel_vs), 'bd')
        # data = data['variable_stiffness']
        # sns.relplot(x="time", y="hv", kind="line", data=data)

    ax1[0].set_ylim([-1.0, 1.0])
    ax1[1].set_ylim([-1.0, 1.0])

    plt.show()


# -------------------------Hammering with new code------------------------ #
# Time optimal trajectory evaluation - 3 Weights
with skip_run('skip', 'optimize_Tf_and_Vel') as check, check():    
    alphas          = [0.8]

    Data = collections.defaultdict()

    counter = 0
    exps    = ['weights_1', 'weights_2', 'weights_3']
    # weights for multi-objective optimization
    weights = [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]

    for weight in weights:
        for setting in config['stiff_config']:
            print(setting)
            if (setting == 'high_stiffness'):
                pyo_model= robot_linear_dynamics_optimize_tf(weight, config)
            else:
                pyo_model = optimize_tf_and_vel( setting, weight, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
            data, _ , tf    = extract_results_for_optimal_tf(solved_model, setting, config)
            
            print("Tf value for " + setting + " is: " + str(tf))

            plot_disp_vel(data, 'Both_magnets_moving', config)
            plt.suptitle('Tf: ' + str(tf) + ', Stiffness: ' + setting)

            # save the csv files         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ exps[counter] / setting) + '.csv'
            data.to_csv(filepath, index=False)
            Data[setting] = data.to_dict()
        
        counter += 1
        plot_settings_together(Data, config)
        
    # save the optimal data
    # filepath = str(Path(__file__).parents[1] / config['two_moving_mags_optim_tf_data'])
    # dd.io.save(filepath, Data)
            
    plt.show()

with skip_run('skip', 'plot_the_optimal_values_LS_VS') as check, check():
    exps    = ['weights_1', 'weights_2', 'weights_3']
    counter = 0

    for i in exps:
        _, ax = plt.subplots(3,1)

        filepath1 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'low_stiffness') + '.csv' 
        data1     = pd.read_csv(filepath1, delimiter=',')

        filepath2 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'variable_stiffness') + '.csv' 
        data2     = pd.read_csv(filepath2, delimiter=',')
        
        ax[0].plot(data1['time'], data1['hd'] + data1['bd'], 'b-', label='LS',  linewidth=2)
        ax[0].plot(data2['time'], data2['hd'] + data2['bd'], 'r-' , label='VS', linewidth=2)
        ax[0].plot([max(data1['time']), max(data1['time'])], [-0.12, 0.1], 'b--')
        ax[0].plot([max(data2['time']), max(data2['time'])], [-0.12, 0.1], 'r--')
        
        ax[0].set_xlim([0, max(max(data1['time']), max(data2['time']))])
        ax[0].set_ylim([-0.12, 0.1])
        ax[0].legend(loc='lower left')

        ax[1].plot(data1['time'], data1['hv'] + data1['bv'], 'b-', label='LS', linewidth=2)
        ax[1].plot(data2['time'], data2['hv'] + data2['bv'], 'r-', label='VS', linewidth=2)
        ax[1].plot([max(data1['time']), max(data1['time'])], [-1, 1], 'b--')
        ax[1].plot([max(data2['time']), max(data2['time'])], [-1, 1], 'r--')
        ax[1].plot(data1['time'], 0.5 + 0 * data1['time'], 'k--')
        ax[1].plot(data1['time'], -0.5 + 0 * data1['time'], 'k--')
        ax[1].set_xlim([0, max(max(data1['time']), max(data2['time'])) ])
        ax[1].set_ylim([-0.8, 0.8])

        ax[2].plot(data1['time'],  data1['md2'], 'b-', linewidth=1.5)
        ax[2].plot(data1['time'], -data1['md2'], 'b-', linewidth=1.5)
        ax[2].plot(data2['time'],  data2['md2'], 'r-', linewidth=1.5)
        ax[2].plot(data2['time'], -data2['md2'], 'r-', linewidth=1.5)
        ax[2].plot([max(data1['time']), max(data1['time'])], [-0.06, 0.06], 'b--')
        ax[2].plot([max(data2['time']), max(data2['time'])], [-0.06, 0.06], 'r--')
        ax[2].set_xlim([0, max(max(data1['time']), max(data2['time'])) ])
        ax[2].set_xlabel('Time (s)')  

        if counter < 1:
            ax[0].set_ylabel('Displacement (m)')
            ax[1].set_ylabel('Velocity (m/s)')
            ax[2].set_ylabel('Magnet (m)') 
        
        if counter == 1:
            _, ax1 = plt.subplots(2,1)
            ax1[0].plot(data1['time'], data1['bd'], 'k-.')
            ax1[0].plot(data2['time'], data2['bd'], 'k-.')
            ax1[0].plot(data1['time'], data1['bd'] + data1['hd'], 'b-')
            ax1[0].plot(data2['time'], data2['bd'] + data2['hd'], 'r-')
            
            ax1[1].plot(data1['time'], data1['bv'], 'k-.')
            ax1[1].plot(data2['time'], data2['bv'], 'k-.')
            ax1[1].plot(data1['time'], data1['bv'] + data1['hv'], 'b-')
            ax1[1].plot(data2['time'], data2['bv'] + data2['hv'], 'r-')

        counter += 1

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        plt.tight_layout()
    plt.show()

# Time optimal trajectory evaluation - All weights
with skip_run('skip', 'optimize_tf_and_vel_all_weights') as check, check():    
    Data = collections.defaultdict()
    Pareto = collections.defaultdict()

    counter = 0

    # weights for multi-objective optimization
    weights = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
    
    for weight in weights:
        temp        = collections.defaultdict()

        for setting in config['stiff_config']:
            if (setting == 'high_stiffness'):
                pyo_model= robot_linear_dynamics_optimize_tf(weight, config)
            else:
                pyo_model = optimize_tf_and_vel( setting, weight, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
            data, _ , tf    = extract_results_for_optimal_tf(solved_model, setting, config)
            
            print("Tf value for " + setting + " is: " + str(tf))

            # save the csv files         
            filepath = str(Path(__file__).parents[1] / config['pareto_path'] / str(counter) / setting) + '.csv'
            data.to_csv(filepath, index=False)

            temp_time       = data['time']

            temp['t_hit']   = temp_time[temp_time == 0.8*tf]
            temp['tf']      = tf
            temp[setting]   = data['bv'][temp_time == 0.8*tf] + data['hv'][temp_time == 0.8*tf]

        counter += 1

        Pareto[counter] = temp

    # path to save the data
    filepath = str(Path(__file__).parents[1] / config['pareto_front'])
    dd.io.save(filepath, Pareto)

with skip_run('skip', 'plot_the_pareto_front') as check, check():
    
    # path to save the data
    filepath = str(Path(__file__).parents[1] / config['pareto_front'])
    pareto_data = dd.io.load(filepath)

    plot_pareto_front(pareto_data,  config)

# Separation sweep optimal solution
with skip_run('skip', 'data_for_two_magnets_sep_sweep') as check, check():
   
    tf = 1.0
    Separations = np.around(np.arange(0.035, 0.06, 0.01), decimals=3)
    
    df      = pd.DataFrame()

    for sep in Separations:
        pyo_model = user_provided_magnet_separation(tf, sep, config)

        # Solve the optimization problem and return the optimal results
        solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
        data, _ , tf    = extract_results(solved_model, '', config)

        temp    = pd.DataFrame({'sep': ["sep_" + str(round(i,3)) for i in data['md2']]})
        data    = data.join(temp, lsuffix='_caller', rsuffix='_other')

        tol_vel = pd.DataFrame({'total_vel' : data['bv'] + data['hv']})
        data    = data.join(tol_vel, lsuffix='_caller', rsuffix='_other')

        tol_disp= pd.DataFrame({'total_disp' : data['bd'] + data['hd']})
        data    = data.join(tol_disp, lsuffix='_caller', rsuffix='_other')

        if df.size > 0:
            df = df.append(data, ignore_index=True)
        else:
            df = data
        
    # save the data
    filepath = str(Path(__file__).parents[1] / config['two_moving_mags_sep_sweep_data'])
    df.to_hdf(filepath, key='df', mode='w')

with skip_run('skip', 'plot_sep_sweep_data') as check, check():
    # load the data
    filepath = str(Path(__file__).parents[1] / config['two_moving_mags_sep_sweep_data'])
    df = pd.read_hdf(filepath, key='df')

    fig,ax  = plt.subplots(2,1)

    sns_palette = sns.color_palette("viridis", n_colors=3, desat=0.5)
    # sns_palette = sns.cubehelix_palette(5, start=2, rot=-.75)

    g1 = sns.relplot(x="time", y="total_vel", hue="sep", kind="line", palette=sns_palette, data=df, ax=ax[0])
    g2 = sns.relplot(x="time", y="total_disp", hue="sep", kind="line", palette=sns_palette, data=df, ax=ax[1])
    
    plt.close(g1.fig)
    plt.close(g2.fig) 
    plt.tight_layout()

    plt.show()



with skip_run('skip', 'run_trajectory_optimization_of_the_robot_given_tf') as check, check():        
    pyo_model = robot_linear_dynamics(0.5, config)

    # Solve the optimization problem and return the optimal results
    solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
    data, _ , tf    = extract_results(solved_model, setting, config)

    _,ax = plt.subplots(2,1)
    ax[0].plot(data['time'], data['bd'], 'b-', label = '0.5 s')
    ax[0].plot([0.5, 0.5], [-0.1, 0.08], 'k--')
    ax[0].plot([3, 3], [-0.1, 0.08], 'k--')

    ax[1].plot(data['time'], data['bv'], 'b-', label = '0.5 s')
    ax[1].plot([0.5, 0.5], [-0.5, 0.5], 'k--')
    ax[1].plot([3, 3], [-0.5, 0.5], 'k--')

    pyo_model = robot_linear_dynamics(3, config)

    # Solve the optimization problem and return the optimal results
    solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
    data, _ , tf    = extract_results(solved_model, setting, config)

    ax[0].plot(data['time'], data['bd'], 'r-', label = '3.0 s')
    ax[0].set_ylabel('Displacement (m)')
    ax[0].grid()
    ax[0].legend(loc='upper center')

    ax[1].plot(data['time'], data['bv'], 'r-', label = '3.0 s')
    # ax[1].plot(data['time'], 0.5 + 0*data['time'], 'k-')
    # ax[1].plot(data['time'], -0.5 + 0*data['time'], 'k-')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].grid()

    plt.tight_layout()
    plt.show()

with skip_run('skip', 'run_trajectory_optimization_of_the_robot_for_tf') as check, check():       
    
    alphas = np.round(np.arange(0, 1.0, 0.1), decimals=1)
    weights = [0.5, 0.5]

    for alpha in alphas:
        pyo_model= robot_linear_dynamics_optimize_tf(weight, config)
        
        # Solve the optimization problem and return the optimal results
        solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
        data, _ , tf    = extract_results_for_optimal_tf(solved_model, setting, config)

        _,ax = plt.subplots(3,1)
        ax[0].plot(data['t'], data['bd'])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Displacement (m)')

        ax[1].plot(data['t'], data['bv'])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (m/s)')

        ax[2].plot(data['t'], data['ba'])
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Acceleration (m/s)')

    plt.show()
    #TODO:
    # plot the pareto front for optimal values corresponding to time and velocity of the robot


# FIXME:
# Optimizing for alpha is not working
with skip_run('skip', 'run_trajectory_optimization_of_the_robot_for_tf_alpha') as check, check():       
    # weights for multi-objective optimization
    weights = [0.5, 0.5]

    pyo_model= robot_linear_dynamics_optimize_tf(weights, config)
    
    # Solve the optimization problem and return the optimal results
    solved_model = pyomo_neos_solver(pyo_model, 'snopt', config)
    data, _ , tf    = extract_results_for_optimal_tf(solved_model, setting, config)

    _,ax = plt.subplots(3,1)
    ax[0].plot(data['t'], data['bd'])
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Displacement (m)')

    ax[1].plot(data['t'], data['bv'])
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (m/s)')

    ax[2].plot(data['t'], data['ba'])
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Acceleration (m/s)')

    plt.show()


# FIXME:
# I have no idea what is going on here
with skip_run('skip', 'optimal_two_magnets_moving_CASADI') as check, check():

    # Example to save the PYOMO model for use in CASADI
    # filepath = str(Path(__file__).parents[1] / config['two_moving_mags_optim_tf_model'])
    # pyo_model.write(filepath)

    # load the model saved by PYOMO
    filepath = str(Path(__file__).parents[1] / config['two_moving_mags_optim_tf_model'])
    
    # Create an NLP instance
    nl = NlpBuilder()

    # Parse an NL-file
    nl.import_nl(filepath, {"verbose":False})

    # NLP solver options
    opts = {}
    # opts["expand"] = True

    # Create an NLP solver
    nlpsol = nlpsol("nlpsol", "ipopt", nl, opts)

    # Solve NLP
    res = nlpsol(lbx=nl.x_lb,
                ubx=nl.x_ub,
                lbg=nl.g_lb,
                ubg=nl.g_ub,
                x0=nl.x_init)
    plt.figure()
    plt.plot(res['x'])
    plt.show()


# -------------Differential flatness for second paper--------------
with skip_run('skip', 'optimal_one_magnet_fixed') as check, check():
    stiff_config  =  ['variable_stiffness'] #['high_stiffness','low_stiffness','variable_stiffness']

    Tf = [1.0] #[1.0, 2.0, 3.0, 4.0, 5.0]

    for tf in Tf:
        for setting in stiff_config:
            for mag_cond in config['magnet_cond']:
                data1 = one_fixed_magnet(tf, setting, mag_cond, config)
                plot_disp_vel(data1, mag_cond, config)

            # path to save the file
            # filepath = str(Path(__file__).parents[1] / 'data/processed/experiment_RSS') + '/' + setting + '_sri.csv'
            # df.to_csv(filepath, index=False)                

            # print('base disp at 0.8tf: %.4f, velocity: %.4f' %(baseD[t == round(alpha*100*tf)/100], baseV[t == round(alpha*100*tf)/100]))
            # print('Hammer displacement: %.4f, velocity: %.4f' %(hammerD[t == round(alpha*100*tf)/100], hammerV[t == round(alpha*100*tf)/100]))
            # print('Time at which the hammer hits the nail: %.2f*tf %.4f'%(alpha, round(alpha*100*tf)/100))

    plt.show()
