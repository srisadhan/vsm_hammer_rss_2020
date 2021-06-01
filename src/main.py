from pathlib import Path
import sys

import numpy as np
import pandas as pd
import collections
import deepdish as dd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import seaborn as sns

import pyomo.environ as pyo
import pyomo.dae as pyodae
sys.path.append(r"G:\Courses_UB\Spring_19\Optimal Controls\Project\casadi-windows-py38-v3.5.5-64bit")
from casadi import *

from models.fixed_magnet_model import one_fixed_magnet
from models.two_moving_magnets_model import trajectory_optimization, optimize_tf_and_vel, user_provided_magnet_separation, maximize_vel_minimize_impulse, maximize_vel_minimize_time_impulse
from models.vsm_plots import plot_disp_vel, plot_settings_together, plot_pareto_front
from models.linear_robot_model import (robot_linear_dynamics, robot_linear_dynamics_optimize_tf, robot_linear_dynamics_optimize_tf_and_alpha)
from models.pyomo_optimizer import pyomo_solver
from data.pyomo_parse_data import extract_results, extract_results_for_optimal_tf

from utils import skip_run, parse_time
import yaml
import io
import csv
import matplotlib.pylab as pylab
from dateutil import parser 

params = {'legend.fontsize': 12,
        #   'figure.figsize': (10, 5),
         'axes.labelsize': 16,
        #  'axes.titlesize':'x-large',
         'xtick.labelsize':14,
         'ytick.labelsize':14}
pylab.rcParams.update(params)

# Load the configurations from the yml file
config = yaml.safe_load(io.open('src/config.yml'))

##############################################################
#  Optimal velocity trajectory for hammering task (given Tf) 
##############################################################
# --------- ICRA extension to IROS --------#
with skip_run('run', 'optimize_vel_given_tf') as check, check():    
    Tf              = [0.3, 2.0]# [0.3, 0.5, 1.0, 1.5, 2.0]
    Tf_folder       = ['tf_03', 'tf_20']#['tf_05', 'tf_10', 'tf_15', 'tf_20']

    counter = 0

    for tf in Tf: 
        for setting in config['stiff_config']:
            if (setting == 'high_stiffness'):
                pyo_model = robot_linear_dynamics(tf, config)
            else:
                pyo_model = trajectory_optimization(tf, setting, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=True)

            data, _ , tf = extract_results(solved_model, setting, config)
            
            # save the csv files         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ Tf_folder[counter] / setting) + '.csv'
            data.to_csv(filepath, index=False)

            plot_disp_vel(data, setting, config)

        counter += 1
                
with skip_run('skip', 'plot_optimal_results_of_hammer_given_tf') as check, check():

    Tf_folder        = ['tf_03', 'tf_20'] #['tf_05', 'tf_10', 'tf_15', 'tf_20']
    plot_magnet_disp = False
      
    for folder_name in Tf_folder:
        if plot_magnet_disp: 
            fig, ax = plt.subplots(3,1)
        else:
            fig, ax = plt.subplots(2,1)

        if folder_name == 'tf_03' or folder_name == 'tf_05':
            ylim      = [-0.1, 0.1]
            ax[0].set_ylabel('Displacement (m)')
            ax[1].set_ylabel('Velocity (m/s)')    

            if plot_magnet_disp:
                ax[2].set_ylabel('magnet (mm)')

        else:
            ylim      = [-0.2, 0.1]
        for setting in config['stiff_config']:
            # load the data         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ folder_name / setting) + '.csv'
            data = pd.read_csv(filepath, delimiter=',')
            tf   = data['time'].iloc[-1]
            w    = 0.03

            if setting == 'high_stiffness':
                linewidth = 2
            else : 
                linewidth = 1.5

            ax[0].plot(data['time'], data['bd']+data['hd'],config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[0].plot(data['time'], 0.05 + 0 * data['time'], 'k:')
            ax[0].plot([0.8 * tf, 0.8 * tf], [-0.2, 0.1], 'k-.')

            ax[0].grid()
            ax[0].set_xlim([0, tf])
            ax[0].set_ylim(ylim)            
            ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax[1].plot(data['time'], data['bv']+data['hv'],config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[1].plot([0.8 * tf, 0.8 * tf], [-1, 1.25], 'k-.')

            ax[1].grid()
            ax[1].set_xlim([0, tf])
            ax[1].set_ylim([-1, 1.25])  
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
            
            if plot_magnet_disp:
                ax[2].plot(data['time'], (data['md1'] + w) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
                ax[2].plot(data['time'], (data['md2'] - w) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
                
                ax[2].plot([0.8 * tf, 0.8 * tf], [-.040 * 1000, .040 * 1000], 'k-.')
                
                ax[2].grid()
                ax[2].set_xlim([0, tf])
                ax[2].set_ylim([-.040 * 1000, .040 * 1000])
                ax[2].set_xlabel('Time (s)')  
                # ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f')) 

                for i in range(2):
                    ax[i].tick_params(axis='x',        # changes apply to the x-axis
                                    which='both',      # both major and minor ticks are affected
                                    bottom=False,      # ticks along the bottom edge are off
                                    top=False,         # ticks along the top edge are off
                                    labelbottom=False) # labels along the bottom edge are off
            else:
                ax[1].set_xlabel('Time (s)')
                
                ax[0].tick_params(axis='x',        # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off

            ax[0].legend(loc='lower right')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=0.0)
   
with skip_run('skip', 'plot_optimal_results_of_robot_given_tf') as check, check():

    Tf_folder        = ['tf_03', 'tf_20'] #['tf_05', 'tf_10', 'tf_15', 'tf_20']
    plot_magnet_disp = False
    k = 0 # counter for plot axis

    fig, ax = plt.subplots(2,2, figsize=(10,5))
    for folder_name in Tf_folder:        

        if folder_name == 'tf_03' or folder_name == 'tf_05':
            ylim      = [-0.2, 0.1] #[-0.02, 0.08]
            ax[0,k].set_ylabel('Displacement (m)')
            ax[1,k].set_ylabel('magnet (mm)')

        else:
            ylim      = [-0.2, 0.1]
        for setting in config['stiff_config']:
            # load the data         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ folder_name / setting) + '.csv'
            data = pd.read_csv(filepath, delimiter=',')
            tf   = data['time'].iloc[-1]
            w    = 0.03

            # if setting == 'high_stiffness':
            #     linewidth = 2
            # else : 
            linewidth = 2

            ax[0,k].plot(data['time'], data['bd'],config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[0,k].plot(data['time'], 0.05 + 0 * data['time'], 'k:')
            ax[0,k].plot([0.8 * tf, 0.8 * tf], [-0.2, 0.1], 'k-.')

            ax[0,k].grid()
            ax[0,k].set_xlim([0, tf])
            ax[0,k].set_ylim(ylim)            
            ax[0,k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # ax[0,k].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            # ax[0,k].xaxis.set_major_locator(ticker.MultipleLocator(1))
            
            ax[1,k].plot(data['time'], (data['md1'] + 0.03) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[1,k].plot(data['time'], (data['md2'] - 0.03) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[1,k].plot([0.8 * tf, 0.8 * tf], [-.035 * 1000, .035 * 1000], 'k-.')
            
            ax[1,k].grid()
            ax[1,k].set_xlim([0, tf])
            ax[1,k].set_ylim([-.035 * 1000, .035 * 1000])
            ax[1,k].set_xlabel('Time (s)')  
            ax[1,k].xaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
            
            ax[0,k].tick_params(axis='x',        # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off

            ax[0,k].legend(loc='lower right')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=0.1)
        
        k += 1

    # remove the yticks for the second plot
    for j in range(0,2):
        ax[j,1].tick_params(axis='y',       
                        which='both',      
                        left=False,    
                        # top=False,         # ticks along the top edge are off
                        labelleft=False)
        ax[j,0].set_xticks(np.arange(0, 0.3 + 0.1, step=0.1))
        ax[j,1].set_xticks(np.arange(0, 2.0 + 0.5, step=0.5))

with skip_run('skip', 'plot_energy_stored') as check, check():

    Tf_folder        = ['tf_03', 'tf_20', 'optimal_tf'] #['tf_05', 'tf_10', 'tf_15', 'tf_20']
      
    for folder_name in Tf_folder:
        fig, ax = plt.subplots(2,1)

        ax[0].set_ylabel('Power (W)')
        ax[1].set_ylabel('Cummulative Energy (J)')
        
        for setting in config['stiff_config']:
            # load the data         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ folder_name / setting) + '.csv'
            data = pd.read_csv(filepath, delimiter=',')
            if folder_name == 'optimal_tf':
                tf = 2.0
            else:
                tf   = data['time'].iloc[-1]
            w    = 0.03

            linewidth = 1.5

            Sep1  = data['hd'] - data['md1'] - w
            Sep2  = data['md2'] - data['hd'] - w 
            
            # calculate the power stored in the magnets
            force  = config['C1'] * (np.exp(-config['C2'] * Sep2) - np.exp(-config['C2'] * Sep1))
            power  = np.multiply(force, data['hv'])
            
            # calculate the cummulative energy
            energy = np.zeros(power.shape)
            for count, _ in enumerate(power):
                energy[count] = np.trapz(abs(power[:count]), data['time'][:count])
            
            ax[0].plot(data['time'], power, config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            
            ax[0].grid()
            ax[0].set_xlim([0, tf])
            ax[0].set_ylim([np.min(power), np.max(power)])
            ax[0].tick_params(axis='x',        # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off
            
            ax[1].plot(data['time'], energy, config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            if folder_name != 'optimal_tf':
                ax[1].plot([0.8 * tf, 0.8 * tf], [0, np.max(energy)], 'k-.')

            ax[1].grid()
            ax[1].set_xlim([0, tf])
            ax[1].set_ylim([0, np.max(energy)])            
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax[1].set_xlabel('Time (s)')

            ax[1].legend(loc='upper left')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=0.0)

    
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
            solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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

    
##############################################################
#  Time optimal hammering task for two moving magnets 
##############################################################
# --------- Used for IROS paper ---------- #
# Time optimal trajectory evaluation - 3 Weights
with skip_run('skip', 'optimize_Tf_and_Vel') as check, check():    
    Data = collections.defaultdict()

    counter = 0
    exps    = ['optimal_tf'] #['weights_1', 'weights_2', 'weights_3']
    # weights for multi-objective optimization
    weights = [[0.5,0.5]] #[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]

    for weight in weights:
        for setting in config['stiff_config']:
            # if (setting == 'high_stiffness'):
            #     pyo_model= robot_linear_dynamics_optimize_tf(weight, config)
            # else:
            pyo_model = optimize_tf_and_vel( setting, weight, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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
            
with skip_run('skip', 'plot_the_optimal_values_LS_VS') as check, check():
    exps    = ['optimal_tf'] #['weights_1', 'weights_2', 'weights_3']
    counter = 0

    plot_magnet_disp = False

    for i in exps:
        if plot_magnet_disp:
            fig, ax = plt.subplots(3,1)
        else:
            fig, ax = plt.subplots(2,1)

        # plt.tight_layout()
        # fig.subplots_adjust(hspace=0.10, wspace=2.25)
        
        filepath1 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'low_stiffness') + '.csv' 
        data1     = pd.read_csv(filepath1, delimiter=',')

        filepath2 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'variable_stiffness') + '.csv' 
        data2     = pd.read_csv(filepath2, delimiter=',')
        
        filepath3 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'high_stiffness') + '.csv' 
        data3     = pd.read_csv(filepath3, delimiter=',')

        ax[0].plot(data1['time'], data1['hd'] + data1['bd'], 'b-', label='LS',  linewidth=1.5)
        ax[0].plot(data2['time'], data2['hd'] + data2['bd'], 'r-' , label='VS', linewidth=1.5)
        ax[0].plot(data3['time'], data3['hd'] + data3['bd'], 'g-', label='HS',  linewidth=2)

        ax[0].plot(data1['time'], 0.05 + 0 * data1['time'], 'k:')
        ax[0].plot([0.8 * data1['time'].iloc[-1], 0.8 * data1['time'].iloc[-1]], [-0.2, 0.1], 'k-.')
        ax[0].plot([0.8 * data2['time'].iloc[-1], 0.8 * data2['time'].iloc[-1]], [-0.2, 0.1], 'k-.')
        ax[0].plot([0.8 * data3['time'].iloc[-1], 0.8 * data3['time'].iloc[-1]], [-0.2, 0.1], 'k-.')

        # ax[0].plot([max(data1['time']), max(data1['time'])], [-0.2, 0.1], 'b:')
        # ax[0].plot([max(data2['time']), max(data2['time'])], [-0.2, 0.1], 'r:')
        # ax[0].plot([max(data3['time']), max(data3['time'])], [-0.2, 0.1], 'g:')
        
        ax[0].set_xlim([0, max(max(data1['time']), max(data2['time']))])
        ax[0].set_ylim([-0.2, 0.1])
        ax[0].legend(loc='lower right')
        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax[1].plot(data1['time'], data1['hv'] + data1['bv'], 'b-', label='LS', linewidth=1.5)
        ax[1].plot(data2['time'], data2['hv'] + data2['bv'], 'r-', label='VS', linewidth=1.5)
        ax[1].plot(data3['time'], data3['hv'] + data3['bv'], 'g-', label='HS', linewidth=1.5)

        ax[1].plot([0.8 * data1['time'].iloc[-1], 0.8 * data1['time'].iloc[-1]], [-1.2, 1.2], 'k-.')
        ax[1].plot([0.8 * data2['time'].iloc[-1], 0.8 * data2['time'].iloc[-1]], [-1.2, 1.2], 'k-.')
        ax[1].plot([0.8 * data3['time'].iloc[-1], 0.8 * data3['time'].iloc[-1]], [-1.2, 1.2], 'k-.')

        # ax[1].plot([max(data1['time']), max(data1['time'])], [-1.2, 1.2], 'b:')
        # ax[1].plot([max(data2['time']), max(data2['time'])], [-1.2, 1.2], 'r:')
        # ax[1].plot([max(data3['time']), max(data3['time'])], [-1.2, 1.2], 'g:')

        # ax[1].plot(data1['time'],  0.5 + 0 * data1['time'], 'k--')
        # ax[1].plot(data1['time'], -0.5 + 0 * data1['time'], 'k--')
        ax[1].set_xlim([0, max(max(data1['time']), max(data2['time'])) ])
        ax[1].set_ylim([-1.2, 1.2])
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if plot_magnet_disp:
            ax[2].plot(data1['time'], 1000 * ( data1['md2'] - 0.03), 'b-', linewidth=1.5)
            ax[2].plot(data1['time'], 1000 * (-data1['md2'] + 0.03), 'b-', linewidth=1.5)
            ax[2].plot(data2['time'], 1000 * ( data2['md2'] - 0.03), 'r-', linewidth=1.5)
            ax[2].plot(data2['time'], 1000 * (-data2['md2'] + 0.03), 'r-', linewidth=1.5)
            ax[2].plot(data3['time'], 1000 * ( data3['md2'] - 0.03), 'g-', linewidth=1.5)
            ax[2].plot(data3['time'], 1000 * (-data3['md2'] + 0.03), 'g-', linewidth=1.5)

            # ax[2].plot([max(data1['time']), max(data1['time'])], [-0.06, 0.06], 'b--')
            # ax[2].plot([max(data2['time']), max(data2['time'])], [-0.06, 0.06], 'r--')
            ax[2].set_xlim([0, max(max(data1['time']), max(data2['time'])) ])
            ax[2].set_xlabel('Time (s)')  

            ax[2].plot([0.8 * data1['time'].iloc[-1], 0.8 * data1['time'].iloc[-1]], [-1000 * 0.045, 1000 * 0.045], 'k-.')
            ax[2].plot([0.8 * data2['time'].iloc[-1], 0.8 * data2['time'].iloc[-1]], [-1000 * 0.045, 1000 * 0.045], 'k-.')
            ax[2].plot([0.8 * data3['time'].iloc[-1], 0.8 * data3['time'].iloc[-1]], [-1000 * 0.045, 1000 * 0.045], 'k-.')
            for i in range(2):
                ax[i].tick_params(axis='x',        # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off
                                
        else:
            ax[1].set_xlabel('Time (s)')
            ax[0].tick_params(axis='x',        # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off

        if counter < 1:
            ax[0].set_ylabel('Displacement (m)')
            ax[1].set_ylabel('Velocity (m/s)')
            if plot_magnet_disp:
                ax[2].set_ylabel('Magnet (m)') 
                ax[2].grid()
        
        # if counter == 1:
        #     _, ax1 = plt.subplots(2,1)
        #     ax1[0].plot(data1['time'], data1['bd'], 'k-.')
        #     ax1[0].plot(data2['time'], data2['bd'], 'k-.')
        #     ax1[0].plot(data1['time'], data1['bd'] + data1['hd'], 'b-')
        #     ax1[0].plot(data2['time'], data2['bd'] + data2['hd'], 'r-')
            
        #     ax1[1].plot(data1['time'], data1['bv'], 'k-.')
        #     ax1[1].plot(data2['time'], data2['bv'], 'k-.')
        #     ax1[1].plot(data1['time'], data1['bv'] + data1['hv'], 'b-')
        #     ax1[1].plot(data2['time'], data2['bv'] + data2['hv'], 'r-')

        counter += 1

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=2.25)
        
        ax[0].grid()
        ax[1].grid()
           
# Time optimal trajectory evaluation - All weights for pareto front
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
            solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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

# Optimal solutions for constant separations
with skip_run('skip', 'data_for_two_magnets_sep_sweep') as check, check():
   
    tf = 1.0
    Separations = np.around(np.arange(0.035, 0.06, 0.01), decimals=3)
    
    df      = pd.DataFrame()

    for sep in Separations:
        pyo_model = user_provided_magnet_separation(tf, sep, config)

        # Solve the optimization problem and return the optimal results
        solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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
    
    # plt.close(g1.fig)
    # plt.close(g2.fig) 
    plt.tight_layout()

    
# Optimal solution for given Tf
with skip_run('skip', 'run_trajectory_optimization_of_the_robot_given_tf') as check, check():        
    pyo_model = robot_linear_dynamics(0.5, config)

    # Solve the optimization problem and return the optimal results
    solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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
    solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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
    
with skip_run('skip', 'run_trajectory_optimization_of_the_robot_for_tf') as check, check():       
    
    alphas = np.round(np.arange(0, 1.0, 0.1), decimals=1)
    weights = [0.5, 0.5]

    for alpha in alphas:
        pyo_model= robot_linear_dynamics_optimize_tf(weight, config)
        
        # Solve the optimization problem and return the optimal results
        solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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

        #TODO:
    # plot the pareto front for optimal values corresponding to time and velocity of the robot


##############################################################
# Impulse and velocity optimization
##############################################################
# --------- Extension of IROS paper ---------- #
# Optimal velocity trajectory given Tf
with skip_run('skip', 'optimize_vel_given_tf_using_impulse') as check, check():    
    Tf              = [0.3, 2.0]# [0.3, 0.5, 1.0, 1.5, 2.0]
    Tf_folder       = ['tf_03', 'tf_20']#['tf_05', 'tf_10', 'tf_15', 'tf_20']

    counter = 0

    for tf in Tf: 
        for setting in config['stiff_config']:
            print('Tf: {}, {}'.format(tf, setting))
            if (setting == 'high_stiffness'):
                pyo_model = robot_linear_dynamics(tf, config)
            else:
                pyo_model = maximize_vel_minimize_impulse(tf, setting, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)

            data, _ , tf = extract_results(solved_model, setting, config)
            
            # save the csv files         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ 'impulse' / Tf_folder[counter] / setting) + '.csv'
            data.to_csv(filepath, index=False)

            plot_disp_vel(data, setting, config)

        counter += 1
                
with skip_run('skip', 'plot_optimal_results_of_hammer_given_tf') as check, check():

    Tf_folder        = ['tf_03', 'tf_20'] #['tf_05', 'tf_10', 'tf_15', 'tf_20']
    plot_magnet_disp = False
      
    for folder_name in Tf_folder:
        if plot_magnet_disp: 
            fig, ax = plt.subplots(3,1)
        else:
            fig, ax = plt.subplots(2,1)

        if folder_name == 'tf_03' or folder_name == 'tf_05':
            ylim      = [-0.1, 0.1]
            ax[0].set_ylabel('Displacement (m)')
            ax[1].set_ylabel('Velocity (m/s)')    

            if plot_magnet_disp:
                ax[2].set_ylabel('magnet (mm)')

        else:
            ylim      = [-0.2, 0.1]
        for setting in config['stiff_config']:
            # load the data         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ folder_name / setting) + '.csv'
            data = pd.read_csv(filepath, delimiter=',')
            tf   = data['time'].iloc[-1]
            w    = 0.03

            if setting == 'high_stiffness':
                linewidth = 2
            else : 
                linewidth = 1.5

            ax[0].plot(data['time'], data['bd']+data['hd'],config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[0].plot(data['time'], 0.05 + 0 * data['time'], 'k:')
            ax[0].plot([0.8 * tf, 0.8 * tf], [-0.2, 0.1], 'k-.')

            ax[0].grid()
            ax[0].set_xlim([0, tf])
            ax[0].set_ylim(ylim)            
            ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax[1].plot(data['time'], data['bv']+data['hv'],config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[1].plot([0.8 * tf, 0.8 * tf], [-1, 1.25], 'k-.')

            ax[1].grid()
            ax[1].set_xlim([0, tf])
            ax[1].set_ylim([-1, 1.25])  
            ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
            
            if plot_magnet_disp:
                ax[2].plot(data['time'], (data['md1'] + w) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
                ax[2].plot(data['time'], (data['md2'] - w) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
                
                ax[2].plot([0.8 * tf, 0.8 * tf], [-.040 * 1000, .040 * 1000], 'k-.')
                
                ax[2].grid()
                ax[2].set_xlim([0, tf])
                ax[2].set_ylim([-.040 * 1000, .040 * 1000])
                ax[2].set_xlabel('Time (s)')  
                # ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f')) 

                for i in range(2):
                    ax[i].tick_params(axis='x',        # changes apply to the x-axis
                                    which='both',      # both major and minor ticks are affected
                                    bottom=False,      # ticks along the bottom edge are off
                                    top=False,         # ticks along the top edge are off
                                    labelbottom=False) # labels along the bottom edge are off
            else:
                ax[1].set_xlabel('Time (s)')
                
                ax[0].tick_params(axis='x',        # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off

            ax[0].legend(loc='lower right')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=0.0)
   
with skip_run('skip', 'plot_optimal_results_of_robot_given_tf') as check, check():

    Tf_folder        = ['tf_03', 'tf_20'] #['tf_05', 'tf_10', 'tf_15', 'tf_20']
    plot_magnet_disp = False
    k = 0 # counter for plot axis

    fig, ax = plt.subplots(2,2, figsize=(10,5))
    for folder_name in Tf_folder:        

        if folder_name == 'tf_03' or folder_name == 'tf_05':
            ylim      = [-0.2, 0.1] #[-0.02, 0.08]
            ax[0,k].set_ylabel('Displacement (m)')
            ax[1,k].set_ylabel('magnet (mm)')

        else:
            ylim      = [-0.2, 0.1]
        for setting in config['stiff_config']:
            # load the data         
            filepath = str(Path(__file__).parents[1] / config['csv_path']/ folder_name / setting) + '.csv'
            data = pd.read_csv(filepath, delimiter=',')
            tf   = data['time'].iloc[-1]
            w    = 0.03

            # if setting == 'high_stiffness':
            #     linewidth = 2
            # else : 
            linewidth = 2

            ax[0,k].plot(data['time'], data['bd'],config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[0,k].plot(data['time'], 0.05 + 0 * data['time'], 'k:')
            ax[0,k].plot([0.8 * tf, 0.8 * tf], [-0.2, 0.1], 'k-.')

            ax[0,k].grid()
            ax[0,k].set_xlim([0, tf])
            ax[0,k].set_ylim(ylim)            
            ax[0,k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # ax[0,k].xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            # ax[0,k].xaxis.set_major_locator(ticker.MultipleLocator(1))
            
            ax[1,k].plot(data['time'], (data['md1'] + 0.03) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[1,k].plot(data['time'], (data['md2'] - 0.03) * 1000,config[setting]['plot_style'], label=config[setting]['plot_label'],  linewidth=linewidth)
            ax[1,k].plot([0.8 * tf, 0.8 * tf], [-.035 * 1000, .035 * 1000], 'k-.')
            
            ax[1,k].grid()
            ax[1,k].set_xlim([0, tf])
            ax[1,k].set_ylim([-.035 * 1000, .035 * 1000])
            ax[1,k].set_xlabel('Time (s)')  
            ax[1,k].xaxis.set_major_formatter(FormatStrFormatter('%.1f')) 
            
            ax[0,k].tick_params(axis='x',        # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off

            ax[0,k].legend(loc='lower right')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=0.1)
        
        k += 1

    # remove the yticks for the second plot
    for j in range(0,2):
        ax[j,1].tick_params(axis='y',       
                        which='both',      
                        left=False,    
                        # top=False,         # ticks along the top edge are off
                        labelleft=False)
        ax[j,0].set_xticks(np.arange(0, 0.3 + 0.1, step=0.1))
        ax[j,1].set_xticks(np.arange(0, 2.0 + 0.5, step=0.5))


# Time optimal trajectory evaluation using Impulse information
with skip_run('skip', 'optimize_Tf_and_Vel_using_impulse') as check, check():    
    Data = collections.defaultdict()

    counter = 0
    exps    = ['optimal_tf'] #['weights_1', 'weights_2', 'weights_3']
    # weights for multi-objective optimization
    weights = [[0.5,0.5]] #[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]

    for weight in weights:
        for setting in config['stiff_config']:
            # if (setting == 'high_stiffness'):
            #     pyo_model= robot_linear_dynamics_optimize_tf(weight, config)
            # else:
            pyo_model       = maximize_vel_minimize_time_impulse(setting, weight, config)
            
            # Solve the optimization problem and return the optimal results
            solved_model    = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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
            
with skip_run('skip', 'plot_the_optimal_values_LS_VS') as check, check():
    exps    = ['optimal_tf'] #['weights_1', 'weights_2', 'weights_3']
    counter = 0

    plot_magnet_disp = False

    for i in exps:
        if plot_magnet_disp:
            fig, ax = plt.subplots(3,1)
        else:
            fig, ax = plt.subplots(2,1)

        # plt.tight_layout()
        # fig.subplots_adjust(hspace=0.10, wspace=2.25)
        
        filepath1 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'low_stiffness') + '.csv' 
        data1     = pd.read_csv(filepath1, delimiter=',')

        filepath2 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'variable_stiffness') + '.csv' 
        data2     = pd.read_csv(filepath2, delimiter=',')
        
        filepath3 = str(Path(__file__).parents[1] / config['csv_path'] / i / 'high_stiffness') + '.csv' 
        data3     = pd.read_csv(filepath3, delimiter=',')

        ax[0].plot(data1['time'], data1['hd'] + data1['bd'], 'b-', label='LS',  linewidth=1.5)
        ax[0].plot(data2['time'], data2['hd'] + data2['bd'], 'r-' , label='VS', linewidth=1.5)
        ax[0].plot(data3['time'], data3['hd'] + data3['bd'], 'g-', label='HS',  linewidth=2)

        ax[0].plot(data1['time'], 0.05 + 0 * data1['time'], 'k:')
        ax[0].plot([0.8 * data1['time'].iloc[-1], 0.8 * data1['time'].iloc[-1]], [-0.2, 0.1], 'k-.')
        ax[0].plot([0.8 * data2['time'].iloc[-1], 0.8 * data2['time'].iloc[-1]], [-0.2, 0.1], 'k-.')
        ax[0].plot([0.8 * data3['time'].iloc[-1], 0.8 * data3['time'].iloc[-1]], [-0.2, 0.1], 'k-.')

        # ax[0].plot([max(data1['time']), max(data1['time'])], [-0.2, 0.1], 'b:')
        # ax[0].plot([max(data2['time']), max(data2['time'])], [-0.2, 0.1], 'r:')
        # ax[0].plot([max(data3['time']), max(data3['time'])], [-0.2, 0.1], 'g:')
        
        ax[0].set_xlim([0, max(max(data1['time']), max(data2['time']))])
        ax[0].set_ylim([-0.2, 0.1])
        ax[0].legend(loc='lower right')
        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax[1].plot(data1['time'], data1['hv'] + data1['bv'], 'b-', label='LS', linewidth=1.5)
        ax[1].plot(data2['time'], data2['hv'] + data2['bv'], 'r-', label='VS', linewidth=1.5)
        ax[1].plot(data3['time'], data3['hv'] + data3['bv'], 'g-', label='HS', linewidth=1.5)

        ax[1].plot([0.8 * data1['time'].iloc[-1], 0.8 * data1['time'].iloc[-1]], [-1.2, 1.2], 'k-.')
        ax[1].plot([0.8 * data2['time'].iloc[-1], 0.8 * data2['time'].iloc[-1]], [-1.2, 1.2], 'k-.')
        ax[1].plot([0.8 * data3['time'].iloc[-1], 0.8 * data3['time'].iloc[-1]], [-1.2, 1.2], 'k-.')

        # ax[1].plot([max(data1['time']), max(data1['time'])], [-1.2, 1.2], 'b:')
        # ax[1].plot([max(data2['time']), max(data2['time'])], [-1.2, 1.2], 'r:')
        # ax[1].plot([max(data3['time']), max(data3['time'])], [-1.2, 1.2], 'g:')

        # ax[1].plot(data1['time'],  0.5 + 0 * data1['time'], 'k--')
        # ax[1].plot(data1['time'], -0.5 + 0 * data1['time'], 'k--')
        ax[1].set_xlim([0, max(max(data1['time']), max(data2['time'])) ])
        ax[1].set_ylim([-1.2, 1.2])
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if plot_magnet_disp:
            ax[2].plot(data1['time'], 1000 * ( data1['md2'] - 0.03), 'b-', linewidth=1.5)
            ax[2].plot(data1['time'], 1000 * (-data1['md2'] + 0.03), 'b-', linewidth=1.5)
            ax[2].plot(data2['time'], 1000 * ( data2['md2'] - 0.03), 'r-', linewidth=1.5)
            ax[2].plot(data2['time'], 1000 * (-data2['md2'] + 0.03), 'r-', linewidth=1.5)
            ax[2].plot(data3['time'], 1000 * ( data3['md2'] - 0.03), 'g-', linewidth=1.5)
            ax[2].plot(data3['time'], 1000 * (-data3['md2'] + 0.03), 'g-', linewidth=1.5)

            # ax[2].plot([max(data1['time']), max(data1['time'])], [-0.06, 0.06], 'b--')
            # ax[2].plot([max(data2['time']), max(data2['time'])], [-0.06, 0.06], 'r--')
            ax[2].set_xlim([0, max(max(data1['time']), max(data2['time'])) ])
            ax[2].set_xlabel('Time (s)')  

            ax[2].plot([0.8 * data1['time'].iloc[-1], 0.8 * data1['time'].iloc[-1]], [-1000 * 0.045, 1000 * 0.045], 'k-.')
            ax[2].plot([0.8 * data2['time'].iloc[-1], 0.8 * data2['time'].iloc[-1]], [-1000 * 0.045, 1000 * 0.045], 'k-.')
            ax[2].plot([0.8 * data3['time'].iloc[-1], 0.8 * data3['time'].iloc[-1]], [-1000 * 0.045, 1000 * 0.045], 'k-.')
            for i in range(2):
                ax[i].tick_params(axis='x',        # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) # labels along the bottom edge are off
                                
        else:
            ax[1].set_xlabel('Time (s)')
            ax[0].tick_params(axis='x',        # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off

        if counter < 1:
            ax[0].set_ylabel('Displacement (m)')
            ax[1].set_ylabel('Velocity (m/s)')
            if plot_magnet_disp:
                ax[2].set_ylabel('Magnet (m)') 
                ax[2].grid()
        
        # if counter == 1:
        #     _, ax1 = plt.subplots(2,1)
        #     ax1[0].plot(data1['time'], data1['bd'], 'k-.')
        #     ax1[0].plot(data2['time'], data2['bd'], 'k-.')
        #     ax1[0].plot(data1['time'], data1['bd'] + data1['hd'], 'b-')
        #     ax1[0].plot(data2['time'], data2['bd'] + data2['hd'], 'r-')
            
        #     ax1[1].plot(data1['time'], data1['bv'], 'k-.')
        #     ax1[1].plot(data2['time'], data2['bv'], 'k-.')
        #     ax1[1].plot(data1['time'], data1['bv'] + data1['hv'], 'b-')
        #     ax1[1].plot(data2['time'], data2['bv'] + data2['hv'], 'r-')

        counter += 1

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.10, wspace=2.25)
        
        ax[0].grid()
        ax[1].grid()


# Plot the impact force profiles for different stiffness settings
# experiment: hitting the nail in wood
with skip_run('skip', 'impact_force_profiles_for_stiffness_settings') as check, check():
    Tf        = ['tf_03', 'tf_20']
    
    mag_sep   = ['0 mm', '0.5mm', '1 mm', '1.5 mm', '2 mm', '2.5 mm', '3 mm']
    stiff_set = ['140', '220', '300', '380', '460'] #, '540', '620']
    
    fig, ax = plt.subplots(2,1)
    for tf_count, tf in enumerate(Tf): 
        for set_count, setting in enumerate(stiff_set):
            filename_const = str(Path(__file__).parents[1] / 'data/experimental_results/IROS_extension' / tf / 'high_stiffness_output_') + setting + '.csv'
            filename_var   = str(Path(__file__).parents[1] / 'data/experimental_results/IROS_extension' / tf / 'variable_stiffness_output.csv')
            data_const = pd.read_csv(filename_const, sep=',')
            data_var   = pd.read_csv(filename_var, sep=',')
            
            time_const = parse_time(data_const['Time'])
            time_var   = parse_time(data_var['Time'])

            plot_const = data_const['Fy']
            plot_var   = data_var['Fy']
            
            plot_const = plot_const - plot_const[0] 
            plot_var = plot_var - plot_var[0] 
            
            if tf == 'tf_03':
                ind = 0.27
            else:
                ind = 2.0
            ax[tf_count].plot(time_var[time_var < ind], plot_var[time_var < ind], 'k-.')
            ax[tf_count].plot(time_const[time_const < ind], plot_const[time_const < ind], label=mag_sep[set_count])
        
            ax[tf_count].grid()
            ax[tf_count].set_ylabel('$F_y$ (N)')
            
    plt.tight_layout()
    plt.legend()

# FIXME:
# Optimizing for alpha is not working
with skip_run('skip', 'run_trajectory_optimization_of_the_robot_for_tf_alpha') as check, check():       
    # weights for multi-objective optimization
    weights = [0.5, 0.5]

    pyo_model= robot_linear_dynamics_optimize_tf(weights, config)
    
    # Solve the optimization problem and return the optimal results
    solved_model = pyomo_solver(pyo_model, 'ipopt', config, neos=False)
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