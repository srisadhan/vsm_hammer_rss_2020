import io
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# if compact printing is required
np.set_printoptions(precision=2)

def friction_func(xdot, mass, mu, damping):
    return np.sign(xdot) *(mu * mass * 9.8) + damping * xdot

def parse_time(time_dataframe):
    split_time_str = pd.Series.str(time_dataframe).split(':')
    # convert the hours*3600, minutes*60, seconds*1, and microseconds*1e-6 format to seconds
    time_array = np.array([int(x[0])*3600 + int(x[1])*60 + int(x[2]) + int(x[3])*1e-6 for x in split_time_str]).reshape(-1, )
    
    return time_array

def calculate_friction(mass, dataframe, start, stop, setting):
    dataframe['Time']  = parse_time(dataframe['Time'])
    dataframe['Time']  = dataframe['Time'] - dataframe['Time'][0]

    dt = np.round(np.mean(np.diff(dataframe['Time'].to_numpy())), 3)
    
    # -----------------------------------------------
    # The window length and polynomial order for pos, vel, and acc are selected iteratively 
    # to match the filtered data with the original data as much as possible
    # -----------------------------------------------
    # hammer position obtained from the potentiometer
    hammer_pos = dataframe['Handle_position'].to_numpy()
    hammer_pos = savgol_filter(hammer_pos, window_length=21, polyorder=5)
    
    
    
    # Filter the hammer velocity and acceleration using Savitzky-Golay filter
    hammer_vel = np.diff(hammer_pos, n=1) / dt
    hammer_vel = savgol_filter(hammer_vel, window_length=21, polyorder=5)
    
    hammer_acc = np.diff(hammer_pos, n=2) / dt**2
    hammer_acc = savgol_filter(hammer_acc, window_length=21, polyorder=5)
    

    # resultant force = F1 - F2 (here 1 is the magnet in negative x direction and 2 on the other side)
    res_force = dataframe['Resultant_force'].to_numpy()
    # filter the magnet force which is based on the hammer position
    res_force = savgol_filter(res_force, window_length=21, polyorder=5)
    
    friction_force = res_force[:-2] - mass * hammer_acc
    
    popt, _ = curve_fit(friction_func, hammer_vel[start:stop], friction_force[start:stop], p0=[mass, 0.01, 2.0], bounds=([mass-0.001, 0.005, 0.4], [mass+0.001, 0.5, 5]))
    print(popt)
    
    estimated_friction = friction_func(hammer_vel[start:stop], mass, 0.1, 2)
    
    residual = friction_force[start:stop] - friction_func(hammer_vel[start:stop], mass, 0.1, 2)
    
    # _, ax = plt.subplots(2,1)
    # ax[0].plot(friction_force[start:stop], 'b', label='actual friction')
    # ax[0].plot(estimated_friction, 'r', label='estimated friction')
    # ax[0].set_title(setting + ' friction force estimation')
    # ax[0].legend()
    
    # ax[1].plot(residual)
    # ax[1].set_ylabel('residual')
    
    plt.figure()
    plt.plot(hammer_pos)    
    plt.plot(res_force/20)
    plt.title(setting)

config  = yaml.safe_load(io.open('src/config.yml'))
df_low1  = pd.read_csv(config['low_mass_file_1'], delimiter=',')
df_low2  = pd.read_csv(config['low_mass_file_2'], delimiter=',')
df_high1 = pd.read_csv(config['high_mass_file_1'], delimiter=',')
df_high2 = pd.read_csv(config['high_mass_file_2'], delimiter=',')

m_low  = 0.06 # as the hammer weight is 140 grams
m_high = 0.2

print('mass,    mu,     damping')

# friction estimation for two different outer magnet settings and without hammer 
calculate_friction(m_low, df_low1, start=430, stop=489, setting='0.06 kg') # start=414 
calculate_friction(m_low, df_low2, start=370, stop=416, setting='0.06 kg') # start=357

# friction estimation for two different outer magnet settings and with hammer 
calculate_friction(m_high, df_high1, start=660, stop=799, setting='0.2 kg') # start=629
calculate_friction(m_high, df_high2, start=400, stop=536, setting='0.2 kg') # start=371

print('Final values of mu:{}, damping:{} used'.format(0.1, 2))
plt.show()