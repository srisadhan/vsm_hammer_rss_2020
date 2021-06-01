import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# import math

def plot_disp_vel(data, plot_title, config):
    """Plot the displacement and velocity of the end-effector, hammer and the magnets
    
    Arguments:
        data {dataframe}    -- [a pandas dataframe consisting of the optimal results for hammering]
        plot_title {string} -- [title for the plot]
        config {yaml}       -- [configurations provided in the config.yaml file]
    """

    t           =   data['time'] 
    baseD       =   data['bd' ] 
    baseV       =   data['bv' ] 
    hammerD     =   data['hd' ] 
    hammerV     =   data['hv' ] 
    magnetD1    =   data['md1'] 
    magnetD2    =   data['md2'] 
    
    if 'impulseR' in data.keys():
        impulseR = data['impulseR']
        
    tf          = max(t)

    fig,ax = plt.subplots(3,1)

    ax[0].plot(t,baseD,color='k',linestyle='solid')
    ax[0].plot(t,baseD+hammerD,color='b',linestyle='dashed')
    ax[0].plot(t, config['nail_pos'] + 0 * t, 'k--')
    ax[0].set_ylabel('Displacement (m)')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_xlim([0,tf])
    ax[0].set_ylim([-0.1, 0.1])
    ax[0].legend(['end-effector','end-effector+hammer'])
    ax[0].grid()

    ax[1].plot(t,baseV,color='k',linestyle='solid')
    ax[1].plot(t,baseV+hammerV,color='b',linestyle='dashed')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_xlim([0,tf])
    ax[1].set_ylim([-1.2, 1.2])
    ax[1].legend(['end-effector','end-effector+hammer'])
    ax[1].grid()

    ax[2].plot(t, magnetD1 + config['w'], t, magnetD2 - config['w'], color='k', linestyle='solid')
    ax[2].plot(t, hammerD)
    ax[2].set_xlim([0,tf])
    ax[2].set_ylim([-0.07, 0.07])
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Displacement (m)')
    ax[2].legend(['magnet_1','magnet_2','hammer'])
    ax[2].set_xlabel('Time (s)')

    fig.suptitle(plot_title)
    
    # plt.figure(10)
    # plt.plot(t, impulseR)


def plot_settings_together(Data, config):
    """Plot the displacement and velocity of the end-effector, hammer and the magnets
    
    Arguments:
        Data {dataframe}    -- [a pandas dataframe consisting of the optimal results for hammering]
        config {yaml}       -- [configurations provided in the config.yaml file]
    """
    max_tf = 0

    fig,ax = plt.subplots(2,1)
    # ax[0].hold(true)
    # ax[1].hold(true)

    for setting in config['stiff_config']:
        if setting != 'high_stiffness':
            data        = Data[setting]

            t           =   [x for x in data['time'].values()] 
            baseD       =   [x for x in data['bd' ].values()]  
            baseV       =   [x for x in data['bv' ].values()]  
            hammerD     =   [x for x in data['hd' ].values()]  
            hammerV     =   [x for x in data['hv' ].values()]  
            magnetD1    =   [x for x in data['md1'].values()]  
            magnetD2    =   [x for x in data['md2'].values()]  
            totalD      =   [x + y for x, y in zip(baseD, hammerD)]
            totalV      =   [x + y for x, y in zip(baseV, hammerV)]

            tf          = max(t)        

            if tf > max_tf:
                max_tf = tf

            if setting == 'low_stiffness':
                ax[0].plot(t,totalD,color='b', label= 'hammer')
                ax[1].plot(t,totalV,color='b', label= 'hammer')
            else:
                ax[0].plot(t,totalD,color='r', label= 'hammer')
                ax[1].plot(t,totalV,color='r', label= 'hammer')
            
            ax[0].plot(t,baseD,color='k', label= 'robot')
            ax[1].plot(t,baseV,color='k', label= 'robot')

            ax[0].plot([t[-1], t[-1]], [-0.1, 0.1], 'k', linestyle='dashed')
            ax[1].plot([t[-1], t[-1]], [-1.2, 1.2], 'k', linestyle='dashed')

        # ax[2].plot(t, magnetD1, t, magnetD2, linestyle='solid')
        # ax[2].plot(t, hammerD)
        # ax[2].set_xlim([0,tf])
        # ax[2].set_ylim([-0.07, 0.07])
        # ax[2].set_xlabel('Time (s)')
        # ax[2].set_ylabel('Displacement (m)')
        # ax[2].legend(['magnet_1','magnet_2','hammer'])
        # ax[2].set_xlabel('Time (s)')

    ax[0].set_xlim([0,max_tf+0.5])
    ax[0].set_ylim([-0.1, 0.1])
    ax[0].set_ylabel('Displacement (m)')
    ax[0].set_xlabel('Time (s)')    
    ax[0].legend(['end-effector','end-effector+hammer'])
    ax[0].grid()

    ax[1].set_xlim([0,max_tf+0.5])
    ax[1].set_ylim([-1.2, 1.2])
    ax[1].set_ylabel('Velocity (m/s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend(['end-effector','end-effector+hammer'])
    ax[1].grid()


def plot_pareto_front(data,  config):
    """Plot the displacement and velocity of the end-effector, hammer and the magnets
    
    Arguments:
        data {dataframe}    -- [a pandas dataframe consisting of the optimal results for hammering]
        plot_title {string} -- [title for the plot]
        config {yaml}       -- [configurations provided in the config.yaml file]
    """
    
    plot_data = pd.DataFrame()

    fig, ax = plt.subplots()

    LS      = np.zeros((9,2))
    VS      = np.zeros((9,2))

    for counter in range(1, 10):
        
        for setting in config['stiff_config']:

            tf      = data[counter]['tf']
            vmax    = data[counter][setting]

            temp        = pd.DataFrame({'tf': tf, 'vmax': vmax, 'setting': setting, 'weights': 0.1*counter}, index=[0])
            plot_data   = plot_data.join(temp, lsuffix='_caller', rsuffix='_other')
            
            # print(counter, setting, tf, vmax)
            # if setting == 'high_stiffness':
                # plt.plot(tf, vmax, color = (0, 0, 0), marker='.')
            # el
            if setting == 'low_stiffness':
                LS[counter-1, :]  = np.array([tf, vmax])
                plt.plot(tf, vmax,  color = (0, 0, 1), marker='o')
                if (counter > 1) and (counter <= 8):
                    ax.text(tf, vmax-0.01, str(counter), fontsize=14, verticalalignment='top')
                elif counter > 8:
                    ax.text(tf+0.05, vmax+0.01, str(counter), fontsize=14, verticalalignment='top')

            elif setting == 'variable_stiffness':
                VS[counter-1, :]  = np.array([tf, vmax])
                plt.plot(tf, vmax,  color = (1, 0, 0), marker='o')
                if (counter > 1) and (counter <= 8):
                    ax.text(tf, vmax+0.02, str(counter), fontsize=14, verticalalignment='top')
                elif counter > 8:
                    ax.text(tf, vmax-0.01, str(counter), fontsize=14, verticalalignment='top')
    
            

    # fig,ax  = plt.subplots()

    # sns_palette = sns.color_palette("viridis", n_colors=3, desat=0.5)
    # sns_palette = sns.cubehelix_palette(5, start=2, rot=-.75)

    # g1 = sns.relplot(x="tf", y="vmax", hue="setting", kind="line", palette=sns_palette, data=plot_data)
    # g2 = sns.relplot(x="tf", y="vmax", hue="weights", kind="line", palette=sns_palette, data=plot_data, ax=ax[1])
    
    # plt.close(g1.fig)
    # plt.close(g2.fig) 
    # plt.tight_layout()
    ind1 = np.argsort(LS[:,0], axis=0)
    ind2 = np.argsort(VS[:,0], axis=0)

    LS = LS[ind1,:]
    VS = VS[ind2,:]

    print(LS, LS.shape)

    plt.plot(LS[:,0], LS[:,1], 'b--', label='LS')
    plt.plot(VS[:,0], VS[:,1], 'r--', label='VS')

    plt.xlabel('tf (sec)')
    plt.ylabel('maximum hammer velocity (m/s)')

    plt.legend()
    plt.tight_layout()
    plt.show()

    
