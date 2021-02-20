import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import re

def plot_graph(y1, y2, x1, ENV_NAME, NO_OF_TRAINERS, ROLLING, HOW_MANY_VALUES = None):
    if(HOW_MANY_VALUES == None):
        HOW_MANY_VALUES = len(y1)
    x1 = x1[:HOW_MANY_VALUES]
    y1 = y1[:HOW_MANY_VALUES]
    y2 = y2[:HOW_MANY_VALUES]
    plt.figure(figsize=[12, 9])
    plt.subplot(1, 1, 1)
    plt.title(ENV_NAME)
    plt.xlabel('epochs')
    plt.ylabel('mse loss')
    plt.plot(x1, y1, color='lightgreen')
    plt.plot(x1, y2, color='pink')
    plt.plot(x1, pd.DataFrame(y1)[0].rolling(ROLLING).mean(), color='green')
    plt.plot(x1, pd.DataFrame(y2)[0].rolling(ROLLING).mean(), color='red')
    plt.grid()
    # plt.legend()

    
    plt.savefig('/home/niranth/Desktop/Work/USC_Task/rl_algos/plot_data/Sprites-v0_avg_rewards_enc.png')
    # plt.close()
    plt.show()

if __name__ == "__main__":
    # ENV_NAME = 'Acrobot-v1'
    # ENV_NAME = 'CartPole-v0'
    ENV_NAME = 'SHG'
    # import pdb; pdb.set_trace()
    # y1 = torch.load('plot_data/Sprites-v0_avg_rewards_enc.pt')
    # y2 = torch.load('plot_data/Sprites-v0_avg_rewards_enc.pt')
    import pickle
    with open('/home/niranth/Desktop/Work/Intern pose and shape estimation/pytorch_stacked_hourglass/exp/kp2/log') as p:
    	data = p.read()
    points = re.findall("\d+\.\d+", data)

    # import pdb; pdb.set_trace()
    data2 = [float(i) for i in points]
    y1 = data2[0::1250]
    y2 = data2[0::1250]
    # y1 = torch.load('/home/niranth/Desktop/Work/USC_Task/rl_algos/plot_data/Sprites-v0_avg_rewards_enc.pt')
    # y2 = torch.load('/home/niranth/Desktop/Work/USC_Task/rl_algos/plot_data/Sprites-v0_avg_rewards_enc.pt')
    # x1 = np.loadtxt('arrays/steps_'+ENV_NAME+'.csv')
    x1 = np.arange(0,len(y1),1) 

    plot_graph(y1, y2, x1, ENV_NAME, 3, 1)