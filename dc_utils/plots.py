import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

def _run_plot(times,states,params,ninter,model):

    _full_time = np.concatenate(states)
    nums=np.random.random((10,ninter))
    colors = cm.rainbow(np.linspace(0, 1, nums.shape[0]))  # generate the colors for each data set
    fig, ax = plt.subplots(states.shape[2],1,figsize=(16,10))

    for state, i in zip(states, range(ninter)):
        for j in range(states.shape[2]):
            ax[j].set_ylim(0, np.max(_full_time[:,j])+10)
            ax[j].plot(times[i],state[:,j], color = colors[i],  linestyle = '-',linewidth=2)
            ax[j].set_xlabel('Time t, [days]',fontsize =12)
            ax[j].set_ylabel('State {}'.format(j+1),fontsize =12)
            _add_values(ax[j],times[i],params[i],colors[i],i,ninter,model)

    return plt.show()
    
    
def _add_values(ax,time,params,c,idx,ninter,model):
    
    params = np.round(params,3)
    x = ((ax.get_xlim()[1]//ninter)*idx)
    y = ax.get_ylim()[1]-10

    if model == 'lv':
        ax.text(x,y,
                '$\\alpha = $'+str(params[0])+ '\n' +\
                '$\\delta = $'+str(params[1])+ '\n' +\
                '$\\gamma = $'+str(params[2])+ '\n'+\
                '$\\beta = $'+str(params[3])+ '\n',
                fontsize = 12,
                color = c,
                )
      
    if model =='sir':
        ax.text(x,y,
            '$R_0 = $'+str(params)+ '\n',
            fontsize = 12,
            color = c,
            )
#  bbox=dict(boxstyle="square,pad=0.1", fc="gray", ec="b", lw=2)