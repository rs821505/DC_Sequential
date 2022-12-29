import numpy as np
import pandas as pd
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
        
def _print_dims(model):
    """
    """
    dims = ['states dim','parameters dim', 'times dim']
    for label,quantity in zip(dims,model._get_outputs()):
        print(label,quantity.shape)
        
def _create_paramdf(params,cols,names=None):
    """
    """
    if cols.__eq__('pp'):
        names = [r"$\alpha$",r"$\beta$",r"$\delta$",r"$\gamma$"]
    elif cols.__eq__('sir'):
        names = [r"$R_0$"]
    elif cols.__eq__('rlc'):
        names = [r"$R$",r"$L$",r"$C$"]

    pdf = pd.DataFrame(params,columns=names)
    pdf.index.name ='Drift_Window'
    return pdf
        
def _plot_intervals(states,times):
    """
    """
    intervals = times[:-1,-1]
    for i in range(intervals.size):
        plt.axvline(intervals[i],0,np.max(states)+1, linestyle='--',linewidth=1,color='g')
        
        
def _plot_states(states,times,labels,intervals=True):
    """
    """
    plt.plot(np.concatenate(states))
    plt.xlabel('Time'); plt.ylabel('States')
    plt.xlim(0,np.round(times.max()))

    if intervals:
        _plot_intervals(states,times)
        
    plt.legend(labels)
    plt.show()