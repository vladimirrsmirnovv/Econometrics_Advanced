"""Standalone retrospective before/after figure at identical full tree counts."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
plot_cache=HERE/'cache/matplotlib'
plot_cache.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(plot_cache)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main():
    f=pd.read_csv(HERE/'confirmation_model_comparison.csv')
    labels=['CatBoost\nуровни','LightGBM\nприросты','LightGBM\nуровни','XGBoost\nприросты'];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,fold,title in zip(axes,['2025-01','2025-02'],['Январь 2025','Февраль 2025']):
        before=f[fold+'_before'].to_numpy();after=f[fold+'_WX'].to_numpy()
        ax.bar(x-.19,before,.38,label='Исходная модель',color='#697d94')
        ax.bar(x+.19,after,.38,label='Учебный WX',color='#22a69a')
        for positions,values in [(x-.19,before),(x+.19,after)]:
            for pos,val in zip(positions,values):ax.text(pos,val+.04,f'{val:.3f}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(x,labels);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(0,max(f.filter(regex='_(before|WX)$').max())*1.15)
    axes[0].set_ylabel('MAPE, % (ниже — лучше)')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2)
    fig.suptitle('Учебный WX при исходных полных числах деревьев',fontsize=14)
    fig.text(.5,.015,'Ретроспективная проверка. WX выбран на августе/ноябре/декабре; истинный март неизвестен.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.86));fig.savefig(HERE/'before_after.png',dpi=180);fig.savefig(HERE/'before_after.svg');plt.close(fig)
    print('COURSE_FIGURE_SAVED',flush=True)
if __name__=='__main__':main()
