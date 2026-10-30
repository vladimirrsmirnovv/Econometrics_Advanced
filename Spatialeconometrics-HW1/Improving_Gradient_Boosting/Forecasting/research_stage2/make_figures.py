"""Static scientific comparison of monthly matched validation deltas."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS

def main():
    frozen=json.loads((HERE/'frozen_final_selection.json').read_text())
    dev=pd.read_csv(HERE/'development_model_comparison.csv')
    audit=pd.read_csv(HERE/'reserved_model_comparison.csv')
    rows=[]
    for model in MODELS:
        part=dev[dev.model==model].set_index('configuration')
        for fold in frozen['development_months']:
            base=part.loc['baseline',fold]
            rows.append(dict(model=model,fold=fold,previous=part.loc['previous',fold]-base,
                selected=part.loc[frozen['individual'][model],fold]-base))
        for fold in frozen['reserved_months']:
            part=audit[(audit.model==model)&(audit.fold==fold)].set_index('variant').mape
            rows.append(dict(model=model,fold=fold,previous=part['previous_rolling']-part['original_rolling'],
                selected=part['frozen_new']-part['original_rolling']))
    frame=pd.DataFrame(rows);out=HERE/'figures';out.mkdir(exist_ok=True)
    frame.to_csv(out/'monthly_deltas.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    titles=['CatBoost: уровень','LightGBM: приросты','LightGBM: уровень','XGBoost: приросты']
    for ax,model,title in zip(axes.ravel(),MODELS,titles):
        part=frame[frame.model==model];x=np.arange(len(part))
        ax.bar(x-.18,part.previous,width=.35,color='#9ca3af',label='Прежняя модификация')
        ax.bar(x+.18,part.selected,width=.35,color='#247a8a',label='Новый выбранный вариант')
        ax.axhline(0,color='#374151',lw=.9);ax.axvline(2.5,color='#6b7280',ls='--',lw=1)
        ax.set_title(title,loc='left',weight='bold');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_xticks(x,['Июн 24','Июл 24','Авг 24','Ноя 24','Дек 24'])
    axes[0,0].legend(loc='lower left',frameon=False)
    fig.suptitle('Изменение MAPE относительно исходной модели при одинаковом rolling-протоколе\n'
        'Слева — разработка; справа от пунктирной линии — контроль после фиксации выбора',fontsize=13)
    fig.supylabel('Δ MAPE, п.п.  •  отрицательные значения лучше')
    fig.savefig(out/'monthly_deltas.png',dpi=180)
    fig.savefig(out/'monthly_deltas.svg');plt.close(fig)
    print('MONTHLY_COMPARISON_FIGURES_WRITTEN')

if __name__=='__main__':main()
