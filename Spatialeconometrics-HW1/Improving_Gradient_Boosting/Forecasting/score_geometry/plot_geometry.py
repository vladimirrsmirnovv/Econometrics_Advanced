"""Render analytical metric curves from the saved calculations (no model fitting)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12})
fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
w=np.linspace(0,1,401)
p0=np.ones(3);delta=np.array([1.,2.,3.])
cases=json.loads((HERE/'metric_diagnostics.json').read_text(encoding='utf-8'))['counterexample_same_predictions_and_endpoint_MAPE']
for j,case in enumerate(cases):
 y=np.array(case['targets']);curve=100*np.abs((p0[:,None]+delta[:,None]*w-y[:,None])/y[:,None]).mean(axis=0)
 axes[0].plot(w,curve,label=f'Выборка {j+1}: оптимум {case["optimal_weight"]:.3f}')
axes[0].set(title='Одинаковые концы, разные минимумы',ylabel='MAPE, %',xlabel='Вес поправки')
axes[0].legend(fontsize=9)
curve=next(x for x in json.loads((HERE/'local_curves.json').read_text(encoding='utf-8')) if x['fold']=='2024-12' and x['direction']=='D')
axes[1].plot(curve['weights'],curve['mape'],label='Фактическая MAPE')
axes[1].plot(curve['weights'],curve['quadratic'],'--',label='Парабола по трем точкам')
axes[1].set(title='Реальный срез: декабрь 2024, D',ylabel='MAPE, %',xlabel='Вес от U к D')
axes[1].ticklabel_format(axis='y',style='plain',useOffset=False)
axes[1].legend(fontsize=9)
weights=np.array([0,1/3,1]);scores=np.array([90.83,90.83,90.82])
q=100*(1-np.sqrt(scores/100));coef=np.polyfit(weights,q,2)
axes[2].plot(w,100*(1-np.polyval(coef,w)/100)**2,'--',label='Условная интерполяция MAPE')
axes[2].errorbar(weights,scores,yerr=.005,fmt='o',capsize=5,label='Баллы ± округление 0,005')
axes[2].set(title='Март: точная кривая неизвестна',ylabel='Баллы X5',xlabel='Вес от U к C или D',ylim=(90.812,90.838))
axes[2].ticklabel_format(axis='y',style='plain',useOffset=False)
axes[2].legend(fontsize=9)
for ax in axes:
 ax.grid(alpha=.2)
 ax.spines[['top','right']].set_visible(False)
fig.savefig(HERE/'metric_geometry.png',dpi=180)
fig.savefig(HERE/'metric_geometry.svg')
print('Saved metric_geometry.png and .svg')
