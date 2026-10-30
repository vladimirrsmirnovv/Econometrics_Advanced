"""Build report tables/figures from saved evidence; never fit forecasting models."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import hashlib
import json
import shutil
import os

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / 'tmp/pdfs/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORECASTING = ROOT / 'Forecasting'
COURSE = FORECASTING / 'course_only'
MODELS = ['catboost_log_level', 'lightgbm_log_diff', 'lightgbm_log_level', 'xgboost_log_diff']
NAMES = ['CatBoost: уровни', 'LightGBM: приросты', 'LightGBM: уровни', 'XGBoost: приросты']
TEX_NAMES = ['CatBoost, уровни', 'LGB, приросты', 'LGB, уровни', 'XGBoost, приросты']
BLUE, RED = '#254e70', '#a04646'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def num(value, places=3):
    return f'{value:.{places}f}'.replace('.', ',')


def line(*cells):
    return ' & '.join(str(c) for c in cells) + r' \\' + '\n'


def saveplot(fig, name):
    fig.savefig(HERE / 'assets' / (name + '.png'), dpi=260, bbox_inches='tight', facecolor='white')
    fig.savefig(HERE / 'assets' / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    for name in ['assets', 'tables', 'sources', 'output/pdf']:
        (HERE / name).mkdir(parents=True, exist_ok=True)
    tracked = [resource_path('Paper/main.tex'), resource_path('Paper/Пространственная_эконометрика_ДЗ1.pdf'),
               resource_path('Analysis/analysis.R'), COURSE/'FINAL_DECISION.json', FORECASTING/'run_validation.py']
    before = {artifact_relative(p): sha(p) for p in tracked}
    description = Path('C:/Users/smirn/.codex/attachments/d54358bb-b2bc-42cc-8fcf-156d55b7569f/Вставленный текст.txt')
    condition = Path('C:/Users/smirn/.codex/attachments/fdb334a3-280f-4730-bdc2-ac15b84d4d3a/Вставленный текст.txt')
    scheme = Path('C:/Users/smirn/Downloads/model_scheme.jpg')
    for source, target in [(description,HERE/'sources/competition_description.md'),
                           (condition,HERE/'sources/competition_condition.txt'),
                           (scheme,HERE/'assets/original_model_scheme.jpg')]:
        if source.exists():
            shutil.copyfile(source, target)
        assert target.exists(), target
    description = HERE/'sources/competition_description.md'
    condition = HERE/'sources/competition_condition.txt'

    registries = [
        ('Модели и первые ансамбли', COURSE/'leaderboard_results_2026-10-11.json'),
        ('Три смеси', FORECASTING/'submissions/2025-03-course-ensemble-alternatives/public_results.json'),
        ('Семь смесей', FORECASTING/'submissions/2025-03-course-ensemble-followups/public_results.json'),
        ('Десять смесей', FORECASTING/'submissions/2025-03-course-final-ten/contest_results.json'),
        ('Пять финальных смесей', FORECASTING/'submissions/2025-03-course-final-five/contest_results.json'),
    ]
    evidence, records = [], []
    for batch, path in registries:
        data = json.loads(path.read_text(encoding='utf-8'))
        evidence.append(dict(file=str(path.relative_to(ROOT)), sha256=sha(path)))
        for row in data['submissions']:
            score = row.get('contest_score', row.get('public_score'))
            assert score is not None and row['verdict']=='OK'
            records.append(dict(batch=batch, **row, report_score=score))
    assert len(records)==36
    pd.DataFrame(records).to_csv(HERE/'tables/submission_registry.csv', index=False)

    comp = pd.read_csv(COURSE/'confirmation_model_comparison.csv').set_index('model').loc[MODELS]
    dev = pd.read_csv(COURSE/'development_model_comparison.csv')
    comp.to_csv(HERE/'tables/model_before_after_local.csv')
    initial = json.loads(registries[0][1].read_text(encoding='utf-8'))['submissions']
    initial_by_order = {r.get('order', i+1): r for i,r in enumerate(initial)}
    score_pairs = []
    confirmation = ''
    private = ''
    for i, model in enumerate(MODELS):
        row=comp.loc[model]
        confirmation += line(TEX_NAMES[i], f"{int(row.trees):,}".replace(',', r'\,'),
                             row.WX.replace('wx_knn', ''),
                             num(row['2025-01_before']), num(row['2025-01_WX']),
                             num(row['2025-02_before']), num(row['2025-02_WX']))
        b=initial_by_order[2*i+1]['contest_score']
        n=initial_by_order[2*i+2]['contest_score']
        score_pairs.append(dict(model=model, before=b, WX=n, gain=n-b))
        private += line(TEX_NAMES[i], num(b,2), num(n,2), num(n-b,2))
    (HERE/'tables/confirmation.tex').write_text(confirmation, encoding='utf-8')
    (HERE/'tables/final_individual.tex').write_text(private, encoding='utf-8')
    pd.DataFrame(score_pairs).to_csv(HERE/'tables/model_before_after_final.csv', index=False)
    development=''
    for i,model in enumerate(MODELS):
        part=dev[dev.model==model].set_index('configuration')
        development += line(TEX_NAMES[i], *[num(part.loc[c,'mean_mape']) for c in ['baseline','wx_knn6','wx_knn12']],
                            comp.loc[model,'WX'].replace('wx_knn', r'$k=' )+'$')
    (HERE/'tables/development.tex').write_text(development, encoding='utf-8')

    diagnostic=pd.read_csv(COURSE/'moran_diagnostics.csv')
    moran=''
    for i, model in enumerate(MODELS):
        k='knn'+comp.loc[model,'WX'].removeprefix('wx_knn')
        part=diagnostic[(diagnostic.model==model)&(diagnostic.fold=='2025-02')&(diagnostic.W==k)].set_index('configuration')
        moran+=line(TEX_NAMES[i],k.replace('knn', ''),num(part.loc['baseline','moran_I'],4),
                    num(part.loc[comp.loc[model,'WX'],'moran_I'],4), '0,001')
    (HERE/'tables/moran.tex').write_text(moran, encoding='utf-8')

    scores={str(row['submission_id']):row['report_score'] for row in records}
    grid_ids=[['167916936','167918237','167918232','167918244','167918251'],
              ['167918221','167921138','167918232','167921159','167921148'],
              ['167918261','167921212','167921206',None,'167921228'],
              ['167921168','167921175','167921206','167921196','167921188']]
    grid=np.array([[scores[key] if key is not None else np.nan for key in row] for row in grid_ids])
    pd.DataFrame(grid,index=['XGB WX','XGB+LGB diff WX','XGB+LGB level WX','XGB+both LGB WX'],
                 columns=['meta','gated_all','gated_no_ld','equal','log']).to_csv(HERE/'tables/ensemble_grid.csv')
    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes=plt.subplots(1,2,figsize=(8.3,4.2),gridspec_kw={'width_ratios':[1.1,1]})
    before_mean=comp[['2025-01_before','2025-02_before']].mean(axis=1).to_numpy()
    after_mean=comp[['2025-01_WX','2025-02_WX']].mean(axis=1).to_numpy()
    y=np.arange(4)
    axes[0].barh(y-.17,before_mean,height=.3,color=BLUE,label='Исходная')
    axes[0].barh(y+.17,after_mean,height=.3,color=RED,label='С WX')
    axes[0].set_yticks(y, ['CatBoost','LGB: приросты','LGB: уровни','XGBoost']); axes[0].invert_yaxis()
    axes[0].set_xlim(0,5.5);axes[0].set_xlabel('Средняя MAPE\nянваря/февраля, %')
    axes[0].set_title('Локальная оценка')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.25),ncol=2,frameon=False)
    for j in y:
        axes[0].text(before_mean[j]+.035,j-.17,num(before_mean[j]),va='center',fontsize=10)
        axes[0].text(after_mean[j]+.035,j+.17,num(after_mean[j]),va='center',fontsize=10)
    gain=np.array([r['gain'] for r in score_pairs])
    axes[1].barh(y,gain,color=[BLUE if x>0 else RED for x in gain],height=.45)
    axes[1].set_yticks(y,[]);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',lw=.8)
    axes[1].set_xlim(-.24,.22);axes[1].set_xticks([-.2,-.1,0,.1,.2])
    axes[1].set_xlabel('Изменение итогового\nбалла после WX')
    axes[1].set_title('Полный мартовский тест')
    for j,x in enumerate(gain):
        axes[1].text(x+(.01 if x>=0 else -.01),j,f'{x:+.2f}'.replace('.',','),va='center',ha='left' if x>=0 else 'right')
    fig.tight_layout(w_pad=2.2);saveplot(fig,'model_comparison')

    fig,ax=plt.subplots(figsize=(10.2,4.1))
    ax.imshow(np.ma.masked_invalid(grid),cmap='Blues',vmin=90.76,vmax=90.84,aspect='auto')
    ax.set_xticks(range(5),['Исходная\nmeta-смесь','Порог 1,08\n4 модели','Порог 1,08\nбез LGB\nприростов','Обычное\nсреднее','Логарифм.\nсреднее'])
    ax.set_yticks(range(4),['WX только у XGB','WX: XGB + LGB приростов','WX: XGB + LGB уровней','WX: XGB + оба LGB'])
    for i in range(4):
        for j in range(5):
            x=grid[i,j]
            ax.text(j,i,'Нет оценки' if np.isnan(x) else num(x,2),ha='center',va='center',
                    color='black' if np.isnan(x) or x<90.815 else 'white',fontweight='bold' if x==90.83 else 'normal')
    ax.tick_params(length=0);ax.set_title('Итоговый балл. CatBoost во всех составах исходный',pad=14)
    fig.tight_layout();saveplot(fig,'ensemble_grid')

    fig,ax=plt.subplots(figsize=(10.5,3.25));ax.set_xlim(0,10.5);ax.set_ylim(0,3.25);ax.axis('off')
    def box(x,y,w,h,text,color='#eef1f4'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#5a6570',lw=.9))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#52616b',lw=1.1))
    box(.1,1.8,2.0,1.05,'История магазинов\nдо февраля 2025\nИсходные признаки X')
    box(2.6,1.8,2.35,1.05,'Средние X поселений\nМатрица W: kNN6/12\nПространственные WX')
    box(5.5,1.8,2.2,1.05,'Четыре бустинга\nИсходные параметры\nПолное обучение')
    box(8.25,1.8,2.0,1.05,'Исходные правила\nансамблирования\nПрогноз марта')
    box(2.75,.2,4.75,.7,'Сравнение версий: локальные месяцы\nи полный итоговый тест марта',color='#e8eef3')
    arrow((2.18,2.325),(2.51,2.325));arrow((5.03,2.325),(5.42,2.325));arrow((7.78,2.325),(8.17,2.325))
    arrow((1.1,1.72),(3.8,.97));arrow((9.25,1.72),(6.5,.97))
    fig.tight_layout();saveplot(fig,'workflow')

    # Evidence appendix: 36 actual score records, not a count of new model fits.
    labels_first=['CatBoost: исходный','CatBoost: WX','LGB приростов: исходный','LGB приростов: WX',
                  'LGB уровней: исходный','LGB уровней: WX','XGB: исходный','XGB: WX',
                  'Исходная meta-смесь','Все четыре WX: meta','Только LGB приростов WX: meta']
    labels_three=['Только XGB WX: meta','Все четыре WX: среднее','Все четыре WX: лог-среднее']
    labels_seven=['CatBoost WX + XGB WX: meta','XGB + LGB приростов WX: meta',
                  'XGB WX: порог без LGB приростов','XGB WX: порог четырех','XGB WX: среднее',
                  'XGB WX: лог-среднее','XGB + LGB уровней WX: meta']
    labels_ten=['XGB + LGB приростов WX: порог','XGB + LGB приростов WX: лог-среднее',
                'XGB + LGB приростов WX: среднее','XGB + оба LGB WX: meta',
                'XGB + оба LGB WX: порог','XGB + оба LGB WX: лог-среднее',
                'XGB + оба LGB WX: среднее','XGB + LGB уровней WX: без LGB приростов',
                'XGB + LGB уровней WX: порог','XGB + LGB уровней WX: лог-среднее']
    labels_five=['Лидеры 04+05: среднее','Лидеры 04+05: лог-среднее','Лидеры + прежний 01: среднее',
                 'Лидеры + прежний 08: среднее','Лидеры + прежний 08: лог-среднее']
    appendix=''
    for j,(row,label) in enumerate(zip(records,labels_first+labels_three+labels_seven+labels_ten+labels_five),1):
        appendix+=line(j,label,row['submission_id'],num(row['report_score'],2))
    (HERE/'tables/submissions.tex').write_text(appendix,encoding='utf-8')
    for p in [COURSE/'confirmation_model_comparison.csv',COURSE/'development_model_comparison.csv',
              COURSE/'moran_diagnostics.csv', COURSE/'cache/weights_manifest.json',
              COURSE/'runner.py', FORECASTING/'run_validation.py',FORECASTING/'baseline/original_solution.ipynb']:
        evidence.append(dict(file=artifact_relative(p),sha256=sha(p)))
    dump(HERE/'evidence_manifest.json',dict(status='passed',report_date='2026-10-11',
         forecast_month='2025-03',new_forecast_fits=0,total_final_score_records=36,
         historical_public_score_reported_by_author=90.64,baseline_final_score=90.78,best_final_score=90.83,
         final_evaluation_scope='Entire March test: public plus private, explicitly confirmed by user',
         public_private_split_fraction_provided=False,source_files=evidence,
         source_documents=[dict(file='sources/competition_description.md',sha256=sha(description)),
                           dict(file='sources/competition_condition.txt',sha256=sha(condition))],
         original_econometric_files_unchanged=before))
    assert all(sha(resource_path(p))==h for p,h in before.items())
    print(json.dumps(dict(status='passed',records=len(records),tables_written=6,figures=3,
                          new_models_trained=0,original_econometric_files_unchanged=True)))


if __name__=='__main__':
    main()
