"""Publish the completed experiment, including negative results and the audit gate."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from notebook_api import MODELS
from selection import load

def table(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.6f}'
        return str(v).replace('|','/').replace('\n',' ')
    header='| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'
    return header+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    frozen=json.loads((HERE/'frozen_final_selection.json').read_text());decision=json.loads((HERE/'final_decision.json').read_text())
    manifest=json.loads((HERE.parent/'submissions/2025-03-stage2/manifest.json').read_text())
    march=[]
    for model in MODELS:
        folder=HERE.parent/'submissions/2025-03-stage2/individual'
        info=json.loads((folder/('test_'+model+'.json')).read_text())
        inner=pd.read_csv(folder/('february_inner_'+model+'.csv'))
        march.append(dict(model=model,configuration=info['configuration'],march_trees=info['trees'],
            inner_february_mape=float((abs(inner.prediction-inner.pto)/inner.pto).mean()*100),
            native_reload_max_relative_difference=info['native_reload_max_relative_difference']))
    pd.DataFrame(march).to_csv(HERE/'march_training_summary.csv',index=False)
    development=pd.read_csv(HERE/'development_model_comparison.csv');reserved=pd.read_csv(HERE/'reserved_model_comparison.csv')
    dev=[];audit=[];protocol=[];ablation=[];all_rows=[]
    for model in MODELS:
        part=development[development.model==model].set_index('configuration');chosen=frozen['individual'][model]
        dev.append(dict(model=model,baseline=part.loc['baseline','mean_mape'],previous=part.loc['previous','mean_mape'],selected=part.loc[chosen,'mean_mape'],configuration=chosen,
            delta=part.loc[chosen,'mean_delta'],months_won=part.loc[chosen,'months_won']))
        means=reserved[reserved.model==model].groupby('variant').mape.mean()
        audit.append(dict(model=model,original_full=means['original_full'],baseline_rolling=means['original_rolling'],previous_rolling=means['previous_rolling'],selected_rolling=means['frozen_new'],
            selected_minus_rolling=means['frozen_new']-means['original_rolling'],selected_minus_full=means['frozen_new']-means['original_full']))
        protocol.append(dict(model=model,original_full_aug=load('2024-08',model,'baseline','full')[1]['mape'],
            previous_full_aug=load('2024-08',model,'previous','full')[1]['mape'],
            baseline_rolling_aug=load('2024-08',model,'baseline')[1]['mape'],selected_rolling_aug=load('2024-08',model,chosen)[1]['mape']))
        for mechanism,control in [('seasonal_spatial','seasonal_control'),('innovations_spatial','innovations_control'),('dynamic_history','geo8_control'),
            ('hybrid8_history+seasonal_spatial','seasonal_spatial'),('hybrid8_history+seasonal_spatial','hybrid8_history'),
            ('hybrid8_history+seasonal_spatial','seasonal_control'),('knn8_history+eigenvectors','knn8_history'),('knn8_history+eigenvectors','eigenvectors')]:
            records=[]
            for fold in frozen['development_months']:
                try:records.append(load(fold,model,mechanism)[1]['mape']-load(fold,model,control)[1]['mape'])
                except FileNotFoundError:pass
            if records:ablation.append(dict(model=model,spatial=mechanism,control=control,months=len(records),mean_delta_to_control=float(np.mean(records)),wins=sum(d<0 for d in records),worst_delta=float(max(records))))
    for p in (HERE/'results').rglob('*.json'):
        row=json.loads(p.read_text());all_rows.append({k:row.get(k) for k in ['model','fold','configuration','underlying_configuration','protocol','mape','mae','trees','inner_selected_trees','seconds','model_file','model_sha256','prediction_sha256']})
    all_metrics=pd.DataFrame(all_rows);all_metrics.to_csv(HERE/'all_stage2_metrics.csv',index=False)
    pd.DataFrame(ablation).to_csv(HERE/'matched_ablation_summary.csv',index=False)
    transfers=development[['model','configuration','mean_mape','mean_delta','months_won','worst_delta','eligible']]
    screen=pd.read_csv(HERE/'screen_new_methods.csv').pivot(index='configuration',columns='model',values='mape')
    for model in MODELS:screen[model]-=load('2024-08',model,'baseline')[1]['mape']
    prior=pd.read_csv(HERE/'prior_month_effects.csv');cat=prior[prior.model==MODELS[0]];cold_share=float(-cat.loc[cat.fold=='2024-03','delta_mape'].sum()/-cat.delta_mape.sum()*100)
    moran=pd.read_csv(HERE/'prior_residual_spatial_structure.csv').groupby('variant')[['market_moran_raw','market_moran_region_removed']].mean().reset_index()
    persistence=pd.read_csv(HERE/'prior_residual_persistence.csv')
    support=pd.read_csv(HERE/'seasonal_feature_support.csv')
    national=support[support.feature=='seasonal_control__national_same_month'][['forecast_month','train_min','train_max','validation_median','outside_training_minmax_share']]
    seasonal_bias=pd.read_csv(HERE/'seasonal_error_direction.csv')
    dev_ensemble=pd.read_csv(HERE/'development_ensemble_comparison.csv').groupby(['composition','rule']).mape.mean().reset_index()
    audit_ensemble=pd.read_csv(HERE/'reserved_ensemble_comparison.csv').groupby(['composition','rule']).mape.mean().reset_index()
    full_ensemble=pd.read_csv(HERE/'full_recipe_ensemble_comparison.csv')
    full_comparison=full_ensemble[full_ensemble.rule=='final_meta_logmean'].pivot(index='fold',columns='composition',values='mape').reset_index()
    rule=frozen['ensemble']['rule'];composition=frozen['ensemble']['composition']
    graph=pd.DataFrame(json.loads((HERE/'cache/feature_manifest.json').read_text())['graph_diagnostics'])
    active_overlap=graph.loc[graph.time>=5,'mean_neighbour_overlap']
    if decision['status']=='passed':
        conclusion='Замороженный новый ансамбль прошёл заранее заданную проверку. Рекомендован `test.csv`, содержащий этот ансамбль.'
    else:
        conclusion='Замороженный новый ансамбль не прошёл заранее заданную проверку. Рекомендован `test.csv`, содержащий исходный ансамбль полного обучения; новые варианты сохранены как экспериментальные альтернативы.'
    text=f'''# Финальный этап прогнозирования X5 — 10 октября 2026

{conclusion} Точный источник файла: `{manifest['source']}` в `Forecasting/submissions/2025-03-stage2`. Прогноз относится к марту **2025**, 18 657 известных магазинов. Посылки в соревнование автоматически не отправлялись.

Средняя контрольная MAPE нового ансамбля — **{np.mean([r['selected_ape'] for r in decision['monthly']]):.6f}%**, исходного полного ансамбля — **{np.mean([r['original_full_ape'] for r in decision['monthly']]):.6f}%**. На ноябре исходный полный ансамбль лучше; основной выигрыш нового приходится на декабрь. Это допуск по заранее установленному протоколу, а не доказательство превосходства на марте. Новая мартовская XGBoost выбрала на феврале одно дерево; зимнее ограничение и остановочные оценки всех компонентов приведены ниже.

## Что изменено и сохранено

Исходный ноутбук, эконометрическое исследование, предыдущие модели и архив мартовских CSV сохранены. Дополнения находятся в `Forecasting/research-stage2`; новые submission — в отдельной папке. Исходные гиперпараметры четырёх алгоритмов сохранены. Отдельное вмешательство — число деревьев выбирается на предыдущем месяце, затем модель заново обучается на всей доступной истории. Его эффект отделён от пространственных признаков.

Вычисления выполнены: {len(all_metrics)} сохранённых внешних прогнозов второго этапа; это число оценок конфигураций, а не число различных эконометрических методов. Внутренние остановочные модели, исходные локальные расчёты и технические повторы сохранены отдельно. Все итоговые метрики — обычная невзвешенная по магазинам MAPE, в процентах.

## Почему прежний прирост нельзя было считать переносимым

1. Февральская валидация использовала 672/685/369/441 дерево, мартовское полное обучение — 6500/7000/8000/3750. Это различие протоколов, а не ошибка выполнения. Проверка полного рецепта показывает, что эффект зависит от алгоритма:

{table(pd.DataFrame(protocol))}

2. На март 2024 пришлось {cold_share:.1f}% суммарного старого прироста CatBoost на четырёх старых фолдах. Шесть специальных мартовских якорей в том фолде были полностью NaN; к марту 2025 они доступны. История и набор доступных сигналов существенно различаются. Это объясняет ограниченность сравнения, но не доказывает причину мартовского ухудшения.
3. Часть пространственной структуры старых ошибок связана с региональными средними. После их удаления индекс Морана заметно снижается:

{table(moran)}

Между январём и февралём 2025 регионально очищенные рыночные ошибки имеют отрицательную корреляцию:

{table(persistence[['model','variant','correlation_region_removed']])}

Это описательные результаты одной последовательной пары; они не доказывают причинность и не оправдывают автоматическое положительное перенесение вчерашнего остатка. Ноябрь/декабрь в эту диагностику до замораживания не входили.
4. Прежний выбор включал много проверок и неодинаковые по условиям месяцы. Сам по себе низкий февральский MAPE не гарантирует мартовский прирост. Реальные помагазинные ответы марта отсутствуют, поэтому причины его результата остаются гипотезами.

## Что дополнительно проверено

Первый этап уже содержал kNN8, разные радиусы/расстояния, гибридные соседства, WX/SLX, пространственные собственные векторы, LISA, локальный ridge, остаточные коррекции и SEM/SDEM. Неустойчивый SDM был отвергнут. Эти подходы не объявляются вновь открытыми.

Новые проверки:

- **Календарная сезонность:** собственное, региональное и общее среднее изменение дневного РТО в том же календарном месяце прошлого года, затем отдельное добавление прошлогодней динамики восьми соседних рынков. Годовой переход — t−12; первые 13 месяцев остаются NaN.
- **Локальные инновации:** вычитание регионального среднего роста, стабилизация малых рынков множителем n/(n+3); лаги, среднее трёх месяцев, разброс средних инноваций соседей и разрыв со своим изменением. Сначала контроль без соседей, затем пространственная добавка.
- **Адаптивная W:** восемь соседей среди 32 ближайших, фиксированный score exp(−distance/50)×(1+max(past correlation,0)); только шесть прошлых месяцев. Отдельный geo8-control сохраняет кандидатов, расстояния, направленность и нормировку, убирая корреляцию.
- **Переносы:** exp50, hybrid8, kNN8 и eigenvectors между всеми четырьмя алгоритмами. Комбинации заранее ограничены hybrid8+seasonal-spatial и kNN8+eigenvectors.
- **Ансамбли:** пять составов, шесть фиксированных правил; без подбора новых весов и порогов. Дополнительно обязательный вариант трёх исходных полных моделей + прежнего полного hybrid8 XGBoost.

Положительный результат непространственного сезонного/регионального контроля не приписывается матрице W. Новые W применяются для предсказания и не являются оценкой причинного SAR-коэффициента. Неизвестный текущий Wy марта не используется.

Адаптивное соседство после появления четырёх прошлых переходов сохраняет в среднем {active_overlap.mean():.1%} восьми соседей между месяцами; минимум {active_overlap.min():.1%}, к марту 2025 — {graph.iloc[-1].mean_neighbour_overlap:.1%}. Все графы имеют нулевую диагональ и 26 880 направленных связей. Это наблюдаемая устойчивость конструкции, а не доказательство стабильности экономических эффектов. Адаптивная W не стала победителем отбора; автоматическое усложнение графа не оправдалось.

## Выбор и временная проверка

Разработка: июнь, июль и август 2024. Август — экран 13 конфигураций; CPU-модели подтверждают два лучших новых механизма, согласованные контроли и переносы на июне/июле. XGBoost дополнительно имеет все 13 конфигураций на всех трёх месяцах; в выборе используются только заранее ограниченные кандидаты. Выбор: средний прирост минимум 0.01 п.п., минимум два выигранных месяца, худшее ухудшение ≤0.10 п.п.; критерий mean(delta)+0.25·std(delta). Исходная версия всегда разрешена.

Конфигурации и состав/правило ансамбля заморожены в `frozen-final-selection.json` **до** оценки ноября/декабря. Ноябрь раньше не оценивался как внешний фолд; декабрь участвовал в старых вспомогательных OOF, поэтому полностью новым его считать нельзя. Проверка ретроспективная, на той же панели. Только предыдущий месяц выбирает число деревьев; текущий внешний месяц не подаётся в eval-set. Для декабря ноябрь законно становится уже известной историей.

### Средняя MAPE разработки: одинаковый rolling-протокол

{table(pd.DataFrame(dev))}

### Средняя MAPE контрольных ноября/декабря

{table(pd.DataFrame(audit))}

`original-full` — исходные фиксированные 6500/7000/8000/3750 деревьев. Остальные столбцы — число деревьев по предыдущему месяцу. Пространственный вклад оценивается по `selected-minus-rolling`; выигрыш перед `original-full` включает изменение обучения.

Внутри rolling число деревьев выбирается отдельно для каждой конфигурации. Разница характеризует предиктивный рецепт с данным набором признаков и одинаковым правилом остановки, а не причинный эффект W или сравнение при одном фиксированном числе деревьев.

Для сопоставимости сохранена также особенность исходного LightGBM log-diff: строки первого месяца с недоступным целевым лог-приростом остаются в обучающем входе. XGBoost, как в исходном коде, исключает такие цели. Этот этап не меняет фильтрацию целевых наблюдений и не проверяет её альтернативы; качество сравнивается внутри одного исходного рецепта.

Помесячный график и его исходные значения: `figures/monthly-deltas.png`, `figures/monthly-deltas.svg`, `figures/monthly-deltas.csv`. Столбцы разработки и контрольных месяцев разделены; отрицательная delta означает улучшение относительно rolling-baseline.

### Вклад именно пространственного добавления

{table(pd.DataFrame(ablation))}

Отрицательная delta означает пользу пространственной добавки против согласованного контроля. `months` показывает число реально сопоставленных месяцев; экран одного августа не равен трёхмесячному подтверждению. Худший месяц и отрицательные результаты сохраняются, даже если средняя ошибка уменьшилась.

Строки сочетаний показывают прибавку относительно каждого отдельного блока. Сравнение hybrid8+seasonal-spatial с seasonal-control добавляет сразу гибридную историю и пространственную сезонность; оно не изолирует эти два механизма друг от друга. Для этого рядом показаны сравнения с hybrid8-history и seasonal-spatial.

### Все подтверждённые переносы и кандидаты

{table(transfers)}

### Полный экран новых механизмов на августе: delta MAPE к baseline

{table(screen.reset_index())}

Это первоначальный экран, а не независимое подтверждение. Например, комбинация hybrid8+seasonal-spatial у XGBoost выглядит сильной в августе, но нарушает ограничение на худшее ухудшение в другом месяце разработки и не выбирается. Прежние eigenvectors у LightGBM темпов и density-weight у LightGBM уровней также не проходят новый критерий устойчивости. Отрицательная средняя delta без допуска по худшему месяцу не достаточна.

Полный экран, включая не прошедшие отбор методы и полные refit, доступен в `all-stage2-metrics.csv`. Таблица не скрывает ухудшения. Региональные и рыночные ошибки контрольных месяцев сохранены отдельно.

### Почему календарный перенос требует осторожности

Для декабря 2024 сезонные признаки впервые содержат прошлогоднее декабрьское изменение: у декабря 2023 соответствующая годовая ссылка ещё отсутствовала. Общий дневной лог-темп 0.131607 превышает максимум 0.074424, наблюдавшийся у этого признака в обучении. Диапазоны общего сезонного признака:

{table(national)}

Это подтверждённый выход признака за обучающий диапазон, а не доказанная причина ошибки. На летних месяцах общий признак тоже выходит за нижнюю границу, но качество улучшается. Совместные признаки, число деревьев и уровень целевой переменной также меняются между месяцами; отдельного контрфактического эксперимента, изолирующего декабрьскую причину, здесь нет.

В контрольных ноябре/декабре оценивается замороженный полный пакет seasonal-spatial; отдельный seasonal-control там не переобучался. Поэтому зимнее ухудшение пакета не приписывается автоматически именно соседям: отдельный вклад пространственного блока проверен на месяцах разработки, а здесь проверяется перенос всего рецепта.

Направление ошибок LightGBM уровней при одинаковом rolling-протоколе:

{table(seasonal_bias)}

Положительный signed error означает завышение, отрицательный — занижение. Эта диагностика выполнена после фиксации выбора и не использовалась для настройки новых признаков, весов или числа деревьев. Полная таблица шести сезонных признаков находится в `seasonal-feature-support.csv`.

## Ансамбли и окончательное решение

Заморожен состав `{composition}`, правило `{rule}`. Средняя MAPE разработки:

Правило `gated-1-08-no-weak` использует три компонента: CatBoost, LightGBM уровней и XGBoost. При отношении максимального к минимальному прогнозу >1.08 берётся их медиана, иначе среднее. LightGBM приростов в этом правиле не участвует; её индивидуальный CSV и сравнения сохраняются. Исходный `final-meta-logmean` объединяет четыре фиксированных кандидатных ансамбля, включая варианты на всех четырёх моделях.

{table(dev_ensemble)}

Средняя MAPE контрольных месяцев:

{table(audit_ensemble)}

`mandatory-mixed` использует одинаковый rolling-протокол. `mandatory-mixed-full` проверяет именно готовый мартовский вариант: три исходных полных модели + прежний полный hybrid8 XGBoost. Их нельзя смешивать в одном сравнении без указания режима обучения.

### Ансамбли с исходными фиксированными числами деревьев

{table(full_comparison)}

Здесь везде одна формула `final-meta-logmean`. Прежние полные четыре модифицированные модели повторены на августе и ноябре; для декабря повторён полный hybrid8 XGBoost, необходимый смешанному ансамблю. Поэтому полный прежний ансамбль декабря не рассчитан, пропуск не заменяется rolling-оценкой. Все шесть фиксированных правил полных ансамблей сохранены в `full-recipe-ensemble-comparison.csv`.

Заранее заданная контрольная граница: улучшение rolling-baseline в обоих месяцах, в среднем ≥0.01 п.п., и отсутствие среднего проигрыша исходному полному ансамблю. Результат: **{decision['status']}**; средняя delta к rolling = {decision['mean_delta_rolling']:.6f} п.п., к исходному полному = {decision['mean_delta_original_full']:.6f} п.п. Другой кандидат по контрольным результатам не выбирался.

Парный региональный bootstrap, 2000 повторов, описывает неопределённость фиксированного выбора: `{json.dumps(decision['regional_paired_bootstrap_95_percent'])}`. Он не учитывает весь поиск конфигураций, малое число временных месяцев или будущие мартовские шоки и не доказывает перенос на март.

## Что означают мартовские баллы

Условие задаёт score = 100·((100−min(MAPE,100))/100)². Для положительных баллов MAPE = 100·(1−sqrt(score/100)). Поэтому 90.79 ≈4.7162%, 90.78 ≈4.7215%; разница примерно 0.0052 п.п. MAPE. Баллы округлены, точный прирост из них не восстанавливается. Пример 80%→20 баллов в присланном условии противоречит формуле: по ней получается 4.

Во время соревнования оценивается подвыборка, финально — все магазины; текущий состав отображаемого leaderboard неизвестен. Баллы используются только для диагностики старых вариантов и не входят в новый отбор. Специальные мартовские признаки исходного CatBoost и различие сезонов дополнительно ограничивают перенос летних/ноябрьских результатов на март.

## Воспроизводимость и файлы

CatBoost и LightGBM сохраняют CPU-бэкенд; XGBoost — CUDA 3.1.3. Перенос на Kaggle проверен на одинаковых исходных fit: расхождения только в последнем разряде double, MAPE совпадает в пределах машинной точности. В версиях библиотек, параметрах, исходных данных и остановке нет скрытой смены модели.

Разрешение равных географических расстояний различалось между Windows и Linux. Итог использует общий сохранённый артефакт признаков. Сохранённая структура соседей точно воспроизводит его и на Windows; старые локальные расчёты не удалены и исключены из выбора. Подстановка фиктивной выручки марта 1e12 не меняет признаки. Старые W kNN8/hybrid8 сохранены явно, а продление старых признаков проверено на феврале.

Первый финальный CUDA-запуск остановился в служебной проверке мартовской цели: исходный numeric-generator возвращает y-val=None для неизвестного месяца, а fixed-tree fit ошибочно вызывал для него isfinite. Исправлено только условие этой проверки: известная внутренняя цель проверяется при early stopping; при фиксированном мартовском обучении неизвестная цель не оценивается. Обучающие цели, матрицы, параметры и замороженный выбор не менялись. Неудачный запуск и его прогнозы сохранены в `sources/reserved-cuda-partial`; повтор выполнен с тем же frozen SHA. Подтверждение — `march-target-mask-checks.json`, `march-failure-recovery.json`.

Мартовские модели выбирают число деревьев на известном феврале, обучаются на январе 2023 — феврале 2025. Исходный мартовский split повторён и проверен; текущая мартовская цель везде NaN. Native-модели перезагружаются и воспроизводят CSV. Проверки схемы: только new-id,rto; 18 657 уникальных исходных ID; конечные положительные прогнозы; ≤1 000 000 байт каждый CSV. Предыдущие файлы сохранены.

### Обучение новых мартовских компонентов

{table(pd.DataFrame(march))}

Февральская внутренняя ошибка здесь используется для остановки, поэтому не является независимой контрольной оценкой. Новая XGBoost seasonal-spatial выбрала всего одно дерево и имеет внутреннюю MAPE 6.0614%; это слабый результат переноса на конец зимы, несмотря на летний выигрыш. Число деревьев вручную не заменялось, новые веса по этой диагностике не подбирались. Смещения мартовских прогнозов относительно исходных сохранены в `march-forecast-changes.csv`; без истинных ответов они не характеризуют качество марта.

Повтор CUDA после исправления неизвестной цели дал **побайтовое совпадение всех 17 ранее готовых прогнозов**, максимальное относительное расхождение 0; `cuda-repeat-checks.json`.

При сборке найден пустой прежний отдельный `2025-03/spatial/test-xgboost-log-diff.csv`. Исходный ZIP сохранил его полные байты с SHA-256 из старого манифеста. Контрольная копия в `2025-03-stage2/reference-previous` и обязательный смешанный ансамбль используют именно эти проверенные байты. Старая папка не изменялась; остальные 18 старых CSV совпадают с манифестом. Причина обнуления не установлена. Проверка и источник восстановления сохранены в `previous-submission-integrity.json`.

Основные артефакты: `final-decision.json`, `all-stage2-metrics.csv`, `matched-ablation-summary.csv`, `reserved-region-comparison.csv`, `reserved-market-comparison.csv`, `freeze-provenance.json`, `reserved-import-checks.json`, `artifact-checks.json`. В новой папке submission находятся `test.csv`, четыре индивидуальных CSV, замороженный новый ансамбль, обязательные смешанные ансамбли и исходные контрольные варианты; `march-submissions-stage2.zip` содержит готовые файлы.

Методологический контекст: [Roberts et al. — проверка временно/пространственно зависимых данных](https://epub.uni-regensburg.de/39299/), [Elhorst — пространственные панели и общие факторы](https://www.rug.nl/staff/j.p.elhorst/spatial-panel-models-common-factors-improved-feb2022.pdf), [официальная документация XGBoost 3.1](https://xgboost.readthedocs.io/en/release-3.1.0/python/python-api.html), [PySAL: неоднозначность равных расстояний kNN](https://pysal.org/libpysal/stable/generated/libpysal.weights.KNN.html). Собственные численные выводы основаны на сохранённых прогнозах, а не на этих источниках.
'''
    review=HERE/'POST_SUBMISSION_REVIEW.md'
    if review.exists():
        text='> **Актуальное уточнение:** этот отчёт описывает исторический эксперимент. Новые индивидуальные версии ухудшились на феврале; результаты мартовских посылок и фактический одно-деревный XGBoost разобраны в `POST_SUBMISSION_REVIEW.md`. Прежняя рекомендация не считается установленным мартовским улучшением. Дальнейшая работа ограничена материалами курса: `../course-only/README.md`. Уже выданные CSV не менялись.\n\n'+text
    (HERE/'REPORT.md').write_text(text,encoding='utf-8')
    summary=dict(recommendation=manifest['source'],decision=decision['status'],forecast_month='2025-03',
        development_models=dev,reserved_models=audit,reserved_ensemble_decision=decision,march_training=march,
        frozen_individual=frozen['individual'],frozen_ensemble=frozen['ensemble'],
        external_prediction_sets=len(all_metrics),competition_submissions_sent=False,
        report='REPORT.md',submissions='../submissions/2025-03-stage2',
        original_research_and_notebook_preserved=True,
        original_submission_directory_modified_by_stage2=False,
        previous_submission_reference_recovery=manifest['previous_reference_recovery'],
        current_status='historical_experiment_superseded_by_course_only_scope' if review.exists() else 'completed',
        post_submission_review='POST_SUBMISSION_REVIEW.md' if review.exists() else None)
    (HERE/'final_result_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=lambda value:value.item()),encoding='utf-8')
    print('FINAL_REPORT_WRITTEN',len(all_metrics),flush=True)

if __name__=='__main__':main()
