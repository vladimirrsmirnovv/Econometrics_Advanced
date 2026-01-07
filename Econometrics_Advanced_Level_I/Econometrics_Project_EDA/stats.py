import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def describe_vars(df_sub, vars_list):
    """Возвращает describe() для списка переменных (transpose)."""
    return df_sub[vars_list].describe().T

def mean_se(df_sub, vars_list):
    """Возвращает mean и стандартную ошибку для списка переменных."""
    means = df_sub[vars_list].mean()
    ses = df_sub[vars_list].std() / np.sqrt(df_sub.shape[0])
    return pd.DataFrame({"mean": means, "se": ses})

def group_compare_ttest(df_sub, group_col, vars_list, group0=0, group1=1):
    """
    Сравнение двух групп (group_col принимает 0/1).
    Возвращает таблицу: mean0, mean1, diff, se0, se1, t_stat, p_value.
    """
    g0 = df_sub[df_sub[group_col] == group0]
    g1 = df_sub[df_sub[group_col] == group1]
    out = []
    for v in vars_list:
        a = g0[v].dropna()
        b = g1[v].dropna()
        mean0 = a.mean()
        mean1 = b.mean()
        se0 = a.std() / np.sqrt(max(1, a.shape[0]))
        se1 = b.std() / np.sqrt(max(1, b.shape[0]))
        # Welch t-test
        try:
            tstat, pval = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        except Exception:
            tstat, pval = (np.nan, np.nan)
        out.append({
            "var": v,
            "mean0": mean0,
            "mean1": mean1,
            "diff": mean0 - mean1,
            "se0": se0,
            "se1": se1,
            "t_stat": tstat,
            "p_value": pval,
            "n0": a.shape[0],
            "n1": b.shape[0]
        })
    return pd.DataFrame(out).set_index("var")
