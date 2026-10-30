import pandas as pd

def check_data(data):

    print(f'Размер данных: {data.shape}')
    print(f'Типы столбцов {data.dtypes}')

    miss = data.isnull().sum().sort_values(ascending=False)
    miss_pct = (miss / len(data) * 100).round(2)
    print("Пропуски", pd.concat([miss, miss_pct], axis=1).rename(columns={0:'missing_count',1:'missing_pct'}).head(50))

    # quantity == 0 и value>0
    if 'quantity' in data.columns and 'value' in data.columns:
        bad_q = data[(data['quantity']==0) & (data['value']>0)]
        print(f'Количество записей с quantity==0 и value>0:, {len(bad_q)}')
        if len(bad_q)>0:
            print(bad_q.head(10).to_string(index=False))

def check_obs(data):

    macro_cols = [
        "gdp_partner", "cpi_partner", "ex_rate_partner",
        "bilateral_er", "iip_partner", "dist", "contig", "comlang_off"
    ]

    partner_stats = data.groupby("partner").agg(
        total_obs=("partner", "count")
    )

    missing_macro = (
        data.groupby("partner")[macro_cols]
            .apply(lambda df: df.isna().sum().sum())
    )

    total_macro = (
        data.groupby("partner")[macro_cols]
            .apply(lambda df: df.size)
    )

    partner_stats["missing_macro"] = missing_macro
    partner_stats["total_macro"] = total_macro
    partner_stats["missing_pct"] = missing_macro / total_macro

    problematic = partner_stats[
        (partner_stats["total_obs"] < 3) | (partner_stats["missing_pct"] > 0.5)
    ].sort_values(["total_obs", "missing_pct"])
    problematic = partner_stats[
        (partner_stats["total_obs"] < 3) | (partner_stats["missing_pct"] > 0.5)
    ].sort_values(
        by=["total_obs", "missing_pct"],
        ascending=[True, False]
    )

    return problematic, partner_stats
