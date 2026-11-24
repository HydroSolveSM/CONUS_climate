
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
import data_loader as dl
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault('MPLBACKEND', 'Agg')


def display(value):
    print(value)


data_raw_dir = Path('./data_raw')
out_dir = Path('./data_intermediate_review')
out_dir.mkdir(parents=True, exist_ok=True)
df_neon = dl.df_neon.copy()


def compute_water_year(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series)
    return dates.apply(lambda d: d.year + 1 if d.month >= 10 else d.year)


def clip_spei_spi(df: pd.DataFrame, lower_bound: float = -3.0, upper_bound: float = 3.0) -> pd.DataFrame:
    spei_spi_cols = [col for col in df.columns if 'spei' in col.lower() or 'spi' in col.lower()]
    for col in spei_spi_cols:
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def modified_mk_test(series, alpha: float = 0.05) -> dict:
    s = pd.to_numeric(pd.Series(series), errors='coerce').dropna()
    result = {'trend': None, 'p': np.nan, 'slope': np.nan, 'h': None, 'z': np.nan,
              'tau': np.nan, 's': np.nan, 'var_s': np.nan, 'intercept': np.nan}

    if len(s) < 3:
        return result

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = mk.hamed_rao_modification_test(s, alpha=alpha)

        if (
            np.isfinite(res.p) and 0 <= res.p <= 1
            and np.isfinite(res.z)
            and np.isfinite(res.var_s)
            and res.var_s >= 0
        ):
            result.update(
                trend=res.trend,
                p=res.p,
                slope=res.slope,
                h=res.h,
                z=res.z,
                tau=res.Tau,
                s=res.s,
                var_s=res.var_s,
                intercept=res.intercept
            )

    except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
        pass

    return result


def mk_for_dataframe(df: pd.DataFrame, value_cols, group_col: str = 'ID',
                     time_col: str | None = None, alpha: float = 0.05) -> pd.DataFrame:
    if time_col is not None:
        df = df.sort_values([group_col, time_col])

    records = []

    for gid, g in df.groupby(group_col):
        for col in value_cols:
            series = g[col].values
            res = modified_mk_test(series, alpha=alpha)
            records.append({group_col: gid, 'variable': col, **res, 'n': len(series)})

    return pd.DataFrame.from_records(records)


def mk_two_timeframes(df: pd.DataFrame, value_cols, group_col: str = 'ID',
                      time_col: str = 'W_YEAR', alpha: float = 0.05,
                      full_start: int = 1926, full_end: int = 2025,
                      recent_start: int = 1996, recent_end: int = 2025) -> pd.DataFrame:
    df_full = df[(df[time_col] >= full_start) & (df[time_col] <= full_end)].copy()
    res_full = mk_for_dataframe(df_full, value_cols, group_col, time_col, alpha)
    res_full['window'] = f'{full_start}-{full_end}'
    res_full['window_type'] = f'full_{full_end - full_start + 1}yr'

    df_recent = df[(df[time_col] >= recent_start) & (df[time_col] <= recent_end)].copy()
    res_recent = mk_for_dataframe(df_recent, value_cols, group_col, time_col, alpha)
    res_recent['window'] = f'{recent_start}-{recent_end}'
    res_recent['window_type'] = f'recent_{recent_end - recent_start + 1}yr'

    return pd.concat([res_full, res_recent], ignore_index=True)


def summarize_trends_by_ecoregion(mk_df: pd.DataFrame, area_df: pd.DataFrame,
                                   variable: str = 'PRCP', id_col: str = 'ID',
                                   domain_col: str = 'domainName', area_col: str = 'Area',
                                   slope_factor: float = 10.0) -> pd.DataFrame:
    mk_var = mk_df[mk_df['variable'] == variable].copy()
    merged = mk_var.merge(area_df[[id_col, domain_col, area_col]], on=id_col, how='left')
    records = []

    for grouping in ([domain_col, 'window_type', 'window'], ['window_type', 'window']):
        for key, g in merged.groupby(grouping):
            if len(grouping) == 3:
                dom, win_type, win = key
            else:
                win_type, win = key
                dom = 'CONUS'

            total_area = g[area_col].sum()

            for tdir in ['increasing', 'decreasing']:
                g_tr = g[(g['h'] == True) & (g['trend'] == tdir)]
                PA = 100.0 * g_tr[area_col].sum() / total_area if total_area > 0 else np.nan
                slopes = g_tr['slope'] * slope_factor

                records.append({
                    domain_col: dom,
                    'window': win,
                    'window_type': win_type,
                    'trend_dir': tdir,
                    'PA': PA,
                    'slope_min': slopes.min(),
                    'slope_max': slopes.max(),
                    'slope_mean': slopes.mean(),
                    'slope_median': slopes.median(),
                    'n_sig': len(g_tr)
                })

    return pd.DataFrame.from_records(records)


def make_grouped_pivot(summary_df: pd.DataFrame, domain_col: str = 'domainName') -> pd.DataFrame:
    pivot = summary_df.pivot_table(
        index=domain_col,
        columns=['window_type', 'trend_dir'],
        values=['PA', 'slope_min', 'slope_max', 'slope_mean', 'slope_median']
    )

    ordered_cols = [
        ('PA', 'recent_30yr', 'increasing'),
        ('PA', 'recent_30yr', 'decreasing'),
        ('slope_min', 'recent_30yr', 'increasing'),
        ('slope_max', 'recent_30yr', 'increasing'),
        ('slope_mean', 'recent_30yr', 'increasing'),
        ('slope_median', 'recent_30yr', 'increasing'),
        ('slope_min', 'recent_30yr', 'decreasing'),
        ('slope_max', 'recent_30yr', 'decreasing'),
        ('slope_mean', 'recent_30yr', 'decreasing'),
        ('slope_median', 'recent_30yr', 'decreasing'),
        ('PA', 'full_100yr', 'increasing'),
        ('PA', 'full_100yr', 'decreasing'),
        ('slope_min', 'full_100yr', 'increasing'),
        ('slope_max', 'full_100yr', 'increasing'),
        ('slope_mean', 'full_100yr', 'increasing'),
        ('slope_median', 'full_100yr', 'increasing'),
        ('slope_min', 'full_100yr', 'decreasing'),
        ('slope_max', 'full_100yr', 'decreasing'),
        ('slope_mean', 'full_100yr', 'decreasing'),
        ('slope_median', 'full_100yr', 'decreasing')
    ]

    return pivot.reindex(columns=[c for c in ordered_cols if c in pivot.columns])


def summarize_and_save_for_variable(mk_two: pd.DataFrame, df_neon_area: pd.DataFrame,
                                    variable: str, out_dir: Path, id_col: str = 'ID',
                                    domain_col: str = 'domainName', area_col: str = 'Area',
                                    slope_factor: float = 10.0) -> None:
    print(f'\n=== Summarizing trends for variable: {variable} ===')

    summary_var = summarize_trends_by_ecoregion(
        mk_two, df_neon_area, variable,
        id_col, domain_col, area_col, slope_factor
    )

    cols_to_round = [c for c in summary_var.columns if 'PA' in c or 'slope_' in c]
    summary_var[cols_to_round] = summary_var[cols_to_round].round(4)

    pivot_var = make_grouped_pivot(summary_var, domain_col)
    var_lower = variable.lower()
    summary_path = out_dir / f'summary_{var_lower}_by_domain_conus.csv'
    pivot_path = out_dir / f'pivot_{var_lower}_for_word_conus.csv'

    summary_var.to_csv(summary_path, index=False)
    pivot_var.to_csv(pivot_path)

    print(f'  Saved summary -> {summary_path}')
    print(f'  Saved pivot   -> {pivot_path}')


def main() -> None:
    print('pyMannKendall location:', mk.__file__)
    print('pyMannKendall version:', getattr(mk, '__version__', 'unknown'))

    variables = ['PRCP', 'TMAX', 'TMIN', 'PCI', 'WSDI', 'CSDI', 'SDII']

    results = mk_two_timeframes(
        dl.df_yearly.copy(),
        value_cols=variables,
        group_col='ID',
        time_col='W_YEAR',
        alpha=0.05
    )

    results.to_csv(out_dir / 'mk_two_non_AI_indices_review.csv', index=False)
    print('Saved results to:', out_dir.resolve())


if __name__ == '__main__':
    main()


REVIEW_DIR = Path('data_intermediate_review')
TREND_FILE = REVIEW_DIR / 'mk_two_non_AI_indices_review.csv'
FDR_ALPHA = 0.05

if not TREND_FILE.exists():
    raise FileNotFoundError(f'Saved review trend table not found: {TREND_FILE}')

fdr_data = pd.read_csv(TREND_FILE, dtype={'ID': str})
station_metadata = pd.read_csv('data_raw/ID_ecoregion.csv', dtype={'ID': str})
station_metadata['domainName'] = station_metadata['domainName'].astype('string').str.strip()

station_ids = set(
    station_metadata.loc[
        station_metadata.domainName.notna() & station_metadata.domainName.ne(''),
        'ID'
    ].dropna()
)

if len(station_ids) != 861:
    raise ValueError(f'Expected 861 stations with ecoregions, found {len(station_ids)}.')

available_ids = set(fdr_data.ID.dropna())

if station_ids - available_ids:
    raise ValueError(
        f'{len(station_ids - available_ids)} study stations are missing from the trend table.'
    )

fdr_data = fdr_data[
    fdr_data.ID.isin(station_ids) & fdr_data.variable.ne('AI')
].copy().reset_index(drop=True)

print(
    'Stations in FDR analysis:', fdr_data.ID.nunique(),
    'Indices:', sorted(fdr_data.variable.unique())
)

required = {'ID', 'variable', 'window', 'trend', 'p', 'slope'}
missing = required - set(fdr_data.columns)

if missing:
    raise ValueError(f'Missing required columns: {sorted(missing)}')

if fdr_data.duplicated(['ID', 'variable', 'window']).any():
    raise ValueError('Duplicate station, index and period records in the saved trend table.')

fdr_data['p'] = pd.to_numeric(fdr_data['p'], errors='coerce')
fdr_data['slope'] = pd.to_numeric(fdr_data['slope'], errors='coerce')

valid_p = np.isfinite(fdr_data['p']) & fdr_data['p'].between(0, 1)

fdr_data['p_fdr_bh'] = np.nan
fdr_data['fdr_significant'] = pd.Series(pd.NA, index=fdr_data.index, dtype='boolean')
fdr_data['fdr_trend'] = None
fdr_data['fdr_family_size'] = 0

for (variable, window), group in fdr_data.groupby(['variable', 'window'], sort=True):
    eligible = group.index[valid_p.loc[group.index]]
    fdr_data.loc[group.index, 'fdr_family_size'] = len(eligible)

    if len(eligible) == 0:
        continue

    significant, adjusted, _, _ = multipletests(
        fdr_data.loc[eligible, 'p'].to_numpy(dtype=float),
        alpha=FDR_ALPHA,
        method='fdr_bh'
    )

    fdr_data.loc[eligible, 'p_fdr_bh'] = adjusted
    fdr_data.loc[eligible, 'fdr_significant'] = significant
    fdr_data.loc[eligible, 'fdr_trend'] = 'no trend'

    sig_index = eligible[significant]
    slopes = fdr_data.loc[sig_index, 'slope']

    fdr_data.loc[slopes.index[slopes > 0], 'fdr_trend'] = 'increasing'
    fdr_data.loc[slopes.index[slopes < 0], 'fdr_trend'] = 'decreasing'
    fdr_data.loc[slopes.index[slopes.isna() | slopes.eq(0)], 'fdr_trend'] = None

fdr_path = REVIEW_DIR / 'all_indices_station_trends_BH_FDR_review.csv'
fdr_data.to_csv(fdr_path, index=False)

summary = fdr_data.groupby(['variable', 'window'], as_index=False).agg(
    total_stations=('ID', 'size'),
    valid_p_values=('p_fdr_bh', 'count'),
    original_increasing=('trend', lambda x: x.eq('increasing').sum()),
    original_decreasing=('trend', lambda x: x.eq('decreasing').sum()),
    fdr_increasing=('fdr_trend', lambda x: x.eq('increasing').sum()),
    fdr_decreasing=('fdr_trend', lambda x: x.eq('decreasing').sum()),
    fdr_no_trend=('fdr_trend', lambda x: x.eq('no trend').sum())
)

summary.to_csv(REVIEW_DIR / 'all_indices_BH_FDR_summary_review.csv', index=False)

fdr_comparison = summary[
    [
        'variable', 'window',
        'original_increasing', 'fdr_increasing',
        'original_decreasing', 'fdr_decreasing'
    ]
].copy()

fdr_comparison.to_csv(
    REVIEW_DIR / 'all_indices_BH_FDR_comparison_861_no_AI_review.csv',
    index=False
)

print('Saved station results:', fdr_path)
display(fdr_comparison)

comparison_path = REVIEW_DIR / 'all_indices_BH_FDR_comparison_861_no_AI_review.csv'

if not comparison_path.exists():
    raise FileNotFoundError(f'Run the preceding BH-FDR section first: {comparison_path}')

fdr_counts = pd.read_csv(comparison_path)

percent_columns = [
    'original_increasing',
    'fdr_increasing',
    'original_decreasing',
    'fdr_decreasing'
]

fdr_percentages = fdr_counts.copy()
fdr_percentages[percent_columns] = (
    fdr_percentages[percent_columns] / 861 * 100
).round(1)

fdr_percentages = fdr_percentages.rename(
    columns={column: column + '_percent' for column in percent_columns}
)

fdr_percentages.to_csv(
    REVIEW_DIR / 'all_indices_BH_FDR_percentages_861_no_AI_review.csv',
    index=False,
    float_format='%.1f'
)

print('Percent of all 861 study stations.')

display(
    fdr_percentages.style.format({
        column + '_percent': '{:.1f}%'
        for column in percent_columns
    })
)

trend_path = REVIEW_DIR / 'mk_two_non_AI_indices_review.csv'
metadata_path = Path('df_neon_filter.csv')

if not trend_path.exists() or not metadata_path.exists():
    raise FileNotFoundError(
        'The saved review trend table or 861-station metadata file is missing.'
    )

trends = pd.read_csv(trend_path, dtype={'ID': str})
stations = pd.read_csv(
    metadata_path,
    usecols=['ID', 'Lat', 'Lon'],
    dtype={'ID': str}
)

if stations.ID.duplicated().any() or len(stations) != 861:
    raise ValueError('Expected one metadata row for each of 861 stations.')

if trends.duplicated(['ID', 'variable', 'window']).any():
    raise ValueError('Duplicate station, variable and period in the trend table.')

trends = trends[trends.variable.ne('AI')].copy()
trends['slope'] = pd.to_numeric(trends['slope'], errors='coerce')
trends['p'] = pd.to_numeric(trends['p'], errors='coerce')

arcgis = trends.merge(
    stations,
    on='ID',
    how='inner',
    validate='many_to_one'
)

arcgis['Lat'] = pd.to_numeric(arcgis.Lat, errors='coerce')
arcgis['Lon'] = pd.to_numeric(arcgis.Lon, errors='coerce')

arcgis = arcgis[
    np.isfinite(arcgis.slope)
    & np.isfinite(arcgis.Lat)
    & np.isfinite(arcgis.Lon)
].copy()

arcgis = arcgis.rename(
    columns={'window': 'period', 'p': 'p_value'}
)[
    ['ID', 'Lat', 'Lon', 'variable', 'period', 'slope', 'p_value', 'trend']
].sort_values(['variable', 'period', 'ID']).reset_index(drop=True)

arcgis_dir = REVIEW_DIR / 'ArcGIS_inputs_review'
arcgis_dir.mkdir(parents=True, exist_ok=True)

arcgis_path = arcgis_dir / 'all_indices_station_trends_861_no_AI_review.csv'
arcgis.to_csv(arcgis_path, index=False)

print('Saved:', arcgis_path.resolve())
print('Slope units: index units per year.')
print('Records:', len(arcgis), 'Stations:', arcgis.ID.nunique())

display(
    arcgis.groupby(['variable', 'period'])
    .size()
    .rename('stations')
    .reset_index()
)
