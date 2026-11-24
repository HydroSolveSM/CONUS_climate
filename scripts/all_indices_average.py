from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

def display(value):
    print(value)


import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import data_loader as dl

def compute_water_year(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series)
    return dates.apply(lambda d: d.year + 1 if d.month >= 10 else d.year)

def clip_spei_spi(df: pd.DataFrame, lower_bound: float=-3.0, upper_bound: float=3.0) -> pd.DataFrame:
    spei_spi_cols = [c for c in df.columns if 'spei' in c.lower() or 'spi' in c.lower()]
    for col in spei_spi_cols:
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def _window_mean_per_station(df: pd.DataFrame, value_col: str, time_col: str='W_YEAR', id_col: str='ID', start_year: int | None=None, end_year: int | None=None, suffix: str='full') -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if start_year is not None:
        mask &= df[time_col] >= start_year
    if end_year is not None:
        mask &= df[time_col] <= end_year
    df_win = df.loc[mask]
    mean_col_name = f'{value_col}_{suffix}_mean'
    out = df_win.groupby(id_col)[value_col].mean().reset_index().rename(columns={value_col: mean_col_name})
    return out

def compute_period_means_per_station(df: pd.DataFrame, value_col: str='PRCP', time_col: str='W_YEAR', id_col: str='ID', full_start: int=1926, full_end: int=2025, recent_start: int=1996, recent_end: int=2025, penult_start: int | None=None, penult_end: int | None=None) -> pd.DataFrame:
    if penult_start is None or penult_end is None:
        penult_end = recent_start - 1
        penult_start = penult_end - (recent_end - recent_start)
    df = df.copy()
    full_df = _window_mean_per_station(df, value_col, time_col, id_col, start_year=int(full_start), end_year=int(full_end), suffix='full')
    recent_df = _window_mean_per_station(df, value_col, time_col, id_col, start_year=int(recent_start), end_year=int(recent_end), suffix='recent')
    penult_df = _window_mean_per_station(df, value_col, time_col, id_col, start_year=int(penult_start), end_year=int(penult_end), suffix='penult')
    out = full_df.merge(recent_df, on=id_col, how='outer')
    out = out.merge(penult_df, on=id_col, how='outer')
    return out

def add_change_columns(df_means: pd.DataFrame, value_col: str='PRCP', id_col: str='ID') -> pd.DataFrame:
    df = df_means.copy()
    full_col = f'{value_col}_full_mean'
    recent_col = f'{value_col}_recent_mean'
    penult_col = f'{value_col}_penult_mean'
    df[f'{value_col}_recent_minus_full'] = df[recent_col] - df[full_col]
    df[f'{value_col}_recent_minus_penult'] = df[recent_col] - df[penult_col]
    return df

def attach_ecoregions(station_df: pd.DataFrame, domain_df: pd.DataFrame, id_col: str='ID', domain_col: str='domainName') -> pd.DataFrame:
    return station_df.merge(domain_df[[id_col, domain_col]], on=id_col, how='left')

def summarize_metrics_by_region_and_conus(df_with_domain: pd.DataFrame, metrics: List[str], domain_col: str='domainName') -> pd.DataFrame:
    df = df_with_domain.copy()
    agg_dict = {m: ['min', 'max', 'mean', 'median'] for m in metrics}
    region_summary = df.groupby(domain_col).agg(agg_dict)
    region_summary.columns = [f'{metric}_{stat}' for metric, stat in region_summary.columns]
    region_summary = region_summary.reset_index()
    conus_row = df[metrics].agg(['min', 'max', 'mean', 'median']).T
    conus_data: dict[str, float] = {}
    for metric in metrics:
        conus_data[f'{metric}_min'] = conus_row.loc[metric, 'min']
        conus_data[f'{metric}_max'] = conus_row.loc[metric, 'max']
        conus_data[f'{metric}_mean'] = conus_row.loc[metric, 'mean']
        conus_data[f'{metric}_median'] = conus_row.loc[metric, 'median']
    conus_df = pd.DataFrame([{domain_col: 'CONUS', **conus_data}])
    summary_all = pd.concat([region_summary, conus_df], ignore_index=True)
    return summary_all

def climate_change_pipeline(df_annual: pd.DataFrame, df_domains: pd.DataFrame, value_col: str, time_col: str='W_YEAR', id_col: str='ID', full_start: int=1926, full_end: int=2025, recent_start: int=1996, recent_end: int=2025, penult_start: int | None=None, penult_end: int | None=None, domain_col: str='domainName') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means_df = compute_period_means_per_station(df=df_annual, value_col=value_col, time_col=time_col, id_col=id_col, full_start=full_start, full_end=full_end, recent_start=recent_start, recent_end=recent_end, penult_start=penult_start, penult_end=penult_end)
    station_stats = add_change_columns(means_df, value_col=value_col, id_col=id_col)
    station_stats = attach_ecoregions(station_stats, df_domains, id_col=id_col, domain_col=domain_col)
    diff_cols = [f'{value_col}_recent_minus_full', f'{value_col}_recent_minus_penult']
    diff_summary = summarize_metrics_by_region_and_conus(station_stats, metrics=diff_cols, domain_col=domain_col)
    mean_cols = [f'{value_col}_full_mean', f'{value_col}_recent_mean']
    mean_summary = summarize_metrics_by_region_and_conus(station_stats, metrics=mean_cols, domain_col=domain_col)
    return (station_stats, diff_summary, mean_summary)

def run_and_save_for_variable(value_col: str, df_annual: pd.DataFrame, df_domains: pd.DataFrame, df_neon_area: pd.DataFrame, out_dir: Path, full_start: int=1926, full_end: int=2025, recent_start: int=1996, recent_end: int=2025) -> None:
    print(f'\n=== Processing variable: {value_col} ===')
    station_stats, diff_summary, mean_summary = climate_change_pipeline(df_annual=df_annual, df_domains=df_domains, value_col=value_col, full_start=full_start, full_end=full_end, recent_start=recent_start, recent_end=recent_end)
    mean_summary_rounded = mean_summary.round(0)
    diff_summary_rounded = diff_summary.round(0)
    var_lower = value_col.lower()
    mean_summary_path = out_dir / f'{var_lower}_mean_summary_review.csv'
    diff_summary_path = out_dir / f'{var_lower}_diff_summary_review.csv'
    station_stats_path = out_dir / f'{var_lower}_station_stats_review.csv'
    mean_summary_rounded.to_csv(mean_summary_path, index=False)
    diff_summary_rounded.to_csv(diff_summary_path, index=False)
    station_stats_with_area = station_stats.merge(df_neon_area, on=['ID', 'domainName'], how='left')
    station_stats_with_area.to_csv(station_stats_path, index=False)
    print(f'  Saved mean   summary -> {mean_summary_path}')
    print(f'  Saved diff   summary -> {diff_summary_path}')
    print(f'  Saved station stats  -> {station_stats_path}')

def main() -> None:
    data_raw_dir = Path('./data_raw')
    data_intermediate_dir = Path('./data_intermediate_review/averages_review')
    data_intermediate_dir.mkdir(parents=True, exist_ok=True)
    df_daily = dl.df_daily.copy()
    df_monthly = dl.df_monthly_updated.copy()
    print('Daily data shape:', df_daily.shape)
    print('Monthly data shape:', df_monthly.shape)
    df_monthly = df_monthly[(df_monthly['DATE'] >= '1925-10-01') & (df_monthly['DATE'] <= '2025-09-30')].copy()
    df_neon = pd.read_csv(data_raw_dir / 'Thiessen_Polygons_100yrs.csv')
    df_neon = df_neon[['ID_1', 'Lat', 'Lon', 'Area_sqkm']]
    df_neon.columns = ['ID', 'Lat', 'Lon', 'Area']
    df_id_neon = pd.read_csv(data_raw_dir / 'ID_ecoregion.csv')
    df_id_neon_updated = df_id_neon.copy()
    replace_map = {'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'}
    df_id_neon_updated['domainName'] = df_id_neon_updated['domainName'].replace(replace_map)
    df_id_neon_updated = df_id_neon_updated[['ID', 'domainName']]
    study_stations = pd.read_csv('df_neon_filter.csv', dtype={'ID': str})
    if study_stations.ID.duplicated().any() or len(study_stations) != 861:
        raise ValueError('Expected 861 unique study stations in df_neon_filter.csv.')
    study_ids = set(study_stations.ID)
    df_id_neon_updated['domainName'] = df_id_neon_updated['domainName'].astype('string').str.strip()
    df_id_neon_updated = df_id_neon_updated[df_id_neon_updated.ID.isin(study_ids) & df_id_neon_updated.domainName.notna() & df_id_neon_updated.domainName.ne('')].copy()
    if df_id_neon_updated.ID.duplicated().any() or set(df_id_neon_updated.ID) != study_ids:
        raise ValueError('Study station IDs and unique named ecoregion assignments do not match.')
    df_neon_area = df_neon[['ID', 'Area']].merge(df_id_neon_updated, on='ID', how='inner')
    df_annual = dl.df_yearly.copy()
    df_annual_updated = df_annual[df_annual['W_YEAR'].between(1926, 2025) & df_annual.ID.isin(study_ids)].copy()
    if set(df_annual_updated.ID) != study_ids:
        raise ValueError('The annual data do not contain all 861 study stations.')
    print('Study stations in annual analysis:', df_annual_updated.ID.nunique())
    variables_to_run = ['PRCP', 'TMAX', 'TMIN', 'PCI', 'WSDI', 'CSDI', 'SDII']
    for var in variables_to_run:
        run_and_save_for_variable(value_col=var, df_annual=df_annual_updated, df_domains=df_id_neon_updated, df_neon_area=df_neon_area, out_dir=data_intermediate_dir, full_start=1926, full_end=2025, recent_start=1996, recent_end=2025)
    print('\nAll variables processed. Outputs in:', data_intermediate_dir.resolve())
if __name__ == '__main__':
    main()


average_dir = Path('data_intermediate_review/averages_review')
arcgis_dir = Path('data_intermediate_review/ArcGIS_inputs_review')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected one metadata record for each of the 861 study stations.')
variables = ['PRCP', 'TMAX', 'TMIN', 'PCI', 'WSDI', 'CSDI', 'SDII']
parts = []
for variable in variables:
    path = average_dir / f'{variable.lower()}_station_stats_review.csv'
    if not path.exists():
        raise FileNotFoundError(f'Run the review averages first; missing: {path}')
    source = pd.read_csv(path, dtype={'ID': str})
    columns = {f'{variable}_recent_mean': 'recent_average', f'{variable}_full_mean': 'centennial_average', f'{variable}_recent_minus_full': 'recent_minus_centennial', f'{variable}_recent_minus_penult': 'recent_minus_prior'}
    missing = set(columns) - set(source.columns)
    if missing:
        raise ValueError(f'Missing columns for {variable}: {sorted(missing)}')
    if source.ID.duplicated().any():
        raise ValueError(f'Duplicate station IDs in {path}')
    panel = stations.merge(source[['ID', *columns]], on='ID', how='left', validate='one_to_one', indicator=True)
    if not panel['_merge'].eq('both').all():
        raise ValueError(f'Some study stations are missing from {path}')
    panel = panel.drop(columns='_merge').rename(columns=columns)
    panel['variable'] = variable
    parts.append(panel)
arcgis_averages = pd.concat(parts, ignore_index=True)
arcgis_averages = arcgis_averages[['ID', 'Lat', 'Lon', 'variable', 'recent_average', 'centennial_average', 'recent_minus_centennial', 'recent_minus_prior']]
arcgis_dir.mkdir(parents=True, exist_ok=True)
export_path = arcgis_dir / 'all_indices_station_averages_861_no_AI_review.csv'
arcgis_averages.to_csv(export_path, index=False)
print('Saved:', export_path.resolve())
print('Periods: recent WY 1996-2025; centennial WY 1926-2025; prior WY 1966-1995.')
display(arcgis_averages.groupby('variable').size().rename('stations'))
