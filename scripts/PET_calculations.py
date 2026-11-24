
from __future__ import annotations

import os
import calendar
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymannkendall as mk
import data_loader as dl
import geopandas as gpd
from climate_indices import eto, indices, compute

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")


def display(value):
    print(value)


df_daily = dl.df_daily.copy()
df_monthly = dl.df_monthly_updated.copy()
data_raw_dir = Path('./data_raw')
df_neon = pd.read_csv(data_raw_dir / 'Thiessen_Polygons_100yrs.csv')
df_neon = df_neon[['ID_1', 'Lat', 'Lon', 'Area_sqkm']]
df_neon.columns = ['ID', 'Lat', 'Lon', 'Area']
df_id_neon = pd.read_csv(data_raw_dir / 'ID_ecoregion.csv')
df_id_neon_updated = df_id_neon.copy()
replace_map = {'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'}
df_id_neon_updated['domainName'] = df_id_neon_updated['domainName'].replace(replace_map)
df_id_neon_updated = df_id_neon_updated[['ID', 'domainName']]
df_neon_area = df_neon[['ID', 'Area']].merge(df_id_neon_updated, on='ID', how='inner')

us_shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp')

OUTPUT_DIR = Path('Reviewer_PET_Sensitivity')
DAILY_CSV_PATH = None
SHAPEFILE_PATH = 'US_CONUS_Neon_shapefile/US_Neon.shp'
MAP_WATER_YEAR = 2025
MEAN_START_WY, MEAN_END_WY = (1996, 2025)
STUDY_START_WY, STUDY_END_WY = (1926, 2025)
SAVE_INDIVIDUAL_WY_MAPS = False
SHOW_FIGURES = False
FIG_DPI = 250

TABLES = OUTPUT_DIR / 'Tables'
FIGURES = OUTPUT_DIR / 'Figures'
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)


def require_cols(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{label} is missing required columns: {missing}')


def clean_lat(df):
    meta = df[['ID', 'Lat'] + (['Lon'] if 'Lon' in df else [])].copy()
    meta['Lat'] = pd.to_numeric(meta['Lat'], errors='coerce')
    if 'Lon' in meta:
        meta['Lon'] = pd.to_numeric(meta['Lon'], errors='coerce')
    meta = meta.drop_duplicates('ID')
    if meta['Lat'].dropna().abs().gt(90).any():
        raise ValueError('Latitude outside -90 to 90 degrees')
    return meta


if 'df_monthly' not in globals() or 'df_neon' not in globals():
    raise ValueError('Please load df_monthly and df_neon in the notebook first.')

require_cols(df_monthly, ['ID', 'YEAR', 'MONTH', 'TAVG'], 'df_monthly')
require_cols(df_neon, ['ID', 'Lat'], 'df_neon')
meta = clean_lat(df_neon)
monthly = df_monthly.drop(columns=['Unnamed: 0'], errors='ignore').copy()
monthly['YEAR'] = pd.to_numeric(monthly['YEAR'], errors='raise').astype(int)
monthly['MONTH'] = pd.to_numeric(monthly['MONTH'], errors='raise').astype(int)

if not monthly['MONTH'].between(1, 12).all():
    raise ValueError('MONTH must be 1..12')

for c in ('TAVG', 'TMAX', 'TMIN', 'PRCP'):
    if c in monthly:
        monthly[c] = pd.to_numeric(monthly[c], errors='coerce')

monthly['DATE'] = pd.to_datetime(dict(year=monthly.YEAR, month=monthly.MONTH, day=1))

if monthly.duplicated(['ID', 'YEAR', 'MONTH']).any():
    raise ValueError('df_monthly has duplicated station/year/month values. Resolve before PET.')

monthly = monthly.merge(meta, on='ID', how='left', validate='many_to_one')
monthly = monthly.sort_values(['ID', 'DATE']).reset_index(drop=True)
monthly['PET_Thornthwaite_mm'] = np.nan

for station, indexes in monthly.groupby('ID').groups.items():
    sub = monthly.loc[indexes].sort_values('DATE')
    lat = sub['Lat'].iloc[0]

    if not np.isfinite(lat):
        continue

    first_year, last_year = (int(sub.YEAR.min()), int(sub.YEAR.max()))
    month_dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    lookup = sub.set_index('DATE')['TAVG'].reindex(month_dates)

    if lookup.notna().sum() == 0:
        continue

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        pet = eto.eto_thornthwaite(
            monthly_temps_celsius=lookup.to_numpy(dtype=float),
            latitude_degrees=float(lat),
            data_start_year=first_year
        )

    mapped = pd.Series(np.asarray(pet).reshape(-1), index=month_dates)
    mapped[lookup.isna()] = np.nan
    monthly.loc[indexes, 'PET_Thornthwaite_mm'] = sub['DATE'].map(mapped).to_numpy()

monthly['WY'] = np.where(monthly['MONTH'] >= 10, monthly['YEAR'] + 1, monthly['YEAR'])
monthly.to_csv(TABLES / 'PET_monthly_Thornthwaite.csv', index=False)
print(f'Thornthwaite PET calculated for {monthly.ID.nunique()} stations.')

if 'df_daily' in globals() and isinstance(globals()['df_daily'], pd.DataFrame):
    daily = globals()['df_daily'].copy()
elif DAILY_CSV_PATH is not None:
    daily = pd.read_csv(DAILY_CSV_PATH)
else:
    print('\nThe Thornthwaite-only CSV was saved in:', TABLES)
    raise ValueError(
        'Hargreaves-Samani requires daily data. Load df_daily with ID, DATE, '
        'TMAX, TMIN (C), or set DAILY_CSV_PATH in this script. '
        'The monthly dataframe cannot produce an exact daily Hargreaves PET.'
    )

require_cols(daily, ['ID', 'TMAX', 'TMIN'], 'df_daily')

if 'DATE' not in daily.columns:
    require_cols(daily, ['YEAR', 'MONTH', 'DAY'], 'df_daily')
    daily['DATE'] = pd.to_datetime(daily[['YEAR', 'MONTH', 'DAY']])
else:
    daily['DATE'] = pd.to_datetime(daily['DATE'], errors='raise')

daily['DATE'] = daily['DATE'].dt.normalize()

if daily.duplicated(['ID', 'DATE']).any():
    raise ValueError('Duplicate ID/DATE observations in df_daily. Resolve before PET.')

for c in ['TMIN', 'TMAX']:
    daily[c] = pd.to_numeric(daily[c], errors='coerce')

daily['temperature_order_issue'] = (
    daily['TMIN'].notna()
    & daily['TMAX'].notna()
    & (daily['TMIN'] > daily['TMAX'])
)

invalid_n = int(daily.temperature_order_issue.sum())
daily.loc[daily.temperature_order_issue, ['TMIN', 'TMAX']] = np.nan
print('Daily TMIN > TMAX records excluded:', invalid_n)

daily = daily.merge(meta[['ID', 'Lat']], on='ID', how='left', validate='many_to_one')
daily = daily[daily['ID'].isin(monthly['ID'].unique())].copy()
daily = daily.sort_values(['ID', 'DATE']).reset_index(drop=True)

blocks = []
quality = []

for station, group in daily.groupby('ID', sort=False):
    lat = group['Lat'].iloc[0]

    if not np.isfinite(lat):
        continue

    yrs = np.arange(group.DATE.dt.year.min(), group.DATE.dt.year.max() + 1)
    yearly_blocks = []

    for year in yrs:
        observed = group[group.DATE.dt.year == year].set_index('DATE')
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
        a = observed.reindex(dates)

        tmp = pd.DataFrame({
            'ID': station,
            'DATE': dates,
            'TMIN': a.TMIN.to_numpy(dtype=float),
            'TMAX': a.TMAX.to_numpy(dtype=float)
        })

        if not calendar.isleap(int(year)):
            tmp = pd.concat([
                tmp,
                pd.DataFrame({
                    'ID': [station],
                    'DATE': [pd.NaT],
                    'TMIN': [np.nan],
                    'TMAX': [np.nan]
                })
            ], ignore_index=True)

        yearly_blocks.append(tmp)

    block = pd.concat(yearly_blocks, ignore_index=True)
    mean_t = (block.TMIN.to_numpy() + block.TMAX.to_numpy()) / 2

    results = eto.eto_hargreaves(
        daily_tmin_celsius=block.TMIN.to_numpy(dtype=float),
        daily_tmax_celsius=block.TMAX.to_numpy(dtype=float),
        daily_tmean_celsius=mean_t,
        latitude_degrees=float(lat)
    )

    block['PET_Hargreaves_mm_day'] = np.asarray(results).reshape(-1)
    block.loc[
        block['TMIN'].isna() | block['TMAX'].isna(),
        'PET_Hargreaves_mm_day'
    ] = np.nan

    block = block[block.DATE.notna()].copy()
    blocks.append(block[['ID', 'DATE', 'PET_Hargreaves_mm_day']])

    quality.append({
        'ID': station,
        'n_daily_rows': len(group),
        'n_daily_PET_valid': int(block.PET_Hargreaves_mm_day.notna().sum())
    })

if not blocks:
    raise ValueError('No valid daily data with station latitude for Hargreaves PET.')

pet_daily = pd.concat(blocks, ignore_index=True)
pet_daily.to_csv(TABLES / 'PET_daily_Hargreaves.csv', index=False)
pd.DataFrame(quality).to_csv(TABLES / 'Hargreaves_daily_coverage.csv', index=False)

pet_daily['DATE_MONTH'] = pet_daily.DATE.dt.to_period('M').dt.to_timestamp()

pet_mon = pet_daily.groupby(['ID', 'DATE_MONTH'], as_index=False).agg(
    PET_Hargreaves_mm=('PET_Hargreaves_mm_day', lambda x: x.sum(min_count=1)),
    n_valid_days=('PET_Hargreaves_mm_day', 'count'),
    n_days=('DATE', 'count')
)

pet_mon.loc[
    pet_mon.n_valid_days != pet_mon.n_days,
    'PET_Hargreaves_mm'
] = np.nan

monthly = monthly.merge(
    pet_mon.rename(columns={'DATE_MONTH': 'DATE'}),
    on=['ID', 'DATE'],
    how='left',
    validate='one_to_one'
)

monthly['PET_difference_mm'] = monthly.PET_Thornthwaite_mm - monthly.PET_Hargreaves_mm
monthly.to_csv(TABLES / 'PET_monthly_both_methods.csv', index=False)

annual = monthly.groupby(['ID', 'WY'], as_index=False).agg(
    PET_Thornthwaite_mm=('PET_Thornthwaite_mm', lambda x: x.sum(min_count=12)),
    PET_Hargreaves_mm=('PET_Hargreaves_mm', lambda x: x.sum(min_count=12)),
    n_months_Thornthwaite=('PET_Thornthwaite_mm', 'count'),
    n_months_Hargreaves=('PET_Hargreaves_mm', 'count')
)

annual = annual[annual.WY.between(STUDY_START_WY, STUDY_END_WY)].copy()
annual['PET_difference_mm'] = annual.PET_Thornthwaite_mm - annual.PET_Hargreaves_mm
annual = annual.merge(meta, on='ID', how='left', validate='many_to_one')
annual.to_csv(TABLES / 'PET_annual_water_year_both_methods.csv', index=False)

paired = annual[
    annual.WY.between(MEAN_START_WY, MEAN_END_WY)
    & annual.PET_difference_mm.notna()
]

station_mean = paired.groupby('ID', as_index=False).agg(
    mean_PET_Thornthwaite_mm=('PET_Thornthwaite_mm', 'mean'),
    mean_PET_Hargreaves_mm=('PET_Hargreaves_mm', 'mean'),
    mean_difference_mm=('PET_difference_mm', 'mean'),
    n_paired_WY=('PET_difference_mm', 'count')
).merge(meta, on='ID', how='left')

station_mean.to_csv(
    TABLES / f'PET_station_mean_paired_WY{MEAN_START_WY}_{MEAN_END_WY}.csv',
    index=False
)

conus_annual = annual.groupby('WY', as_index=False).agg(
    mean_difference_mm=('PET_difference_mm', 'mean'),
    n_paired_stations=('PET_difference_mm', 'count')
)

conus_annual.to_csv(TABLES / 'PET_annual_CONUS_station_mean_difference.csv', index=False)

shape = globals().get('us_shape', None)

if shape is None and SHAPEFILE_PATH and Path(SHAPEFILE_PATH).exists():
    shape = gpd.read_file(SHAPEFILE_PATH)

if shape is not None:
    shape = shape.to_crs('EPSG:4326')


def make_map(data, title, stem):
    data = data.dropna(subset=['Lon', 'Lat', 'PET_difference_mm']).copy()

    if len(data) == 0:
        print('No mappable PET differences:', title)
        return

    (FIGURES / stem).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    if shape is not None:
        shape.plot(
            ax=ax,
            facecolor='#f3f3f3',
            edgecolor='#aaaaaa',
            linewidth=0.5
        )

    limit = np.nanpercentile(np.abs(data.PET_difference_mm), 98)
    limit = max(limit, 1.0)

    pts = ax.scatter(
        data.Lon,
        data.Lat,
        c=data.PET_difference_mm,
        cmap='RdBu_r',
        vmin=-limit,
        vmax=limit,
        s=15,
        alpha=0.9,
        edgecolors='none'
    )

    plt.colorbar(
        pts,
        ax=ax,
        label='Thornthwaite − Hargreaves-Samani (mm/year)',
        shrink=0.75
    )

    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.12)
    fig.tight_layout()

    fig.savefig(
        FIGURES / f'{stem}.png',
        dpi=FIG_DPI,
        bbox_inches='tight'
    )

    fig.savefig(
        FIGURES / f'{stem}.pdf',
        bbox_inches='tight'
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


make_map(
    annual.loc[annual.WY == MAP_WATER_YEAR],
    f'Annual PET difference, WY {MAP_WATER_YEAR}',
    f'PET_difference_WY{MAP_WATER_YEAR}'
)

make_map(
    station_mean.rename(columns={'mean_difference_mm': 'PET_difference_mm'}),
    f'Mean annual PET difference, WY {MEAN_START_WY}–{MEAN_END_WY}',
    f'PET_mean_difference_WY{MEAN_START_WY}_{MEAN_END_WY}'
)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(conus_annual.WY, conus_annual.mean_difference_mm, lw=1.5)
ax.axhline(0, color='k', linewidth=0.8)

ax.set(
    xlabel='Water year',
    ylabel='Mean station PET difference (mm/year)',
    title='Annual PET differences (paired stations)'
)

ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURES / 'CONUS_annual_PET_difference.png', dpi=FIG_DPI)
fig.savefig(FIGURES / 'CONUS_annual_PET_difference.pdf')

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)

if SAVE_INDIVIDUAL_WY_MAPS:
    for wy in sorted(annual.WY.unique()):
        make_map(
            annual.loc[annual.WY == wy],
            f'Annual PET difference, WY {wy}',
            f'Annual_Maps/PET_difference_WY{wy}'
        )

print('\nFinished. Results saved under:', OUTPUT_DIR.resolve())
print('Paired station-years:', int(annual.PET_difference_mm.notna().sum()))
print('Important: review method differences and PET formulation before reporting.')

pet_tables = Path('Reviewer_PET_Sensitivity') / 'Tables'
pet_figures = Path('Reviewer_PET_Sensitivity') / 'Figures'
monthly_path = pet_tables / 'PET_monthly_both_methods.csv'
annual_pet_path = pet_tables / 'PET_annual_water_year_both_methods.csv'

for path in (monthly_path, annual_pet_path):
    if not path.exists():
        raise FileNotFoundError(
            f'Required saved PET output not found: {path}. '
            'Run the PET sensitivity cell first.'
        )

pet_monthly = pd.read_csv(
    monthly_path,
    usecols=['ID', 'YEAR', 'MONTH', 'WY', 'PRCP'],
    dtype={'ID': str}
)

pet_annual = pd.read_csv(annual_pet_path, dtype={'ID': str})

if (
    pet_monthly.duplicated(['ID', 'YEAR', 'MONTH']).any()
    or pet_annual.duplicated(['ID', 'WY']).any()
):
    raise ValueError(
        'Duplicate station-month or station-water-year records in saved PET tables.'
    )

pet_monthly['PRCP'] = pd.to_numeric(pet_monthly['PRCP'], errors='coerce')

precip_wy = pet_monthly[
    pet_monthly['WY'].between(1926, 2025)
].groupby(['ID', 'WY'], as_index=False).agg(
    n_months=('MONTH', 'nunique'),
    n_prcp_months=('PRCP', 'count'),
    PRCP_WY_mm=('PRCP', lambda x: x.sum(min_count=12))
)

precip_wy.loc[
    (precip_wy.n_months != 12) | (precip_wy.n_prcp_months != 12),
    'PRCP_WY_mm'
] = np.nan

ai_water_year = precip_wy.merge(
    pet_annual,
    on=['ID', 'WY'],
    how='left',
    validate='one_to_one'
)

for method in ('Thornthwaite', 'Hargreaves'):
    pet_col = f'PET_{method}_mm'
    valid = (
        (ai_water_year[pet_col] > 0)
        & (ai_water_year[f'n_months_{method}'] == 12)
    )

    ai_water_year[f'AI_{method}'] = (
        ai_water_year['PRCP_WY_mm'] / ai_water_year[pet_col]
    ).where(valid)

ai_water_year.to_csv(
    pet_tables / 'AI_station_water_year_1926_2025.csv',
    index=False
)

periods = [(1926, 2025), (1996, 2025)]
ai_periods = []

for start, end in periods:
    subset = ai_water_year[ai_water_year.WY.between(start, end)].copy()
    subset = subset.dropna(subset=['AI_Thornthwaite', 'AI_Hargreaves'])

    summary = subset.groupby('ID', as_index=False).agg(
        AI_Thornthwaite=('AI_Thornthwaite', 'mean'),
        AI_Hargreaves=('AI_Hargreaves', 'mean'),
        n_paired_WY=('WY', 'nunique'),
        Lat=('Lat', 'first'),
        Lon=('Lon', 'first')
    )

    summary['start_WY'], summary['end_WY'] = (start, end)
    ai_periods.append(summary)

ai_period_summary = pd.concat(ai_periods, ignore_index=True)

ai_period_summary.to_csv(
    pet_tables / 'AI_station_period_means_paired_WY1926_2025_WY1996_2025.csv',
    index=False
)

print(f'Saved water-year AI for {ai_water_year.ID.nunique()} stations to {pet_tables.resolve()}')

print(
    ai_period_summary.groupby(['start_WY', 'end_WY']).agg(
        stations=('ID', 'nunique'),
        min_paired_WY=('n_paired_WY', 'min'),
        max_paired_WY=('n_paired_WY', 'max')
    )
)

trend_records = []

for start, end in [(1926, 2025), (1996, 2025)]:
    for station, group in ai_water_year.groupby('ID', sort=True):
        yearly = group.set_index('WY').reindex(range(start, end + 1))
        paired = yearly['AI_Thornthwaite'].notna() & yearly['AI_Hargreaves'].notna()
        n = int(paired.sum())

        for method in ('Thornthwaite', 'Hargreaves'):
            record = {
                'ID': station,
                'method': method,
                'start_WY': start,
                'end_WY': end,
                'n_paired_WY': n,
                'trend': None,
                'p': np.nan,
                'h': False,
                'slope_per_year': np.nan,
                'slope_per_decade': np.nan,
                'z': np.nan,
                'tau': np.nan,
                's': np.nan,
                'var_s': np.nan,
                'intercept': np.nan
            }

            if n >= 3:
                series = yearly[f'AI_{method}'].where(paired).to_numpy(dtype=float)

                try:
                    result = mk.hamed_rao_modification_test(series, alpha=0.05)

                    if (
                        np.isfinite(result.p)
                        and 0 <= result.p <= 1
                        and np.isfinite(result.z)
                        and np.isfinite(result.var_s)
                        and result.var_s >= 0
                    ):
                        record.update(
                            trend=result.trend,
                            p=result.p,
                            h=bool(result.h),
                            slope_per_year=result.slope,
                            slope_per_decade=10 * result.slope,
                            z=result.z,
                            tau=result.Tau,
                            s=result.s,
                            var_s=result.var_s,
                            intercept=result.intercept
                        )

                except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
                    pass

            trend_records.append(record)

ai_trends = pd.DataFrame(trend_records)

station_coordinates = ai_water_year[
    ['ID', 'Lat', 'Lon']
].dropna(subset=['Lat', 'Lon']).drop_duplicates('ID')

ai_trends = ai_trends.merge(
    station_coordinates,
    on='ID',
    how='left',
    validate='many_to_one'
)

ai_trends.to_csv(
    pet_tables / 'AI_station_modified_MK_trends_WY1926_2025_WY1996_2025.csv',
    index=False
)

print('Station trend results saved to', pet_tables.resolve())

print(
    ai_trends.groupby(
        ['method', 'start_WY', 'end_WY', 'trend']
    ).size().rename('stations').to_string()
)

if not hasattr(indices.Distribution, 'loglogistic'):
    raise RuntimeError(
        'This climate_indices version does not support log-logistic SPEI. '
        'A version with Distribution.loglogistic is needed to match the '
        'original distribution; no package was installed automatically.'
    )

spei_input = pd.read_csv(
    pet_tables / 'PET_monthly_both_methods.csv',
    usecols=['ID', 'YEAR', 'MONTH', 'PRCP', 'PET_Hargreaves_mm'],
    dtype={'ID': str}
)

if spei_input.duplicated(['ID', 'YEAR', 'MONTH']).any():
    raise ValueError('Duplicate station-month records in the PET monthly table.')

spei_input['DATE'] = pd.to_datetime(
    dict(year=spei_input.YEAR, month=spei_input.MONTH, day=1)
)

spei_monthly_parts, spei_fit_errors = ([], [])

for station, group in spei_input.groupby('ID', sort=True):
    group = group.sort_values('DATE').set_index('DATE')
    first_year, last_year = (group.index.year.min(), group.index.year.max())
    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    group = group.reindex(dates)

    prcp = pd.to_numeric(
        group.PRCP, errors='coerce'
    ).to_numpy(dtype=float)

    pet = pd.to_numeric(
        group.PET_Hargreaves_mm, errors='coerce'
    ).to_numpy(dtype=float)

    try:
        values = indices.spei(
            precips_mm=prcp,
            pet_mm=pet,
            scale=12,
            distribution=indices.Distribution.loglogistic,
            periodicity=compute.Periodicity.monthly,
            data_start_year=int(first_year),
            calibration_year_initial=1926,
            calibration_year_final=2025
        )

        values = np.asarray(values).reshape(-1)[:len(dates)]

    except Exception as exc:
        values = np.full(len(dates), np.nan)
        spei_fit_errors.append({'ID': station, 'error': str(exc)})

    spei_monthly_parts.append(
        pd.DataFrame({
            'ID': station,
            'DATE': dates,
            'SPEI12_Hargreaves': values
        })
    )

spei12_hargreaves = pd.concat(spei_monthly_parts, ignore_index=True)

spei12_hargreaves.to_csv(
    pet_tables / 'SPEI12_Hargreaves_python_monthly.csv',
    index=False
)

spei_fit_errors = pd.DataFrame(spei_fit_errors, columns=['ID', 'error'])

spei_fit_errors.to_csv(
    pet_tables / 'SPEI12_Hargreaves_python_fit_errors.csv',
    index=False
)

print(
    'Saved Python Hargreaves SPEI-12; '
    'stations with fitting errors:', len(spei_fit_errors)
)

if not spei_fit_errors.empty:
    display(spei_fit_errors)
