
from __future__ import annotations

import os
import warnings
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
from climate_indices import indices, compute
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")


def display(value):
    print(value)


PET_PATH = Path('Reviewer_PET_Sensitivity/Tables/PET_monthly_Thornthwaite.csv')
OUTPUT_DIR = Path('AI_SPEI_python')
TABLES = OUTPUT_DIR / 'Tables'
PERIODS = {'centennial': (1926, 2025), 'recent': (1996, 2025), 'prior': (1966, 1995)}
TREND_PERIODS = {k: PERIODS[k] for k in ('centennial', 'recent')}
CALIBRATION_START, CALIBRATION_END = (1926, 2025)
ALPHA = 0.05

if not PET_PATH.exists():
    raise FileNotFoundError(f'Saved Thornthwaite PET table not found: {PET_PATH}')
if not hasattr(indices.Distribution, 'loglogistic'):
    raise RuntimeError('The active climate_indices version lacks log-logistic SPEI support.')

TABLES.mkdir(parents=True, exist_ok=True)
print('PET input:', PET_PATH.resolve())
print('Output folder:', OUTPUT_DIR.resolve())
print('pyMannKendall:', mk.__file__, getattr(mk, '__version__', 'unknown'))
print('climate_indices:', importlib.metadata.version('climate-indices'))

columns = ['ID', 'YEAR', 'MONTH', 'PRCP', 'PET_Thornthwaite_mm', 'Lat', 'Lon']
monthly = pd.read_csv(PET_PATH, usecols=columns, dtype={'ID': str})

for col in ['YEAR', 'MONTH']:
    monthly[col] = pd.to_numeric(monthly[col], errors='raise').astype(int)

if not monthly.MONTH.between(1, 12).all():
    raise ValueError('Invalid calendar month in PET input.')

if monthly.duplicated(['ID', 'YEAR', 'MONTH']).any():
    raise ValueError('Duplicate station-month records in PET input.')

for col in ['PRCP', 'PET_Thornthwaite_mm', 'Lat', 'Lon']:
    monthly[col] = pd.to_numeric(monthly[col], errors='coerce')

monthly['DATE'] = pd.to_datetime(dict(year=monthly.YEAR, month=monthly.MONTH, day=1))
monthly['W_YEAR'] = monthly.YEAR + (monthly.MONTH >= 10).astype(int)
monthly = monthly.sort_values(['ID', 'DATE']).reset_index(drop=True)
coordinates = monthly[['ID', 'Lat', 'Lon']].dropna(subset=['Lat', 'Lon']).drop_duplicates('ID')

print('Stations:', monthly.ID.nunique(), 'Monthly records:', len(monthly))
print('Available dates:', monthly.DATE.min(), 'to', monthly.DATE.max())
print('Missing PRCP:', monthly.PRCP.isna().sum(), 'Missing Thornthwaite PET:', monthly.PET_Thornthwaite_mm.isna().sum())
print('Stations with coordinates:', coordinates.ID.nunique())

annual_ai = monthly[monthly.W_YEAR.between(1926, 2025)].groupby(['ID', 'W_YEAR'], as_index=False).agg(
    n_months=('MONTH', 'nunique'),
    n_prcp=('PRCP', 'count'),
    n_pet=('PET_Thornthwaite_mm', 'count'),
    PRCP_mm=('PRCP', lambda x: x.sum(min_count=12)),
    PET_mm=('PET_Thornthwaite_mm', lambda x: x.sum(min_count=12))
)

valid = (
    (annual_ai.n_months == 12)
    & (annual_ai.n_prcp == 12)
    & (annual_ai.n_pet == 12)
    & (annual_ai.PET_mm > 0)
)

annual_ai['AI'] = (annual_ai.PRCP_mm / annual_ai.PET_mm).where(valid)
annual_ai = annual_ai.merge(coordinates, on='ID', how='left', validate='many_to_one')
annual_ai.to_csv(TABLES / 'AI_station_water_year.csv', index=False)

ai_means = []

for name, (start, end) in PERIODS.items():
    subset = annual_ai[annual_ai.W_YEAR.between(start, end)]
    mean = subset.groupby('ID', as_index=False).agg(
        mean_AI=('AI', 'mean'),
        n_valid_WY=('AI', 'count')
    )
    mean['period'], mean['start_WY'], mean['end_WY'] = (name, start, end)
    ai_means.append(mean)

ai_period_means = pd.concat(ai_means, ignore_index=True).merge(
    coordinates, on='ID', how='left', validate='many_to_one'
)

ai_period_means.to_csv(TABLES / 'AI_station_period_means.csv', index=False)
print('Saved annual AI and period means.')
display(ai_period_means.groupby('period')['n_valid_WY'].describe())


def run_mk(values, min_count):
    x = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    record = {
        'n': len(x),
        'trend': None,
        'p': np.nan,
        'z': np.nan,
        'variance': np.nan,
        'slope_per_step': np.nan
    }

    if len(x) < min_count:
        return record

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = mk.hamed_rao_modification_test(x, alpha=ALPHA)

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
                z=result.z,
                variance=result.var_s,
                slope_per_step=result.slope
            )

    except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
        pass

    return record


ai_records = []

for name, (start, end) in TREND_PERIODS.items():
    window = annual_ai[annual_ai.W_YEAR.between(start, end)].sort_values(['ID', 'W_YEAR'])

    for station, group in window.groupby('ID'):
        rec = run_mk(group.AI, min_count=3)
        rec.update(ID=station, period=name, start_WY=start, end_WY=end)
        rec['slope_per_decade'] = rec['slope_per_step'] * 10
        ai_records.append(rec)

ai_trends = pd.DataFrame(ai_records).merge(
    coordinates, on='ID', how='left', validate='many_to_one'
)

ai_trends.to_csv(TABLES / 'AI_station_modified_MK_trends.csv', index=False)
print('Saved AI trend results.')
display(ai_trends.groupby(['period', 'trend']).size().rename('stations'))

SPEI_SCALES = [1, 3, 6, 12]
spei_parts, fit_errors = ([], [])

for station, group in monthly.groupby('ID', sort=True):
    group = group.set_index('DATE').sort_index()
    first_year, last_year = (
        int(group.index.year.min()),
        int(group.index.year.max())
    )

    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    aligned = group.reindex(dates)
    prcp = aligned.PRCP.to_numpy(dtype=float)
    pet = aligned.PET_Thornthwaite_mm.to_numpy(dtype=float)
    out = pd.DataFrame({'ID': station, 'DATE': dates})

    for scale in SPEI_SCALES:
        try:
            values = indices.spei(
                precips_mm=prcp,
                pet_mm=pet,
                scale=scale,
                distribution=indices.Distribution.loglogistic,
                periodicity=compute.Periodicity.monthly,
                data_start_year=first_year,
                calibration_year_initial=CALIBRATION_START,
                calibration_year_final=CALIBRATION_END
            )

            values = np.asarray(values).reshape(-1)

            if len(values) < len(dates):
                raise ValueError('SPEI output is shorter than the monthly input.')

            out[f'SPEI{scale}'] = values[:len(dates)]

        except Exception as exc:
            out[f'SPEI{scale}'] = np.nan
            fit_errors.append({
                'ID': station,
                'scale': scale,
                'error': f'{type(exc).__name__}: {exc}'
            })

    spei_parts.append(out)

spei_monthly = pd.concat(spei_parts, ignore_index=True)
spei_monthly['W_YEAR'] = (
    spei_monthly.DATE.dt.year
    + (spei_monthly.DATE.dt.month >= 10).astype(int)
)

spei_monthly.to_csv(TABLES / 'SPEI_python_Thornthwaite_monthly.csv', index=False)

spei_errors = pd.DataFrame(fit_errors, columns=['ID', 'scale', 'error'])
spei_errors.to_csv(TABLES / 'SPEI_python_Thornthwaite_fit_errors.csv', index=False)

if not spei_errors.empty:
    display(spei_errors)

spei_records, drought_records = ([], [])

for station, group in spei_monthly.groupby('ID', sort=True):
    group = group.sort_values('DATE')

    for name, (start, end) in PERIODS.items():
        window = group[group.W_YEAR.between(start, end)]

        for scale in SPEI_SCALES:
            values = pd.to_numeric(
                window[f'SPEI{scale}'],
                errors='coerce'
            ).clip(-3, 3)

            valid = values.notna()

            drought_records.append({
                'ID': station,
                'period': name,
                'start_WY': start,
                'end_WY': end,
                'scale': scale,
                'n_valid_months': int(valid.sum()),
                'drought_months': int((values <= -1).sum()),
                'drought_months_per_decade': (
                    float((values <= -1).sum()) / (end - start + 1) * 10
                    if valid.sum() >= 12 else np.nan
                )
            })

            if name in TREND_PERIODS:
                rec = run_mk(values, min_count=12)
                rec.update(
                    ID=station,
                    period=name,
                    start_WY=start,
                    end_WY=end,
                    scale=scale
                )
                rec['slope_per_decade'] = rec['slope_per_step'] * 120
                spei_records.append(rec)

spei_trends = pd.DataFrame(spei_records).merge(
    coordinates, on='ID', how='left', validate='many_to_one'
)

spei_drought = pd.DataFrame(drought_records).merge(
    coordinates, on='ID', how='left', validate='many_to_one'
)

spei_trends.to_csv(TABLES / 'SPEI_station_modified_MK_trends.csv', index=False)
spei_drought.to_csv(TABLES / 'SPEI_station_drought_months_per_decade.csv', index=False)

print('Saved SPEI trend and drought-frequency results.')
display(spei_trends.groupby(['scale', 'period', 'trend']).size().rename('stations'))

FDR_ALPHA = 0.05


def add_bh_fdr(frame, family_columns):
    result = frame.copy()

    required = {
        'ID', 'period', 'trend', 'p',
        'slope_per_step', *family_columns
    }

    missing = required - set(result.columns)

    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    keys = ['ID'] + family_columns

    if result.duplicated(keys).any():
        raise ValueError(f'Duplicate records for {keys}')

    result['p'] = pd.to_numeric(result['p'], errors='coerce')
    result['slope_per_step'] = pd.to_numeric(result['slope_per_step'], errors='coerce')

    eligible = np.isfinite(result.p) & result.p.between(0, 1)

    result['p_fdr_bh'] = np.nan
    result['fdr_significant'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    result['fdr_trend'] = None
    result['fdr_family_size'] = 0

    for _, group in result.groupby(family_columns, sort=True):
        selected = group.index[eligible.loc[group.index]]
        result.loc[group.index, 'fdr_family_size'] = len(selected)

        if len(selected) == 0:
            continue

        reject, adjusted, _, _ = multipletests(
            result.loc[selected, 'p'].to_numpy(dtype=float),
            alpha=FDR_ALPHA,
            method='fdr_bh'
        )

        result.loc[selected, 'p_fdr_bh'] = adjusted
        result.loc[selected, 'fdr_significant'] = reject
        result.loc[selected, 'fdr_trend'] = 'no trend'

        sig_index = selected[reject]
        slopes = result.loc[sig_index, 'slope_per_step']

        result.loc[slopes.index[slopes > 0], 'fdr_trend'] = 'increasing'
        result.loc[slopes.index[slopes < 0], 'fdr_trend'] = 'decreasing'
        result.loc[slopes.index[slopes.isna() | slopes.eq(0)], 'fdr_trend'] = None

    return result


def summarize_fdr(frame, family_columns):
    return frame.groupby(family_columns, as_index=False).agg(
        total_stations=('ID', 'size'),
        valid_p_values=('p_fdr_bh', 'count'),
        original_increasing=('trend', lambda x: x.eq('increasing').sum()),
        original_decreasing=('trend', lambda x: x.eq('decreasing').sum()),
        fdr_increasing=('fdr_trend', lambda x: x.eq('increasing').sum()),
        fdr_decreasing=('fdr_trend', lambda x: x.eq('decreasing').sum()),
        fdr_no_trend=('fdr_trend', lambda x: x.eq('no trend').sum())
    )


ai_path = TABLES / 'AI_station_modified_MK_trends.csv'
spei_path = TABLES / 'SPEI_station_modified_MK_trends.csv'

for path in (ai_path, spei_path):
    if not path.exists():
        raise FileNotFoundError(f'Required saved trend results not found: {path}')

ai_fdr = add_bh_fdr(
    pd.read_csv(ai_path, dtype={'ID': str}),
    ['period']
)

spei_fdr = add_bh_fdr(
    pd.read_csv(spei_path, dtype={'ID': str}),
    ['scale', 'period']
)

ai_fdr_summary = summarize_fdr(ai_fdr, ['period'])
spei_fdr_summary = summarize_fdr(spei_fdr, ['scale', 'period'])

ai_fdr.to_csv(TABLES / 'AI_station_trends_BH_FDR.csv', index=False)
spei_fdr.to_csv(TABLES / 'SPEI_station_trends_BH_FDR.csv', index=False)
ai_fdr_summary.to_csv(TABLES / 'AI_BH_FDR_summary.csv', index=False)
spei_fdr_summary.to_csv(TABLES / 'SPEI_BH_FDR_summary.csv', index=False)

print('Saved BH-FDR results to', TABLES.resolve())
print('AI:')
display(ai_fdr_summary)
print('SPEI:')
display(spei_fdr_summary)

percent_columns = [
    'original_increasing',
    'fdr_increasing',
    'original_decreasing',
    'fdr_decreasing'
]


def percentages_for_study(summary_path):
    if not summary_path.exists():
        raise FileNotFoundError(
            f'Run the preceding BH-FDR section first: {summary_path}'
        )

    frame = pd.read_csv(summary_path)

    if not frame.total_stations.eq(861).all():
        raise ValueError(
            'A family does not contain 861 station records; '
            'check the station set before reporting percentages.'
        )

    result = frame.drop(
        columns=['total_stations', 'valid_p_values', 'fdr_no_trend'],
        errors='ignore'
    ).copy()

    result[percent_columns] = (
        result[percent_columns] / 861 * 100
    ).round(1)

    return result.rename(
        columns={col: col + '_percent' for col in percent_columns}
    )


ai_percentages = percentages_for_study(TABLES / 'AI_BH_FDR_summary.csv')
spei_percentages = percentages_for_study(TABLES / 'SPEI_BH_FDR_summary.csv')

ai_percentages.to_csv(
    TABLES / 'AI_BH_FDR_percentages_861.csv',
    index=False,
    float_format='%.1f'
)

spei_percentages.to_csv(
    TABLES / 'SPEI_BH_FDR_percentages_861.csv',
    index=False,
    float_format='%.1f'
)

print('Percent of all 861 study stations.')

format_percent = {
    col + '_percent': '{:.1f}%'
    for col in percent_columns
}

print('AI:')
display(ai_percentages.style.format(format_percent))
print('SPEI:')
display(spei_percentages.style.format(format_percent))

input_dir = Path('AI_SPEI_python/Tables')
arcgis_dir = Path('data_intermediate_review/ArcGIS_inputs_review')

stations = pd.read_csv(
    'df_neon_filter.csv',
    usecols=['ID', 'Lat', 'Lon'],
    dtype={'ID': str}
)

if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations in df_neon_filter.csv.')

ai = pd.read_csv(
    input_dir / 'AI_station_modified_MK_trends.csv',
    dtype={'ID': str}
)

spei = pd.read_csv(
    input_dir / 'SPEI_station_modified_MK_trends.csv',
    dtype={'ID': str}
)

ai['variable'] = 'AI'
spei['variable'] = (
    'SPEI'
    + pd.to_numeric(spei.scale, errors='raise').astype(int).astype(str)
)

ai['slope_unit'] = 'per year'
spei['slope_unit'] = 'per month'

trend_source = pd.concat([ai, spei], ignore_index=True)

if trend_source.duplicated(['ID', 'variable', 'period']).any():
    raise ValueError(
        'Duplicate station, variable and period in saved trend results.'
    )

trend_source['p'] = pd.to_numeric(trend_source.p, errors='coerce')
trend_source['slope_per_step'] = pd.to_numeric(
    trend_source.slope_per_step,
    errors='coerce'
)

trend_source['period'] = (
    trend_source.start_WY.astype(int).astype(str)
    + '-'
    + trend_source.end_WY.astype(int).astype(str)
)

trend_source = trend_source.drop(columns=['Lat', 'Lon'], errors='ignore')

arcgis_trends = trend_source.merge(
    stations,
    on='ID',
    how='inner',
    validate='many_to_one'
)

for col in ['Lat', 'Lon']:
    arcgis_trends[col] = pd.to_numeric(
        arcgis_trends[col],
        errors='coerce'
    )

arcgis_trends = arcgis_trends[
    np.isfinite(arcgis_trends.slope_per_step)
    & np.isfinite(arcgis_trends.Lat)
    & np.isfinite(arcgis_trends.Lon)
]

arcgis_trends = arcgis_trends.rename(
    columns={
        'slope_per_step': 'slope',
        'p': 'p_value'
    }
)[
    [
        'ID', 'Lat', 'Lon', 'variable', 'period',
        'slope', 'slope_unit', 'p_value', 'trend'
    ]
].sort_values(
    ['variable', 'period', 'ID']
).reset_index(drop=True)

arcgis_dir.mkdir(parents=True, exist_ok=True)

trend_export_path = (
    arcgis_dir / 'AI_SPEI_python_station_trends_861.csv'
)

arcgis_trends.to_csv(trend_export_path, index=False)

print('Saved:', trend_export_path.resolve())
print('Raw AI slope is per year; raw SPEI slope is per retained monthly observation.')
display(
    arcgis_trends.groupby(['variable', 'period'])
    .size()
    .rename('stations')
    .reset_index()
)

ai_means = pd.read_csv(
    input_dir / 'AI_station_period_means.csv',
    dtype={'ID': str}
)

spei_drought = pd.read_csv(
    input_dir / 'SPEI_station_drought_months_per_decade.csv',
    dtype={'ID': str}
)

ai_periods = ai_means[
    ['ID', 'period', 'mean_AI']
].rename(columns={'mean_AI': 'value'})

ai_periods['variable'] = 'AI'

spei_periods = spei_drought[
    ['ID', 'period', 'scale', 'drought_months_per_decade']
].rename(columns={'drought_months_per_decade': 'value'})

spei_periods['variable'] = (
    'SPEI'
    + pd.to_numeric(spei_periods.scale, errors='raise')
    .astype(int)
    .astype(str)
)

spei_periods = spei_periods.drop(columns='scale')

average_source = pd.concat(
    [ai_periods, spei_periods],
    ignore_index=True
)

if average_source.duplicated(['ID', 'variable', 'period']).any():
    raise ValueError(
        'Duplicate station, variable and period in saved average data.'
    )

wide = average_source.pivot(
    index=['ID', 'variable'],
    columns='period',
    values='value'
).reset_index()

missing_periods = {'centennial', 'recent', 'prior'} - set(wide.columns)

if missing_periods:
    raise ValueError(f'Missing periods: {sorted(missing_periods)}')

wide['recent_minus_centennial'] = wide.recent - wide.centennial
wide['recent_minus_prior'] = wide.recent - wide.prior

wide = wide.rename(
    columns={
        'recent': 'recent_average',
        'centennial': 'centennial_average'
    }
)

arcgis_averages = wide.merge(
    stations,
    on='ID',
    how='inner',
    validate='many_to_one'
)

arcgis_averages['measure'] = np.where(
    arcgis_averages.variable == 'AI',
    'mean annual AI',
    'drought months per decade'
)

arcgis_averages = arcgis_averages[
    [
        'ID', 'Lat', 'Lon', 'variable', 'measure',
        'recent_average', 'centennial_average',
        'recent_minus_centennial', 'recent_minus_prior'
    ]
]

arcgis_averages = arcgis_averages.sort_values(
    ['variable', 'ID']
).reset_index(drop=True)

arcgis_dir.mkdir(parents=True, exist_ok=True)

average_export_path = (
    arcgis_dir / 'AI_SPEI_python_station_averages_861.csv'
)

arcgis_averages.to_csv(average_export_path, index=False)

print('Saved:', average_export_path.resolve())
print('Recent: WY 1996-2025; centennial: WY 1926-2025; prior: WY 1966-1995.')
display(arcgis_averages.groupby('variable').size().rename('stations'))

trend_path = TABLES / 'SPEI_station_modified_MK_trends.csv'

if not trend_path.exists():
    raise FileNotFoundError(f'Saved SPEI trends not found: {trend_path}')

spei_results = pd.read_csv(
    trend_path,
    dtype={'ID': str}
)

stations = pd.read_csv(
    'df_neon_filter.csv',
    usecols=['ID', 'Area'],
    dtype={'ID': str}
)

domains = pd.read_csv(
    'data_raw/ID_ecoregion.csv',
    usecols=['ID', 'domainName'],
    dtype={'ID': str}
)

domains['domainName'] = (
    domains.domainName.astype('string')
    .str.strip()
    .replace({
        'Atlantic Neotropical': 'Southeast',
        'Appalachians / Cumberland Plateau': 'Appalachians',
        'Southern Rockies / Colorado Plateau': 'Colorado Plateau'
    })
)

domains = domains[
    domains.domainName.notna()
    & domains.domainName.ne('')
].copy()

if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected exactly 861 unique study stations.')

if domains[domains.ID.isin(stations.ID)].ID.duplicated().any():
    raise ValueError(
        'Duplicate named ecoregion assignments for a study station.'
    )

metadata = stations.merge(
    domains,
    on='ID',
    how='left',
    validate='one_to_one'
)

metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')

if (
    metadata.domainName.isna().any()
    or (~np.isfinite(metadata.Area)).any()
    or (metadata.Area <= 0).any()
):
    raise ValueError(
        'Every study station must have an ecoregion and positive Thiessen area.'
    )

if spei_results.duplicated(['ID', 'scale', 'period']).any():
    raise ValueError(
        'Duplicate station, SPEI scale and period in trend results.'
    )

expected_ids = set(metadata.ID)

spei_results = spei_results[
    spei_results.ID.isin(expected_ids)
].copy()

expected_families = {
    (scale, period)
    for scale in (1, 3, 6, 12)
    for period in ('centennial', 'recent')
}

if set(zip(spei_results.scale, spei_results.period)) != expected_families:
    raise ValueError(
        'Expected four SPEI scales for both centennial and recent periods.'
    )

for key, group in spei_results.groupby(['scale', 'period']):
    if set(group.ID) != expected_ids:
        raise ValueError(
            f'Trend family {key} does not contain exactly the 861 study stations.'
        )

spei_results['p'] = pd.to_numeric(spei_results.p, errors='coerce')
spei_results['status'] = spei_results.trend

if not set(spei_results.status.dropna()).issubset(
    {'increasing', 'decreasing', 'no trend'}
):
    raise ValueError('Unexpected trend classification.')

spei_results = spei_results.merge(
    metadata,
    on='ID',
    how='left',
    validate='many_to_one'
)

station_rows, area_rows = ([], [])

for (scale, period), family in spei_results.groupby(
    ['scale', 'period'],
    sort=True
):
    groups = [('CONUS', family)] + list(
        family.groupby('domainName', sort=True)
    )

    for region, group in groups:
        record = {
            'region': region,
            'SPEI_scale': scale,
            'period': period,
            'start_WY': int(group.start_WY.iloc[0]),
            'end_WY': int(group.end_WY.iloc[0])
        }

        count = len(group)
        total_area = group.Area.sum()

        increasing = group.status.eq('increasing')
        decreasing = group.status.eq('decreasing')

        station_rows.append({
            **record,
            'total_stations': count,
            'significant_increasing_stations': int(increasing.sum()),
            'significant_decreasing_stations': int(decreasing.sum()),
            'significant_increasing_percent': 100 * increasing.sum() / count,
            'significant_decreasing_percent': 100 * decreasing.sum() / count
        })

        area_rows.append({
            **record,
            'total_area_km2': total_area,
            'significant_increasing_area_km2': group.loc[increasing, 'Area'].sum(),
            'significant_decreasing_area_km2': group.loc[decreasing, 'Area'].sum(),
            'significant_increasing_percent': (
                100 * group.loc[increasing, 'Area'].sum() / total_area
            ),
            'significant_decreasing_percent': (
                100 * group.loc[decreasing, 'Area'].sum() / total_area
            )
        })

output_columns = [
    'region',
    'SPEI_scale',
    'period',
    'start_WY',
    'end_WY',
    'significant_increasing_percent',
    'significant_decreasing_percent'
]

spei_station_percentages = pd.DataFrame(
    station_rows
)[output_columns].copy()

spei_area_percentages = pd.DataFrame(
    area_rows
)[output_columns].copy()

for table in (spei_station_percentages, spei_area_percentages):
    table[
        [
            'significant_increasing_percent',
            'significant_decreasing_percent'
        ]
    ] = table[
        [
            'significant_increasing_percent',
            'significant_decreasing_percent'
        ]
    ].round(1)

spei_station_percentages.to_csv(
    TABLES / 'SPEI_significant_trends_percent_stations_CONUS_NEON.csv',
    index=False,
    float_format='%.1f'
)

spei_area_percentages.to_csv(
    TABLES / 'SPEI_significant_trends_percent_area_CONUS_NEON.csv',
    index=False,
    float_format='%.1f'
)

print('Percentages use all stations or total Thiessen area in each region.')
print('Unadjusted modified Mann-Kendall results, alpha = 0.05; 861 CONUS stations.')
print('Station percentages:')
display(spei_station_percentages.round(2))
print('Area percentages:')
display(spei_area_percentages.round(2))

drought = pd.read_csv(
    TABLES / 'SPEI_station_drought_months_per_decade.csv',
    dtype={'ID': str}
)

stations = pd.read_csv(
    'df_neon_filter.csv',
    usecols=['ID', 'Area'],
    dtype={'ID': str}
)

if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')

stations['Area'] = pd.to_numeric(stations.Area, errors='coerce')

if (~np.isfinite(stations.Area)).any() or (stations.Area <= 0).any():
    raise ValueError('All study stations must have positive Thiessen areas.')

drought = drought[drought.ID.isin(set(stations.ID))].copy()

if drought.duplicated(['ID', 'scale', 'period']).any():
    raise ValueError('Duplicate station, scale and period in the drought table.')

expected = {
    (scale, period)
    for scale in (1, 3, 6, 12)
    for period in ('centennial', 'recent', 'prior')
}

if set(zip(drought.scale, drought.period)) != expected:
    raise ValueError('Expected all four SPEI scales and all three periods.')

for family, group in drought.groupby(['scale', 'period']):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'Incomplete station set for {family}.')

drought['drought_months'] = pd.to_numeric(
    drought.drought_months,
    errors='coerce'
)

drought['n_valid_months'] = pd.to_numeric(
    drought.n_valid_months,
    errors='coerce'
)

if drought.drought_months.isna().any() or (drought.n_valid_months < 12).any():
    raise ValueError(
        'At least one study station has an unavailable drought-frequency result.'
    )

counts = drought.pivot(
    index=['ID', 'scale'],
    columns='period',
    values='drought_months'
).reset_index()

counts = counts.merge(
    stations,
    on='ID',
    how='left',
    validate='many_to_one'
)

records = []

for scale, group in counts.groupby('scale', sort=True):
    total_area = group.Area.sum()

    for comparison, increased in [
        (
            'recent minus centennial',
            group.recent * 100 > group.centennial * 30
        ),
        (
            'recent minus prior',
            group.recent > group.prior
        )
    ]:
        records.append({
            'SPEI_scale': scale,
            'comparison': comparison,
            'recent_WY': '1996-2025',
            'reference_WY': (
                '1926-2025'
                if comparison.endswith('centennial')
                else '1966-1995'
            ),
            'CONUS_area_increase_percent': round(
                100 * group.loc[increased, 'Area'].sum() / total_area,
                1
            )
        })

drought_area_increase = pd.DataFrame(records)

output_path = (
    TABLES / 'SPEI_drought_frequency_increase_percent_CONUS_area.csv'
)

drought_area_increase.to_csv(
    output_path,
    index=False,
    float_format='%.1f'
)

print('Saved:', output_path.resolve())
display(drought_area_increase)

ai_results = pd.read_csv(
    TABLES / 'AI_station_modified_MK_trends.csv',
    dtype={'ID': str}
)

stations = pd.read_csv(
    'df_neon_filter.csv',
    usecols=['ID', 'Area'],
    dtype={'ID': str}
)

domains = pd.read_csv(
    'data_raw/ID_ecoregion.csv',
    usecols=['ID', 'domainName'],
    dtype={'ID': str}
)

domains['domainName'] = (
    domains.domainName.astype('string')
    .str.strip()
    .replace({
        'Atlantic Neotropical': 'Southeast',
        'Appalachians / Cumberland Plateau': 'Appalachians',
        'Southern Rockies / Colorado Plateau': 'Colorado Plateau'
    })
)

domains = domains[
    domains.domainName.notna()
    & domains.domainName.ne('')
].copy()

if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')

domains = domains[domains.ID.isin(stations.ID)]

if domains.ID.duplicated().any():
    raise ValueError(
        'Duplicate named ecoregions for a study station.'
    )

metadata = stations.merge(
    domains,
    on='ID',
    how='left',
    validate='one_to_one'
)

metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')

if (
    metadata.domainName.isna().any()
    or (~np.isfinite(metadata.Area)).any()
    or (metadata.Area <= 0).any()
):
    raise ValueError(
        'Each study station needs a named ecoregion and positive Thiessen area.'
    )

if ai_results.duplicated(['ID', 'period']).any():
    raise ValueError('Duplicate AI station-period trend results.')

expected_ids = set(metadata.ID)

ai_results = ai_results[
    ai_results.ID.isin(expected_ids)
].copy()

if set(ai_results.period) != {'centennial', 'recent'}:
    raise ValueError('Expected centennial and recent AI trend periods.')

for period, group in ai_results.groupby('period'):
    if set(group.ID) != expected_ids:
        raise ValueError(
            f'AI period {period} does not contain all 861 stations.'
        )

ai_results['p'] = pd.to_numeric(ai_results.p, errors='coerce')
ai_results['status'] = ai_results.trend

if not set(ai_results.status.dropna()).issubset(
    {'increasing', 'decreasing', 'no trend'}
):
    raise ValueError('Unexpected trend classification.')

ai_results = ai_results.merge(
    metadata,
    on='ID',
    how='left',
    validate='many_to_one'
)

station_rows, area_rows = ([], [])

for period, family in ai_results.groupby('period', sort=True):
    for region, group in [
        ('CONUS', family)
    ] + list(family.groupby('domainName', sort=True)):

        increasing = group.status.eq('increasing')
        decreasing = group.status.eq('decreasing')

        fields = {
            'region': region,
            'period': period,
            'start_WY': int(group.start_WY.iloc[0]),
            'end_WY': int(group.end_WY.iloc[0])
        }

        station_rows.append({
            **fields,
            'significant_increasing_percent': round(
                100 * increasing.sum() / len(group),
                1
            ),
            'significant_decreasing_percent': round(
                100 * decreasing.sum() / len(group),
                1
            )
        })

        area_rows.append({
            **fields,
            'significant_increasing_percent': round(
                100 * group.loc[increasing, 'Area'].sum() / group.Area.sum(),
                1
            ),
            'significant_decreasing_percent': round(
                100 * group.loc[decreasing, 'Area'].sum() / group.Area.sum(),
                1
            )
        })

ai_station_percentages = pd.DataFrame(station_rows)
ai_area_percentages = pd.DataFrame(area_rows)

station_path = (
    TABLES / 'AI_significant_trends_percent_stations_CONUS_NEON.csv'
)

area_path = (
    TABLES / 'AI_significant_trends_percent_area_CONUS_NEON.csv'
)

ai_station_percentages.to_csv(
    station_path,
    index=False,
    float_format='%.1f'
)

ai_area_percentages.to_csv(
    area_path,
    index=False,
    float_format='%.1f'
)

print('Saved:', station_path.resolve(), 'and', area_path.resolve())
print('Unadjusted modified Mann-Kendall, alpha = 0.05; all 861 study stations in CONUS denominator.')

display(ai_station_percentages)
display(ai_area_percentages)
