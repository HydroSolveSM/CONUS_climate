from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
import warnings
import pymannkendall as mk
from climate_indices import indices, compute
import shutil
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

def display(value):
    print(value)


OUTPUT = Path('supplementary_review')
trend_path = Path('AI_SPEI_python/Tables/AI_station_modified_MK_trends.csv')
stations_path = Path('df_neon_filter.csv')
domains_path = Path('data_raw/ID_ecoregion.csv')
source_tex = Path('scripts/supplementary_template.tex')
for path in (trend_path, stations_path, domains_path, source_tex):
    if not path.exists():
        raise FileNotFoundError(f'Required input not found: {path}')
trends = pd.read_csv(trend_path, dtype={'ID': str})
stations = pd.read_csv(stations_path, usecols=['ID', 'Area'], dtype={'ID': str})
domains = pd.read_csv(domains_path, usecols=['ID', 'domainName'], dtype={'ID': str})
domains['domainName'] = domains.domainName.astype('string').str.strip().replace({'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'})
domains = domains[domains.domainName.notna() & domains.domainName.ne('') & domains.ID.isin(stations.ID)]
if len(stations) != 861 or stations.ID.duplicated().any() or domains.ID.duplicated().any():
    raise ValueError('Expected 861 unique stations and unique named ecoregion assignments.')
metadata = stations.merge(domains, on='ID', how='left', validate='one_to_one')
metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')
if metadata.domainName.isna().any() or (~np.isfinite(metadata.Area)).any() or (metadata.Area <= 0).any():
    raise ValueError('Every station needs a named ecoregion and positive Thiessen area.')
if trends.duplicated(['ID', 'period']).any():
    raise ValueError('Duplicate AI station-period trend records.')
trends = trends[trends.ID.isin(set(metadata.ID))].copy()
if set(trends.period) != {'recent', 'centennial'}:
    raise ValueError('Expected recent and centennial AI periods.')
for period, group in trends.groupby('period'):
    if set(group.ID) != set(metadata.ID):
        raise ValueError(f'Missing study stations for {period}.')
trends['p'] = pd.to_numeric(trends.p, errors='coerce')
trends['slope_per_decade'] = pd.to_numeric(trends.slope_per_decade, errors='coerce')
valid = np.isfinite(trends.p) & trends.p.between(0, 1) & np.isfinite(trends.slope_per_decade)
trends['status'] = trends.trend.where(valid)
trends = trends.merge(metadata, on='ID', how='left', validate='many_to_one')
records = []
for period, family in trends.groupby('period', sort=True):
    for region, group in [('CONUS', family)] + list(family.groupby('domainName', sort=True)):
        for direction in ('increasing', 'decreasing'):
            subset = group[(group.status == direction) & (group.p < 0.05)]
            slopes = subset.slope_per_decade
            records.append({'region': region, 'period': period, 'direction': direction, 'total_stations': len(group), 'significant_stations': len(subset), 'total_area_km2': group.Area.sum(), 'significant_area_km2': subset.Area.sum(), 'percent_area': 100 * subset.Area.sum() / group.Area.sum(), 'slope_min': slopes.min(), 'slope_max': slopes.max(), 'slope_mean': slopes.mean(), 'slope_median': slopes.median()})
ai_long = pd.DataFrame(records)
region_order = ['CONUS'] + sorted(metadata.domainName.unique())
wide_rows = []
for region in region_order:
    row = {'region': region}
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            item = ai_long[(ai_long.region == region) & (ai_long.period == period) & (ai_long.direction == direction)].iloc[0]
            for metric in ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median'):
                row[f'{period}_{direction}_{metric}'] = item[metric]
    wide_rows.append(row)
ai_table_s8 = pd.DataFrame(wide_rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
ai_long.to_csv(OUTPUT / 'AI_Table_S8_significant_trends_long_review.csv', index=False)
ai_table_s8.to_csv(OUTPUT / 'AI_Table_S8_regional_trends_review.csv', index=False)
print('AI S8 summary saved:', len(ai_table_s8), 'regions; 861 stations; unadjusted p < 0.05.')
display(ai_table_s8.round(2))

text = source_tex.read_text(encoding='utf-8')
caption_marker = '\\small Table S8:'
end_marker = '\\end{landscape}'
start = text.find(caption_marker)
if start < 0:
    raise ValueError('Could not find Table S8 in the original supplementary source.')
table_start = text.rfind('\\begin{landscape}', 0, start)
table_end = text.find(end_marker, start)
if table_start < 0 or table_end < 0:
    raise ValueError('Could not identify the full Table S8 landscape block.')
table_end += len(end_marker)
old_block = text[table_start:table_end]
head_end = old_block.find('\\midrule')
body_end = old_block.find('\\bottomrule', head_end)
if head_end < 0 or body_end < 0:
    raise ValueError('Could not identify Table S8 data rows.')
header = old_block[:head_end + len('\\midrule')]
footer = old_block[body_end:]
metrics = ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median')

def format_value(value):
    if pd.isna(value):
        return ''
    if abs(value) < 0.005:
        value = 0.0
    return f'{value:.2f}'
lines = []
for number, row in enumerate(ai_table_s8.to_dict('records'), 1):
    fields = [str(number), row['region']]
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            fields.extend((format_value(row[f'{period}_{direction}_{metric}']) for metric in metrics))
    lines.append(' & '.join(fields) + ' \\\\')
review_block = header + '\n' + '\n'.join(lines) + '\n' + footer
review_text = text[:table_start] + review_block + text[table_end:]
review_text = review_text.replace('{figures/', '{../supplementary/figures/')
review_path = OUTPUT / 'Supplementary_review.tex'
review_path.write_text(review_text, encoding='utf-8')
(OUTPUT / 'AI_Table_S8_review.tex').write_text(review_block + '\n', encoding='utf-8')
print('Updated Table S8 in:', review_path.resolve())
print('Supplementary source updated.')

OUTPUT = Path('supplementary_review')
trend_path = Path('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv')
review_path = OUTPUT / 'Supplementary_review.tex'
if not trend_path.exists() or not review_path.exists():
    raise FileNotFoundError('Saved SPEI trends or Supplementary_review.tex is missing; run Table S8 section first.')
trends = pd.read_csv(trend_path, dtype={'ID': str})
trends = trends[trends.scale == 1].copy()
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Area'], dtype={'ID': str})
domains = pd.read_csv('data_raw/ID_ecoregion.csv', usecols=['ID', 'domainName'], dtype={'ID': str})
domains['domainName'] = domains.domainName.astype('string').str.strip().replace({'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'})
domains = domains[domains.domainName.notna() & domains.domainName.ne('') & domains.ID.isin(stations.ID)]
if len(stations) != 861 or stations.ID.duplicated().any() or domains.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations and named ecoregions.')
metadata = stations.merge(domains, on='ID', how='left', validate='one_to_one')
metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')
if metadata.domainName.isna().any() or (~np.isfinite(metadata.Area)).any() or (metadata.Area <= 0).any():
    raise ValueError('Each station needs an ecoregion and positive Thiessen area.')
if trends.duplicated(['ID', 'period']).any():
    raise ValueError('Duplicate SPEI-1 station-period records.')
trends = trends[trends.ID.isin(set(metadata.ID))].copy()
if set(trends.period) != {'recent', 'centennial'}:
    raise ValueError('Expected recent and centennial SPEI-1 periods.')
for period, group in trends.groupby('period'):
    if set(group.ID) != set(metadata.ID):
        raise ValueError(f'Missing study stations for {period}.')
for col in ['p', 'slope_per_step', 'slope_per_decade']:
    trends[col] = pd.to_numeric(trends[col], errors='coerce')
finite_slopes = np.isfinite(trends.slope_per_step) & np.isfinite(trends.slope_per_decade)
if not np.allclose(trends.loc[finite_slopes, 'slope_per_decade'], 120 * trends.loc[finite_slopes, 'slope_per_step'], rtol=1e-10, atol=1e-12):
    raise ValueError('Saved per-decade SPEI slopes do not equal monthly slopes times 120.')
valid = np.isfinite(trends.p) & trends.p.between(0, 1) & finite_slopes
trends['status'] = trends.trend.where(valid)
trends = trends.merge(metadata, on='ID', how='left', validate='many_to_one')
records = []
for period, family in trends.groupby('period', sort=True):
    for region, group in [('CONUS', family)] + list(family.groupby('domainName', sort=True)):
        for direction in ('increasing', 'decreasing'):
            subset = group[(group.status == direction) & (group.p < 0.05)]
            slopes = subset.slope_per_decade
            records.append({'region': region, 'period': period, 'direction': direction, 'total_stations': len(group), 'significant_stations': len(subset), 'total_area_km2': group.Area.sum(), 'significant_area_km2': subset.Area.sum(), 'percent_area': 100 * subset.Area.sum() / group.Area.sum(), 'slope_min': slopes.min(), 'slope_max': slopes.max(), 'slope_mean': slopes.mean(), 'slope_median': slopes.median()})
spei1_long = pd.DataFrame(records)
region_order = ['CONUS'] + sorted(metadata.domainName.unique())
wide_rows = []
for region in region_order:
    row = {'region': region}
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            item = spei1_long[(spei1_long.region == region) & (spei1_long.period == period) & (spei1_long.direction == direction)].iloc[0]
            for metric in ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median'):
                row[f'{period}_{direction}_{metric}'] = item[metric]
    wide_rows.append(row)
spei1_table_s9 = pd.DataFrame(wide_rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
spei1_long.to_csv(OUTPUT / 'SPEI1_Table_S9_significant_trends_long_review.csv', index=False)
spei1_table_s9.to_csv(OUTPUT / 'SPEI1_Table_S9_regional_trends_review.csv', index=False)
print('Saved SPEI-1 S9 summary; verified slope per decade = 120 x slope per month.')
display(spei1_table_s9.round(2))

text = review_path.read_text(encoding='utf-8')
caption_marker = '\\small Table S9:'
start = text.find(caption_marker)
table_start = text.rfind('\\begin{landscape}', 0, start)
table_end = text.find('\\end{landscape}', start)
if start < 0 or table_start < 0 or table_end < 0:
    raise ValueError('Could not identify the Table S9 landscape block.')
table_end += len('\\end{landscape}')
block = text[table_start:table_end]
head_end = block.find('\\midrule')
body_end = block.find('\\bottomrule', head_end)
if head_end < 0 or body_end < 0:
    raise ValueError('Could not identify Table S9 data rows.')
header = block[:head_end + len('\\midrule')]
footer = block[body_end:]
metrics = ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median')

def format_spei1_value(value):
    if pd.isna(value):
        return ''
    if abs(value) < 0.005:
        value = 0.0
    return f'{value:.2f}'
lines = []
for number, row in enumerate(spei1_table_s9.to_dict('records'), 1):
    fields = [str(number), row['region']]
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            fields.extend((format_spei1_value(row[f'{period}_{direction}_{metric}']) for metric in metrics))
    lines.append(' & '.join(fields) + ' \\\\')
review_block = header + '\n' + '\n'.join(lines) + '\n' + footer
review_path.write_text(text[:table_start] + review_block + text[table_end:], encoding='utf-8')
(OUTPUT / 'SPEI1_Table_S9_review.tex').write_text(review_block + '\n', encoding='utf-8')
print('Updated Table S9 in:', review_path.resolve())

OUTPUT = Path('supplementary_review')
trend_path = Path('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv')
review_path = OUTPUT / 'Supplementary_review.tex'
if not trend_path.exists() or not review_path.exists():
    raise FileNotFoundError('Saved SPEI trends or Supplementary_review.tex is missing; run Table S8 section first.')
trends = pd.read_csv(trend_path, dtype={'ID': str})
trends = trends[trends.scale == 3].copy()
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Area'], dtype={'ID': str})
domains = pd.read_csv('data_raw/ID_ecoregion.csv', usecols=['ID', 'domainName'], dtype={'ID': str})
domains['domainName'] = domains.domainName.astype('string').str.strip().replace({'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'})
domains = domains[domains.domainName.notna() & domains.domainName.ne('') & domains.ID.isin(stations.ID)]
if len(stations) != 861 or stations.ID.duplicated().any() or domains.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations and named ecoregions.')
metadata = stations.merge(domains, on='ID', how='left', validate='one_to_one')
metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')
if metadata.domainName.isna().any() or (~np.isfinite(metadata.Area)).any() or (metadata.Area <= 0).any():
    raise ValueError('Each station needs an ecoregion and positive Thiessen area.')
if trends.duplicated(['ID', 'period']).any():
    raise ValueError('Duplicate SPEI-3 station-period records.')
trends = trends[trends.ID.isin(set(metadata.ID))].copy()
if set(trends.period) != {'recent', 'centennial'}:
    raise ValueError('Expected recent and centennial SPEI-3 periods.')
for period, group in trends.groupby('period'):
    if set(group.ID) != set(metadata.ID):
        raise ValueError(f'Missing study stations for {period}.')
for col in ['p', 'slope_per_step', 'slope_per_decade']:
    trends[col] = pd.to_numeric(trends[col], errors='coerce')
finite_slopes = np.isfinite(trends.slope_per_step) & np.isfinite(trends.slope_per_decade)
if not np.allclose(trends.loc[finite_slopes, 'slope_per_decade'], 120 * trends.loc[finite_slopes, 'slope_per_step'], rtol=1e-10, atol=1e-12):
    raise ValueError('Saved per-decade SPEI slopes do not equal monthly slopes times 120.')
valid = np.isfinite(trends.p) & trends.p.between(0, 1) & finite_slopes
trends['status'] = trends.trend.where(valid)
trends = trends.merge(metadata, on='ID', how='left', validate='many_to_one')
records = []
for period, family in trends.groupby('period', sort=True):
    for region, group in [('CONUS', family)] + list(family.groupby('domainName', sort=True)):
        for direction in ('increasing', 'decreasing'):
            subset = group[(group.status == direction) & (group.p < 0.05)]
            slopes = subset.slope_per_decade
            records.append({'region': region, 'period': period, 'direction': direction, 'total_stations': len(group), 'significant_stations': len(subset), 'total_area_km2': group.Area.sum(), 'significant_area_km2': subset.Area.sum(), 'percent_area': 100 * subset.Area.sum() / group.Area.sum(), 'slope_min': slopes.min(), 'slope_max': slopes.max(), 'slope_mean': slopes.mean(), 'slope_median': slopes.median()})
spei3_long = pd.DataFrame(records)
region_order = ['CONUS'] + sorted(metadata.domainName.unique())
wide_rows = []
for region in region_order:
    row = {'region': region}
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            item = spei3_long[(spei3_long.region == region) & (spei3_long.period == period) & (spei3_long.direction == direction)].iloc[0]
            for metric in ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median'):
                row[f'{period}_{direction}_{metric}'] = item[metric]
    wide_rows.append(row)
spei3_table_s9 = pd.DataFrame(wide_rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
spei3_long.to_csv(OUTPUT / 'SPEI3_Table_S10_significant_trends_long_review.csv', index=False)
spei3_table_s9.to_csv(OUTPUT / 'SPEI3_Table_S10_regional_trends_review.csv', index=False)
print('Saved SPEI-3 S10 summary; verified slope per decade = 120 x slope per month.')
display(spei3_table_s9.round(2))

text = review_path.read_text(encoding='utf-8')
caption_marker = '\\small Table S10:'
start = text.find(caption_marker)
table_start = text.rfind('\\begin{landscape}', 0, start)
table_end = text.find('\\end{landscape}', start)
if start < 0 or table_start < 0 or table_end < 0:
    raise ValueError('Could not identify the Table S10 landscape block.')
table_end += len('\\end{landscape}')
block = text[table_start:table_end]
head_end = block.find('\\midrule')
body_end = block.find('\\bottomrule', head_end)
if head_end < 0 or body_end < 0:
    raise ValueError('Could not identify Table S10 data rows.')
header = block[:head_end + len('\\midrule')]
footer = block[body_end:]
metrics = ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median')

def format_spei3_value(value):
    if pd.isna(value):
        return ''
    if abs(value) < 0.005:
        value = 0.0
    return f'{value:.2f}'
lines = []
for number, row in enumerate(spei3_table_s9.to_dict('records'), 1):
    fields = [str(number), row['region']]
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            fields.extend((format_spei3_value(row[f'{period}_{direction}_{metric}']) for metric in metrics))
    lines.append(' & '.join(fields) + ' \\\\')
review_block = header + '\n' + '\n'.join(lines) + '\n' + footer
review_path.write_text(text[:table_start] + review_block + text[table_end:], encoding='utf-8')
(OUTPUT / 'SPEI3_Table_S10_review.tex').write_text(review_block + '\n', encoding='utf-8')
print('Updated Table S10 in:', review_path.resolve())

OUTPUT = Path('supplementary_review')
trend_path = Path('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv')
review_path = OUTPUT / 'Supplementary_review.tex'
if not trend_path.exists() or not review_path.exists():
    raise FileNotFoundError('Saved SPEI trends or Supplementary_review.tex is missing; run Table S8 section first.')
trends = pd.read_csv(trend_path, dtype={'ID': str})
trends = trends[trends.scale == 6].copy()
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Area'], dtype={'ID': str})
domains = pd.read_csv('data_raw/ID_ecoregion.csv', usecols=['ID', 'domainName'], dtype={'ID': str})
domains['domainName'] = domains.domainName.astype('string').str.strip().replace({'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'})
domains = domains[domains.domainName.notna() & domains.domainName.ne('') & domains.ID.isin(stations.ID)]
if len(stations) != 861 or stations.ID.duplicated().any() or domains.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations and named ecoregions.')
metadata = stations.merge(domains, on='ID', how='left', validate='one_to_one')
metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')
if metadata.domainName.isna().any() or (~np.isfinite(metadata.Area)).any() or (metadata.Area <= 0).any():
    raise ValueError('Each station needs an ecoregion and positive Thiessen area.')
if trends.duplicated(['ID', 'period']).any():
    raise ValueError('Duplicate SPEI-6 station-period records.')
trends = trends[trends.ID.isin(set(metadata.ID))].copy()
if set(trends.period) != {'recent', 'centennial'}:
    raise ValueError('Expected recent and centennial SPEI-6 periods.')
for period, group in trends.groupby('period'):
    if set(group.ID) != set(metadata.ID):
        raise ValueError(f'Missing study stations for {period}.')
for col in ['p', 'slope_per_step', 'slope_per_decade']:
    trends[col] = pd.to_numeric(trends[col], errors='coerce')
finite_slopes = np.isfinite(trends.slope_per_step) & np.isfinite(trends.slope_per_decade)
if not np.allclose(trends.loc[finite_slopes, 'slope_per_decade'], 120 * trends.loc[finite_slopes, 'slope_per_step'], rtol=1e-10, atol=1e-12):
    raise ValueError('Saved per-decade SPEI slopes do not equal monthly slopes times 120.')
valid = np.isfinite(trends.p) & trends.p.between(0, 1) & finite_slopes
trends['status'] = trends.trend.where(valid)
trends = trends.merge(metadata, on='ID', how='left', validate='many_to_one')
records = []
for period, family in trends.groupby('period', sort=True):
    for region, group in [('CONUS', family)] + list(family.groupby('domainName', sort=True)):
        for direction in ('increasing', 'decreasing'):
            subset = group[(group.status == direction) & (group.p < 0.05)]
            slopes = subset.slope_per_decade
            records.append({'region': region, 'period': period, 'direction': direction, 'total_stations': len(group), 'significant_stations': len(subset), 'total_area_km2': group.Area.sum(), 'significant_area_km2': subset.Area.sum(), 'percent_area': 100 * subset.Area.sum() / group.Area.sum(), 'slope_min': slopes.min(), 'slope_max': slopes.max(), 'slope_mean': slopes.mean(), 'slope_median': slopes.median()})
spei6_long = pd.DataFrame(records)
region_order = ['CONUS'] + sorted(metadata.domainName.unique())
wide_rows = []
for region in region_order:
    row = {'region': region}
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            item = spei6_long[(spei6_long.region == region) & (spei6_long.period == period) & (spei6_long.direction == direction)].iloc[0]
            for metric in ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median'):
                row[f'{period}_{direction}_{metric}'] = item[metric]
    wide_rows.append(row)
spei6_table_s9 = pd.DataFrame(wide_rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
spei6_long.to_csv(OUTPUT / 'SPEI6_Table_S11_significant_trends_long_review.csv', index=False)
spei6_table_s9.to_csv(OUTPUT / 'SPEI6_Table_S11_regional_trends_review.csv', index=False)
print('Saved SPEI-6 S11 summary; verified slope per decade = 120 x slope per month.')
display(spei6_table_s9.round(2))

text = review_path.read_text(encoding='utf-8')
caption_marker = '\\small Table S11:'
start = text.find(caption_marker)
table_start = text.rfind('\\begin{landscape}', 0, start)
table_end = text.find('\\end{landscape}', start)
if start < 0 or table_start < 0 or table_end < 0:
    raise ValueError('Could not identify the Table S11 landscape block.')
table_end += len('\\end{landscape}')
block = text[table_start:table_end]
head_end = block.find('\\midrule')
body_end = block.find('\\bottomrule', head_end)
if head_end < 0 or body_end < 0:
    raise ValueError('Could not identify Table S11 data rows.')
header = block[:head_end + len('\\midrule')]
footer = block[body_end:]
metrics = ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median')

def format_spei6_value(value):
    if pd.isna(value):
        return ''
    if abs(value) < 0.005:
        value = 0.0
    return f'{value:.2f}'
lines = []
for number, row in enumerate(spei6_table_s9.to_dict('records'), 1):
    fields = [str(number), row['region']]
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            fields.extend((format_spei6_value(row[f'{period}_{direction}_{metric}']) for metric in metrics))
    lines.append(' & '.join(fields) + ' \\\\')
review_block = header + '\n' + '\n'.join(lines) + '\n' + footer
review_path.write_text(text[:table_start] + review_block + text[table_end:], encoding='utf-8')
(OUTPUT / 'SPEI6_Table_S11_review.tex').write_text(review_block + '\n', encoding='utf-8')
print('Updated Table S11 in:', review_path.resolve())

OUTPUT = Path('supplementary_review')
trend_path = Path('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv')
review_path = OUTPUT / 'Supplementary_review.tex'
if not trend_path.exists() or not review_path.exists():
    raise FileNotFoundError('Saved SPEI trends or Supplementary_review.tex is missing; run Table S8 section first.')
trends = pd.read_csv(trend_path, dtype={'ID': str})
trends = trends[trends.scale == 12].copy()
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Area'], dtype={'ID': str})
domains = pd.read_csv('data_raw/ID_ecoregion.csv', usecols=['ID', 'domainName'], dtype={'ID': str})
domains['domainName'] = domains.domainName.astype('string').str.strip().replace({'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'})
domains = domains[domains.domainName.notna() & domains.domainName.ne('') & domains.ID.isin(stations.ID)]
if len(stations) != 861 or stations.ID.duplicated().any() or domains.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations and named ecoregions.')
metadata = stations.merge(domains, on='ID', how='left', validate='one_to_one')
metadata['Area'] = pd.to_numeric(metadata.Area, errors='coerce')
if metadata.domainName.isna().any() or (~np.isfinite(metadata.Area)).any() or (metadata.Area <= 0).any():
    raise ValueError('Each station needs an ecoregion and positive Thiessen area.')
if trends.duplicated(['ID', 'period']).any():
    raise ValueError('Duplicate SPEI-12 station-period records.')
trends = trends[trends.ID.isin(set(metadata.ID))].copy()
if set(trends.period) != {'recent', 'centennial'}:
    raise ValueError('Expected recent and centennial SPEI-12 periods.')
for period, group in trends.groupby('period'):
    if set(group.ID) != set(metadata.ID):
        raise ValueError(f'Missing study stations for {period}.')
for col in ['p', 'slope_per_step', 'slope_per_decade']:
    trends[col] = pd.to_numeric(trends[col], errors='coerce')
finite_slopes = np.isfinite(trends.slope_per_step) & np.isfinite(trends.slope_per_decade)
if not np.allclose(trends.loc[finite_slopes, 'slope_per_decade'], 120 * trends.loc[finite_slopes, 'slope_per_step'], rtol=1e-10, atol=1e-12):
    raise ValueError('Saved per-decade SPEI slopes do not equal monthly slopes times 120.')
valid = np.isfinite(trends.p) & trends.p.between(0, 1) & finite_slopes
trends['status'] = trends.trend.where(valid)
trends = trends.merge(metadata, on='ID', how='left', validate='many_to_one')
records = []
for period, family in trends.groupby('period', sort=True):
    for region, group in [('CONUS', family)] + list(family.groupby('domainName', sort=True)):
        for direction in ('increasing', 'decreasing'):
            subset = group[(group.status == direction) & (group.p < 0.05)]
            slopes = subset.slope_per_decade
            records.append({'region': region, 'period': period, 'direction': direction, 'total_stations': len(group), 'significant_stations': len(subset), 'total_area_km2': group.Area.sum(), 'significant_area_km2': subset.Area.sum(), 'percent_area': 100 * subset.Area.sum() / group.Area.sum(), 'slope_min': slopes.min(), 'slope_max': slopes.max(), 'slope_mean': slopes.mean(), 'slope_median': slopes.median()})
spei12_long = pd.DataFrame(records)
region_order = ['CONUS'] + sorted(metadata.domainName.unique())
wide_rows = []
for region in region_order:
    row = {'region': region}
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            item = spei12_long[(spei12_long.region == region) & (spei12_long.period == period) & (spei12_long.direction == direction)].iloc[0]
            for metric in ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median'):
                row[f'{period}_{direction}_{metric}'] = item[metric]
    wide_rows.append(row)
spei12_table_s9 = pd.DataFrame(wide_rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
spei12_long.to_csv(OUTPUT / 'SPEI12_Table_S12_significant_trends_long_review.csv', index=False)
spei12_table_s9.to_csv(OUTPUT / 'SPEI12_Table_S12_regional_trends_review.csv', index=False)
print('Saved SPEI-12 S12 summary; verified slope per decade = 120 x slope per month.')
display(spei12_table_s9.round(2))

text = review_path.read_text(encoding='utf-8')
caption_marker = '\\small Table S12:'
start = text.find(caption_marker)
table_start = text.rfind('\\begin{landscape}', 0, start)
table_end = text.find('\\end{landscape}', start)
if start < 0 or table_start < 0 or table_end < 0:
    raise ValueError('Could not identify the Table S12 landscape block.')
table_end += len('\\end{landscape}')
block = text[table_start:table_end]
head_end = block.find('\\midrule')
body_end = block.find('\\bottomrule', head_end)
if head_end < 0 or body_end < 0:
    raise ValueError('Could not identify Table S12 data rows.')
header = block[:head_end + len('\\midrule')]
footer = block[body_end:]
metrics = ('percent_area', 'slope_min', 'slope_max', 'slope_mean', 'slope_median')

def format_spei12_value(value):
    if pd.isna(value):
        return ''
    if abs(value) < 0.005:
        value = 0.0
    return f'{value:.2f}'
lines = []
for number, row in enumerate(spei12_table_s9.to_dict('records'), 1):
    fields = [str(number), row['region']]
    for period in ('recent', 'centennial'):
        for direction in ('increasing', 'decreasing'):
            fields.extend((format_spei12_value(row[f'{period}_{direction}_{metric}']) for metric in metrics))
    lines.append(' & '.join(fields) + ' \\\\')
review_block = header + '\n' + '\n'.join(lines) + '\n' + footer
review_path.write_text(text[:table_start] + review_block + text[table_end:], encoding='utf-8')
(OUTPUT / 'SPEI12_Table_S12_review.tex').write_text(review_block + '\n', encoding='utf-8')
print('Updated Table S12 in:', review_path.resolve())

OUTPUT = Path('supplementary_review')
trend_path = Path('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv')
review_path = OUTPUT / 'Supplementary_review.tex'
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
trends = pd.read_csv(trend_path, dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
trends = trends[(trends.period == 'recent') & trends.ID.isin(set(stations.ID)) & trends.scale.isin([1, 3, 6, 12])].copy()
if trends.duplicated(['ID', 'scale']).any():
    raise ValueError('Duplicate station-scale trends.')
for scale, group in trends.groupby('scale'):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'SPEI-{scale} does not contain all 861 stations.')
if set(trends.scale) != {1, 3, 6, 12}:
    raise ValueError('Missing one or more SPEI scales.')
trends['p'] = pd.to_numeric(trends.p, errors='coerce')
valid = np.isfinite(trends.p) & trends.p.between(0, 1)
trends['significant_direction'] = 'none'
for direction in ('increasing', 'decreasing'):
    trends.loc[valid & trends.p.lt(0.05) & trends.trend.eq(direction), 'significant_direction'] = direction
wide = trends.pivot(index='ID', columns='scale', values='significant_direction').reindex(columns=[1, 3, 6, 12])
station_classes = stations.merge(wide.reset_index(), on='ID', how='left', validate='one_to_one')
station_classes['n_increasing'] = station_classes[[1, 3, 6, 12]].eq('increasing').sum(axis=1)
station_classes['n_decreasing'] = station_classes[[1, 3, 6, 12]].eq('decreasing').sum(axis=1)
station_classes['category'] = 'not displayed'
station_classes.loc[station_classes.n_increasing.eq(4), 'category'] = 'all increasing'
station_classes.loc[station_classes.n_decreasing.eq(4), 'category'] = 'all decreasing'
station_classes.loc[station_classes.n_decreasing.between(1, 3), 'category'] = 'one to three decreasing'
OUTPUT.mkdir(parents=True, exist_ok=True)
station_classes.rename(columns={1: 'SPEI1', 3: 'SPEI3', 6: 'SPEI6', 12: 'SPEI12'}).to_csv(OUTPUT / 'Figure_S1_SPEI_consistency_station_classes_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
shape.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.65)
styles = [('all increasing', '^', '#17851b', 'All SPEI increasing'), ('all decreasing', 'v', '#f01818', 'All SPEI decreasing'), ('one to three decreasing', 'v', '#f4a261', 'One to three SPEI decreasing')]
handles = []
for category, marker, color, label in styles:
    subset = station_classes[station_classes.category == category].dropna(subset=['Lat', 'Lon'])
    ax.scatter(subset.Lon, subset.Lat, marker=marker, s=42, c=color, edgecolors='none', zorder=3)
    handles.append(Line2D([0], [0], marker=marker, linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=9, label=f'{label} (n={len(subset)})'))
ax.set_title('SPEI (1, 3, 6, 12) consistency stations', fontsize=17)
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')
ax.legend(handles=handles, loc='lower left', fontsize=12, frameon=True)
fig.savefig(OUTPUT / 'Figure_S1_SPEI_consistency_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S1_SPEI_consistency_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print('Category counts:')
print(station_classes.category.value_counts().to_string())
if not review_path.exists():
    raise FileNotFoundError('Supplementary_review.tex not found; run the S8 review section first.')
text = review_path.read_text(encoding='utf-8')
old = '{../supplementary/figures/image1.jpeg}'
new = '{Figure_S1_SPEI_consistency_review.png}'
if text.count(old) == 1 and text.count(new) == 0:
    review_path.write_text(text.replace(old, new, 1), encoding='utf-8')
elif text.count(old) == 0 and text.count(new) == 1:
    pass
else:
    raise ValueError('Could not uniquely identify Figure S1 in the review source.')
print('Updated Figure S1 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
PET_TABLES = Path('Reviewer_PET_Sensitivity/Tables')
AI_TABLES = Path('AI_SPEI_python/Tables')
review_path = OUTPUT / 'Supplementary_review.tex'
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
paired_means = pd.read_csv(PET_TABLES / 'AI_station_period_means_paired_WY1926_2025_WY1996_2025.csv', dtype={'ID': str})
current_means = pd.read_csv(AI_TABLES / 'AI_station_period_means.csv', dtype={'ID': str})
current_trends = pd.read_csv(AI_TABLES / 'AI_station_modified_MK_trends.csv', dtype={'ID': str})
pet_trends = pd.read_csv(PET_TABLES / 'AI_station_modified_MK_trends_WY1926_2025_WY1996_2025.csv', dtype={'ID': str})
periods = [('centennial', 1926, 2025), ('recent', 1996, 2025)]
expected_ids = set(stations.ID)
regression_rows, plot_means = ([], [])
for name, start, end in periods:
    pair = paired_means[(paired_means.start_WY == start) & (paired_means.end_WY == end)].copy()
    current = current_means[current_means.period == name][['ID', 'mean_AI']]
    if pair.ID.duplicated().any() or current.ID.duplicated().any() or set(pair.ID) != expected_ids or (set(current.ID) != expected_ids):
        raise ValueError(f'Incomplete or duplicated paired AI station means for {name}.')
    if not pair.n_paired_WY.eq(end - start + 1).all():
        raise ValueError(f'Not all paired station-years are available for {name}; review before comparing.')
    pair = pair.merge(current, on='ID', validate='one_to_one')
    if not np.allclose(pair.AI_Thornthwaite, pair.mean_AI, rtol=1e-09, atol=1e-10):
        raise ValueError(f'PET sensitivity Thornthwaite AI does not match current AI analysis for {name}.')
    fit = linregress(pair.mean_AI, pair.AI_Hargreaves)
    regression_rows.append({'period': name, 'start_WY': start, 'end_WY': end, 'stations': len(pair), 'slope': fit.slope, 'intercept': fit.intercept, 'r_squared': fit.rvalue ** 2})
    pair['period'] = name
    plot_means.append(pair)
ai_pet_means = pd.concat(plot_means, ignore_index=True)
ai_pet_regression = pd.DataFrame(regression_rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
ai_pet_regression.to_csv(OUTPUT / 'AI_PET_regression_statistics_review.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
for i, (name, start, end) in enumerate(periods):
    ax = axes[i]
    panel = ai_pet_means[ai_pet_means.period == name]
    fit = ai_pet_regression[ai_pet_regression.period == name].iloc[0]
    x = panel.mean_AI.to_numpy()
    ax.scatter(x, panel.AI_Hargreaves, s=17, color='#1a9850', alpha=0.55, edgecolors='none')
    line_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(line_x, fit.slope * line_x + fit.intercept, color='#c9142a', linewidth=1.8)
    ax.text(0.05, 0.95, f'y = {fit.slope:.3f}x {fit.intercept:+.3f}\nR-squared = {fit.r_squared:.3f}', transform=ax.transAxes, va='top', fontsize=13)
    ax.set_title(f'{chr(97 + i)}) AI (WY {start}-{end})', loc='left', fontsize=16)
    ax.set_xlabel('Mean AI (from Thornthwaite)', fontsize=14)
    ax.set_ylabel('Mean AI (from Hargreaves-Samani)', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.2)
fig.savefig(OUTPUT / 'Figure_S6_AI_PET_regression_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S6_AI_PET_regression_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
trend_panels = []
for name, start, end in periods:
    t = current_trends[current_trends.period == name][['ID', 'trend', 'p', 'slope_per_step']].copy()
    t = t.rename(columns={'slope_per_step': 'slope'})
    h = pet_trends[(pet_trends.method == 'Hargreaves') & (pet_trends.start_WY == start) & (pet_trends.end_WY == end)][['ID', 'trend', 'p', 'slope_per_year', 'n_paired_WY']].copy()
    h = h.rename(columns={'slope_per_year': 'slope'})
    if not h.n_paired_WY.eq(end - start + 1).all():
        raise ValueError(f'Incomplete Hargreaves paired years for {name}.')
    for method, frame in [('Thornthwaite', t), ('Hargreaves-Samani', h)]:
        if frame.ID.duplicated().any() or set(frame.ID) != expected_ids:
            raise ValueError(f'Incomplete or duplicated {method} trend stations for {name}.')
        frame['p'] = pd.to_numeric(frame.p, errors='coerce')
        frame['slope'] = pd.to_numeric(frame.slope, errors='coerce')
        valid = np.isfinite(frame.p) & frame.p.between(0, 1)
        frame['category'] = None
        frame.loc[valid & frame.p.ge(0.05), 'category'] = 'no trend'
        frame.loc[valid & frame.p.lt(0.05) & frame.trend.eq('increasing'), 'category'] = 'increasing'
        frame.loc[valid & frame.p.lt(0.05) & frame.trend.eq('decreasing'), 'category'] = 'decreasing'
        frame['method'], frame['period'] = (method, name)
        trend_panels.append(frame[['ID', 'method', 'period', 'slope', 'p', 'category']])
ai_pet_trends = pd.concat(trend_panels, ignore_index=True).merge(stations, on='ID', validate='many_to_one')
ai_pet_trends.to_csv(OUTPUT / 'AI_PET_trend_comparison_stations_review.csv', index=False)
trend_summary = []
for (method, period), group in ai_pet_trends.groupby(['method', 'period']):
    if not np.isfinite(group.slope).all():
        raise ValueError(f'Missing Sen slopes for {method}, {period}.')
    trend_summary.append({'method': method, 'period': period, 'stations': len(group), 'slope_increasing_percent': 100 * group.slope.gt(0).sum() / len(group), 'significant_increasing_percent': 100 * group.category.eq('increasing').sum() / len(group), 'slope_decreasing_percent': 100 * group.slope.lt(0).sum() / len(group), 'significant_decreasing_percent': 100 * group.category.eq('decreasing').sum() / len(group)})
ai_pet_summary = pd.DataFrame(trend_summary)
ai_pet_summary.to_csv(OUTPUT / 'AI_PET_trend_comparison_summary_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
colors = {'decreasing': '#d73027', 'no trend': '#bdbdbd', 'increasing': '#1a9850'}
fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
for col, (name, start, end) in enumerate(periods):
    for row, method in enumerate(['Thornthwaite', 'Hargreaves-Samani']):
        ax = axes[row, col]
        panel = ai_pet_trends[(ai_pet_trends.period == name) & (ai_pet_trends.method == method)]
        shape.plot(ax=ax, facecolor='#f3f3f3', edgecolor='black', linewidth=0.7)
        for category, color in colors.items():
            subset = panel[panel.category == category]
            ax.scatter(subset.Lon, subset.Lat, color=color, s=16, edgecolors='none')
        ax.set_title(f'{chr(97 + row * 2 + col)}) AI (from {method}), WY {start}-{end}', fontsize=15, loc='left')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
handles = [Line2D([0], [0], marker='o', linestyle='none', color=color, label=category.title(), markersize=7) for category, color in colors.items()]
fig.legend(handles=handles, title='Modified Mann-Kendall trend', loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=12, title_fontsize=13)
fig.savefig(OUTPUT / 'Figure_S7_AI_PET_trends_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S7_AI_PET_trends_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print('Regression statistics:')
display(ai_pet_regression)
print('Trend percentages:')
display(ai_pet_summary)

text = review_path.read_text(encoding='utf-8')
start = text.find('\\section*{S2.2. Sensitivity of the Aridity Index}')
end = text.find('\\section*{S2.3. Sensitivity of SPEI-1}', start)
if start < 0 or end < 0:
    raise ValueError('Could not uniquely identify Section S2.2.')

def percentage(value):
    return f'{value:.0f}\\%'

def result(method, period):
    row = ai_pet_summary[(ai_pet_summary.method == method) & (ai_pet_summary.period == period)]
    if len(row) != 1:
        raise ValueError('Trend summary family is not unique.')
    return row.iloc[0]

def wording(method_label, row):
    return f'{method_label} increasing slopes occurred at {percentage(row.slope_increasing_percent)} of stations, with statistically significant increases at {percentage(row.significant_increasing_percent)}; decreasing slopes occurred at {percentage(row.slope_decreasing_percent)}, with significant decreases at {percentage(row.significant_decreasing_percent)}.'
cent_t, cent_h = (result('Thornthwaite', 'centennial'), result('Hargreaves-Samani', 'centennial'))
rec_t, rec_h = (result('Thornthwaite', 'recent'), result('Hargreaves-Samani', 'recent'))
rcent = ai_pet_regression[ai_pet_regression.period == 'centennial'].r_squared.iloc[0]
rrecent = ai_pet_regression[ai_pet_regression.period == 'recent'].r_squared.iloc[0]
section = '\\section*{S2.2. Sensitivity of the Aridity Index}' + '\n' + f'Mean AI estimates from Thornthwaite and Hargreaves-Samani PET were compared across 861 stations using the same water years for both methods. The coefficients of determination were {rcent:.3f} for WY 1926--2025 and {rrecent:.3f} for WY 1996--2025 (Figure S6).' + '\\par' + '\n' + '\\begin{center}' + '\n' + '\\includegraphics[width=\\linewidth,height=0.70\\textheight,keepaspectratio]{Figure_S6_AI_PET_regression_review.png}' + '\n' + '\\end{center}' + '\n' + 'Figure S6: Relationship between mean aridity index (AI) estimates derived using Thornthwaite and Hargreaves-Samani PET for (a) WY 1926--2025 and (b) WY 1996--2025 across 861 NOAA stations. Red lines represent linear regression fits.' + '\\par' + '\n' + 'For AI and SPEI, percentages were calculated using all 861 stations. Positive and negative slope directions and locally significant trends are reported separately for each PET method and period.' + '\\par' + '\n' + 'For the centennial period (WY 1926--2025), ' + wording('Thornthwaite-based AI', cent_t) + ' ' + wording('Hargreaves-Samani-based AI', cent_h) + '\\par' + '\n' + 'For the recent period (WY 1996--2025), ' + wording('Thornthwaite-based AI', rec_t) + ' ' + wording('Hargreaves-Samani-based AI', rec_h) + '\\par' + '\n' + '\\begin{center}' + '\n' + '\\includegraphics[width=\\linewidth,height=0.70\\textheight,keepaspectratio]{Figure_S7_AI_PET_trends_review.png}' + '\n' + '\\end{center}' + '\n' + 'Figure S7: Spatial distribution of increasing (green), decreasing (red), and no trend (gray) aridity index (AI) trends across 861 NOAA stations using Thornthwaite PET for (a) WY 1926--2025 and (b) WY 1996--2025, and Hargreaves-Samani PET for (c) WY 1926--2025 and (d) WY 1996--2025. Significance is based on the unadjusted modified Mann--Kendall test at p < 0.05.' + '\\par' + '\n'
review_path.write_text(text[:start] + section + text[end:], encoding='utf-8')
print('Updated Section S2.2 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
pet_path = Path('Reviewer_PET_Sensitivity/Tables/PET_monthly_both_methods.csv')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
if getattr(mk, '__version__', None) != '1.4.3':
    raise RuntimeError('Use pyMannKendall 1.4.3 from the advance environment, as in the Thornthwaite results.')
if not hasattr(indices.Distribution, 'loglogistic'):
    raise RuntimeError('The climate_indices version must support log-logistic SPEI.')
monthly_pet = pd.read_csv(pet_path, usecols=['ID', 'YEAR', 'MONTH', 'PRCP', 'PET_Hargreaves_mm'], dtype={'ID': str})
monthly_pet = monthly_pet[monthly_pet.ID.isin(set(stations.ID))].copy()
if set(monthly_pet.ID) != set(stations.ID) or monthly_pet.duplicated(['ID', 'YEAR', 'MONTH']).any():
    raise ValueError('Missing study stations or duplicate Hargreaves PET station-months.')
monthly_pet['DATE'] = pd.to_datetime(dict(year=monthly_pet.YEAR, month=monthly_pet.MONTH, day=1))
monthly_parts, failures = ([], [])
for station, group in monthly_pet.groupby('ID', sort=True):
    group = group.sort_values('DATE').set_index('DATE')
    first_year, last_year = (int(group.index.year.min()), int(group.index.year.max()))
    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    aligned = group.reindex(dates)
    prcp = pd.to_numeric(aligned.PRCP, errors='coerce').to_numpy(dtype=float)
    pet = pd.to_numeric(aligned.PET_Hargreaves_mm, errors='coerce').to_numpy(dtype=float)
    try:
        values = indices.spei(precips_mm=prcp, pet_mm=pet, scale=1, distribution=indices.Distribution.loglogistic, periodicity=compute.Periodicity.monthly, data_start_year=first_year, calibration_year_initial=1926, calibration_year_final=2025)
        values = np.asarray(values).reshape(-1)
        if len(values) < len(dates):
            raise ValueError('SPEI output shorter than input.')
        values = values[:len(dates)]
    except Exception as exc:
        failures.append({'ID': station, 'error': f'{type(exc).__name__}: {exc}'})
        values = np.full(len(dates), np.nan)
    monthly_parts.append(pd.DataFrame({'ID': station, 'DATE': dates, 'SPEI1_Hargreaves': values}))
harg_monthly = pd.concat(monthly_parts, ignore_index=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
harg_monthly.to_csv(OUTPUT / 'SPEI1_Hargreaves_python_monthly_review.csv', index=False)
pd.DataFrame(failures, columns=['ID', 'error']).to_csv(OUTPUT / 'SPEI1_Hargreaves_fit_errors_review.csv', index=False)
if failures:
    raise RuntimeError(f'Hargreaves SPEI-1 fitting failed for {len(failures)} stations; see saved error file.')
print('Calculated Hargreaves SPEI-1 for', harg_monthly.ID.nunique(), 'stations.', flush=True)
harg_monthly['W_YEAR'] = harg_monthly.DATE.dt.year + (harg_monthly.DATE.dt.month >= 10).astype(int)
harg_monthly['SPEI1_Hargreaves'] = pd.to_numeric(harg_monthly.SPEI1_Hargreaves, errors='coerce').clip(-3, 3)
records = []
for name, start, end in [('centennial', 1926, 2025), ('recent', 1996, 2025)]:
    window = harg_monthly[harg_monthly.W_YEAR.between(start, end)].sort_values(['ID', 'DATE'])
    for station, group in window.groupby('ID'):
        x = group.SPEI1_Hargreaves.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        rec = {'ID': station, 'period': name, 'start_WY': start, 'end_WY': end, 'n': len(x), 'trend': None, 'p': np.nan, 'z': np.nan, 'variance': np.nan, 'slope_per_step': np.nan}
        if len(x) >= 12:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = mk.hamed_rao_modification_test(x, alpha=0.05)
                valid = bool(np.isfinite(result.p) and np.isfinite(result.z) and np.isfinite(result.var_s) and (result.var_s >= 0))
                
                if valid and 0 <= result.p <= 1:
                    rec.update(trend=result.trend, p=result.p, z=result.z, variance=result.var_s, slope_per_step=result.slope)
            except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
                pass
        records.append(rec)
    print('Completed Hargreaves SPEI-1', name, flush=True)
harg_wy_trends = pd.DataFrame(records)
harg_wy_trends.to_csv(OUTPUT / 'SPEI1_Hargreaves_water_year_trends_review.csv', index=False)

review_path = OUTPUT / 'Supplementary_review.tex'
if not review_path.exists():
    raise FileNotFoundError('Supplementary review source not found.')
current = pd.read_csv('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv', dtype={'ID': str})
current = current[current.scale == 1].copy()
harg = pd.read_csv(OUTPUT / 'SPEI1_Hargreaves_water_year_trends_review.csv', dtype={'ID': str})
expected_ids = set(stations.ID)
periods = [('centennial', 1926, 2025), ('recent', 1996, 2025)]
trend_parts = []
for method, frame in [('Thornthwaite', current), ('Hargreaves-Samani', harg)]:
    for name, start, end in periods:
        part = frame[(frame.period == name) & frame.ID.isin(expected_ids)].copy()
        if part.ID.duplicated().any() or set(part.ID) != expected_ids:
            raise ValueError(f'Incomplete or duplicated {method} SPEI-1 results for {name}.')
        if not (part.start_WY.eq(start) & part.end_WY.eq(end)).all():
            raise ValueError(f'Unexpected water-year bounds for {method}, {name}.')
        part['p'] = pd.to_numeric(part.p, errors='coerce')
        part['slope_per_step'] = pd.to_numeric(part.slope_per_step, errors='coerce')
        valid = np.isfinite(part.p) & part.p.between(0, 1)
        part['category'] = None
        part.loc[valid & part.p.ge(0.05), 'category'] = 'no trend'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('increasing'), 'category'] = 'increasing'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('decreasing'), 'category'] = 'decreasing'
        part['method'] = method
        trend_parts.append(part[['ID', 'period', 'method', 'slope_per_step', 'p', 'category']])
spei1_pet_trends = pd.concat(trend_parts, ignore_index=True).merge(stations, on='ID', validate='many_to_one')
spei1_pet_trends.to_csv(OUTPUT / 'SPEI1_PET_trend_comparison_stations_review.csv', index=False)
summary_rows = []
for (method, period), group in spei1_pet_trends.groupby(['method', 'period']):
    if not np.isfinite(group.slope_per_step).all():
        raise ValueError(f'Missing Sen slopes for {method}, {period}.')
    n = len(group)
    summary_rows.append({'method': method, 'period': period, 'stations': n, 'slope_increasing_percent': 100 * group.slope_per_step.gt(0).sum() / n, 'significant_increasing_percent': 100 * group.category.eq('increasing').sum() / n, 'slope_decreasing_percent': 100 * group.slope_per_step.lt(0).sum() / n, 'significant_decreasing_percent': 100 * group.category.eq('decreasing').sum() / n})
spei1_pet_summary = pd.DataFrame(summary_rows)
spei1_pet_summary.to_csv(OUTPUT / 'SPEI1_PET_trend_comparison_summary_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
colors = {'decreasing': '#d73027', 'no trend': '#bdbdbd', 'increasing': '#1a9850'}
fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
for col, (name, start, end) in enumerate(periods):
    for row, method in enumerate(['Thornthwaite', 'Hargreaves-Samani']):
        ax = axes[row, col]
        panel = spei1_pet_trends[(spei1_pet_trends.period == name) & (spei1_pet_trends.method == method)]
        shape.plot(ax=ax, facecolor='#f3f3f3', edgecolor='black', linewidth=0.7)
        for category, color in colors.items():
            subset = panel[panel.category == category]
            ax.scatter(subset.Lon, subset.Lat, color=color, s=16, edgecolors='none')
        ax.set_title(f'{chr(97 + row * 2 + col)}) SPEI-1 (from {method}), WY {start}-{end}', fontsize=15, loc='left')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
handles = [Line2D([0], [0], marker='o', linestyle='none', color=color, label=category.title(), markersize=7) for category, color in colors.items()]
fig.legend(handles=handles, title='Modified Mann-Kendall trend', loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=12, title_fontsize=13)
fig.savefig(OUTPUT / 'Figure_S9_SPEI1_PET_trends_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S9_SPEI1_PET_trends_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print('SPEI-1 trend percentages:')
display(spei1_pet_summary)

text = review_path.read_text(encoding='utf-8')
marker = '\\section*{S2.3. Sensitivity of SPEI-1}'
start = text.find(marker)
if start < 0:
    start = text.find('\\end{document}')
    end = start
else:
    next_section = text.find('\\section*{S2.4.', start)
    end = next_section if next_section >= 0 else text.find('\\end{document}', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify the insertion point for Section S2.4.')

def find_spei1_summary(method, period):
    row = spei1_pet_summary[(spei1_pet_summary.method == method) & (spei1_pet_summary.period == period)]
    if len(row) != 1:
        raise ValueError('Trend family is not unique.')
    return row.iloc[0]

def pct_spei1(value):
    return f'{value:.0f}\\%'

def describe_spei1(method, row):
    return f'{method}-based SPEI-1 showed increasing slopes at {pct_spei1(row.slope_increasing_percent)} of stations, including significant increases at {pct_spei1(row.significant_increasing_percent)}, and decreasing slopes at {pct_spei1(row.slope_decreasing_percent)}, including significant decreases at {pct_spei1(row.significant_decreasing_percent)}.'
ct = find_spei1_summary('Thornthwaite', 'centennial')
ch = find_spei1_summary('Hargreaves-Samani', 'centennial')
rt = find_spei1_summary('Thornthwaite', 'recent')
rh = find_spei1_summary('Hargreaves-Samani', 'recent')
section = marker + '\n' + 'For the centennial period (WY 1926--2025), ' + describe_spei1('Thornthwaite', ct) + ' ' + describe_spei1('Hargreaves-Samani', ch) + '\\par' + '\n' + 'For the recent period (WY 1996--2025), ' + describe_spei1('Thornthwaite', rt) + ' ' + describe_spei1('Hargreaves-Samani', rh) + '\\par' + '\n' + '\\begin{center}' + '\n' + '\\includegraphics[width=\\linewidth,height=0.70\\textheight,keepaspectratio]{Figure_S9_SPEI1_PET_trends_review.png}' + '\n' + '\\end{center}' + '\n' + 'Figure S8: Spatial distribution of increasing (green), decreasing (red), and no trend (gray) SPEI-1 trends across 861 NOAA stations using Thornthwaite PET for (a) WY 1926--2025 and (b) WY 1996--2025, and Hargreaves-Samani PET for (c) WY 1926--2025 and (d) WY 1996--2025. Significance is based on the unadjusted modified Mann--Kendall test at p < 0.05.' + '\\par' + '\n'
review_path.write_text(text[:start] + section + text[end:], encoding='utf-8')
print('Updated Section S2.3 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
pet_path = Path('Reviewer_PET_Sensitivity/Tables/PET_monthly_both_methods.csv')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
if getattr(mk, '__version__', None) != '1.4.3':
    raise RuntimeError('Use pyMannKendall 1.4.3 from the advance environment, as in the Thornthwaite results.')
if not hasattr(indices.Distribution, 'loglogistic'):
    raise RuntimeError('The climate_indices version must support log-logistic SPEI.')
monthly_pet = pd.read_csv(pet_path, usecols=['ID', 'YEAR', 'MONTH', 'PRCP', 'PET_Hargreaves_mm'], dtype={'ID': str})
monthly_pet = monthly_pet[monthly_pet.ID.isin(set(stations.ID))].copy()
if set(monthly_pet.ID) != set(stations.ID) or monthly_pet.duplicated(['ID', 'YEAR', 'MONTH']).any():
    raise ValueError('Missing study stations or duplicate Hargreaves PET station-months.')
monthly_pet['DATE'] = pd.to_datetime(dict(year=monthly_pet.YEAR, month=monthly_pet.MONTH, day=1))
monthly_parts, failures = ([], [])
for station, group in monthly_pet.groupby('ID', sort=True):
    group = group.sort_values('DATE').set_index('DATE')
    first_year, last_year = (int(group.index.year.min()), int(group.index.year.max()))
    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    aligned = group.reindex(dates)
    prcp = pd.to_numeric(aligned.PRCP, errors='coerce').to_numpy(dtype=float)
    pet = pd.to_numeric(aligned.PET_Hargreaves_mm, errors='coerce').to_numpy(dtype=float)
    try:
        values = indices.spei(precips_mm=prcp, pet_mm=pet, scale=3, distribution=indices.Distribution.loglogistic, periodicity=compute.Periodicity.monthly, data_start_year=first_year, calibration_year_initial=1926, calibration_year_final=2025)
        values = np.asarray(values).reshape(-1)
        if len(values) < len(dates):
            raise ValueError('SPEI output shorter than input.')
        values = values[:len(dates)]
    except Exception as exc:
        failures.append({'ID': station, 'error': f'{type(exc).__name__}: {exc}'})
        values = np.full(len(dates), np.nan)
    monthly_parts.append(pd.DataFrame({'ID': station, 'DATE': dates, 'SPEI3_Hargreaves': values}))
harg_monthly = pd.concat(monthly_parts, ignore_index=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
harg_monthly.to_csv(OUTPUT / 'SPEI3_Hargreaves_python_monthly_review.csv', index=False)
pd.DataFrame(failures, columns=['ID', 'error']).to_csv(OUTPUT / 'SPEI3_Hargreaves_fit_errors_review.csv', index=False)
if failures:
    raise RuntimeError(f'Hargreaves SPEI-3 fitting failed for {len(failures)} stations; see saved error file.')
print('Calculated Hargreaves SPEI-3 for', harg_monthly.ID.nunique(), 'stations.', flush=True)
harg_monthly['W_YEAR'] = harg_monthly.DATE.dt.year + (harg_monthly.DATE.dt.month >= 10).astype(int)
harg_monthly['SPEI3_Hargreaves'] = pd.to_numeric(harg_monthly.SPEI3_Hargreaves, errors='coerce').clip(-3, 3)
records = []
for name, start, end in [('centennial', 1926, 2025), ('recent', 1996, 2025)]:
    window = harg_monthly[harg_monthly.W_YEAR.between(start, end)].sort_values(['ID', 'DATE'])
    for station, group in window.groupby('ID'):
        x = group.SPEI3_Hargreaves.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        rec = {'ID': station, 'period': name, 'start_WY': start, 'end_WY': end, 'n': len(x), 'trend': None, 'p': np.nan, 'z': np.nan, 'variance': np.nan, 'slope_per_step': np.nan}
        if len(x) >= 12:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = mk.hamed_rao_modification_test(x, alpha=0.05)
                valid = bool(np.isfinite(result.p) and np.isfinite(result.z) and np.isfinite(result.var_s) and (result.var_s >= 0))
                
                if valid and 0 <= result.p <= 1:
                    rec.update(trend=result.trend, p=result.p, z=result.z, variance=result.var_s, slope_per_step=result.slope)
            except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
                pass
        records.append(rec)
    print('Completed Hargreaves SPEI-3', name, flush=True)
harg_wy_trends = pd.DataFrame(records)
harg_wy_trends.to_csv(OUTPUT / 'SPEI3_Hargreaves_water_year_trends_review.csv', index=False)

review_path = OUTPUT / 'Supplementary_review.tex'
if not review_path.exists():
    raise FileNotFoundError('Supplementary review source not found.')
current = pd.read_csv('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv', dtype={'ID': str})
current = current[current.scale == 3].copy()
harg = pd.read_csv(OUTPUT / 'SPEI3_Hargreaves_water_year_trends_review.csv', dtype={'ID': str})
expected_ids = set(stations.ID)
periods = [('centennial', 1926, 2025), ('recent', 1996, 2025)]
trend_parts = []
for method, frame in [('Thornthwaite', current), ('Hargreaves-Samani', harg)]:
    for name, start, end in periods:
        part = frame[(frame.period == name) & frame.ID.isin(expected_ids)].copy()
        if part.ID.duplicated().any() or set(part.ID) != expected_ids:
            raise ValueError(f'Incomplete or duplicated {method} SPEI-3 results for {name}.')
        if not (part.start_WY.eq(start) & part.end_WY.eq(end)).all():
            raise ValueError(f'Unexpected water-year bounds for {method}, {name}.')
        part['p'] = pd.to_numeric(part.p, errors='coerce')
        part['slope_per_step'] = pd.to_numeric(part.slope_per_step, errors='coerce')
        valid = np.isfinite(part.p) & part.p.between(0, 1)
        part['category'] = None
        part.loc[valid & part.p.ge(0.05), 'category'] = 'no trend'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('increasing'), 'category'] = 'increasing'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('decreasing'), 'category'] = 'decreasing'
        part['method'] = method
        trend_parts.append(part[['ID', 'period', 'method', 'slope_per_step', 'p', 'category']])
spei3_pet_trends = pd.concat(trend_parts, ignore_index=True).merge(stations, on='ID', validate='many_to_one')
spei3_pet_trends.to_csv(OUTPUT / 'SPEI3_PET_trend_comparison_stations_review.csv', index=False)
summary_rows = []
for (method, period), group in spei3_pet_trends.groupby(['method', 'period']):
    if not np.isfinite(group.slope_per_step).all():
        raise ValueError(f'Missing Sen slopes for {method}, {period}.')
    n = len(group)
    summary_rows.append({'method': method, 'period': period, 'stations': n, 'slope_increasing_percent': 100 * group.slope_per_step.gt(0).sum() / n, 'significant_increasing_percent': 100 * group.category.eq('increasing').sum() / n, 'slope_decreasing_percent': 100 * group.slope_per_step.lt(0).sum() / n, 'significant_decreasing_percent': 100 * group.category.eq('decreasing').sum() / n})
spei3_pet_summary = pd.DataFrame(summary_rows)
spei3_pet_summary.to_csv(OUTPUT / 'SPEI3_PET_trend_comparison_summary_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
colors = {'decreasing': '#d73027', 'no trend': '#bdbdbd', 'increasing': '#1a9850'}
fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
for col, (name, start, end) in enumerate(periods):
    for row, method in enumerate(['Thornthwaite', 'Hargreaves-Samani']):
        ax = axes[row, col]
        panel = spei3_pet_trends[(spei3_pet_trends.period == name) & (spei3_pet_trends.method == method)]
        shape.plot(ax=ax, facecolor='#f3f3f3', edgecolor='black', linewidth=0.7)
        for category, color in colors.items():
            subset = panel[panel.category == category]
            ax.scatter(subset.Lon, subset.Lat, color=color, s=16, edgecolors='none')
        ax.set_title(f'{chr(97 + row * 2 + col)}) SPEI-3 (from {method}), WY {start}-{end}', fontsize=15, loc='left')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
handles = [Line2D([0], [0], marker='o', linestyle='none', color=color, label=category.title(), markersize=7) for category, color in colors.items()]
fig.legend(handles=handles, title='Modified Mann-Kendall trend', loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=12, title_fontsize=13)
fig.savefig(OUTPUT / 'Figure_S10_SPEI3_PET_trends_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S10_SPEI3_PET_trends_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print('SPEI-3 trend percentages:')
display(spei3_pet_summary)

text = review_path.read_text(encoding='utf-8')
marker = '\\section*{S2.4. Sensitivity of SPEI-3}'
start = text.find(marker)
if start < 0:
    start = text.find('\\end{document}')
    end = start
else:
    next_section = text.find('\\section*{S2.5.', start)
    end = next_section if next_section >= 0 else text.find('\\end{document}', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify the insertion point for Section S2.6.')

def find_spei3_summary(method, period):
    row = spei3_pet_summary[(spei3_pet_summary.method == method) & (spei3_pet_summary.period == period)]
    if len(row) != 1:
        raise ValueError('Trend family is not unique.')
    return row.iloc[0]

def pct_spei3(value):
    return f'{value:.0f}\\%'

def describe_spei3(method, row):
    return f'{method}-based SPEI-3 showed increasing slopes at {pct_spei3(row.slope_increasing_percent)} of stations, including significant increases at {pct_spei3(row.significant_increasing_percent)}, and decreasing slopes at {pct_spei3(row.slope_decreasing_percent)}, including significant decreases at {pct_spei3(row.significant_decreasing_percent)}.'
ct = find_spei3_summary('Thornthwaite', 'centennial')
ch = find_spei3_summary('Hargreaves-Samani', 'centennial')
rt = find_spei3_summary('Thornthwaite', 'recent')
rh = find_spei3_summary('Hargreaves-Samani', 'recent')
section = marker + '\n' + 'For the centennial period (WY 1926--2025), ' + describe_spei3('Thornthwaite', ct) + ' ' + describe_spei3('Hargreaves-Samani', ch) + '\\par' + '\n' + 'For the recent period (WY 1996--2025), ' + describe_spei3('Thornthwaite', rt) + ' ' + describe_spei3('Hargreaves-Samani', rh) + '\\par' + '\n' + '\\begin{center}' + '\n' + '\\includegraphics[width=\\linewidth,height=0.70\\textheight,keepaspectratio]{Figure_S10_SPEI3_PET_trends_review.png}' + '\n' + '\\end{center}' + '\n' + 'Figure S9: Spatial distribution of increasing (green), decreasing (red), and no trend (gray) SPEI-3 trends across 861 NOAA stations using Thornthwaite PET for (a) WY 1926--2025 and (b) WY 1996--2025, and Hargreaves-Samani PET for (c) WY 1926--2025 and (d) WY 1996--2025. Significance is based on the unadjusted modified Mann--Kendall test at p < 0.05.' + '\\par' + '\n'
review_path.write_text(text[:start] + section + text[end:], encoding='utf-8')
print('Updated Section S2.4 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
pet_path = Path('Reviewer_PET_Sensitivity/Tables/PET_monthly_both_methods.csv')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
if getattr(mk, '__version__', None) != '1.4.3':
    raise RuntimeError('Use pyMannKendall 1.4.3 from the advance environment, as in the Thornthwaite results.')
if not hasattr(indices.Distribution, 'loglogistic'):
    raise RuntimeError('The climate_indices version must support log-logistic SPEI.')
monthly_pet = pd.read_csv(pet_path, usecols=['ID', 'YEAR', 'MONTH', 'PRCP', 'PET_Hargreaves_mm'], dtype={'ID': str})
monthly_pet = monthly_pet[monthly_pet.ID.isin(set(stations.ID))].copy()
if set(monthly_pet.ID) != set(stations.ID) or monthly_pet.duplicated(['ID', 'YEAR', 'MONTH']).any():
    raise ValueError('Missing study stations or duplicate Hargreaves PET station-months.')
monthly_pet['DATE'] = pd.to_datetime(dict(year=monthly_pet.YEAR, month=monthly_pet.MONTH, day=1))
monthly_parts, failures = ([], [])
for station, group in monthly_pet.groupby('ID', sort=True):
    group = group.sort_values('DATE').set_index('DATE')
    first_year, last_year = (int(group.index.year.min()), int(group.index.year.max()))
    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    aligned = group.reindex(dates)
    prcp = pd.to_numeric(aligned.PRCP, errors='coerce').to_numpy(dtype=float)
    pet = pd.to_numeric(aligned.PET_Hargreaves_mm, errors='coerce').to_numpy(dtype=float)
    try:
        values = indices.spei(precips_mm=prcp, pet_mm=pet, scale=6, distribution=indices.Distribution.loglogistic, periodicity=compute.Periodicity.monthly, data_start_year=first_year, calibration_year_initial=1926, calibration_year_final=2025)
        values = np.asarray(values).reshape(-1)
        if len(values) < len(dates):
            raise ValueError('SPEI output shorter than input.')
        values = values[:len(dates)]
    except Exception as exc:
        failures.append({'ID': station, 'error': f'{type(exc).__name__}: {exc}'})
        values = np.full(len(dates), np.nan)
    monthly_parts.append(pd.DataFrame({'ID': station, 'DATE': dates, 'SPEI6_Hargreaves': values}))
harg_monthly = pd.concat(monthly_parts, ignore_index=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
harg_monthly.to_csv(OUTPUT / 'SPEI6_Hargreaves_python_monthly_review.csv', index=False)
pd.DataFrame(failures, columns=['ID', 'error']).to_csv(OUTPUT / 'SPEI6_Hargreaves_fit_errors_review.csv', index=False)
if failures:
    raise RuntimeError(f'Hargreaves SPEI-6 fitting failed for {len(failures)} stations; see saved error file.')
print('Calculated Hargreaves SPEI-6 for', harg_monthly.ID.nunique(), 'stations.', flush=True)
harg_monthly['W_YEAR'] = harg_monthly.DATE.dt.year + (harg_monthly.DATE.dt.month >= 10).astype(int)
harg_monthly['SPEI6_Hargreaves'] = pd.to_numeric(harg_monthly.SPEI6_Hargreaves, errors='coerce').clip(-3, 3)
records = []
for name, start, end in [('centennial', 1926, 2025), ('recent', 1996, 2025)]:
    window = harg_monthly[harg_monthly.W_YEAR.between(start, end)].sort_values(['ID', 'DATE'])
    for station, group in window.groupby('ID'):
        x = group.SPEI6_Hargreaves.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        rec = {'ID': station, 'period': name, 'start_WY': start, 'end_WY': end, 'n': len(x), 'trend': None, 'p': np.nan, 'z': np.nan, 'variance': np.nan, 'slope_per_step': np.nan}
        if len(x) >= 12:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = mk.hamed_rao_modification_test(x, alpha=0.05)
                valid = bool(np.isfinite(result.p) and np.isfinite(result.z) and np.isfinite(result.var_s) and (result.var_s >= 0))
                
                if valid and 0 <= result.p <= 1:
                    rec.update(trend=result.trend, p=result.p, z=result.z, variance=result.var_s, slope_per_step=result.slope)
            except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
                pass
        records.append(rec)
    print('Completed Hargreaves SPEI-6', name, flush=True)
harg_wy_trends = pd.DataFrame(records)
harg_wy_trends.to_csv(OUTPUT / 'SPEI6_Hargreaves_water_year_trends_review.csv', index=False)

review_path = OUTPUT / 'Supplementary_review.tex'
if not review_path.exists():
    raise FileNotFoundError('Supplementary review source not found.')
current = pd.read_csv('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv', dtype={'ID': str})
current = current[current.scale == 6].copy()
harg = pd.read_csv(OUTPUT / 'SPEI6_Hargreaves_water_year_trends_review.csv', dtype={'ID': str})
expected_ids = set(stations.ID)
periods = [('centennial', 1926, 2025), ('recent', 1996, 2025)]
trend_parts = []
for method, frame in [('Thornthwaite', current), ('Hargreaves-Samani', harg)]:
    for name, start, end in periods:
        part = frame[(frame.period == name) & frame.ID.isin(expected_ids)].copy()
        if part.ID.duplicated().any() or set(part.ID) != expected_ids:
            raise ValueError(f'Incomplete or duplicated {method} SPEI-6 results for {name}.')
        if not (part.start_WY.eq(start) & part.end_WY.eq(end)).all():
            raise ValueError(f'Unexpected water-year bounds for {method}, {name}.')
        part['p'] = pd.to_numeric(part.p, errors='coerce')
        part['slope_per_step'] = pd.to_numeric(part.slope_per_step, errors='coerce')
        valid = np.isfinite(part.p) & part.p.between(0, 1)
        part['category'] = None
        part.loc[valid & part.p.ge(0.05), 'category'] = 'no trend'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('increasing'), 'category'] = 'increasing'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('decreasing'), 'category'] = 'decreasing'
        part['method'] = method
        trend_parts.append(part[['ID', 'period', 'method', 'slope_per_step', 'p', 'category']])
spei6_pet_trends = pd.concat(trend_parts, ignore_index=True).merge(stations, on='ID', validate='many_to_one')
spei6_pet_trends.to_csv(OUTPUT / 'SPEI6_PET_trend_comparison_stations_review.csv', index=False)
summary_rows = []
for (method, period), group in spei6_pet_trends.groupby(['method', 'period']):
    if not np.isfinite(group.slope_per_step).all():
        raise ValueError(f'Missing Sen slopes for {method}, {period}.')
    n = len(group)
    summary_rows.append({'method': method, 'period': period, 'stations': n, 'slope_increasing_percent': 100 * group.slope_per_step.gt(0).sum() / n, 'significant_increasing_percent': 100 * group.category.eq('increasing').sum() / n, 'slope_decreasing_percent': 100 * group.slope_per_step.lt(0).sum() / n, 'significant_decreasing_percent': 100 * group.category.eq('decreasing').sum() / n})
spei6_pet_summary = pd.DataFrame(summary_rows)
spei6_pet_summary.to_csv(OUTPUT / 'SPEI6_PET_trend_comparison_summary_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
colors = {'decreasing': '#d73027', 'no trend': '#bdbdbd', 'increasing': '#1a9850'}
fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
for col, (name, start, end) in enumerate(periods):
    for row, method in enumerate(['Thornthwaite', 'Hargreaves-Samani']):
        ax = axes[row, col]
        panel = spei6_pet_trends[(spei6_pet_trends.period == name) & (spei6_pet_trends.method == method)]
        shape.plot(ax=ax, facecolor='#f3f3f3', edgecolor='black', linewidth=0.7)
        for category, color in colors.items():
            subset = panel[panel.category == category]
            ax.scatter(subset.Lon, subset.Lat, color=color, s=16, edgecolors='none')
        ax.set_title(f'{chr(97 + row * 2 + col)}) SPEI-6 (from {method}), WY {start}-{end}', fontsize=15, loc='left')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
handles = [Line2D([0], [0], marker='o', linestyle='none', color=color, label=category.title(), markersize=7) for category, color in colors.items()]
fig.legend(handles=handles, title='Modified Mann-Kendall trend', loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=12, title_fontsize=13)
fig.savefig(OUTPUT / 'Figure_S11_SPEI6_PET_trends_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S11_SPEI6_PET_trends_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print('SPEI-6 trend percentages:')
display(spei6_pet_summary)

text = review_path.read_text(encoding='utf-8')
marker = '\\section*{S2.5. Sensitivity of SPEI-6}'
start = text.find(marker)
if start < 0:
    start = text.find('\\end{document}')
    end = start
else:
    next_section = text.find('\\section*{S2.6.', start)
    end = next_section if next_section >= 0 else text.find('\\end{document}', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify the insertion point for Section S2.6.')

def find_spei6_summary(method, period):
    row = spei6_pet_summary[(spei6_pet_summary.method == method) & (spei6_pet_summary.period == period)]
    if len(row) != 1:
        raise ValueError('Trend family is not unique.')
    return row.iloc[0]

def pct_spei6(value):
    return f'{value:.0f}\\%'

def describe_spei6(method, row):
    return f'{method}-based SPEI-6 showed increasing slopes at {pct_spei6(row.slope_increasing_percent)} of stations, including significant increases at {pct_spei6(row.significant_increasing_percent)}, and decreasing slopes at {pct_spei6(row.slope_decreasing_percent)}, including significant decreases at {pct_spei6(row.significant_decreasing_percent)}.'
ct = find_spei6_summary('Thornthwaite', 'centennial')
ch = find_spei6_summary('Hargreaves-Samani', 'centennial')
rt = find_spei6_summary('Thornthwaite', 'recent')
rh = find_spei6_summary('Hargreaves-Samani', 'recent')
section = marker + '\n' + 'For the centennial period (WY 1926--2025), ' + describe_spei6('Thornthwaite', ct) + ' ' + describe_spei6('Hargreaves-Samani', ch) + '\\par' + '\n' + 'For the recent period (WY 1996--2025), ' + describe_spei6('Thornthwaite', rt) + ' ' + describe_spei6('Hargreaves-Samani', rh) + '\\par' + '\n' + '\\begin{center}' + '\n' + '\\includegraphics[width=\\linewidth,height=0.70\\textheight,keepaspectratio]{Figure_S11_SPEI6_PET_trends_review.png}' + '\n' + '\\end{center}' + '\n' + 'Figure S10: Spatial distribution of increasing (green), decreasing (red), and no trend (gray) SPEI-6 trends across 861 NOAA stations using Thornthwaite PET for (a) WY 1926--2025 and (b) WY 1996--2025, and Hargreaves-Samani PET for (c) WY 1926--2025 and (d) WY 1996--2025. Significance is based on the unadjusted modified Mann--Kendall test at p < 0.05.' + '\\par' + '\n'
review_path.write_text(text[:start] + section + text[end:], encoding='utf-8')
print('Updated Section S2.5 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
harg_path = Path('Reviewer_PET_Sensitivity/Tables/SPEI12_Hargreaves_python_monthly.csv')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
if getattr(mk, '__version__', None) != '1.4.3':
    raise RuntimeError('Use pyMannKendall 1.4.3 from the advance environment, as in the current Thornthwaite trends.')
monthly_h = pd.read_csv(harg_path, dtype={'ID': str})
monthly_h['DATE'] = pd.to_datetime(monthly_h.DATE, errors='raise')
monthly_h['W_YEAR'] = monthly_h.DATE.dt.year + (monthly_h.DATE.dt.month >= 10).astype(int)
monthly_h['SPEI12_Hargreaves'] = pd.to_numeric(monthly_h.SPEI12_Hargreaves, errors='coerce').clip(-3, 3)
monthly_h = monthly_h[monthly_h.ID.isin(set(stations.ID))].copy()
if monthly_h.duplicated(['ID', 'DATE']).any():
    raise ValueError('Duplicate Hargreaves station-month observations.')
if set(monthly_h.ID) != set(stations.ID):
    raise ValueError('Hargreaves SPEI data do not contain the full 861-station network.')
records = []
for name, start, end in [('centennial', 1926, 2025), ('recent', 1996, 2025)]:
    window = monthly_h[monthly_h.W_YEAR.between(start, end)].sort_values(['ID', 'DATE'])
    for station, group in window.groupby('ID'):
        x = group.SPEI12_Hargreaves.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        rec = {'ID': station, 'period': name, 'start_WY': start, 'end_WY': end, 'n': len(x), 'trend': None, 'p': np.nan, 'z': np.nan, 'variance': np.nan, 'slope_per_step': np.nan}
        if len(x) >= 12:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    result = mk.hamed_rao_modification_test(x, alpha=0.05)
                valid = bool(np.isfinite(result.p) and np.isfinite(result.z) and np.isfinite(result.var_s) and (result.var_s >= 0))
                
                if valid and 0 <= result.p <= 1:
                    rec.update(trend=result.trend, p=result.p, z=result.z, variance=result.var_s, slope_per_step=result.slope)
            except (ValueError, ZeroDivisionError, FloatingPointError, IndexError):
                pass
        records.append(rec)
    print('Completed Hargreaves SPEI-12', name, flush=True)
harg_wy_trends = pd.DataFrame(records)
OUTPUT.mkdir(parents=True, exist_ok=True)
harg_wy_trends.to_csv(OUTPUT / 'SPEI12_Hargreaves_water_year_trends_review.csv', index=False)
print('Saved Hargreaves water-year trends to', OUTPUT.resolve())

review_path = OUTPUT / 'Supplementary_review.tex'
if not review_path.exists():
    raise FileNotFoundError('Supplementary review source not found.')
current = pd.read_csv('AI_SPEI_python/Tables/SPEI_station_modified_MK_trends.csv', dtype={'ID': str})
current = current[current.scale == 12].copy()
harg = pd.read_csv(OUTPUT / 'SPEI12_Hargreaves_water_year_trends_review.csv', dtype={'ID': str})
expected_ids = set(stations.ID)
periods = [('centennial', 1926, 2025), ('recent', 1996, 2025)]
trend_parts = []
for method, frame in [('Thornthwaite', current), ('Hargreaves-Samani', harg)]:
    for name, start, end in periods:
        part = frame[(frame.period == name) & frame.ID.isin(expected_ids)].copy()
        if part.ID.duplicated().any() or set(part.ID) != expected_ids:
            raise ValueError(f'Incomplete or duplicated {method} SPEI-12 results for {name}.')
        if not (part.start_WY.eq(start) & part.end_WY.eq(end)).all():
            raise ValueError(f'Unexpected water-year bounds for {method}, {name}.')
        part['p'] = pd.to_numeric(part.p, errors='coerce')
        part['slope_per_step'] = pd.to_numeric(part.slope_per_step, errors='coerce')
        valid = np.isfinite(part.p) & part.p.between(0, 1)
        part['category'] = None
        part.loc[valid & part.p.ge(0.05), 'category'] = 'no trend'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('increasing'), 'category'] = 'increasing'
        part.loc[valid & part.p.lt(0.05) & part.trend.eq('decreasing'), 'category'] = 'decreasing'
        part['method'] = method
        trend_parts.append(part[['ID', 'period', 'method', 'slope_per_step', 'p', 'category']])
spei12_pet_trends = pd.concat(trend_parts, ignore_index=True).merge(stations, on='ID', validate='many_to_one')
spei12_pet_trends.to_csv(OUTPUT / 'SPEI12_PET_trend_comparison_stations_review.csv', index=False)
summary_rows = []
for (method, period), group in spei12_pet_trends.groupby(['method', 'period']):
    if not np.isfinite(group.slope_per_step).all():
        raise ValueError(f'Missing Sen slopes for {method}, {period}.')
    n = len(group)
    summary_rows.append({'method': method, 'period': period, 'stations': n, 'slope_increasing_percent': 100 * group.slope_per_step.gt(0).sum() / n, 'significant_increasing_percent': 100 * group.category.eq('increasing').sum() / n, 'slope_decreasing_percent': 100 * group.slope_per_step.lt(0).sum() / n, 'significant_decreasing_percent': 100 * group.category.eq('decreasing').sum() / n})
spei12_pet_summary = pd.DataFrame(summary_rows)
spei12_pet_summary.to_csv(OUTPUT / 'SPEI12_PET_trend_comparison_summary_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
colors = {'decreasing': '#d73027', 'no trend': '#bdbdbd', 'increasing': '#1a9850'}
fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
for col, (name, start, end) in enumerate(periods):
    for row, method in enumerate(['Thornthwaite', 'Hargreaves-Samani']):
        ax = axes[row, col]
        panel = spei12_pet_trends[(spei12_pet_trends.period == name) & (spei12_pet_trends.method == method)]
        shape.plot(ax=ax, facecolor='#f3f3f3', edgecolor='black', linewidth=0.7)
        for category, color in colors.items():
            subset = panel[panel.category == category]
            ax.scatter(subset.Lon, subset.Lat, color=color, s=16, edgecolors='none')
        ax.set_title(f'{chr(97 + row * 2 + col)}) SPEI-12 (from {method}), WY {start}-{end}', fontsize=15, loc='left')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
handles = [Line2D([0], [0], marker='o', linestyle='none', color=color, label=category.title(), markersize=7) for category, color in colors.items()]
fig.legend(handles=handles, title='Modified Mann-Kendall trend', loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=12, title_fontsize=13)
fig.savefig(OUTPUT / 'Figure_S8_SPEI12_PET_trends_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S8_SPEI12_PET_trends_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print('SPEI-12 trend percentages:')
display(spei12_pet_summary)

text = review_path.read_text(encoding='utf-8')
start = text.find('\\section*{S2.6. Sensitivity of SPEI-12}')
if start < 0:
    raise ValueError('Section S2.3 not found.')
next_section = text.find('\\section*{S3.', start)
end = next_section if next_section >= 0 else text.find('\\end{document}', start)
if end < 0:
    raise ValueError('End of document not found.')

def find_summary(method, period):
    row = spei12_pet_summary[(spei12_pet_summary.method == method) & (spei12_pet_summary.period == period)]
    if len(row) != 1:
        raise ValueError('Trend family is not unique.')
    return row.iloc[0]

def pct(value):
    return f'{value:.0f}\\%'

def describe(method, row):
    return f'{method}-based SPEI-12 showed increasing slopes at {pct(row.slope_increasing_percent)} of stations, including significant increases at {pct(row.significant_increasing_percent)}, and decreasing slopes at {pct(row.slope_decreasing_percent)}, including significant decreases at {pct(row.significant_decreasing_percent)}.'
ct = find_summary('Thornthwaite', 'centennial')
ch = find_summary('Hargreaves-Samani', 'centennial')
rt = find_summary('Thornthwaite', 'recent')
rh = find_summary('Hargreaves-Samani', 'recent')
section = '\\section*{S2.6. Sensitivity of SPEI-12}' + '\n' + 'For the centennial period (WY 1926--2025), ' + describe('Thornthwaite', ct) + ' ' + describe('Hargreaves-Samani', ch) + '\\par' + '\n' + 'For the recent period (WY 1996--2025), ' + describe('Thornthwaite', rt) + ' ' + describe('Hargreaves-Samani', rh) + '\\par' + '\n' + '\\begin{center}' + '\n' + '\\includegraphics[width=\\linewidth,height=0.70\\textheight,keepaspectratio]{Figure_S8_SPEI12_PET_trends_review.png}' + '\n' + '\\end{center}' + '\n' + 'Figure S11: Spatial distribution of increasing (green), decreasing (red), and no trend (gray) SPEI-12 trends across 861 NOAA stations using Thornthwaite PET for (a) WY 1926--2025 and (b) WY 1996--2025, and Hargreaves-Samani PET for (c) WY 1926--2025 and (d) WY 1996--2025. Significance is based on the unadjusted modified Mann--Kendall test at p < 0.05.' + '\\par' + '\n'
review_path.write_text(text[:start] + section + text[end:], encoding='utf-8')
print('Updated Section S2.6 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
review_path = OUTPUT / 'Supplementary_review.tex'
trend_path = Path('data_intermediate_review/mk_two_non_AI_indices_review.csv')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
trends = pd.read_csv(trend_path, dtype={'ID': str})
trends = trends[trends.ID.isin(set(stations.ID)) & trends.window.eq('1996-2025') & trends.variable.isin(['TMAX', 'TMIN'])].copy()
if trends.duplicated(['ID', 'variable']).any():
    raise ValueError('Duplicate station-variable trends.')
for variable, group in trends.groupby('variable'):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'Incomplete station set for {variable}.')
if set(trends.variable) != {'TMAX', 'TMIN'}:
    raise ValueError('Missing TMAX or TMIN trends.')
trends['p'] = pd.to_numeric(trends.p, errors='coerce')
valid = np.isfinite(trends.p) & trends.p.between(0, 1)
trends['classification'] = None
trends.loc[valid, 'classification'] = 'other valid trend'
trends.loc[valid & trends.p.lt(0.05) & trends.trend.eq('increasing'), 'classification'] = 'increasing'
wide = trends.pivot(index='ID', columns='variable', values='classification').reset_index()
s2_stations = stations.merge(wide, on='ID', validate='one_to_one')
valid_both = s2_stations[['TMAX', 'TMIN']].notna().all(axis=1)
n_increasing = s2_stations[['TMAX', 'TMIN']].eq('increasing').sum(axis=1)
s2_stations['category'] = 'not displayed'
s2_stations.loc[valid_both & n_increasing.eq(1), 'category'] = 'one increasing'
s2_stations.loc[valid_both & n_increasing.eq(2), 'category'] = 'both increasing'
OUTPUT.mkdir(parents=True, exist_ok=True)
s2_stations.to_csv(OUTPUT / 'Figure_S2_TMAX_TMIN_station_classes_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
shape.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.65)
handles = []
for category, color, label in [('one increasing', '#f4a261', 'One temperature increasing'), ('both increasing', '#f01818', 'Both temperatures increasing')]:
    subset = s2_stations[s2_stations.category == category].dropna(subset=['Lat', 'Lon'])
    ax.scatter(subset.Lon, subset.Lat, marker='^', s=42, c=color, edgecolors='none', zorder=3)
    handles.append(Line2D([0], [0], marker='^', linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=9, label=f'{label} (n={len(subset)})'))
ax.set_title('TMAX and TMIN combined trend stations', fontsize=17)
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')
ax.legend(handles=handles, loc='lower left', fontsize=12, frameon=True)
fig.savefig(OUTPUT / 'Figure_S2_TMAX_TMIN_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S2_TMAX_TMIN_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print(s2_stations.category.value_counts().to_string())
text = review_path.read_text(encoding='utf-8')
old = '{../supplementary/figures/image2.jpeg}'
new = '{Figure_S2_TMAX_TMIN_review.png}'
if text.count(old) == 1 and text.count(new) == 0:
    text = text.replace(old, new, 1)
elif not (text.count(old) == 0 and text.count(new) == 1):
    raise ValueError('Could not uniquely identify Figure S2 reference.')
start = text.find('Figure S2:')
end = text.find('\\par', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify Figure S2 caption.')
caption = 'Figure S2: Spatial distribution of station-level trends in annual mean daily maximum air temperature (TMAX) and minimum air temperature (TMIN) during WY 1996--2025 across CONUS. Red upward triangles identify stations with statistically significant increasing trends in both TMAX and TMIN; orange upward triangles identify stations with a significant increase in only one of the two variables. Significance is based on unadjusted modified Mann--Kendall tests at p < 0.05.'
review_path.write_text(text[:start] + caption + text[end:], encoding='utf-8')
print('Updated Figure S2 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
review_path = OUTPUT / 'Supplementary_review.tex'
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
trends = pd.read_csv('data_intermediate_review/mk_two_non_AI_indices_review.csv', dtype={'ID': str})
trends = trends[trends.ID.isin(set(stations.ID)) & trends.window.eq('1996-2025') & trends.variable.isin(['TMAX', 'TMIN', 'PRCP'])].copy()
if trends.duplicated(['ID', 'variable']).any() or set(trends.variable) != {'TMAX', 'TMIN', 'PRCP'}:
    raise ValueError('Duplicate or missing station-variable trends.')
for variable, group in trends.groupby('variable'):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'Incomplete station set for {variable}.')
trends['p'] = pd.to_numeric(trends.p, errors='coerce')
valid = np.isfinite(trends.p) & trends.p.between(0, 1)
trends['classification'] = None
trends.loc[valid, 'classification'] = 'other valid trend'
trends.loc[valid & trends.p.lt(0.05) & trends.trend.eq('increasing'), 'classification'] = 'increasing'
wide = trends.pivot(index='ID', columns='variable', values='classification').reset_index()
s3_stations = stations.merge(wide, on='ID', validate='one_to_one')
valid_all = s3_stations[['TMAX', 'TMIN', 'PRCP']].notna().all(axis=1)
both_temp = s3_stations.TMAX.eq('increasing') & s3_stations.TMIN.eq('increasing')
s3_stations['category'] = 'not displayed'
s3_stations.loc[both_temp & valid_all & s3_stations.PRCP.ne('increasing'), 'category'] = 'temperatures increasing, PRCP not increasing'
s3_stations.loc[both_temp & valid_all & s3_stations.PRCP.eq('increasing'), 'category'] = 'all three increasing'
OUTPUT.mkdir(parents=True, exist_ok=True)
s3_stations.to_csv(OUTPUT / 'Figure_S3_TMAX_TMIN_PRCP_station_classes_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
shape.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.65)
styles = [('temperatures increasing, PRCP not increasing', '^', '#e6a000', 'TMAX and TMIN increasing; PRCP not increasing'), ('all three increasing', 'o', '#e76f00', 'TMAX, TMIN and PRCP increasing')]
handles = []
for category, marker, color, label in styles:
    subset = s3_stations[s3_stations.category == category].dropna(subset=['Lat', 'Lon'])
    ax.scatter(subset.Lon, subset.Lat, marker=marker, s=42, c=color, edgecolors='none', zorder=3)
    handles.append(Line2D([0], [0], marker=marker, linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=9, label=f'{label} (n={len(subset)})'))
ax.set_title('TMAX and TMIN trends with PRCP scenarios', fontsize=17)
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')
ax.legend(handles=handles, loc='lower left', fontsize=12, frameon=True)
fig.savefig(OUTPUT / 'Figure_S3_TMAX_TMIN_PRCP_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S3_TMAX_TMIN_PRCP_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print(s3_stations.category.value_counts().to_string())
text = review_path.read_text(encoding='utf-8')
old, new = ('{../supplementary/figures/image3.jpeg}', '{Figure_S3_TMAX_TMIN_PRCP_review.png}')
if text.count(old) == 1 and text.count(new) == 0:
    text = text.replace(old, new, 1)
elif not (text.count(old) == 0 and text.count(new) == 1):
    raise ValueError('Could not uniquely identify Figure S3 reference.')
start = text.find('Figure S3:')
end = text.find('\\par', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify Figure S3 caption.')
n_not = int(s3_stations.category.eq('temperatures increasing, PRCP not increasing').sum())
n_all = int(s3_stations.category.eq('all three increasing').sum())
caption = f'Figure S3: Spatial distribution of stations with statistically significant increasing trends in annual mean daily maximum air temperature (TMAX) and minimum air temperature (TMIN), classified by annual precipitation (PRCP) trend during WY 1996--2025 across CONUS. Upward triangles identify stations with significant increases in TMAX and TMIN but no significant increase in PRCP (n = {n_not}); circles identify stations with significant increases in all three variables (n = {n_all}). Significance is based on unadjusted modified Mann--Kendall tests at p < 0.05.'
review_path.write_text(text[:start] + caption + text[end:], encoding='utf-8')
print('Updated Figure S3 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
review_path = OUTPUT / 'Supplementary_review.tex'
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
trends = pd.read_csv('data_intermediate_review/mk_two_non_AI_indices_review.csv', dtype={'ID': str})
trends = trends[trends.ID.isin(set(stations.ID)) & trends.window.eq('1996-2025') & trends.variable.isin(['WSDI', 'CSDI'])].copy()
if trends.duplicated(['ID', 'variable']).any() or set(trends.variable) != {'WSDI', 'CSDI'}:
    raise ValueError('Duplicate or missing station-variable trends.')
for variable, group in trends.groupby('variable'):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'Incomplete station set for {variable}.')
trends['p'] = pd.to_numeric(trends.p, errors='coerce')
valid = np.isfinite(trends.p) & trends.p.between(0, 1)
trends['classification'] = None
trends.loc[valid, 'classification'] = 'no significant trend'
for direction in ['increasing', 'decreasing']:
    trends.loc[valid & trends.p.lt(0.05) & trends.trend.eq(direction), 'classification'] = direction
wide = trends.pivot(index='ID', columns='variable', values='classification').reset_index()
s4_stations = stations.merge(wide, on='ID', validate='one_to_one')
valid_both = s4_stations[['WSDI', 'CSDI']].notna().all(axis=1)
n_increasing = s4_stations[['WSDI', 'CSDI']].eq('increasing').sum(axis=1)
n_decreasing = s4_stations[['WSDI', 'CSDI']].eq('decreasing').sum(axis=1)
s4_stations['category'] = 'not displayed'
s4_stations.loc[valid_both & n_decreasing.eq(2), 'category'] = 'both decreasing'
s4_stations.loc[valid_both & n_increasing.eq(1), 'category'] = 'one increasing'
s4_stations.loc[valid_both & n_increasing.eq(2), 'category'] = 'both increasing'
OUTPUT.mkdir(parents=True, exist_ok=True)
s4_stations.to_csv(OUTPUT / 'Figure_S4_WSDI_CSDI_station_classes_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
shape.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.65)
styles = [('both decreasing', 'v', '#17851b', 'Both decreasing'), ('one increasing', '^', '#f4a261', 'One increasing'), ('both increasing', '^', '#f01818', 'Both increasing')]
handles = []
for category, marker, color, label in styles:
    subset = s4_stations[s4_stations.category == category].dropna(subset=['Lat', 'Lon'])
    ax.scatter(subset.Lon, subset.Lat, marker=marker, s=42, c=color, edgecolors='none', zorder=3)
    handles.append(Line2D([0], [0], marker=marker, linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=9, label=f'{label} (n={len(subset)})'))
ax.set_title('WSDI and CSDI combined trend stations', fontsize=17)
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')
ax.legend(handles=handles, loc='lower left', fontsize=12, frameon=True)
fig.savefig(OUTPUT / 'Figure_S4_WSDI_CSDI_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S4_WSDI_CSDI_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print(s4_stations.category.value_counts().to_string())
text = review_path.read_text(encoding='utf-8')
old, new = ('{../supplementary/figures/image4.jpeg}', '{Figure_S4_WSDI_CSDI_review.png}')
if text.count(old) == 1 and text.count(new) == 0:
    text = text.replace(old, new, 1)
elif not (text.count(old) == 0 and text.count(new) == 1):
    raise ValueError('Could not uniquely identify Figure S4 reference.')
start = text.find('Figure S4:')
end = text.find('\\par', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify Figure S4 caption.')
counts = s4_stations.category.value_counts()
caption = f"Figure S4: Spatial distribution of combined station-level trends in the warm spell duration index (WSDI) and cold spell duration index (CSDI) during WY 1996--2025 across CONUS. Stations are classified as having significant decreases in both indices (green downward triangles; n = {counts.get('both decreasing', 0)}), a significant increase in exactly one index (orange upward triangles; n = {counts.get('one increasing', 0)}), or significant increases in both indices (red upward triangles; n = {counts.get('both increasing', 0)}). Significance is based on unadjusted modified Mann--Kendall tests at p < 0.05. WSDI and CSDI represent annual days in warm and cold spells, respectively, relative to the fixed WY 1966--1995 reference period."
review_path.write_text(text[:start] + caption + text[end:], encoding='utf-8')
print('Updated Figure S4 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
review_path = OUTPUT / 'Supplementary_review.tex'
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
if len(stations) != 861 or stations.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations.')
trends = pd.read_csv('data_intermediate_review/mk_two_non_AI_indices_review.csv', dtype={'ID': str})
trends = trends[trends.ID.isin(set(stations.ID)) & trends.window.eq('1996-2025') & trends.variable.isin(['PCI', 'SDII'])].copy()
if trends.duplicated(['ID', 'variable']).any() or set(trends.variable) != {'PCI', 'SDII'}:
    raise ValueError('Duplicate or missing station-variable trends.')
for variable, group in trends.groupby('variable'):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'Incomplete station set for {variable}.')
trends['p'] = pd.to_numeric(trends.p, errors='coerce')
valid = np.isfinite(trends.p) & trends.p.between(0, 1)
trends['classification'] = None
trends.loc[valid, 'classification'] = 'no significant trend'
for direction in ['increasing', 'decreasing']:
    trends.loc[valid & trends.p.lt(0.05) & trends.trend.eq(direction), 'classification'] = direction
wide = trends.pivot(index='ID', columns='variable', values='classification').reset_index()
s5_stations = stations.merge(wide, on='ID', validate='one_to_one')
valid_both = s5_stations[['PCI', 'SDII']].notna().all(axis=1)
n_increasing = s5_stations[['PCI', 'SDII']].eq('increasing').sum(axis=1)
n_decreasing = s5_stations[['PCI', 'SDII']].eq('decreasing').sum(axis=1)
s5_stations['category'] = 'not displayed'
s5_stations.loc[valid_both & n_decreasing.eq(2), 'category'] = 'both decreasing'
s5_stations.loc[valid_both & n_increasing.eq(1), 'category'] = 'one increasing'
s5_stations.loc[valid_both & n_increasing.eq(2), 'category'] = 'both increasing'
OUTPUT.mkdir(parents=True, exist_ok=True)
s5_stations.to_csv(OUTPUT / 'Figure_S5_PCI_SDII_station_classes_review.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
shape.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.65)
styles = [('both decreasing', 'v', '#17851b', 'Both decreasing'), ('one increasing', '^', '#f4a261', 'One increasing'), ('both increasing', '^', '#f01818', 'Both increasing')]
handles = []
for category, marker, color, label in styles:
    subset = s5_stations[s5_stations.category == category].dropna(subset=['Lat', 'Lon'])
    ax.scatter(subset.Lon, subset.Lat, marker=marker, s=42, c=color, edgecolors='none', zorder=3)
    handles.append(Line2D([0], [0], marker=marker, linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=9, label=f'{label} (n={len(subset)})'))
ax.set_title('PCI and SDII combined trend stations', fontsize=17)
ax.set_axis_off()
ax.set_aspect('equal', adjustable='box')
ax.legend(handles=handles, loc='lower left', fontsize=12, frameon=True)
fig.savefig(OUTPUT / 'Figure_S5_PCI_SDII_review.png', dpi=800, bbox_inches='tight')
fig.savefig(OUTPUT / 'Figure_S5_PCI_SDII_review.pdf', dpi=800, bbox_inches='tight')
plt.show()
print(s5_stations.category.value_counts().to_string())
text = review_path.read_text(encoding='utf-8')
old, new = ('{../supplementary/figures/image5.jpeg}', '{Figure_S5_PCI_SDII_review.png}')
if text.count(old) == 1 and text.count(new) == 0:
    text = text.replace(old, new, 1)
elif not (text.count(old) == 0 and text.count(new) == 1):
    raise ValueError('Could not uniquely identify Figure S5 reference.')
start = text.find('Figure S5:')
end = text.find('\\par', start)
if start < 0 or end < 0:
    raise ValueError('Could not identify Figure S5 caption.')
counts = s5_stations.category.value_counts()
caption = f"Figure S5: Spatial distribution of combined station-level trends in the precipitation concentration index (PCI) and simple daily intensity index (SDII) during WY 1996--2025 across CONUS. Stations are classified as having significant decreases in both indices (green downward triangles; n = {counts.get('both decreasing', 0)}), a significant increase in exactly one index (orange upward triangles; n = {counts.get('one increasing', 0)}), or significant increases in both indices (red upward triangles; n = {counts.get('both increasing', 0)}). Significance is based on unadjusted modified Mann--Kendall tests at p < 0.05. PCI describes the intra-annual concentration of precipitation, while SDII describes mean intensity on wet days."
review_path.write_text(text[:start] + caption + text[end:], encoding='utf-8')
print('Updated Figure S5 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
review_path = OUTPUT / 'Supplementary_review.tex'
annual = pd.read_csv('data_intermediate_review/all_indices_station_trends_BH_FDR_review.csv', dtype={'ID': str})
ai = pd.read_csv('AI_SPEI_python/Tables/AI_station_trends_BH_FDR.csv', dtype={'ID': str})
spei = pd.read_csv('AI_SPEI_python/Tables/SPEI_station_trends_BH_FDR.csv', dtype={'ID': str})
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID'], dtype={'ID': str})
domains = pd.read_csv('data_raw/ID_ecoregion.csv', usecols=['ID', 'domainName'], dtype={'ID': str})
domains['domainName'] = domains.domainName.astype('string').str.strip().replace({'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'})
domains = domains[domains.domainName.notna() & domains.domainName.ne('') & domains.ID.isin(stations.ID)]
if len(stations) != 861 or stations.ID.duplicated().any() or domains.ID.duplicated().any():
    raise ValueError('Expected 861 unique study stations and ecoregion assignments.')
metadata = stations.merge(domains, on='ID', how='left', validate='one_to_one')
if metadata.domainName.isna().any():
    raise ValueError('Missing ecoregion for a study station.')
annual = annual[annual.variable.isin(['PRCP', 'TMAX', 'TMIN', 'PCI', 'SDII', 'WSDI', 'CSDI'])].copy()
annual['period'] = annual.window.map({'1926-2025': 'centennial', '1996-2025': 'recent'})
ai['variable'] = 'AI'
spei['variable'] = 'SPEI-' + pd.to_numeric(spei.scale, errors='raise').astype(int).astype(str)
cols = ['ID', 'variable', 'period', 'trend', 'p', 'p_fdr_bh', 'fdr_trend']
data = pd.concat([annual[cols], ai[cols], spei[cols]], ignore_index=True)
data = data[data.ID.isin(set(stations.ID))].copy()
expected_variables = {'PRCP', 'TMAX', 'TMIN', 'PCI', 'SDII', 'WSDI', 'CSDI', 'AI', 'SPEI-1', 'SPEI-3', 'SPEI-6', 'SPEI-12'}
if data.period.isna().any() or set(data.variable) != expected_variables or data.duplicated(['ID', 'variable', 'period']).any():
    raise ValueError('Missing, unexpected or duplicated index-period station records.')
expected_families = {(variable, period) for variable in expected_variables for period in ('recent', 'centennial')}
if set(zip(data.variable, data.period)) != expected_families:
    raise ValueError('Missing index-period families.')
for family, group in data.groupby(['variable', 'period']):
    if set(group.ID) != set(stations.ID):
        raise ValueError(f'Family {family} does not include all 861 study stations.')
for col in ['p', 'p_fdr_bh']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.merge(metadata, on='ID', how='left', validate='many_to_one')
variable_order = ['PRCP', 'SDII', 'PCI', 'TMAX', 'TMIN', 'WSDI', 'CSDI', 'AI', 'SPEI-1', 'SPEI-3', 'SPEI-6', 'SPEI-12']
rows = []
for variable in variable_order:
    for period in ['recent', 'centennial']:
        family = data[(data.variable == variable) & (data.period == period)]
        for region, group in [('CONUS', family)] + list(family.groupby('domainName', sort=True)):
            n = len(group)
            valid_original = np.isfinite(group.p) & group.p.between(0, 1)
            valid_fdr = np.isfinite(group.p_fdr_bh) & group.p_fdr_bh.between(0, 1)
            rows.append({'variable': variable, 'period': period, 'region': region, 'stations': n, 'unadjusted_increasing_percent': 100 * (valid_original & group.p.lt(0.05) & group.trend.eq('increasing')).sum() / n, 'unadjusted_decreasing_percent': 100 * (valid_original & group.p.lt(0.05) & group.trend.eq('decreasing')).sum() / n, 'fdr_increasing_percent': 100 * (valid_fdr & group.p_fdr_bh.le(0.05) & group.fdr_trend.eq('increasing')).sum() / n, 'fdr_decreasing_percent': 100 * (valid_fdr & group.p_fdr_bh.le(0.05) & group.fdr_trend.eq('decreasing')).sum() / n})
fdr_regions = pd.DataFrame(rows)
OUTPUT.mkdir(parents=True, exist_ok=True)
fdr_regions.to_csv(OUTPUT / 'Table_S13_FDR_station_percentages_CONUS_NEON_review.csv', index=False, float_format='%.1f')
print('Created', len(fdr_regions), 'regional index-period rows.')
display(fdr_regions.head(20).round(1))

text = review_path.read_text(encoding='utf-8')
if '\\usepackage{longtable}' not in text:
    text = text.replace('\\usepackage{graphicx,booktabs,pdflscape,geometry,textcomp}', '\\usepackage{graphicx,booktabs,pdflscape,geometry,textcomp}' + '\n' + '\\usepackage{longtable}', 1)
marker = '\\section*{S3. False Discovery Rate Adjustment}'
start = text.find(marker)
if start < 0:
    start = text.find('\\end{document}')
    end = start
else:
    end = text.find('\\section*{S4.', start)
    if end < 0:
        end = text.find('\\end{document}', start)
if start < 0 or end < 0:
    raise ValueError('Could not locate the S3 insertion point.')
table_lines = []
for row in fdr_regions.itertuples(index=False):
    period_label = 'Recent' if row.period == 'recent' else 'Centennial'
    table_lines.append(' & '.join([row.variable, period_label, row.region, f'{row.unadjusted_increasing_percent:.1f}', f'{row.unadjusted_decreasing_percent:.1f}', f'{row.fdr_increasing_percent:.1f}', f'{row.fdr_decreasing_percent:.1f}']) + ' \\\\')
section = marker + '\n' + 'Table S13 compares the percentage of stations with locally significant increasing and decreasing trends before and after Benjamini--Hochberg false discovery rate adjustment. The adjustment uses all valid station p-values across the 861-station network separately for each index and period, at a false discovery rate of 0.05. Regional percentages represent the proportion of study stations in each region. Recent and centennial periods are WY 1996--2025 and WY 1926--2025, respectively.' + '\\par' + '\n' + '\\begingroup\\setcounter{table}{12}\\renewcommand{\\thetable}{S\\arabic{table}}' + '\n' + '\\small' + '\n' + '\\begin{longtable}{lllrrrr}' + '\n' + '\\caption{Station percentages with significant increasing and decreasing trends before and after BH-FDR adjustment by index, period, and ecoregion.}\\\\' + '\n' + '\\toprule' + '\n' + 'Index & Period & Region & Local inc. (\\%) & Local dec. (\\%) & FDR inc. (\\%) & FDR dec. (\\%) \\\\' + '\n' + '\\midrule\\endfirsthead' + '\n' + '\\multicolumn{7}{l}{\\small Table S13 continued}\\\\' + '\n' + '\\toprule' + '\n' + 'Index & Period & Region & Local inc. (\\%) & Local dec. (\\%) & FDR inc. (\\%) & FDR dec. (\\%) \\\\' + '\n' + '\\midrule\\endhead' + '\n' + '\\bottomrule\\endfoot' + '\n' + '\n'.join(table_lines) + '\n' + '\\end{longtable}' + '\n' + '\\normalsize\\endgroup' + '\n'
review_path.write_text(text[:start] + section + text[end:], encoding='utf-8')
(OUTPUT / 'Table_S13_FDR_station_percentages_review.tex').write_text(section, encoding='utf-8')
print('Updated S3 and Table S13 in', review_path.resolve())

OUTPUT = Path('supplementary_review')
OUTPUT.mkdir(parents=True, exist_ok=True)
missing_dir = Path('missing_data_folder')
availability_figure = missing_dir / 'Figures/valid_stations_by_water_year.png'
if not availability_figure.exists():
    raise FileNotFoundError(availability_figure)
shutil.copy2(availability_figure, OUTPUT / 'Figure_S12_observed_data_availability_review.png')
stations = pd.read_csv('df_neon_filter.csv', usecols=['ID', 'Lat', 'Lon'], dtype={'ID': str})
stations.ID = stations.ID.str.strip()
if len(stations) != 861 or stations.ID.duplicated().any() or stations[['Lat', 'Lon']].isna().any().any():
    raise ValueError('Expected 861 unique stations with coordinates.')
variables = ['PRCP', 'SDII', 'TMAX', 'TMIN', 'WSDI', 'CSDI']
original = pd.read_csv('data_intermediate_review/mk_two_non_AI_indices_review.csv', dtype={'ID': str})
original = original[original.ID.isin(stations.ID) & original.variable.isin(variables)].copy()
original['period'] = original.window.map({'1926-2025': 'centennial', '1996-2025': 'recent'})
observed = pd.read_csv(missing_dir / 'Tables/observed_only_station_trends_all_indices.csv', dtype={'ID': str})
observed = observed[observed.ID.isin(stations.ID) & observed.variable.isin(variables)].copy()
families = {(v, p) for v in variables for p in ['centennial', 'recent']}
for label, data in [('original', original), ('observed', observed)]:
    if data.duplicated(['ID', 'variable', 'period']).any() or set(zip(data.variable, data.period)) != families:
        raise ValueError(f'Unexpected or duplicate {label} trend families.')
    for family, group in data.groupby(['variable', 'period']):
        if set(group.ID) != set(stations.ID):
            raise ValueError(f'{label} {family} does not contain all 861 study stations.')
    data['p'] = pd.to_numeric(data.p, errors='coerce')
    data['display_class'] = None
    valid = np.isfinite(data.p) & data.p.between(0, 1)
    data.loc[valid & data.p.ge(0.05), 'display_class'] = 'No trend'
    data.loc[valid & data.p.lt(0.05) & data.trend.eq('increasing'), 'display_class'] = 'Increasing'
    data.loc[valid & data.p.lt(0.05) & data.trend.eq('decreasing'), 'display_class'] = 'Decreasing'
comparison = original[['ID', 'variable', 'period', 'trend', 'p', 'display_class']].merge(observed[['ID', 'variable', 'period', 'trend', 'p', 'display_class', 'n_valid_years']], on=['ID', 'variable', 'period'], suffixes=('_original', '_observed'), validate='one_to_one')
comparison = comparison.merge(stations, on='ID', validate='many_to_one')
comparison.to_csv(OUTPUT / 'S4_missing_data_trend_comparison_stations.csv', index=False)
summary = comparison.groupby(['variable', 'period', 'display_class_original', 'display_class_observed']).size().rename('stations').reset_index()
summary.to_csv(OUTPUT / 'S4_missing_data_trend_comparison_counts.csv', index=False)
shape = gpd.read_file('US_CONUS_Neon_shapefile/US_Neon.shp').to_crs('EPSG:4326')
styles = [('No trend', 'o', '#aaa9a9', 11), ('Increasing', '^', '#16833c', 23), ('Decreasing', 'v', '#d12c2c', 23)]
panels = [('original', 'centennial', 'a) Original (WY 1926-2025)'), ('original', 'recent', 'b) Original (WY 1996-2025)'), ('observed', 'centennial', 'c) Observed data (WY 1926-2025)'), ('observed', 'recent', 'd) Observed data (WY 1996-2025)')]
handles = [Line2D([0], [0], marker=marker, linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=8, label=label) for label, marker, color, size in styles]
for number, variable in enumerate(variables, start=13):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (method, period, title) in zip(axes.flat, panels):
        shape.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.55)
        data = comparison[(comparison.variable == variable) & (comparison.period == period)]
        for label, marker, color, size in styles:
            subset = data[data[f'display_class_{method}'].eq(label)]
            ax.scatter(subset.Lon, subset.Lat, marker=marker, s=size, c=color, linewidths=0, zorder=2 if label == 'No trend' else 3)
        ax.set_title(title, fontsize=13, loc='left')
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.98, wspace=0.04, hspace=0.04)
    stem = OUTPUT / f'Figure_S{number}_{variable}_missing_data_sensitivity_review'
    fig.savefig(stem.with_suffix('.png'), dpi=800, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    display(fig)
    plt.close(fig)
    print('Saved', variable, 'comparison', flush=True)
display(summary)

review_path = OUTPUT / 'Supplementary_review.tex'
text = review_path.read_text(encoding='utf-8')
marker = '\\section*{S4. Sensitivity to Missing Observed Data}'
start = text.find(marker)
if start < 0:
    start = text.find('\\end{document}')
end = text.find('\\end{document}', start)
if start < 0 or end < 0:
    raise ValueError('Could not locate the S4 insertion point.')
section = marker + '\n' + ('To assess sensitivity to missing observations, we recalculated trends using observed daily data for the 861 study stations. For each variable, a water year was retained only when every month had no more than three missing daily observations. Precipitation, maximum temperature, and minimum temperature were assessed separately. Figure S12 shows the number of stations meeting this rule in each water year. Figures S13--S18 compare the original study results with the observed-data sensitivity analysis for the centennial (WY 1926--2025) and recent (WY 1996--2025) periods. Trends are based on locally significant modified Mann--Kendall tests at p < 0.05; gray circles represent stations with no trend. The sensitivity analysis uses observed precipitation totals without adjustment for missing days.' + '\\par' + '\n')
section += '\\begin{center}' + '\n' + '\\includegraphics[width=0.95\\linewidth]{Figure_S12_observed_data_availability_review.png}' + '\n' + 'Figure S12: Number of the 861 stations with valid observed data in each water year, assessed separately for precipitation, maximum temperature, and minimum temperature. A station-water year is valid when no month has more than three missing daily observations.' + '\\par' + '\n' + '\\end{center}' + '\n'
for number, variable in enumerate(variables, start=13):
    section += '\\begin{center}' + '\n' + f'\\includegraphics[width=0.96\\linewidth]{{Figure_S{number}_{variable}_missing_data_sensitivity_review.png}}' + '\n' + f'Figure S{number}: Station-level {variable} trends using the original study data for (a) WY 1926--2025 and (b) WY 1996--2025, and observed data subject to the missing-day rule for (c) WY 1926--2025 and (d) WY 1996--2025. Green upward triangles and red downward triangles show locally significant increases and decreases, respectively (modified Mann--Kendall, p < 0.05); gray circles represent stations with no trend.' + '\\par' + '\n' + '\\end{center}' + '\n'
updated = text[:start] + section + text[end:]
updated = updated.replace('Stations with an unavailable trend classification are omitted from joint classifications.', '')
updated = updated.replace('These area percentages do not represent percentages of stations or independent tests of statistical significance across the region.', 'Area percentages represent Thiessen-assigned land area, whereas station percentages represent the proportion of stations.')
updated = updated.replace('Potential evapotranspiration (PET) represents atmospheric evaporative demand under specified surface and water-availability assumptions and should not be confused with actual evapotranspiration.', 'Potential evapotranspiration (PET) represents atmospheric evaporative demand under specified surface and water-availability assumptions, whereas actual evapotranspiration reflects realized water loss.')
updated = updated.replace('Consequently, Hargreaves-Samani was used as an alternative formulation rather than an assumed error-free reference against which Thornthwaite estimates were evaluated.', 'Hargreaves-Samani provides an alternative PET formulation for assessing the sensitivity of AI and SPEI to PET estimation.')
updated = updated.replace('\\textsuperscript{1})..\\par', '\\textsuperscript{1}).\\par')
references = '\\section*{References}\n\\begin{description}\n\\item Allen, R. G., Pereira, L. S., Raes, D., \\& Smith, M. (1998). \\textit{Crop evapotranspiration: Guidelines for computing crop water requirements} (FAO Irrigation and Drainage Paper No. 56). Food and Agriculture Organization of the United Nations. \\url{https://www.fao.org/4/x0490e/x0490e00.htm}\n\\item Jensen, M. E., \\& Allen, R. G. (Eds.). (2016). \\textit{Evaporation, evapotranspiration, and irrigation water requirements} (2nd ed., ASCE Manuals and Reports on Engineering Practice No. 70). American Society of Civil Engineers.\n\\item Kukal, M. S., Irmak, S., Walia, H., \\& Odhiambo, L. (2020). Spatio-temporal calibration of Hargreaves-Samani model to estimate reference evapotranspiration across U.S. High Plains. \\textit{Agronomy Journal, 112}(5), 4232--4248.\n\\item Thornthwaite, C. W. (1948). An approach toward a rational classification of climate. \\textit{Geographical Review, 38}(1), 55--94.\n\\item United Nations Environment Programme. (1992). \\textit{World atlas of desertification}. Edward Arnold. \\url{https://wedocs.unep.org/handle/20.500.11822/42137}\n\\item van der Schrier, G., Jones, P. D., \\& Briffa, K. R. (2011). The sensitivity of the PDSI to the Thornthwaite and Penman-Monteith parameterizations for potential evapotranspiration. \\textit{Journal of Geophysical Research: Atmospheres, 116}(D3), D03106.\n\\item Vicente-Serrano, S. M., Beguería, S., \\& López-Moreno, J. I. (2010). A multiscalar drought index sensitive to global warming: The standardized precipitation evapotranspiration index. \\textit{Journal of Climate, 23}(7), 1696--1718.\n\\end{description}\n'
if '\\usepackage{url}' not in updated:
    updated = updated.replace('\\usepackage{longtable}', '\\usepackage{longtable}' + '\n' + '\\usepackage{url}', 1)
reference_start = updated.find('\\section*{References}')
document_end = updated.find('\\end{document}')
if document_end < 0:
    raise ValueError('Document end not found.')
if reference_start < 0:
    reference_start = document_end
updated = updated[:reference_start] + references + updated[document_end:]
review_path.write_text(updated, encoding='utf-8')
print('Updated section S4 in', review_path.resolve())
