import pandas as pd
import numpy as np
df_daily = pd.read_csv('./data_raw/updated_df_daily.csv')
df_monthly = pd.read_csv('./data_raw/updated_df_monthy.csv')
df_yearly = pd.read_csv('./data_raw/updated_df_yearly.csv')
df_neon = pd.read_csv('data_raw/Thiessen_Polygons_100yrs.csv')
df_neon = df_neon[['ID_1', 'Lat', 'Lon', 'Area_sqkm']]
df_neon.columns = ['ID', 'Lat', 'Lon', 'Area']
df_monthly_updated = df_monthly.copy()
df_id_neon = pd.read_csv('./data_raw/ID_ecoregion.csv')
df_id_neon_updated = df_id_neon.copy()
replace_map = {'Atlantic Neotropical': 'Southeast', 'Appalachians / Cumberland Plateau': 'Appalachians', 'Southern Rockies / Colorado Plateau': 'Colorado Plateau'}
df_id_neon_updated['domainName'] = df_id_neon_updated['domainName'].replace(replace_map)
df_id_neon_updated = df_id_neon_updated[['ID', 'domainName']]
df_neon_area = df_neon[['ID', 'Area']].merge(df_id_neon_updated, on='ID', how='inner')
df_id_neon_updated.dropna(subset=['domainName'], inplace=True)
df_neon = df_neon[df_neon['ID'].isin(df_id_neon_updated.ID.unique())]
df_monthly_updated = df_monthly_updated[df_monthly_updated['ID'].isin(df_id_neon_updated.ID.unique())]
df_daily = df_daily[df_daily['ID'].isin(df_id_neon_updated.ID.unique())]
