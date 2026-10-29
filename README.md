# CONUS Hydroclimate Workflow

This project is for the manuscript on Recent and Centennial Perspectives on Hydroclimatic Wetting and Drying Across the Contiguous United States.


## Folders

- `data/`: station metadata, including the station list used for downloads.
- `scripts/`: notebooks and scripts used for preprocessing, climate indices, tables, figures, and spatial mapping.
- `data_raw/`: local source and processed CSV inputs needed by the workflow.


## Key inputs

- NOAA daily climate summaries: https://www.ncei.noaa.gov/cdo-web/datasets
- NEON domains shapefile: https://www.neonscience.org/sites/default/files/NEONDomains_2024.zip

## Main workflow order

The workflow is meant to run in this order:

1. `scripts/download_noaa_stations.py`
2. `scripts/Data_preprocessing.ipynb`
3. `scripts/SPEI_updated.R`
4. `scripts/all_indices_average.ipynb`
5. `scripts/all_indices_trend.ipynb`
6. `scripts/spei_indices.ipynb`
7. `scripts/Supplementary_tables.ipynb`
8. `scripts/Supplemnetary_figures.ipynb`
9. `scripts/arcgis_trend_mapper.ipynb` in ArcGIS Pro
10. `scripts/grid_tiff_plot_updated.ipynb`



