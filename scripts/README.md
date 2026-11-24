
# Scripts Guide

This folder contains the notebooks and Python scripts used for this project. The files are arranged so the workflow can be run from this folder using relative paths only.

## Recommended run order

1. `download_noaa_stations.py`
2. `data_loader.py`
3. `all_indices_average.py`
4. `all_indices_trend.py`
5. `PET_calculations.py`
6. `ai_spei.py`
7. `supplementary.py`
8. `arcgis_trend_mapper.ipynb` inside ArcGIS Pro
9. `grid_tiff_plot_updated.ipynb`

Alternatively, `run_all.py` can be used to execute the Python workflow.

## File roles

- `download_noaa_stations.py`: Downloads and assembles NOAA daily station data.
- `data_loader.py`: Loads station metadata and climate datasets.
- `all_indices_average.py`: Computes station and ecoregion averages for all hydroclimatic indices except AI and SPEI.
- `all_indices_trend.py`: Computes modified Mann-Kendall trends for all hydroclimatic indices except AI and SPEI, including BH-FDR sensitivity analysis.
- `PET_calculations.py`: Calculates Thornthwaite and Hargreaves-Samani PET and generates PET sensitivity comparison outputs.
- `ai_spei.py`: Computes AI and SPEI indices, trends, drought-frequency summaries, and BH-FDR sensitivity results.
- `supplementary.py`: Generates supplementary tables, figures, and sensitivity analysis outputs for the manuscript.
- `run_all.py`: Executes the Python scripts in the defined workflow sequence.
- `arcgis_trend_mapper.ipynb`: ArcGIS Pro mapping workflow for trend layers and labels.
- `grid_tiff_plot_updated.ipynb`: Final grid and TIFF figure assembly.
