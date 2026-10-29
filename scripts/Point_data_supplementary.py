"""Export NEON trend percentages and the 12 point figures for the indices.


"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from openpyxl.utils import get_column_letter

# main() selects the backend before importing pyplot.
plt = None
BASE_DIR = Path(__file__).resolve().parent


def load_inputs(base_dir):
    """Load station metadata and derived trends, without loading raw daily data."""
    domains = pd.read_csv(base_dir / "data_raw" / "ID_ecoregion.csv")
    domains = domains[["ID", "domainName"]].dropna(subset=["domainName"]).copy()
    domains["ID"] = domains["ID"].astype(str).str.strip()
    domains["domainName"] = domains["domainName"].str.strip().replace({
        "Atlantic Neotropical": "Southeast",
        "Appalachians / Cumberland Plateau": "Appalachians",
        "Southern Rockies / Colorado Plateau": "Colorado Plateau",
    })
    domains = domains.drop_duplicates()
    if domains["ID"].duplicated().any():
        raise ValueError("Some station IDs have multiple domain assignments.")
    trends = pd.concat([
        pd.read_csv(base_dir / "data_intermediate" / "mk_two_PRCP_TMAX_TMIN_AI.csv"),
        pd.read_csv(base_dir / "data_intermediate_spei_monthly" / "mk_two_spei_monthly.csv"),
    ], ignore_index=True)
    trends["ID"] = trends["ID"].astype(str).str.strip()
    trends = trends[trends["ID"].isin(domains["ID"])].copy()
    return domains, trends


def export_trend_percentages(df_id_neon_updated, df_trend_all, output_file):
    """Write the original three-sheet workbook using station-count percentages."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    domain_order = ['CONUS', 'Appalachians', 'Central Plains', 'Colorado Plateau', 'Desert Southwest', 'Great Basin', 'Great Lakes', 'Mid Atlantic', 'Northeast', 'Northern Plains', 'Northern Rockies', 'Ozarks Complex', 'Pacific Northwest', 'Pacific Southwest', 'Prairie Peninsula', 'Southeast', 'Southern Plains']
    df_domains = df_id_neon_updated[['ID', 'domainName']].copy()
    df_domains['ID'] = df_domains['ID'].astype(str).str.strip()
    df_domains['domainName'] = df_domains['domainName'].astype(str).str.strip()
    df_domains = df_domains.drop_duplicates()
    if df_domains['ID'].duplicated().any():
        raise ValueError('Some station IDs have multiple domain assignments.')
    df_trends = df_trend_all[['ID', 'variable', 'trend', 'window_type']].copy()
    df_trends['ID'] = df_trends['ID'].astype(str).str.strip()
    df_trends['variable'] = df_trends['variable'].astype(str).str.strip()
    df_trends['trend'] = df_trends['trend'].astype(str).str.strip().str.lower()
    df_trends['window_type'] = df_trends['window_type'].astype(str).str.strip()
    df = df_trends.merge(df_domains, on='ID', how='left', validate='many_to_one')
    if df['domainName'].isna().any():
        missing_ids = df.loc[df['domainName'].isna(), 'ID'].unique()
        raise ValueError(f'{len(missing_ids)} stations have no NEON domain assignment.')
    unknown_domains = set(df['domainName'].unique()) - set(domain_order)
    if unknown_domains:
        raise ValueError(f'Unrecognized NEON domains: {unknown_domains}')
    df['variable_window_type'] = df['variable'] + '_' + df['window_type']
    variables = df_trends['variable'].unique().tolist()
    window_types = ['full_100yr', 'recent_30yr']
    other_windows = [w for w in df_trends['window_type'].unique() if w not in window_types]
    window_types = window_types + other_windows
    column_order = [f'{variable}_{window}' for variable in variables for window in window_types if ((df['variable'] == variable) & (df['window_type'] == window)).any()]
    duplicate_cols = ['ID', 'variable_window_type']
    conflicting_results = df.groupby(duplicate_cols)['trend'].nunique()
    if (conflicting_results > 1).any():
        raise ValueError('Some stations have conflicting trend classifications for the same variable and window.')
    df = df.drop_duplicates(subset=duplicate_cols).copy()
    domain_total = df.groupby(['domainName', 'variable_window_type'])['ID'].nunique().unstack()
    domain_total = domain_total.reindex(index=domain_order[1:], columns=column_order)
    conus_total = df.groupby('variable_window_type')['ID'].nunique().reindex(column_order)

    def calculate_trend_percentages(trend_type):
        df_filtered = df[df['trend'] == trend_type].copy()
        domain_counts = df_filtered.groupby(['domainName', 'variable_window_type'])['ID'].nunique().unstack()
        domain_counts = domain_counts.reindex(index=domain_order[1:], columns=column_order, fill_value=0).fillna(0)
        domain_percent = domain_counts.div(domain_total).mul(100)
        conus_counts = df_filtered.groupby('variable_window_type')['ID'].nunique().reindex(column_order, fill_value=0)
        conus_percent = conus_counts.div(conus_total).mul(100)
        conus_df = pd.DataFrame([conus_percent], index=['CONUS'])
        result = pd.concat([conus_df, domain_percent])
        result = result.reindex(index=domain_order, columns=column_order)
        result = result.round(1)
        result = result.reset_index()
        result = result.rename(columns={'index': 'domainName'})
        return result
    df_increasing = calculate_trend_percentages('increasing')
    df_decreasing = calculate_trend_percentages('decreasing')
    df_no_trend = calculate_trend_percentages('no trend')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_increasing.to_excel(writer, sheet_name='Increasing trend', index=False)
        df_decreasing.to_excel(writer, sheet_name='Decreasing trend', index=False)
        df_no_trend.to_excel(writer, sheet_name='No trend', index=False)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = 'B2'
            ws.auto_filter.ref = ws.dimensions
            ws.column_dimensions['A'].width = 24
            for col_idx in range(2, ws.max_column + 1):
                col_letter = get_column_letter(col_idx)
                ws.column_dimensions[col_letter].width = 22
                for row_idx in range(2, ws.max_row + 1):
                    ws.cell(row=row_idx, column=col_idx).number_format = '0.0'
    print(f'Excel file saved successfully:\n{output_file}')


def prepare_point_gdf(df, lon_col="Lon", lat_col="Lat", target_crs=None):
    df = df.copy()
    df = df.dropna(subset=[lon_col, lat_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


## updated generalized figures

# UPDATED GENERALIZED CLIMATE POINT PLOTTING FUNCTION
#
# Improvements:
# 1. Independent class boundaries for all six panels
# 2. Independent colors for each panel
# 3. Independent color orientation
# 4. Custom panel titles and colorbar labels
# 5. State and NEON boundaries above station points
# 6. Trend slope conversion from per year to per decade

def plot_climate_point_panel(
    df_averages_diffs,
    df_trend_all,
    df_neon,
    neon_shape,
    state_shape,

    variable="PRCP",
    trend_variable=None,

    # Individual panel settings
    panel_bounds=None,
    panel_colors=None,
    panel_reverse=None,
    panel_titles=None,
    cbar_labels=None,

    # Point settings
    value_point_size=18,
    trend_point_size=28,

    # Slope conversion
    slope_to_decade=True,

    # Figure settings
    figsize=(14, 16),
    dpi=600,

    # Output
    save_path=None
):

    """
    Generic six-panel climate point plotting function.

    Panel arrangement:

        a) Recent 30-year mean
        b) Centennial mean

        c) Recent 30-year trend
        d) Centennial trend

        e) Recent 30-year minus prior 30-year mean
        f) Recent 30-year minus centennial mean

    Each panel can have its own:
        - Class boundaries
        - Colors
        - Color orientation
        - Title
        - Colorbar label

    Trend marker classification:

        o = No trend
        ^ = Increasing
        v = Decreasing

    Station points: zorder = 2
    State boundaries: zorder = 3
    NEON boundaries: zorder = 4
    """

    # 1. DEFAULT SETTINGS

    if trend_variable is None:
        trend_variable = variable

    if panel_reverse is None:

        panel_reverse = {
            letter: False
            for letter in "abcdef"
        }

    if panel_titles is None:

        panel_titles = {
            letter: ""
            for letter in "abcdef"
        }

    if cbar_labels is None:

        cbar_labels = {
            letter: ""
            for letter in "abcdef"
        }

    # 2. VALIDATE PANEL SETTINGS

    if panel_bounds is None or panel_colors is None:

        raise ValueError(
            "Please provide panel_bounds and panel_colors."
        )

    for letter in "abcdef":

        if letter not in panel_bounds:
            raise ValueError(
                f"Missing class boundaries for panel {letter}"
            )

        if letter not in panel_colors:
            raise ValueError(
                f"Missing colors for panel {letter}"
            )

        if len(panel_bounds[letter]) != (
            len(panel_colors[letter]) + 1
        ):

            raise ValueError(
                f"Panel {letter}: number of boundaries must "
                "equal number of colors + 1."
            )

    # 3. PREPARE STATION COORDINATES

    df_coords = df_neon[
        ["ID", "Lat", "Lon"]
    ].copy()

    df_coords["ID"] = (
        df_coords["ID"]
        .astype(str)
        .str.strip()
    )

    # Prevent duplicate station coordinates
    df_coords = df_coords.drop_duplicates(
        subset=["ID"]
    )

    # 4. PREPARE MEAN AND DIFFERENCE DATA

    recent_mean_col = f"{variable}_recent_mean"

    full_mean_col = f"{variable}_full_mean"

    diff_penult_col = (
        f"{variable}_recent_minus_penult"
    )

    diff_full_col = (
        f"{variable}_recent_minus_full"
    )

    avg_cols = [
        "ID",
        recent_mean_col,
        full_mean_col,
        diff_penult_col,
        diff_full_col
    ]

    df_vals = df_averages_diffs[
        avg_cols
    ].copy()

    df_vals["ID"] = (
        df_vals["ID"]
        .astype(str)
        .str.strip()
    )

    df_vals = df_vals.merge(
        df_coords,
        on="ID",
        how="left",
        validate="many_to_one"
    )

    # Convert values to numeric
    for col in avg_cols[1:]:

        df_vals[col] = pd.to_numeric(
            df_vals[col],
            errors="coerce"
        )

    gdf_vals = prepare_point_gdf(
        df_vals,
        lon_col="Lon",
        lat_col="Lat",
        target_crs=neon_shape.crs
    )

    # 5. PREPARE TREND DATA

    df_tr = df_trend_all[
        [
            "ID",
            "variable",
            "trend",
            "slope",
            "window_type"
        ]
    ].copy()

    df_tr["ID"] = (
        df_tr["ID"]
        .astype(str)
        .str.strip()
    )

    df_tr["variable"] = (
        df_tr["variable"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_tr["trend"] = (
        df_tr["trend"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df_tr["window_type"] = (
        df_tr["window_type"]
        .astype(str)
        .str.strip()
    )

    # Filter selected variable
    df_tr = df_tr[
        df_tr["variable"]
        == str(trend_variable).upper()
    ].copy()

    # Merge coordinates
    df_tr = df_tr.merge(
        df_coords,
        on="ID",
        how="left",
        validate="many_to_one"
    )

    # Convert slope to numeric
    df_tr["slope"] = pd.to_numeric(
        df_tr["slope"],
        errors="coerce"
    )

    # Convert annual slope to decadal slope
    if slope_to_decade:

        df_tr["slope_plot"] = (
            df_tr["slope"] * 10.0
        )

    else:

        df_tr["slope_plot"] = (
            df_tr["slope"]
        )

    # Convert to GeoDataFrame
    gdf_tr = prepare_point_gdf(
        df_tr,
        lon_col="Lon",
        lat_col="Lat",
        target_crs=neon_shape.crs
    )

    # Separate recent and centennial trends
    gdf_tr_recent = gdf_tr[
        gdf_tr["window_type"] == "recent_30yr"
    ].copy()

    gdf_tr_full = gdf_tr[
        gdf_tr["window_type"] == "full_100yr"
    ].copy()

    # 6. ORGANIZE PANEL DATA

    panel_data = {

        "a": {
            "gdf": gdf_vals,
            "value_col": recent_mean_col,
            "type": "value"
        },

        "b": {
            "gdf": gdf_vals,
            "value_col": full_mean_col,
            "type": "value"
        },

        "c": {
            "gdf": gdf_tr_recent,
            "value_col": "slope_plot",
            "type": "trend"
        },

        "d": {
            "gdf": gdf_tr_full,
            "value_col": "slope_plot",
            "type": "trend"
        },

        "e": {
            "gdf": gdf_vals,
            "value_col": diff_penult_col,
            "type": "value"
        },

        "f": {
            "gdf": gdf_vals,
            "value_col": diff_full_col,
            "type": "value"
        }

    }

    # 7. CREATE FIGURE

    # fig, axes = plt.subplots(
    #     3,
    #     2,
    #     figsize=figsize
    # )

    # plt.subplots_adjust(
    #     wspace=0.08,
    #     hspace=0.22
    # )
    fig, axes = plt.subplots(
        3,
        2,
        figsize=figsize
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.98,
        bottom=0.02,
        wspace=0.04,
        hspace=0.02
    )
    # 8. PLOT EACH PANEL

    marker_map = {
        "no trend": "o",
        "increasing": "^",
        "decreasing": "v"
    }

    trend_order = [
        "no trend",
        "increasing",
        "decreasing"
    ]

    for i, letter in enumerate("abcdef"):

        ax = axes.flat[i]

        current = panel_data[letter]

        gdf = current["gdf"]

        value_col = current["value_col"]

        panel_type = current["type"]

        # ----------------------------------------------------
        # A. INDIVIDUAL PANEL COLORMAP
        # ----------------------------------------------------

        bounds = panel_bounds[letter]

        colors = panel_colors[letter].copy()

        reverse = panel_reverse.get(
            letter,
            False
        )

        if reverse:

            colors = colors[::-1]

        cmap = ListedColormap(colors)

        norm = BoundaryNorm(
            bounds,
            cmap.N,
            clip=True
        )

        # ----------------------------------------------------
        # B. PLOT STATION POINTS
        # ----------------------------------------------------

        if panel_type == "value":

            sub = gdf.dropna(
                subset=[value_col]
            )

            ax.scatter(
                sub.geometry.x,
                sub.geometry.y,

                c=sub[value_col],

                cmap=cmap,
                norm=norm,

                s=value_point_size,

                marker="o",

                edgecolor="black",
                linewidth=0.2,

                alpha=0.95,

                zorder=2
            )

        else:

            # Plot all stations according to trend class
            for trend_type in trend_order:

                sub = gdf[
                    gdf["trend"] == trend_type
                ].copy()

                sub = sub.dropna(
                    subset=[value_col]
                )

                if sub.empty:
                    continue

                ax.scatter(
                    sub.geometry.x,
                    sub.geometry.y,

                    c=sub[value_col],

                    cmap=cmap,
                    norm=norm,

                    s=trend_point_size,

                    marker=marker_map[trend_type],

                    edgecolor="black",
                    linewidth=0.25,

                    alpha=0.95,

                    zorder=2
                )

        # ----------------------------------------------------
        # C. STATE BOUNDARIES
        # ----------------------------------------------------

        state_shape.boundary.plot(
            ax=ax,

            color="gray",

            linewidth=0.5,

            linestyle="--",

            alpha=0.65,

            zorder=3
        )

        # ----------------------------------------------------
        # D. NEON BOUNDARIES
        # ----------------------------------------------------

        neon_shape.boundary.plot(
            ax=ax,

            color="black",

            linewidth=1.0,

            linestyle="-",

            alpha=1.0,

            zorder=4
        )

        # ----------------------------------------------------
        # E. PANEL LABEL
        # ----------------------------------------------------

        ax.text(
            0.01,
            0.98,

            f"{letter})",

            transform=ax.transAxes,

            ha="left",
            va="top",

            fontsize=16,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # F. PANEL TITLE
        # ----------------------------------------------------

        ax.set_title(
            panel_titles.get(letter, ""),

            fontsize=11,
            fontweight="bold"
        )

        ax.set_axis_off()

        # ----------------------------------------------------
        # G. COLORBAR
        # ----------------------------------------------------

        sm = ScalarMappable(
            norm=norm,
            cmap=cmap
        )

        sm.set_array([])

        cbar = fig.colorbar(
            sm,

            ax=ax,

            orientation="horizontal",

            fraction=0.045,

            pad=0.02,

            boundaries=bounds,

            ticks=bounds,

            spacing="uniform"
        )

        # Place the label above the colorbar
        cbar.ax.set_title(
            cbar_labels.get(letter, ""),

            fontsize=9,

            fontweight="bold",

            loc="left",

            pad=4
        )

        cbar.ax.tick_params(
            labelsize=8,

            length=0,

            rotation=45
        )

        # ----------------------------------------------------
        # H. TREND MARKER LEGEND
        # ----------------------------------------------------

        if panel_type == "trend":

            handles = [

                Line2D(
                    [0], [0],

                    marker="o",

                    linestyle="None",

                    label="No trend",

                    markerfacecolor="lightgray",

                    markeredgecolor="black",

                    markersize=6
                ),

                Line2D(
                    [0], [0],

                    marker="^",

                    linestyle="None",

                    label="Increasing",

                    markerfacecolor="lightgray",

                    markeredgecolor="black",

                    markersize=7
                ),

                Line2D(
                    [0], [0],

                    marker="v",

                    linestyle="None",

                    label="Decreasing",

                    markerfacecolor="lightgray",

                    markeredgecolor="black",

                    markersize=7
                )

            ]

            ax.legend(
                handles=handles,

                loc="lower left",

                fontsize=8,

                frameon=True
            )

    # 9. SAVE FIGURE

    if save_path is not None:

        out_dir = os.path.dirname(save_path)

        if out_dir:

            os.makedirs(
                out_dir,
                exist_ok=True
            )

        fig.savefig(
            save_path,

            dpi=dpi,

            bbox_inches="tight",

            facecolor="white"
        )

        print(
            f"Figure saved successfully:\n{save_path}"
        )

    # main() handles interactive display and closes figures after saving.

    return fig, axes

# Final notebook settings: class limits, colors, and labels preserved.
PANEL_SETTINGS = {'SDII': {'panel_bounds': {'a': [4, 6, 9, 12, 15, 21],
                           'b': [3.9, 6, 9, 12, 15, 21],
                           'c': [-2.0, 0, 0.3, 0.6, 1.0, 3.6],
                           'd': [-0.9, 0, 0.3, 0.6],
                           'e': [-2.4, 0, 0.3, 0.6, 1.7, 3.8],
                           'f': [-2.9, 0, 0.3, 0.6, 1.7, 2.6]},
          'panel_colors': {'a': ['#FF0000', '#FF8000', '#FFFF00', '#8BD000', '#1F9D00'],
                           'b': ['#FF0000', '#FF8000', '#FFFF00', '#8BD000', '#1F9D00'],
                           'c': ['#FF0000', '#FF8000', '#FFFF00', '#8BD000', '#1F9D00'],
                           'd': ['#FF0000', '#FF8000', '#FFFF00'],
                           'e': ['#FF0000', '#FF8000', '#FFFF00', '#8BD000', '#1F9D00'],
                           'f': ['#FF0000', '#FF8000', '#FFFF00', '#8BD000', '#1F9D00']},
          'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
          'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
          'cbar_labels': {'a': 'SDII (mm day$^{-1}$, 30-y)',
                          'b': 'SDII (mm day$^{-1}$, 100-y)',
                          'c': 'Slope (mm day$^{-1}$ decade$^{-1}$, 30-y)',
                          'd': 'Slope (mm day$^{-1}$ decade$^{-1}$, 100-y)',
                          'e': '$\\Delta$ (mm day$^{-1}$, 30-y - prior 30-y)',
                          'f': '$\\Delta$ (mm day$^{-1}$, 30-y - 100-y)'}},
 'PCI': {'panel_bounds': {'a': [9.5, 10, 13, 15, 20, 35],
                          'b': [9.5, 10, 13, 15, 20, 35],
                          'c': [-3.0, -0.4, 0.2, 0.6, 1.2, 4.0],
                          'd': [-0.8, -0.4, -0.2, 0, 0.2, 0.6],
                          'e': [-2.7, -0.5, 0, 0.5, 1.5, 6.5],
                          'f': [-3.1, -0.5, 0, 0.5, 1.0, 4.4]},
         'panel_colors': {'a': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                          'b': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                          'c': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                          'd': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                          'e': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                          'f': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000']},
         'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
         'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
         'cbar_labels': {'a': 'PCI (unitless, 30-y)',
                         'b': 'PCI (unitless, 100-y)',
                         'c': 'Slope (PCI units decade$^{-1}$, 30-y)',
                         'd': 'Slope (PCI units decade$^{-1}$, 100-y)',
                         'e': '$\\Delta$ (PCI units, 30-y - prior 30-y)',
                         'f': '$\\Delta$ (PCI units, 30-y - 100-y)'}},
 'TMAX': {'panel_bounds': {'a': [9.4, 13, 16, 19, 22, 25, 30],
                           'b': [9, 13, 16, 19, 22, 25, 30],
                           'c': [-0.9, 0, 0.1, 0.2, 0.3, 1.3],
                           'd': [-0.3, 0, 0.1, 0.2, 0.3, 0.4],
                           'e': [-1.1, -0.4, 0, 0.4, 0.8, 2.0],
                           'f': [-1.2, -0.4, 0, 0.4, 0.8, 1.7]},
          'panel_colors': {'a': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'b': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'c': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'd': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'e': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'f': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000']},
          'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
          'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
          'cbar_labels': {'a': 'TMAX (°C, 30-y)',
                          'b': 'TMAX (°C, 100-y)',
                          'c': 'Slope (°C decade$^{-1}$, 30-y)',
                          'd': 'Slope (°C decade$^{-1}$, 100-y)',
                          'e': '$\\Delta$ (°C, 30-y - prior 30-y)',
                          'f': '$\\Delta$ (°C, 30-y - 100-y)'}},
 'TMIN': {'panel_bounds': {'a': [-7, 0, 3, 6, 9, 12, 21],
                           'b': [-7, 0, 3, 6, 9, 12, 20],
                           'c': [-1.2, 0, 0.2, 0.4, 0.6, 2.0],
                           'd': [-0.5, 0, 0.2, 0.4, 0.6, 1.7],
                           'e': [-1.6, 0, 0.4, 0.8, 1.2, 2.5],
                           'f': [0, 0.4, 0.8, 1.2, 2.6]},
          'panel_colors': {'a': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'b': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'c': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'd': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'e': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'f': ['#00A000', '#80D000', '#FFFF00', '#FF8C00']},
          'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
          'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
          'cbar_labels': {'a': 'TMIN (°C, 30-y)',
                          'b': 'TMIN (°C, 100-y)',
                          'c': 'Slope (°C decade$^{-1}$, 30-y)',
                          'd': 'Slope (°C decade$^{-1}$, 100-y)',
                          'e': '$\\Delta$ (°C, 30-y - prior 30-y)',
                          'f': '$\\Delta$ (°C, 30-y - 100-y)'}},
 'WSDI': {'panel_bounds': {'a': [3, 15, 25, 35, 45, 55, 83],
                           'b': [3, 15, 25, 35, 45, 55, 83],
                           'c': [-22, 0, 3, 6, 10, 12],
                           'd': [-8, 0, 3, 6, 10, 12],
                           'e': [-13, 0, 10, 20, 30, 58],
                           'f': [-19, 0, 10, 20, 30, 38]},
          'panel_colors': {'a': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'b': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'c': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'd': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'e': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'f': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000']},
          'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
          'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
          'cbar_labels': {'a': 'WSDI (days, 30-y)',
                          'b': 'WSDI (days, 100-y)',
                          'c': 'Slope (days decade$^{-1}$, 30-y)',
                          'd': 'Slope (days decade$^{-1}$, 100-y)',
                          'e': '$\\Delta$ (days, 30-y - prior 30-y)',
                          'f': '$\\Delta$ (days, 30-y - 100-y)'}},
 'CSDI': {'panel_bounds': {'a': [1, 7, 10, 13, 16, 20, 52],
                           'b': [1, 7, 10, 13, 16, 20, 52],
                           'c': [-15, 0, 2, 4, 7, 19],
                           'd': [-4.4, 0, 2, 4, 7, 19],
                           'e': [-23, -5, 0, 5, 10, 40],
                           'f': [-49, -5, 0, 5, 10, 25]},
          'panel_colors': {'a': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'b': ['#00A000', '#80D000', '#C0E800', '#FFFF00', '#FF8C00', '#FF0000'],
                           'c': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'd': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'e': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000'],
                           'f': ['#00A000', '#80D000', '#FFFF00', '#FF8C00', '#FF0000']},
          'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
          'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
          'cbar_labels': {'a': 'CSDI (days, 30-y)',
                          'b': 'CSDI (days, 100-y)',
                          'c': 'Slope (days decade$^{-1}$, 30-y)',
                          'd': 'Slope (days decade$^{-1}$, 100-y)',
                          'e': '$\\Delta$ (days, 30-y - prior 30-y)',
                          'f': '$\\Delta$ (days, 30-y - 100-y)'}},
 'AI': {'panel_bounds': {'a': [0.08, 0.2, 0.5, 0.65, 1.5, 5.85],
                         'b': [0.08, 0.2, 0.5, 0.65, 1.5, 5.85],
                         'c': [-0.4, -0.08, -0.01, 0, 0.01, 0.2],
                         'd': [-0.083, -0.01, -0.001, 0, 0.01, 0.2],
                         'e': [-0.4, -0.05, 0, 0.01, 0.04, 0.35],
                         'f': [-0.34, -0.05, 0, 0.01, 0.04, 0.3]},
        'panel_colors': {'a': ['#FF0000', '#FF8C00', '#FFFF00', '#C0E800', '#80D000'],
                         'b': ['#FF0000', '#FF8C00', '#FFFF00', '#C0E800', '#80D000'],
                         'c': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                         'd': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                         'e': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                         'f': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000']},
        'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
        'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
        'cbar_labels': {'a': 'AI (unitless, 30-y)',
                        'b': 'AI (unitless, 100-y)',
                        'c': 'Slope (AI units decade$^{-1}$, 30-y)',
                        'd': 'Slope (AI units decade$^{-1}$, 100-y)',
                        'e': '$\\Delta$ (AI units, 30-y - prior 30-y)',
                        'f': '$\\Delta$ (AI units, 30-y - 100-y)'}},
 'SPEI1': {'panel_bounds': {'a': [15, 20, 24, 28, 32, 37],
                            'b': [17, 20, 24, 28, 32, 35],
                            'c': [-0.02, -0.004, 0.002, 0.004, 0.006, 0.16],
                            'd': [-0.02, -0.004, 0.002, 0.004, 0.006, 0.06],
                            'e': [-18, -3, 0, 3, 6, 17],
                            'f': [-12, -3, 0, 3, 6, 13]},
           'panel_colors': {'a': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'b': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'c': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'd': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'e': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'f': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000']},
           'panel_reverse': {'a': False,
                             'b': False,
                             'c': False,
                             'd': False,
                             'e': False,
                             'f': False},
           'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
           'cbar_labels': {'a': 'SPEI-1 (months decade$^{-1}$, 30-y)',
                           'b': 'SPEI-1 (months decade$^{-1}$, 100-y)',
                           'c': 'Slope (SPEI units decade$^{-1}$, 30-y)',
                           'd': 'Slope (SPEI units decade$^{-1}$, 100-y)',
                           'e': '$\\Delta$ (months decade$^{-1}$, 30-y - prior 30-y)',
                           'f': '$\\Delta$ (months decade$^{-1}$, 30-y - 100-y)'}},
 'SPEI3': {'panel_bounds': {'a': [13, 20, 24, 28, 32, 40],
                            'b': [19, 20, 24, 28, 32, 45],
                            'c': [-0.02, -0.006, -0.003, 0, 0.001, 0.03],
                            'd': [-0.02, -0.006, -0.003, 0, 0.001, 0.008],
                            'e': [-17, -3, 0, 3, 6, 15],
                            'f': [-13, -3, 0, 3, 6, 12]},
           'panel_colors': {'a': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'b': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'c': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'd': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'e': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'f': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000']},
           'panel_reverse': {'a': False,
                             'b': False,
                             'c': False,
                             'd': False,
                             'e': False,
                             'f': False},
           'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
           'cbar_labels': {'a': 'SPEI-3 (months decade$^{-1}$, 30-y)',
                           'b': 'SPEI-3 (months decade$^{-1}$, 100-y)',
                           'c': 'Slope (SPEI units decade$^{-1}$, 30-y)',
                           'd': 'Slope (SPEI units decade$^{-1}$, 100-y)',
                           'e': '$\\Delta$ (months decade$^{-1}$, 30-y - prior 30-y)',
                           'f': '$\\Delta$ (months decade$^{-1}$, 30-y - 100-y)'}},
 'SPEI6': {'panel_bounds': {'a': [11, 20, 24, 28, 32, 58],
                            'b': [18, 20, 24, 28, 32, 60],
                            'c': [-0.01, -0.004, -0.002, 0, 0.002, 0.025],
                            'd': [-0.005, -0.004, -0.002, 0, 0.002, 0.009],
                            'e': [-19, -3, 0, 3, 6, 18],
                            'f': [-13, -3, 0, 3, 6, 11]},
           'panel_colors': {'a': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'b': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'c': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'd': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'e': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                            'f': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000']},
           'panel_reverse': {'a': False,
                             'b': False,
                             'c': False,
                             'd': False,
                             'e': False,
                             'f': False},
           'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
           'cbar_labels': {'a': 'SPEI-6 (months decade$^{-1}$, 30-y)',
                           'b': 'SPEI-6 (months decade$^{-1}$, 100-y)',
                           'c': 'Slope (SPEI units decade$^{-1}$, 30-y)',
                           'd': 'Slope (SPEI units decade$^{-1}$, 100-y)',
                           'e': '$\\Delta$ (months decade$^{-1}$, 30-y - prior 30-y)',
                           'f': '$\\Delta$ (months decade$^{-1}$, 30-y - 100-y)'}},
 'SPEI12': {'panel_bounds': {'a': [6, 20, 24, 28, 32, 54],
                             'b': [18, 20, 24, 28, 32, 36],
                             'c': [-0.03, -0.004, 0, 0.002, 0.004, 0.04],
                             'd': [-0.005, -0.004, 0, 0.002, 0.004, 0.01],
                             'e': [-23, -3, 0, 3, 6, 20],
                             'f': [-17, -3, 0, 3, 6, 15]},
            'panel_colors': {'a': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                             'b': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                             'c': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                             'd': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                             'e': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                             'f': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000']},
            'panel_reverse': {'a': False,
                              'b': False,
                              'c': False,
                              'd': False,
                              'e': False,
                              'f': False},
            'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
            'cbar_labels': {'a': 'SPEI-12 (months decade$^{-1}$, 30-y)',
                            'b': 'SPEI-12 (months decade$^{-1}$, 100-y)',
                            'c': 'Slope (SPEI units decade$^{-1}$, 30-y)',
                            'd': 'Slope (SPEI units decade$^{-1}$, 100-y)',
                            'e': '$\\Delta$ (months decade$^{-1}$, 30-y - prior 30-y)',
                            'f': '$\\Delta$ (months decade$^{-1}$, 30-y - 100-y)'}},
 'PRCP': {'panel_bounds': {'a': [91, 350, 500, 700, 1000, 1400, 3080],
                           'b': [91, 350, 500, 700, 1000, 1400, 3080],
                           'c': [-280, -50, 0, 50, 260],
                           'd': [-57, -50, 0, 50, 260],
                           'e': [-205, -50, 0, 50, 100, 243],
                           'f': [-175, -50, 0, 50, 100, 181]},
          'panel_colors': {'a': ['#FF0000', '#FF8C00', '#FFD000', '#FFFF00', '#80D000', '#00A000'],
                           'b': ['#FF0000', '#FF8C00', '#FFD000', '#FFFF00', '#80D000', '#00A000'],
                           'c': ['#FF0000', '#FF8C00', '#FFFF00', '#00A000'],
                           'd': ['#FF0000', '#FF8C00', '#FFFF00', '#00A000'],
                           'e': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000'],
                           'f': ['#FF0000', '#FF8C00', '#FFFF00', '#80D000', '#00A000']},
          'panel_reverse': {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False},
          'panel_titles': {'a': '', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''},
          'cbar_labels': {'a': 'PRCP (mm, 30-y)',
                          'b': 'PRCP (mm, 100-y)',
                          'c': 'Slope (mm decade$^{-1}$, 30-y)',
                          'd': 'Slope (mm decade$^{-1}$, 100-y)',
                          'e': '$\\Delta$ (mm, 30-y - prior 30-y)',
                          'f': '$\\Delta$ (mm, 30-y - 100-y)'}}}


def main(argv=None):
    """Export the workbook and save each final supplementary point figure."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=BASE_DIR,
                        help="Parent directory for Point_data_output and Figures_point.")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Figure resolution (default: 600; use 100 for previews).")
    parser.add_argument("--excel-only", action="store_true", help="Skip figure generation.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    global plt
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    plt = pyplot

    domains, trends = load_inputs(BASE_DIR)
    output_root = args.output_root.resolve()
    workbook = output_root / "Point_data_output" / "NEON_station_trend_percentages.xlsx"
    if not args.excel_only:
        averages = pd.read_csv(BASE_DIR / "data_intermediate" / "all_indices_station_stats.csv")
        averages["ID"] = averages["ID"].astype(str).str.strip()
        averages = averages[averages["ID"].isin(domains["ID"])].copy()
        stations = pd.read_csv(BASE_DIR / "data_raw" / "Thiessen_Polygons_100yrs.csv")
        stations = stations[["ID_1", "Lat", "Lon", "Area_sqkm"]].rename(
            columns={"ID_1": "ID", "Area_sqkm": "Area"})
        stations["ID"] = stations["ID"].astype(str).str.strip()
        stations = stations[stations["ID"].isin(domains["ID"])].copy()
        neon_shape = gpd.read_file(BASE_DIR / "US_CONUS_Neon_shapefile" / "US_Neon.shp")
        state_shape = gpd.read_file(BASE_DIR / "US_CONUS_State_shapefile" / "US_CONUS.shp")
        if state_shape.crs != neon_shape.crs:
            state_shape = state_shape.to_crs(neon_shape.crs)

    export_trend_percentages(domains, trends, workbook)
    if args.excel_only:
        return

    for variable, settings in PANEL_SETTINGS.items():
        filename = variable.replace("SPEI", "SPEI-") + "_point_panel.png"
        print(f"Rendering {variable}...")
        fig, _ = plot_climate_point_panel(
            df_averages_diffs=averages,
            df_trend_all=trends,
            df_neon=stations,
            neon_shape=neon_shape,
            state_shape=state_shape,
            variable=variable,
            trend_variable=variable,
            **settings,
            value_point_size=18,
            trend_point_size=28,
            slope_to_decade=True,
            figsize=(14, 16),
            dpi=args.dpi,
            save_path=output_root / "Figures_point" / filename,
        )
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
