#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3x2 figure (3 rows, 2 columns):
Row 1: a) UVEL diff (FH=0.18Sv, annual)   b) MSF diff (FH=0.18Sv, annual)
Row 2: c) UVEL diff (FH=0.18Sv, DJF)     d) MSF diff (FH=0.18Sv, DJF)
Row 3: e) UVEL diff (FH=0.18Sv, JJA)     f) MSF diff (FH=0.18Sv, JJA)

- NO shared colorbars: each subplot has its own colorbar
- x-labels visible on EVERY subplot
- contours: baseline mean (branch600) in black
- filled: (branch3800 - branch600)
- open circles: NON-significant differences (Welch significance < 0.95)
"""

#%%

import numpy as np
import netCDF4 as netcdf
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- Paths ----------------
directory_data     = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures  = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'
region             = 'global'

# ---------------- Welch test (returns significance level, e.g. 0.95 = 95%) ----------------
def Welch(data_1, data_2):
    """Conducts Welch t-test; returns highest significance level achieved (0..1)."""

    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)

    std_1 = np.sqrt(1.0 / (len(data_1) - 1) * np.sum((data_1 - mean_1) ** 2.0))
    std_2 = np.sqrt(1.0 / (len(data_2) - 1) * np.sum((data_2 - mean_2) ** 2.0))

    t_welch = (mean_1 - mean_2) / np.sqrt((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2)))

    dof = (
        ((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2))) ** 2.0
        / (
            (std_1**4.0 / (len(data_1) ** 2.0 * (len(data_1) - 1)))
            + (std_2**4.0 / (len(data_2) ** 2.0 * (len(data_2) - 1)))
        )
    )

    sig_levels = np.arange(50, 100, 0.5) / 100.0
    t_crit = stats.t.ppf((1.0 + sig_levels) / 2.0, dof)

    sig_index = np.where(np.fabs(t_welch) > t_crit)[0]
    significant = 0.0
    if len(sig_index) > 0:
        significant = sig_levels[sig_index[-1]]

    return significant

# ---------------- Loaders ----------------
def read_uvel(month_start, month_end, branch, year_start, year_end, region="global", time_slice=None):
    fn = (
        f"{directory_data}UVEL_{region}_month_{month_start}-{month_end}_"
        f"branch{branch}_year_{year_start}_{year_end}.nc"
    )
    ds = netcdf.Dataset(fn, "r")
    time = ds.variables["time"][:]
    lev  = ds.variables["lev"][:]
    lat  = ds.variables["lat"][:]
    U    = ds.variables["U"][:]  # (time, lev, lat)
    ds.close()

    if time_slice is not None:
        time = time[time_slice]
        U    = U[time_slice, :, :]

    return time, lev, lat, U

def read_sf(month_start, month_end, branch, year_start, year_end, time_slice=None):
    # NOTE: your SF filenames use year_999-1100 (dash), not year_999_1100 (underscore)
    fn = f"{directory_data}Streamfunction_month_{month_start}-{month_end}_branch{branch}_year_{year_start}-{year_end}.nc"
    ds = netcdf.Dataset(fn, "r")
    time = ds.variables["time"][:]
    lev  = ds.variables["lev"][:]
    lat  = ds.variables["lat"][:]
    SF   = ds.variables["SF"][:]  # (time, lev, lat)
    ds.close()

    if time_slice is not None:
        time = time[time_slice]
        SF   = SF[time_slice, :, :]

    return time, lev, lat, SF

# ---------------- Seasons (your convention: DJF=12-14) ----------------
seasons = [
    ("annual", 1, 12, "a)", "b)"),
    ("DJF",   12, 14, "c)", "d)"),
    ("JJA",    6,  8, "e)", "f)"),
]

# FH=0.18Sv comparison: branch3800 - branch600 (same as your UVEL left case)
base = dict(branch=600,  year_start=999,  year_end=1100, time_slice=None)
hos  = dict(branch=3800, year_start=4199, year_end=4300, time_slice=None)

# ---------------- Plot settings ----------------
uvel_diff_levels = np.linspace(-3, 3, 41)
uvel_base_levels = np.linspace(-35, 35, 15)
uvel_linestyles  = ["dashed" if lvl < 0 else "solid" for lvl in uvel_base_levels]

sf_diff_levels   = np.linspace(-3, 3, 21)  # as in your example
# for SF baseline contours we compute levels from min/max each season (like you did)

fig, axs = plt.subplots(3, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

for row, (season_name, mstart, mend, labU, labS) in enumerate(seasons):

    # ---------- UVEL (left column) ----------
    _, levU, latU, U_base = read_uvel(mstart, mend, region=region, **base)
    _, _,    _,    U_hos  = read_uvel(mstart, mend, region=region, **hos)

    ax = axs[row, 0]

    # baseline contours (branch600)
    cU = ax.contour(
        latU, levU, np.mean(U_base, axis=0),
        levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
    )
    ax.clabel(cU, inline=True, fontsize=8, fmt="%1.0f")

    # difference fill (branch3800 - branch600)
    cfU = ax.contourf(
        latU, levU, np.mean(U_hos, axis=0) - np.mean(U_base, axis=0),
        levels=uvel_diff_levels, cmap="seismic", extend="both"
    )
    cbar = fig.colorbar(cfU, ax=ax)
    cbar.set_label("Zonal velocity difference [m/s]", fontsize=10)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    # axes styling
    ax.set_ylim(1000, 100)
    ax.set_xlim(-70, 70)
    ax.set_xlabel("Latitude [$^\\circ$N]", fontsize=11)  # visible on every subplot
    ax.set_ylabel("Pressure [hPa]", fontsize=11)
    ax.set_title(f"{labU} Zonal velocity ($F_H$=0.18Sv, {season_name})", fontsize=12)

    # non-significance markers (sig < 0.95)
    for lat_i in range(0, len(latU), 3):
        for lev_i in range(0, len(levU), 3):
            sig = Welch(U_base[:, lev_i, lat_i], U_hos[:, lev_i, lat_i])
            if sig < 0.95:
                ax.scatter(latU[lat_i], levU[lev_i], marker="o",
                           edgecolor="k", s=6, facecolors="none")

    # ---------- MSF (right column) ----------
    _, levS, latS, SF_base = read_sf(mstart, mend, **base)
    _, _,    _,    SF_hos  = read_sf(mstart, mend, **hos)

    ax = axs[row, 1]

    # baseline MSF contours (branch600), scale 1e10 like your script
    SF_base_mean = np.mean(SF_base, axis=0) / 1e10
    SF_diff_mean = (np.mean(SF_hos, axis=0) - np.mean(SF_base, axis=0)) / 1e10

    contour_levels = np.linspace(np.min(SF_base_mean), np.max(SF_base_mean), 10)
    cS = ax.contour(
        latS, levS, SF_base_mean,
        levels=contour_levels, colors="black",
        linestyles=["dotted" if lvl < 0 else "solid" for lvl in contour_levels]
    )
    ax.clabel(cS, inline=True, fontsize=8, fmt="%1.0f")

    cfS = ax.contourf(
        latS, levS, SF_diff_mean,
        levels=sf_diff_levels, cmap="seismic", extend="both"
    )
    cbar = fig.colorbar(cfS, ax=ax)
    cbar.set_label("MSF difference [$10^{10}$ kg/s]", fontsize=10)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    ax.set_ylim(1000, 100)
    ax.set_xlim(-70, 70)
    ax.set_xlabel("Latitude [$^\\circ$N]", fontsize=11)  # visible on every subplot
    ax.set_ylabel("Pressure [hPa]", fontsize=11)
    ax.set_title(f"{labS} Meridional streamfunction ($F_H$=0.18Sv, {season_name})", fontsize=12)

    # non-significance markers for SF (sig < 0.95)
    for lat_i in range(0, len(latS), 3):
        for lev_i in range(0, len(levS), 3):
            sig = Welch(SF_base[:, lev_i, lat_i], SF_hos[:, lev_i, lat_i])
            if sig < 0.95:
                ax.scatter(latS[lat_i], levS[lev_i], marker="o",
                           edgecolor="k", s=6, facecolors="none")

outfn = directory_figures + "Figure_3_CD.pdf"
plt.savefig(outfn, dpi=300, bbox_inches="tight")
plt.show()

#%%

plt.figure()
plt.contourf(latU, levU, np.mean(U_base, axis=0), cmap='RdBu_r', levels=np.linspace(-50,50,21))
plt.colorbar()
plt.ylim(1000,100)

#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creates TWO separate 3x2 figures:

FIGURE 1: FH = 0.18Sv (branch3800 - branch600)
  Row 1: a) UVEL diff (annual)   b) MSF diff (annual)
  Row 2: c) UVEL diff (DJF)     d) MSF diff (DJF)
  Row 3: e) UVEL diff (JJA)     f) MSF diff (JJA)

FIGURE 2: FH = 0.45Sv (branch2900 - branch1500)
  Row 1: a) UVEL diff (annual)   b) MSF diff (annual)
  Row 2: c) UVEL diff (DJF)     d) MSF diff (DJF)
  Row 3: e) UVEL diff (JJA)     f) MSF diff (JJA)

Per-subplot colorbars, xlabels on every subplot, black baseline contours,
and open circles marking NON-significant differences (Welch significance < 0.95).

NOTE:
- DJF is assumed to be stored as month_12-14 (your convention).
- For UVEL FH=0.45Sv, your original script cropped UVEL3 with [399:500].
  Here we apply the same crop to branch2900 for UVEL only (edit if not desired).
- For SF FH=0.45Sv, no crop is applied unless you add it similarly.
"""

import numpy as np
import netCDF4 as netcdf
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- Paths ----------------
directory_data     = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures  = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'
region             = 'global'

# ---------------- Welch test (returns significance level, e.g. 0.95 = 95%) ----------------
def Welch(data_1, data_2):
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)

    std_1 = np.sqrt(1.0 / (len(data_1) - 1) * np.sum((data_1 - mean_1) ** 2.0))
    std_2 = np.sqrt(1.0 / (len(data_2) - 1) * np.sum((data_2 - mean_2) ** 2.0))

    t_welch = (mean_1 - mean_2) / np.sqrt((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2)))

    dof = (
        ((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2))) ** 2.0
        / (
            (std_1**4.0 / (len(data_1) ** 2.0 * (len(data_1) - 1)))
            + (std_2**4.0 / (len(data_2) ** 2.0 * (len(data_2) - 1)))
        )
    )

    sig_levels = np.arange(50, 100, 0.5) / 100.0
    t_crit = stats.t.ppf((1.0 + sig_levels) / 2.0, dof)

    sig_index = np.where(np.fabs(t_welch) > t_crit)[0]
    significant = 0.0
    if len(sig_index) > 0:
        significant = sig_levels[sig_index[-1]]

    return significant

# ---------------- Loaders ----------------
def read_uvel(month_start, month_end, branch, year_start, year_end, region="global", time_slice=None):
    fn = (
        f"{directory_data}UVEL_{region}_month_{month_start}-{month_end}_"
        f"branch{branch}_year_{year_start}_{year_end}.nc"
    )
    ds = netcdf.Dataset(fn, "r")
    time = ds.variables["time"][:]
    lev  = ds.variables["lev"][:]
    lat  = ds.variables["lat"][:]
    U    = ds.variables["U"][:]  # (time, lev, lat)
    ds.close()

    if time_slice is not None:
        time = time[time_slice]
        U    = U[time_slice, :, :]

    return time, lev, lat, U

def read_sf(month_start, month_end, branch, year_start, year_end, time_slice=None):
    # Your SF filenames use year_999-1100 (dash)
    fn = f"{directory_data}Streamfunction_month_{month_start}-{month_end}_branch{branch}_year_{year_start}-{year_end}.nc"
    ds = netcdf.Dataset(fn, "r")
    time = ds.variables["time"][:]
    lev  = ds.variables["lev"][:]
    lat  = ds.variables["lat"][:]
    SF   = ds.variables["SF"][:]  # (time, lev, lat)
    ds.close()

    if time_slice is not None:
        time = time[time_slice]
        SF   = SF[time_slice, :, :]

    return time, lev, lat, SF

# ---------------- Seasons ----------------
seasons = [
    ("annual", 1, 12, "a)", "b)"),
    ("DJF",   12, 14, "c)", "d)"),
    ("JJA",    6,  8, "e)", "f)"),
]

# ---------------- Plot settings ----------------
uvel_diff_levels = np.linspace(-3, 3, 41)
uvel_base_levels = np.linspace(-35, 35, 15)
uvel_linestyles  = ["dashed" if lvl < 0 else "solid" for lvl in uvel_base_levels]

sf_diff_levels   = np.linspace(-3, 3, 21)  # scaled by 1e10 in plotting

# ---------------- Core plotting function ----------------
def make_figure(fh_label, base_cfg, hos_cfg, outname, uvel_hos_time_slice=None, sf_hos_time_slice=None):

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    for row, (season_name, mstart, mend, labU, labS) in enumerate(seasons):

        # ---------- UVEL (left column) ----------
        _, levU, latU, U_base = read_uvel(mstart, mend, region=region, time_slice=None, **base_cfg)
        _, _,    _,    U_hos  = read_uvel(mstart, mend, region=region, time_slice=uvel_hos_time_slice, **hos_cfg)

        ax = axs[row, 0]

        cU = ax.contour(
            latU, levU, np.mean(U_base, axis=0),
            levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
        )
        ax.clabel(cU, inline=True, fontsize=8, fmt="%1.0f")

        cfU = ax.contourf(
            latU, levU, np.mean(U_hos, axis=0) - np.mean(U_base, axis=0),
            levels=uvel_diff_levels, cmap="seismic", extend="both"
        )
        cbar = fig.colorbar(cfU, ax=ax)
        cbar.set_label("Zonal velocity difference [m/s]", fontsize=10)
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

        ax.set_ylim(1000, 100)
        ax.set_xlim(-70, 70)
        ax.set_xlabel("Latitude [$^\\circ$N]", fontsize=11)
        ax.set_ylabel("Pressure [hPa]", fontsize=11)
        ax.set_title(f"{labU} Zonal velocity ({fh_label}, {season_name})", fontsize=12)

        for lat_i in range(0, len(latU), 3):
            for lev_i in range(0, len(levU), 3):
                sig = Welch(U_base[:, lev_i, lat_i], U_hos[:, lev_i, lat_i])
                if sig < 0.95:
                    ax.scatter(latU[lat_i], levU[lev_i], marker="o",
                               edgecolor="k", s=6, facecolors="none")

        # ---------- MSF (right column) ----------
        _, levS, latS, SF_base = read_sf(mstart, mend, time_slice=None, **base_cfg)
        _, _,    _,    SF_hos  = read_sf(mstart, mend, time_slice=sf_hos_time_slice, **hos_cfg)

        ax = axs[row, 1]

        SF_base_mean = np.mean(SF_base, axis=0) / 1e10
        SF_diff_mean = (np.mean(SF_hos, axis=0) - np.mean(SF_base, axis=0)) / 1e10

        contour_levels = np.linspace(np.min(SF_base_mean), np.max(SF_base_mean), 10)
        cS = ax.contour(
            latS, levS, SF_base_mean,
            levels=contour_levels, colors="black",
            linestyles=["dotted" if lvl < 0 else "solid" for lvl in contour_levels]
        )
        ax.clabel(cS, inline=True, fontsize=8, fmt="%1.0f")

        cfS = ax.contourf(
            latS, levS, SF_diff_mean,
            levels=sf_diff_levels, cmap="seismic", extend="both"
        )
        cbar = fig.colorbar(cfS, ax=ax)
        cbar.set_label("MSF difference [$10^{10}$ kg/s]", fontsize=10)
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

        ax.set_ylim(1000, 100)
        ax.set_xlim(-70, 70)
        ax.set_xlabel("Latitude [$^\\circ$N]", fontsize=11)
        ax.set_ylabel("Pressure [hPa]", fontsize=11)
        ax.set_title(f"{labS} Meridional streamfunction ({fh_label}, {season_name})", fontsize=12)

        for lat_i in range(0, len(latS), 3):
            for lev_i in range(0, len(levS), 3):
                sig = Welch(SF_base[:, lev_i, lat_i], SF_hos[:, lev_i, lat_i])
                if sig < 0.95:
                    ax.scatter(latS[lat_i], levS[lev_i], marker="o",
                               edgecolor="k", s=6, facecolors="none")

    plt.savefig(directory_figures + outname, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------- FH = 0.18Sv ----------------
base_018 = dict(branch=600,  year_start=999,  year_end=1100)
hos_018  = dict(branch=3800, year_start=4199, year_end=4300)

make_figure(
    fh_label=r"$F_H$=0.18Sv",
    base_cfg=base_018,
    hos_cfg=hos_018,
    outname="Figure_UVEL_MSF_FH018Sv_3x2_annual_DJF_JJA.pdf",
    uvel_hos_time_slice=None,
    sf_hos_time_slice=None
)

# ---------------- FH = 0.45Sv ----------------
# UVEL filenames for branch2900 in your earlier script used year_2900_3500 with underscores,
# and you applied a [399:500] crop. We'll keep that for UVEL hosing.
# For SF you provided filenames with dashes: year_2900-3500.
base_045_uvel = dict(branch=1500, year_start=1899, year_end=2000)
hos_045_uvel  = dict(branch=2900, year_start=2900, year_end=3500)

# For SF: same branch numbers & years, but loader uses dash-format names.
base_045_sf = dict(branch=1500, year_start=1899, year_end=2000)
hos_045_sf  = dict(branch=2900, year_start=2900, year_end=3500)

# We can reuse the same dicts because loader decides filename format.
# But note: read_uvel expects underscores in year part, read_sf expects dashes in year part.

make_figure(
    fh_label=r"$F_H$=0.45Sv",
    base_cfg=base_045_uvel,
    hos_cfg=hos_045_uvel,
    outname="Figure_S3_CD.pdf",
    uvel_hos_time_slice=slice(399, 500),  # as in your earlier UVEL script
    sf_hos_time_slice=None                # set to slice(399,500) too if you want identical windowing
)

#%%



#%%

import numpy as np
import netCDF4 as netcdf
import matplotlib.pyplot as plt
from scipy import stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

# ---------------- Paths ----------------
directory_data     = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures  = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'
region             = 'global'

# ---------------- Welch test (returns significance level, e.g. 0.95 = 95%) ----------------
def Welch(data_1, data_2):
    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)

    std_1 = np.sqrt(1.0 / (len(data_1) - 1) * np.sum((data_1 - mean_1) ** 2.0))
    std_2 = np.sqrt(1.0 / (len(data_2) - 1) * np.sum((data_2 - mean_2) ** 2.0))

    t_welch = (mean_1 - mean_2) / np.sqrt((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2)))

    dof = (
        ((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2))) ** 2.0
        / (
            (std_1**4.0 / (len(data_1) ** 2.0 * (len(data_1) - 1)))
            + (std_2**4.0 / (len(data_2) ** 2.0 * (len(data_2) - 1)))
        )
    )

    sig_levels = np.arange(50, 100, 0.5) / 100.0
    t_crit = stats.t.ppf((1.0 + sig_levels) / 2.0, dof)

    sig_index = np.where(np.fabs(t_welch) > t_crit)[0]
    significant = 0.0
    if len(sig_index) > 0:
        significant = sig_levels[sig_index[-1]]

    return significant

# ---------------- Loaders ----------------
def read_uvel(month_start, month_end, branch, year_start, year_end, region="global"):
    fn = (
        f"{directory_data}UVEL_{region}_month_{month_start}-{month_end}_"
        f"branch{branch}_year_{year_start}_{year_end}.nc"
    )
    ds = netcdf.Dataset(fn, "r")
    lev  = ds.variables["lev"][:]
    lat  = ds.variables["lat"][:]
    U    = ds.variables["U"][:]  # (time, lev, lat)
    ds.close()
    return lev, lat, U

def read_sf(month_start, month_end, branch, year_start, year_end):
    fn = f"{directory_data}Streamfunction_month_{month_start}-{month_end}_branch{branch}_year_{year_start}-{year_end}.nc"
    ds = netcdf.Dataset(fn, "r")
    lev = ds.variables["lev"][:]
    lat = ds.variables["lat"][:]
    SF  = ds.variables["SF"][:]  # (time, lev, lat)
    ds.close()
    return lev, lat, SF

def read_jet(month_start, month_end, branch):
    fn = f"{directory_data}Jet_200_hPa_Atlantic_sector_month_{month_start}-{month_end}_{branch}.nc"
    ds = netcdf.Dataset(fn, "r")
    lon = ds.variables["lon"][:]
    lat = ds.variables["lat"][:]
    U   = ds.variables["U"][:]    # (time, lat, lon)
    UU  = ds.variables["UU"][:]   # (time, lat, lon)
    V   = ds.variables["V"][:]    # (time, lat, lon)
    VV  = ds.variables["VV"][:]   # (time, lat, lon)
    ds.close()
    return lon, lat, U, UU, V, VV

# ======================================================================
#                            SETTINGS
# ======================================================================

# Annual for zonal-mean UVEL and SF
mstart_annual, mend_annual = 1, 12

# Jet month (your example was Jan only)
mstart_jet, mend_jet = 1, 1

# FH = 0.18Sv case: baseline = branch600, hosing/off = branch3800
uvel_base_cfg = dict(branch=600,  year_start=999,  year_end=1100)
uvel_hos_cfg  = dict(branch=3800, year_start=4199, year_end=4300)

sf_base_cfg   = dict(branch=600,  year_start=999,  year_end=1100)
sf_hos_cfg    = dict(branch=3800, year_start=4199, year_end=4300)

jet_base_branch = 600
jet_hos_branch  = 3800

# Plot style
uvel_diff_levels = np.linspace(-3, 3, 41)
uvel_base_levels = np.linspace(-35, 35, 15)
uvel_linestyles  = ["dashed" if lvl < 0 else "solid" for lvl in uvel_base_levels]

sf_diff_levels   = np.linspace(-3, 3, 21)  # after /1e10 scaling

# Jet panel style (you can tune these)
jet_u_levels     = np.linspace(0, 50, 21)         # extend on top if needed
jet_spd_levels   = np.arange(-10, 10.1, 1)
cmap_jet_abs     = "Spectral_r"
cmap_jet_diff    = "PiYG_r"

extent_jet = [-70, 10, 10, 70]  # matches your desired domain
scale_arrow = 4
quiver_scale = 100

# ======================================================================
#                            READ + COMPUTE
# ======================================================================

# --- a) UVEL annual zonal mean diff ---
levU, latU, U_base = read_uvel(mstart_annual, mend_annual, region=region, **uvel_base_cfg)
_,    _,    U_hos  = read_uvel(mstart_annual, mend_annual, region=region, **uvel_hos_cfg)

U_base_mean = np.mean(U_base, axis=0)
U_diff_mean = np.mean(U_hos, axis=0) - np.mean(U_base, axis=0)

# --- b) SF annual diff (scaled by 1e10 like your script) ---
levS, latS, SF_base = read_sf(mstart_annual, mend_annual, **sf_base_cfg)
_,    _,    SF_hos  = read_sf(mstart_annual, mend_annual, **sf_hos_cfg)

SF_base_mean = np.mean(SF_base, axis=0) / 1e10
SF_diff_mean = (np.mean(SF_hos, axis=0) - np.mean(SF_base, axis=0)) / 1e10

sf_contour_levels = np.linspace(np.min(SF_base_mean), np.max(SF_base_mean), 10)

# --- c,d) Jet 200 hPa (month 1-1) ---
lonJ, latJ, U_all, UU_all, V_all, VV_all = read_jet(mstart_jet, mend_jet, jet_hos_branch)
_,    _,    U_ref, UU_ref, V_ref, VV_ref = read_jet(mstart_jet, mend_jet, jet_base_branch)

# time means (absolute reference/baseline for panel c: AMOC on = branch600)
vel_speed_ref	= np.mean(np.sqrt(UU_ref + VV_ref), axis = 0)
u_ref_m = np.mean(U_ref, axis=0)
v_ref_m = np.mean(V_ref, axis=0)
spd_ref_m = np.mean(np.sqrt(UU_ref + VV_ref), axis=0)

# time means (hos/off = branch3800)
u_all_m = np.mean(U_all, axis=0)
v_all_m = np.mean(V_all, axis=0)
spd_all_m = np.mean(np.sqrt(UU_all + VV_all), axis=0)

# diffs (off - on = 3800 - 600) for panel d
u_diff = u_all_m - u_ref_m
v_diff = v_all_m - v_ref_m
spd_diff = spd_all_m - spd_ref_m

# ======================================================================
#                            PLOT 2x2
# ======================================================================

#%%

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

# ---- Panel a (top-left): UVEL zonal mean diff ----
axA = fig.add_subplot(gs[0, 0])

cA_cont = axA.contour(
    latU, levU, U_base_mean,
    levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
)
axA.clabel(cA_cont, inline=True, fontsize=8, fmt="%1.0f")

cA = axA.contourf(
    latU, levU, U_diff_mean,
    levels=uvel_diff_levels, cmap="seismic", extend="both"
)
cbA = fig.colorbar(cA, ax=axA, orientation="vertical")
cbA.set_label("Zonal velocity difference [m/s]")
cbA.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axA.set_ylim(1000, 100)
axA.set_xlim(-70, 70)
axA.set_xlabel("Latitude [$^\\circ$N]")
axA.set_ylabel("Pressure [hPa]")
axA.set_title(r"a) Zonal velocity (annual, $\overline{F_H}=0.18$Sv)")

# optional: non-significance markers
for lat_i in range(0, len(latU), 3):
    for lev_i in range(0, len(levU), 3):
        sig = Welch(U_base[:, lev_i, lat_i], U_hos[:, lev_i, lat_i])
        if sig < 0.95:
            axA.scatter(latU[lat_i], levU[lev_i], marker="o",
                        edgecolor="k", s=6, facecolors="none")

# ---- Panel b (top-right): MSF diff ----
axB = fig.add_subplot(gs[0, 1])

cB_cont = axB.contour(
    latS, levS, SF_base_mean,
    levels=sf_contour_levels, colors="black",
    linestyles=["dotted" if lvl < 0 else "solid" for lvl in sf_contour_levels]
)
axB.clabel(cB_cont, inline=True, fontsize=8, fmt="%1.0f")

cB = axB.contourf(
    latS, levS, SF_diff_mean,
    levels=sf_diff_levels, cmap="seismic", extend="both"
)
cbB = fig.colorbar(cB, ax=axB, orientation="vertical")
cbB.set_label("MSF difference [$10^{10}$ kg/s]")
cbB.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axB.set_ylim(1000, 100)
axB.set_xlim(-70, 70)
axB.set_xlabel("Latitude [$^\\circ$N]")
axB.set_ylabel("Pressure [hPa]")
axB.set_title(r"b) Meridional streamfunction (annual, $\overline{F_H}=0.18$Sv)")

for lat_i in range(0, len(latS), 3):
    for lev_i in range(0, len(levS), 3):
        sig = Welch(SF_base[:, lev_i, lat_i], SF_hos[:, lev_i, lat_i])
        if sig < 0.95:
            axB.scatter(latS[lat_i], levS[lev_i], marker="o",
                        edgecolor="k", s=6, facecolors="none")

# ---- Panel c (bottom-left): Jet absolute (AMOC on / branch600) ----
axC = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())

cC = axC.contourf(
    lonJ, latJ, vel_speed_ref,
    levels=jet_u_levels, cmap=cmap_jet_abs, extend="max",
    transform=ccrs.PlateCarree()
)
cbC = fig.colorbar(cC, ax=axC, orientation="vertical")
cbC.set_label("Zonal velocity [m s$^{-1}$]")

axC.quiver(
    lonJ[::scale_arrow], latJ[::scale_arrow],
    u_ref_m[::scale_arrow, ::scale_arrow],
    v_ref_m[::scale_arrow, ::scale_arrow],
    scale=500,
    transform=ccrs.PlateCarree()
)

axC.set_extent(extent_jet, crs=ccrs.PlateCarree())
axC.coastlines("110m")
axC.add_feature(cfeature.LAND, zorder=0)
axC.set_title("c) 200 hPa velocities (AMOC on, January, $\overline{F_H}=0.18$Sv)")

axC.set_xticks(np.arange(-90, 31, 30), crs=ccrs.PlateCarree())
axC.xaxis.set_major_formatter(cticker.LongitudeFormatter())
axC.set_yticks(np.arange(0, 81, 20), crs=ccrs.PlateCarree())
axC.yaxis.set_major_formatter(cticker.LatitudeFormatter())

# ---- Panel d (bottom-right): Jet difference (off - on) ----
axD = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())

cD = axD.contourf(
    lonJ, latJ, spd_diff,
    levels=jet_spd_levels, cmap=cmap_jet_diff, extend="both",
    transform=ccrs.PlateCarree()
)
cbD = fig.colorbar(cD, ax=axD, orientation="vertical")
cbD.set_label("Wind-speed difference [m s$^{-1}$]")

axD.quiver(
    lonJ[::scale_arrow], latJ[::scale_arrow],
    u_diff[::scale_arrow, ::scale_arrow],
    v_diff[::scale_arrow, ::scale_arrow],
    scale=quiver_scale,
    transform=ccrs.PlateCarree()
)

axD.set_extent(extent_jet, crs=ccrs.PlateCarree())
axD.coastlines("110m")
axD.add_feature(cfeature.LAND, zorder=0)
axD.set_title(r"d) 200 hPa difference (January, $\overline{F_H}=0.18$Sv)")

axD.set_xticks(np.arange(-90, 31, 30), crs=ccrs.PlateCarree())
axD.xaxis.set_major_formatter(cticker.LongitudeFormatter())
axD.set_yticks(np.arange(0, 81, 20), crs=ccrs.PlateCarree())
axD.yaxis.set_major_formatter(cticker.LatitudeFormatter())

# Save
#plt.savefig(directory_figures + "Figure_3_CD_new.pdf", dpi=300, bbox_inches="tight")
plt.show()

#%% Hadley, zonal velocity and ITCZ plot

fh      = netcdf.Dataset(directory_data+'ITCZ_QE_year_0-2200.nc', 'r')

time_forward    = fh.variables['time'][:] #Model years
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
lon             = fh.variables['lon'][:]  #Longitudes used for zonal mean
ITCZ_forward    = fh.variables['ITCZ'][:] #Position of ITCZ

fh.close()

fh      = netcdf.Dataset(directory_data+'ITCZ_QE_year_2201-4400.nc', 'r')

time_backward    = fh.variables['time'][:] #Model years
lat              = fh.variables['lat'][:]  #Array of latitudes [degN]
lon              = fh.variables['lon'][:]  #Longitudes used for zonal mean
ITCZ_backward    = fh.variables['ITCZ'][:] #Position of ITCZ

fh.close()

fh      = netcdf.Dataset(directory_data+'ITCZ_branch600_year_999-1100.nc', 'r')

time_E1         = fh.variables['time'][:] #Model years
time_E1_month   = fh.variables['time_month'][:] #Model years
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
lon             = fh.variables['lon'][:]  #Longitudes used for zonal mean
ITCZ_E1         = fh.variables['ITCZ'][:] #Position of ITCZ
ITCZ_E1_month   = fh.variables['ITCZ_month'][:] #Position of ITCZ

fh.close()

fh      = netcdf.Dataset(directory_data+'ITCZ_branch1500_year_1899-2000.nc', 'r')

time_E2         = fh.variables['time'][:] #Model years
time_E2_month   = fh.variables['time_month'][:] #Model years
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
lon             = fh.variables['lon'][:]  #Longitudes used for zonal mean
ITCZ_E2         = fh.variables['ITCZ'][:] #Position of ITCZ
ITCZ_E2_month   = fh.variables['ITCZ_month'][:] #Position of ITCZ

fh.close()

fh      = netcdf.Dataset(directory_data+'ITCZ_branch2900_year_2900-3500.nc', 'r')

time_E3         = fh.variables['time'][399:501] #Model years
time_E3_month   = fh.variables['time_month'][399*12:501*12 - 1] #Model years
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
lon             = fh.variables['lon'][:]  #Longitudes used for zonal mean
ITCZ_E3         = fh.variables['ITCZ'][399:501] #Position of ITCZ
ITCZ_E3_month   = fh.variables['ITCZ_month'][399*12:501*12 - 1] #Position of ITCZ

fh.close()

fh      = netcdf.Dataset(directory_data+'ITCZ_branch3800_year_4199-4300.nc', 'r')

time_E4         = fh.variables['time'][:] #Model years
time_E4_month   = fh.variables['time_month'][:] #Model years
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
lon             = fh.variables['lon'][:]  #Longitudes used for zonal mean
ITCZ_E4         = fh.variables['ITCZ'][:] #Position of ITCZ
ITCZ_E4_month   = fh.variables['ITCZ_month'][:] #Position of ITCZ

#fh.close()

#%%

region      = 'global'
    
#Read in data
fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch600_year_999-1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon_SLP             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat_SLP             = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_1_annual   = fh.variables['SLP'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch1500_year_1899-2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
SLP_2_annual   = fh.variables['SLP'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch2900_year_2900-3500.nc', 'r')

time3           = fh.variables['time'][400:500] #Model years
SLP_3_annual   = fh.variables['SLP'][400:500,:,:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch3800_year_4199-4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
SLP_4_annual   = fh.variables['SLP'][:]   #Mean reference air SLPerature

fh.close()

slp_diff_18 = np.mean(SLP_4_annual, axis=0) - np.mean(SLP_1_annual, axis=0)
slp_diff_18_prime = slp_diff_18 - np.mean(slp_diff_18, axis=1, keepdims=True)

slp_diff_45 = np.mean(SLP_3_annual, axis=0) - np.mean(SLP_2_annual, axis=0)
slp_diff_45_prime = slp_diff_45 - np.mean(slp_diff_45, axis=1, keepdims=True)

SLP1_prime = SLP_1_annual - np.mean(SLP_1_annual, axis=2, keepdims=True)
SLP4_prime = SLP_4_annual - np.mean(SLP_4_annual, axis=2, keepdims=True)

SLP2_prime = SLP_2_annual - np.mean(SLP_2_annual, axis=2, keepdims=True)
SLP3_prime = SLP_3_annual - np.mean(SLP_3_annual, axis=2, keepdims=True)

#%% Precipitation 

fh      = netcdf.Dataset(directory_data+'PREC_month_1-12_branch600_year_999-1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon_SLP             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat_SLP             = fh.variables['lat'][:]  #Array of latitudes [degN]
PREC_1_annual   = fh.variables['PREC'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'PREC_month_1-12_branch1500_year_1899-2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
PREC_2_annual   = fh.variables['PREC'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'PREC_month_1-12_branch2900_year_3299-3400.nc', 'r')

time3           = fh.variables['time'][:] #Model years
PREC_3_annual   = fh.variables['PREC'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'PREC_month_1-12_branch3800_year_4199-4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
PREC_4_annual   = fh.variables['PREC'][:]   #Mean reference air SLPerature

fh.close()

#%%

freq_on  = 100 * np.mean(ITCZ_E1, axis=0)   # shape (lat, lon)
freq_off = 100 * np.mean(ITCZ_E4, axis=0)  # shape (lat, lon)

freq_diff = freq_off - freq_on

plt.figure()
plt.contourf(lon, lat, freq_diff, levels=np.linspace(-0.00000000000001, 0.00000000000001, 21), cmap="seismic", extend="both")
plt.colorbar()

#%%

# Step 1: Count occurrences (treat NaN as 0)
ITCZ_clean_E1 = ITCZ_E1.filled(0)  # Replace NaN with 0 for counting
counts_E1 = np.nansum(ITCZ_clean_E1, axis=0)  # shape = (lat, lon). Divide by mean as the values are 0.08 (some kind of normalization was applied which actually was unneccesary as we just flagged the gridcells)

# Step 2: Convert to frequency (0–1)
n_time_steps = ITCZ_E1.shape[0]
frequency_E1 = counts_E1 / n_time_steps  # fraction of time

# Step 3: Optional: convert to percentage
frequency_percent_E1 = frequency_E1 * 100

plt.figure()
plt.contourf(lon, lat, frequency_percent_E1, levels=np.linspace(0, 100, 21), cmap="viridis", extend="max")
plt.colorbar(label="Frequency of ITCZ presence (%)")
plt.xlabel("Longitude [degE]")
plt.ylabel("Latitude [degN]")

#%%

# Step 1: Count occurrences (treat NaN as 0)
ITCZ_clean_E2 = ITCZ_E2.filled(0)  # Replace NaN with 0 for counting
counts_E2 = np.nansum(ITCZ_clean_E2, axis=0)  # shape = (lat, lon). Divide by mean as the values are 0.08 (some kind of normalization was applied which actually was unneccesary as we just flagged the gridcells)
n_time_steps = ITCZ_E2.shape[0]
frequency_E2 = counts_E2 / n_time_steps  # fraction of time
frequency_percent_E2 = frequency_E2 * 100

# Step 1: Count occurrences (treat NaN as 0)
ITCZ_clean_E3 = ITCZ_E3.filled(0)  # Replace NaN with 0 for counting
counts_E3 = np.nansum(ITCZ_clean_E3, axis=0)  # shape = (lat, lon). Divide by mean as the values are 0.08 (some kind of normalization was applied which actually was unneccesary as we just flagged the gridcells)
n_time_steps = ITCZ_E3.shape[0]
frequency_E3 = counts_E3 / n_time_steps  # fraction of time
frequency_percent_E3 = frequency_E3 * 100

# Step 1: Count occurrences (treat NaN as 0)
ITCZ_clean_E4 = ITCZ_E4.filled(0)  # Replace NaN with 0 for counting
counts_E4 = np.nansum(ITCZ_clean_E4, axis=0)  # shape = (lat, lon). Divide by mean as the values are 0.08 (some kind of normalization was applied which actually was unneccesary as we just flagged the gridcells)
n_time_steps = ITCZ_E4.shape[0]
frequency_E4 = counts_E4 / n_time_steps  # fraction of time
frequency_percent_E4 = frequency_E4 * 100


#%%
# Step 1: Count occurrences (treat NaN as 0)
counts = np.nansum(ITCZ_forward[600:1500], axis=0)/np.mean(ITCZ_forward[600:1500])  # shape = (lat, lon). Divide by mean as the values are 0.08 (some kind of normalization was applied which actually was unneccesary as we just flagged the gridcells)

# Step 2: Convert to frequency (0–1)
n_time_steps = ITCZ_forward[600:1500].shape[0]
frequency = counts / n_time_steps  # fraction of time

# Step 3: Optional: convert to percentage
frequency_percent_forward = frequency * 100

# Step 1: Count occurrences (treat NaN as 0)
counts = np.nansum(ITCZ_backward[700:1600], axis=0)/np.mean(ITCZ_backward[700:1600])  # shape = (lat, lon). Divide by mean as the values are 0.08 (some kind of normalization was applied which actually was unneccesary as we just flagged the gridcells)

# Step 2: Convert to frequency (0–1)
n_time_steps = ITCZ_backward[700:1600].shape[0]
frequency = counts / n_time_steps  # fraction of time

# Step 3: Optional: convert to percentage
frequency_percent_backward = frequency * 100

#%%

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

axA = fig.add_subplot(gs[0, 0])

cA_cont = axA.contour(
    latU, levU, U_base_mean,
    levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
)
axA.clabel(cA_cont, inline=True, fontsize=8, fmt="%1.0f")

cA = axA.contourf(
    latU, levU, U_diff_mean,
    levels=uvel_diff_levels, cmap="seismic", extend="both"
)
cbA = fig.colorbar(cA, ax=axA, orientation="vertical")
cbA.set_label("Zonal velocity difference [m/s]")
cbA.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axA.set_ylim(1000, 100)
axA.set_xlim(-70, 70)
axA.set_xlabel("Latitude [$^\\circ$N]")
axA.set_ylabel("Pressure [hPa]")
axA.set_title(r"a) Zonal velocity (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

for lat_i in range(0, len(latU), 3):
    for lev_i in range(0, len(levU), 3):
        sig = Welch(U_base[:, lev_i, lat_i], U_hos[:, lev_i, lat_i])
        if sig < 0.95:
            axA.scatter(
                latU[lat_i], levU[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

axB = fig.add_subplot(gs[0, 1])

cB_cont = axB.contour(
    latS, levS, SF_base_mean,
    levels=sf_contour_levels, colors="black",
    linestyles=["dotted" if lvl < 0 else "solid" for lvl in sf_contour_levels])
axB.clabel(cB_cont, inline=True, fontsize=8, fmt="%1.0f")

cB = axB.contourf(
    latS, levS, SF_diff_mean,
    levels=sf_diff_levels, cmap="seismic", extend="both")
cbB = fig.colorbar(cB, ax=axB, orientation="vertical")
cbB.set_label("MSF difference [$10^{10}$ kg/s]")
cbB.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axB.set_ylim(1000, 100)
axB.set_xlim(-70, 70)
axB.set_xlabel("Latitude [$^\\circ$N]")
axB.set_ylabel("Pressure [hPa]")
axB.set_title(r"b) Meridional streamfunction (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

for lat_i in range(0, len(latS), 3):
    for lev_i in range(0, len(levS), 3):
        sig = Welch(SF_base[:, lev_i, lat_i], SF_hos[:, lev_i, lat_i])
        if sig < 0.95:
            axB.scatter(
                latS[lat_i], levS[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

subgs = gs[1, :].subgridspec(1, 3, width_ratios=[-6, 100, -14], hspace=0.5)
axC = fig.add_subplot(subgs[0,1], projection=ccrs.Robinson())

cC = axC.contourf(
    lon, lat, frequency_percent_E4 - frequency_percent_E1,
    levels=np.linspace(-5, 5, 17),
    extend="both",
    cmap="BrBG",
    transform=ccrs.PlateCarree())
cbC = fig.colorbar(cC, ax=axC, orientation="vertical", shrink=0.6)
cbC.set_label("Difference [%]")

axC.add_feature(cfeature.LAND, facecolor="lightgray")
axC.coastlines(resolution="50m")

gl = axC.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        # skip if no occurrences in both
        if ma.is_masked(frequency_percent_E1[lat_i, lon_i]) or ma.is_masked(frequency_percent_E4[lat_i, lon_i]):
            continue

        p_value = Welch(ITCZ_E1[:, lat_i, lon_i], ITCZ_E4[:, lat_i, lon_i])
        if p_value <= 0.95:
            axC.scatter(
                lon[lon_i], lat[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

axC.set_title(r"c) ITCZ location (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

plt.tight_layout()
#plt.savefig(directory_figures + "Figure_2_CD.pdf", dpi=300, bbox_inches="tight")
plt.show()


#%%

# =========================
# 1. Read UVEL and SF data
# =========================

# UVEL: PI_on_45 = branch1500, PI_off_45 = branch2900
levU_45, latU_45, U_base_45 = read_uvel(1, 12, branch=1500, year_start=1899, year_end=2000, region=region)
_,       _,       U_hos_45  = read_uvel(1, 12, branch=2900, year_start=2900, year_end=3500, region=region)

# apply same crop as before for branch2900 if you want consistency
U_hos_45 = U_hos_45[399:500]

U_base_mean_45 = np.mean(U_base_45, axis=0)
U_diff_mean_45 = np.mean(U_hos_45, axis=0) - np.mean(U_base_45, axis=0)

# SF: PI_on_45 = branch1500, PI_off_45 = branch2900
levS_45, latS_45, SF_base_45 = read_sf(1, 12, branch=1500, year_start=1899, year_end=2000)
_,       _,       SF_hos_45  = read_sf(1, 12, branch=2900, year_start=2900, year_end=3500)

# if you also want the same crop for SF, uncomment this:
# SF_hos_45 = SF_hos_45[399:500]

SF_base_mean_45 = np.mean(SF_base_45, axis=0) / 1e10
SF_diff_mean_45 = (np.mean(SF_hos_45, axis=0) - np.mean(SF_base_45, axis=0)) / 1e10

sf_contour_levels_45 = np.linspace(np.min(SF_base_mean_45), np.max(SF_base_mean_45), 10)

# =========================
# 2. Plot
# =========================

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.25)

# ---- Panel a: UVEL ----
axA = fig.add_subplot(gs[0, 0])

cA_cont = axA.contour(
    latU_45, levU_45, U_base_mean_45,
    levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles)

axA.clabel(cA_cont, inline=True, fontsize=8, fmt="%1.0f")

cA = axA.contourf(
    latU_45, levU_45, U_diff_mean_45,
    levels=uvel_diff_levels, cmap="seismic", extend="both")

cbA = fig.colorbar(cA, ax=axA, orientation="vertical")
cbA.set_label("Zonal velocity difference [m/s]")
cbA.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axA.set_ylim(1000, 100)
axA.set_xlim(-70, 70)
axA.set_xlabel("Latitude [$^\\circ$N]")
axA.set_ylabel("Pressure [hPa]")
axA.set_title(r"a) Zonal velocity (PI$^{off}_{45}$ - PI$^{on}_{45}$)")

for lat_i in range(0, len(latU_45), 3):
    for lev_i in range(0, len(levU_45), 3):
        sig = Welch(U_base_45[:, lev_i, lat_i], U_hos_45[:, lev_i, lat_i])
        if sig < 0.95:
            axA.scatter(
                latU_45[lat_i], levU_45[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

# ---- Panel b: MSF ----
axB = fig.add_subplot(gs[0, 1])

cB_cont = axB.contour(
    latS_45, levS_45, SF_base_mean_45,
    levels=sf_contour_levels_45, colors="black",
    linestyles=["dotted" if lvl < 0 else "solid" for lvl in sf_contour_levels_45])

axB.clabel(cB_cont, inline=True, fontsize=8, fmt="%1.0f")

cB = axB.contourf(
    latS_45, levS_45, SF_diff_mean_45,
    levels=sf_diff_levels, cmap="seismic", extend="both")

cbB = fig.colorbar(cB, ax=axB, orientation="vertical")
cbB.set_label("MSF difference [$10^{10}$ kg/s]")
cbB.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axB.set_ylim(1000, 100)
axB.set_xlim(-70, 70)
axB.set_xlabel("Latitude [$^\\circ$N]")
axB.set_ylabel("Pressure [hPa]")
axB.set_title(r"b) Meridional streamfunction (PI$^{off}_{45}$ - PI$^{on}_{45}$)")

for lat_i in range(0, len(latS_45), 3):
    for lev_i in range(0, len(levS_45), 3):
        sig = Welch(SF_base_45[:, lev_i, lat_i], SF_hos_45[:, lev_i, lat_i])
        if sig < 0.95:
            axB.scatter(
                latS_45[lat_i], levS_45[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

# ---- Panel c: ITCZ ----
subgs = gs[1, :].subgridspec(1, 3, width_ratios=[-6, 100, -14], hspace=0.5)
axC = fig.add_subplot(subgs[0, 1], projection=ccrs.Robinson())

cC = axC.contourf(
    lon, lat, frequency_percent_E3 - frequency_percent_E2,
    levels=np.linspace(-60, 60, 17),
    extend="both",
    cmap="BrBG",
    transform=ccrs.PlateCarree())

cbC = fig.colorbar(cC, ax=axC, orientation="vertical", shrink=0.6)
cbC.set_label("Difference [%]")

axC.add_feature(cfeature.LAND, facecolor="lightgray")
axC.coastlines(resolution="50m")

gl = axC.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        p_value = Welch(ITCZ_E2[:, lat_i, lon_i], ITCZ_E3[:, lat_i, lon_i])
        if p_value <= 0.95:
            axC.scatter(
                lon[lon_i], lat[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

axC.set_title(r"c) ITCZ location (PI$^{off}_{45}$ - PI$^{on}_{45}$)")

plt.tight_layout()
#plt.savefig(directory_figures + "Figure_S2_CD.pdf", dpi=300, bbox_inches="tight")
plt.show()

#%% Difference-of-differences: (PI18off - PI18on) - (PI45off - PI45on)

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Read annual data
# =========================

# PI18: on = branch600, off = branch3800
levU_18, latU_18, U_on_18  = read_uvel(1, 12, branch=600,  year_start=999,  year_end=1100, region=region)
_,       _,       U_off_18 = read_uvel(1, 12, branch=3800, year_start=4199, year_end=4300, region=region)

levS_18, latS_18, SF_on_18  = read_sf(1, 12, branch=600,  year_start=999,  year_end=1100)
_,       _,       SF_off_18 = read_sf(1, 12, branch=3800, year_start=4199, year_end=4300)

# PI45: on = branch1500, off = branch2900
levU_45, latU_45, U_on_45  = read_uvel(1, 12, branch=1500, year_start=1899, year_end=2000, region=region)
_,       _,       U_off_45 = read_uvel(1, 12, branch=2900, year_start=2900, year_end=3500, region=region)

# same crop as in your earlier scripts
U_off_45 = U_off_45[399:500]

levS_45, latS_45, SF_on_45  = read_sf(1, 12, branch=1500, year_start=1899, year_end=2000)
_,       _,       SF_off_45 = read_sf(1, 12, branch=2900, year_start=2900, year_end=3500)

# If you want identical time windowing for SF too, uncomment:
# SF_off_45 = SF_off_45[399:500]

# =========================
# 2. Compute mean responses
# =========================

# UVEL responses
U_resp_18 = np.mean(U_off_18, axis=0) - np.mean(U_on_18, axis=0)
U_resp_45 = np.mean(U_off_45, axis=0) - np.mean(U_on_45, axis=0)

# Difference-of-differences
U_diffdiff = U_resp_18 - U_resp_45

# MSF responses (scaled by 1e10)
SF_resp_18 = (np.mean(SF_off_18, axis=0) - np.mean(SF_on_18, axis=0)) / 1e10
SF_resp_45 = (np.mean(SF_off_45, axis=0) - np.mean(SF_on_45, axis=0)) / 1e10

# Difference-of-differences
SF_diffdiff = SF_resp_18 - SF_resp_45

# =========================
# 3. Baseline contours
# =========================
# I use PI18-on as reference contours, but you could also use PI45-on
U_base_contours  = np.mean(U_on_18, axis=0)
SF_base_contours = np.mean(SF_on_18, axis=0) / 1e10

sf_contour_levels_diff = np.linspace(np.min(SF_base_contours), np.max(SF_base_contours), 10)

# =========================
# 4. Optional significance for difference-of-differences
# =========================
# Compare the PI18 response sample to the PI45 response sample:
#   (U_off_18 - U_on_18) vs (U_off_45 - U_on_45)
#
# Because sample lengths differ, we truncate to the minimum length.

nU = min(len(U_on_18), len(U_off_18), len(U_on_45), len(U_off_45))
nS = min(len(SF_on_18), len(SF_off_18), len(SF_on_45), len(SF_off_45))

U_resp_samples_18 = U_off_18[:nU] - U_on_18[:nU]
U_resp_samples_45 = U_off_45[:nU] - U_on_45[:nU]

SF_resp_samples_18 = SF_off_18[:nS] - SF_on_18[:nS]
SF_resp_samples_45 = SF_off_45[:nS] - SF_on_45[:nS]

# =========================
# 5. Plot
# =========================

diffdiff_levels_u  = np.linspace(-2.5, 2.5, 21)
diffdiff_levels_sf = np.linspace(-2.0, 2.0, 21)

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2, wspace=0.25)

# ---- Panel a: UVEL difference-of-differences ----
axA = fig.add_subplot(gs[0, 0])

cA_cont = axA.contour(
    latU_18, levU_18, U_base_contours,
    levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
)
axA.clabel(cA_cont, inline=True, fontsize=8, fmt="%1.0f")

cA = axA.contourf(
    latU_18, levU_18, U_diffdiff,
    levels=diffdiff_levels_u, cmap="seismic", extend="both"
)
cbA = fig.colorbar(cA, ax=axA, orientation="vertical")
cbA.set_label(r"$\Delta$ zonal velocity difference [m/s]")
cbA.set_ticks(np.linspace(-2.5, 2.5, 11))

axA.set_ylim(1000, 100)
axA.set_xlim(-70, 70)
axA.set_xlabel("Latitude [$^\\circ$N]")
axA.set_ylabel("Pressure [hPa]")
axA.set_title(r"a) Zonal velocity: $(PI^{off}_{18}-PI^{on}_{18}) - (PI^{off}_{45}-PI^{on}_{45})$")

# non-significance markers
for lat_i in range(0, len(latU_18), 3):
    for lev_i in range(0, len(levU_18), 3):
        sig = Welch(U_resp_samples_18[:, lev_i, lat_i], U_resp_samples_45[:, lev_i, lat_i])
        if sig < 0.95:
            axA.scatter(
                latU_18[lat_i], levU_18[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none"
            )

# ---- Panel b: MSF difference-of-differences ----
axB = fig.add_subplot(gs[0, 1])

cB_cont = axB.contour(
    latS_18, levS_18, SF_base_contours,
    levels=sf_contour_levels_diff, colors="black",
    linestyles=["dotted" if lvl < 0 else "solid" for lvl in sf_contour_levels_diff]
)
axB.clabel(cB_cont, inline=True, fontsize=8, fmt="%1.0f")

cB = axB.contourf(
    latS_18, levS_18, SF_diffdiff,
    levels=diffdiff_levels_sf, cmap="seismic", extend="both"
)
cbB = fig.colorbar(cB, ax=axB, orientation="vertical")
cbB.set_label(r"$\Delta$ MSF difference [$10^{10}$ kg/s]")
cbB.set_ticks(np.linspace(-2.0, 2.0, 9))

axB.set_ylim(1000, 100)
axB.set_xlim(-70, 70)
axB.set_xlabel("Latitude [$^\\circ$N]")
axB.set_ylabel("Pressure [hPa]")
axB.set_title(r"b) MSF: $(PI^{off}_{18}-PI^{on}_{18}) - (PI^{off}_{45}-PI^{on}_{45})$")

# non-significance markers
for lat_i in range(0, len(latS_18), 3):
    for lev_i in range(0, len(levS_18), 3):
        sig = Welch(SF_resp_samples_18[:, lev_i, lat_i], SF_resp_samples_45[:, lev_i, lat_i])
        if sig < 0.95:
            axB.scatter(
                latS_18[lat_i], levS_18[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none"
            )

plt.tight_layout()
plt.savefig(directory_figures + "Figure_diffdiff_UVEL_MSF_PI18_minus_PI45.pdf",
            dpi=300, bbox_inches="tight")
plt.show()

#%%

plt.figure()
plt.contourf(latU_18, levU_18, np.mean(U_off_18, axis=0) - np.mean(U_on_18, axis=0), levels=np.linspace(-5,5,21), cmap='seismic')
plt.ylim(1000,100)
plt.colorbar()

plt.figure()
plt.contourf(latU_18, levU_18, np.mean(U_off_45, axis=0) - np.mean(U_on_45, axis=0), levels=np.linspace(-5,5,21), cmap='seismic')
plt.ylim(1000,100)
plt.colorbar()

#%%

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.15)

axA = fig.add_subplot(gs[0, 0])

cA_cont = axA.contour(
    latU, levU[10::], U_base_mean_45[10::,:],
    levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
)
axA.clabel(cA_cont, inline=True, fontsize=8, fmt="%1.0f")

cA = axA.contourf(
    latU, levU, U_diff_mean_45,
    levels=uvel_diff_levels, cmap="seismic", extend="both"
)
cbA = fig.colorbar(cA, ax=axA, orientation="vertical")
cbA.set_label("Zonal velocity difference [m/s]")
cbA.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axA.set_ylim(1000, 100)
axA.set_xlim(-70, 70)
axA.set_xlabel("Latitude [$^\\circ$N]")
axA.set_ylabel("Pressure [hPa]")
axA.set_title(r"a) Zonal velocity (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)")

for lat_i in range(0, len(latU), 3):
    for lev_i in range(0, len(levU), 3):
        sig = Welch(U_base_45[:, lev_i, lat_i], U_hos_45[:, lev_i, lat_i])
        if sig < 0.95:
            axA.scatter(
                latU[lat_i], levU[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

axB = fig.add_subplot(gs[0, 1])

cB_cont = axB.contour(
    latS, levS, SF_base_mean_45,
    levels=sf_contour_levels, colors="black",
    linestyles=["dotted" if lvl < 0 else "solid" for lvl in sf_contour_levels])
axB.clabel(cB_cont, inline=True, fontsize=8, fmt="%1.0f")

cB = axB.contourf(
    latS, levS, SF_diff_mean_45,
    levels=sf_diff_levels, cmap="seismic", extend="both")
cbB = fig.colorbar(cB, ax=axB, orientation="vertical")
cbB.set_label("MSF difference [$10^{10}$ kg/s]")
cbB.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axB.set_ylim(1000, 100)
axB.set_xlim(-70, 70)
axB.set_xlabel("Latitude [$^\\circ$N]")
axB.set_ylabel("Pressure [hPa]")
axB.set_title(r"b) Meridional streamfunction (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)")

for lat_i in range(0, len(latS), 3):
    for lev_i in range(0, len(levS), 3):
        sig = Welch(SF_base_45[:, lev_i, lat_i], SF_hos_45[:, lev_i, lat_i])
        if sig < 0.95:
            axB.scatter(
                latS[lat_i], levS[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

#subgs = gs[1, 0].subgridspec(1, 3, width_ratios=[-6, 100, -14], hspace=0.5)
axC = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())#
#axC = fig.add_subplot(subgs[0,1], projection=ccrs.Robinson())

axC.add_feature(cfeature.LAND, facecolor="lightgray")
axC.coastlines(resolution="50m")

gl = axC.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False

cf = axC.contourf(lon_SLP, lat_SLP, np.mean(SLP_3_annual, axis=0) - np.mean(SLP_2_annual, axis=0), transform=ccrs.PlateCarree(), levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
cb = fig.colorbar(cf, ax=axC, orientation='vertical', shrink=0.8)
cb.set_label('SLP difference [hPa]')
cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
axC.set_title(r'c) Annual SLP (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')

for lat_i in range(0, len(lat_SLP), 5):
    for lon_i in range(0, len(lon_SLP), 5):
        p_value = Welch(SLP_2_annual[:, lat_i, lon_i], SLP_3_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axC.scatter(
                lon_SLP[lon_i], lat_SLP[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

# =========================================================            
#subgs = gs[1, 1].subgridspec(1, 4, width_ratios=[-6, 100, -14], hspace=0.5)
axD = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
#axD = fig.add_subplot(subgs[0,2], projection=ccrs.Robinson())

axD.add_feature(cfeature.LAND, facecolor="lightgray")
axD.coastlines(resolution="50m")

gl = axD.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False

cf = axD.contourf(
    lon_SLP, lat_SLP,
    np.mean(PREC_3_annual, axis=0) - np.mean(PREC_2_annual, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-2, 2, 21), cmap='BrBG', extend='both')
cb = fig.colorbar(cf, ax=axD, orientation='vertical', shrink=0.8)
cb.set_label('PREC difference [mm/day]')
cb.set_ticks([-2, -1, 0, 1, 2])
axD.set_title(r'd) Annual precipitation (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')

for lat_i in range(0, len(lat_SLP), 5):
    for lon_i in range(0, len(lon_SLP), 5):
        p_value = Welch(PREC_2_annual[:, lat_i, lon_i], PREC_3_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axD.scatter(lon_SLP[lon_i], lat_SLP[lat_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

# ---- Panel e: ITCZ ----

subgs = gs[2, :].subgridspec(1, 3, width_ratios=[-2, 30, -4.5])
axE = fig.add_subplot(subgs[0, 1], projection=ccrs.Robinson())

cE = axE.contourf(
    lon, lat, frequency_percent_E3 - frequency_percent_E2,
    levels=np.linspace(-5, 5, 17),
    extend="both",
    cmap="BrBG",
    transform=ccrs.PlateCarree()
)

cbE = fig.colorbar(cE, ax=axE, orientation="vertical", shrink=0.6)
cbE.set_label("Difference [%]")

axE.add_feature(cfeature.LAND, facecolor="lightgray")
axE.coastlines(resolution="50m")

gl = axE.gridlines(crs=ccrs.PlateCarree())

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        p_value = Welch(ITCZ_E2[:, lat_i, lon_i], ITCZ_E3[:, lat_i, lon_i])
        if p_value <= 0.95:
            axE.scatter(
                lon[lon_i], lat[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

axE.set_title(r"e) ITCZ location (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)")

plt.tight_layout()
plt.savefig(directory_figures + "Figure_S2_CD.pdf", dpi=300, bbox_inches="tight")
plt.show()
# %%


fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.15)

axA = fig.add_subplot(gs[0, 0])

cA_cont = axA.contour(
    latU, levU[10::], U_base_mean[10::,:],
    levels=uvel_base_levels, colors="black", linestyles=uvel_linestyles
)
axA.clabel(cA_cont, inline=True, fontsize=8, fmt="%1.0f")

cA = axA.contourf(
    latU, levU, U_diff_mean,
    levels=uvel_diff_levels, cmap="seismic", extend="both"
)
cbA = fig.colorbar(cA, ax=axA, orientation="vertical")
cbA.set_label("Zonal velocity difference [m/s]")
cbA.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axA.set_ylim(1000, 100)
axA.set_xlim(-70, 70)
axA.set_xlabel("Latitude [$^\\circ$N]")
axA.set_ylabel("Pressure [hPa]")
axA.set_title(r"a) Zonal velocity (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

for lat_i in range(0, len(latU), 3):
    for lev_i in range(0, len(levU), 3):
        sig = Welch(U_base[:, lev_i, lat_i], U_hos[:, lev_i, lat_i])
        if sig < 0.95:
            axA.scatter(
                latU[lat_i], levU[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

axB = fig.add_subplot(gs[0, 1])

cB_cont = axB.contour(
    latS, levS, SF_base_mean,
    levels=sf_contour_levels, colors="black",
    linestyles=["dotted" if lvl < 0 else "solid" for lvl in sf_contour_levels])
axB.clabel(cB_cont, inline=True, fontsize=8, fmt="%1.0f")

cB = axB.contourf(
    latS, levS, SF_diff_mean,
    levels=sf_diff_levels, cmap="seismic", extend="both")
cbB = fig.colorbar(cB, ax=axB, orientation="vertical")
cbB.set_label("MSF difference [$10^{10}$ kg/s]")
cbB.set_ticks([-3, -2, -1, 0, 1, 2, 3])

axB.set_ylim(1000, 100)
axB.set_xlim(-70, 70)
axB.set_xlabel("Latitude [$^\\circ$N]")
axB.set_ylabel("Pressure [hPa]")
axB.set_title(r"b) Meridional streamfunction (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

for lat_i in range(0, len(latS), 3):
    for lev_i in range(0, len(levS), 3):
        sig = Welch(SF_base[:, lev_i, lat_i], SF_hos[:, lev_i, lat_i])
        if sig < 0.95:
            axB.scatter(
                latS[lat_i], levS[lev_i],
                marker="o", edgecolor="k", s=6, facecolors="none")

#subgs = gs[1, 0].subgridspec(1, 3, width_ratios=[-6, 100, -14], hspace=0.5)
axC = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())#
#axC = fig.add_subplot(subgs[0,1], projection=ccrs.Robinson())

axC.add_feature(cfeature.LAND, facecolor="lightgray")
axC.coastlines(resolution="50m")

gl = axC.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False

cf = axC.contourf(lon_SLP, lat_SLP, np.mean(SLP_4_annual, axis=0) - np.mean(SLP_1_annual, axis=0), transform=ccrs.PlateCarree(), levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
cb = fig.colorbar(cf, ax=axC, orientation='vertical', shrink=0.8)
cb.set_label('SLP difference [hPa]')
cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
axC.set_title(r'c) Annual SLP (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')

for lat_i in range(0, len(lat_SLP), 5):
    for lon_i in range(0, len(lon_SLP), 5):
        p_value = Welch(SLP_1_annual[:, lat_i, lon_i], SLP_4_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axC.scatter(
                lon_SLP[lon_i], lat_SLP[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

# =========================================================            
#subgs = gs[1, 1].subgridspec(1, 4, width_ratios=[-6, 100, -14], hspace=0.5)
axD = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
#axD = fig.add_subplot(subgs[0,2], projection=ccrs.Robinson())

axD.add_feature(cfeature.LAND, facecolor="lightgray")
axD.coastlines(resolution="50m")

gl = axD.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
gl.top_labels = False
gl.right_labels = False

cf = axD.contourf(
    lon_SLP, lat_SLP,
    np.mean(PREC_4_annual, axis=0) - np.mean(PREC_1_annual, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-2, 2, 21), cmap='BrBG', extend='both')
cb = fig.colorbar(cf, ax=axD, orientation='vertical', shrink=0.8)
cb.set_label('PREC difference [mm/day]')
cb.set_ticks([-2, -1, 0, 1, 2])
axD.set_title(r'd) Annual precipitation (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')

for lat_i in range(0, len(lat_SLP), 5):
    for lon_i in range(0, len(lon_SLP), 5):
        p_value = Welch(PREC_1_annual[:, lat_i, lon_i], PREC_4_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axD.scatter(lon_SLP[lon_i], lat_SLP[lat_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

# ---- Panel c: ITCZ ----

subgs = gs[2, :].subgridspec(1, 3, width_ratios=[-2, 30, -4.5])
axE = fig.add_subplot(subgs[0, 1], projection=ccrs.Robinson())

cE = axE.contourf(
    lon, lat, frequency_percent_E4 - frequency_percent_E1,
    levels=np.linspace(-5, 5, 17),
    extend="both",
    cmap="BrBG",
    transform=ccrs.PlateCarree()
)

cbE = fig.colorbar(cE, ax=axE, orientation="vertical", shrink=0.6)
cbE.set_label("Difference [%]")

axE.add_feature(cfeature.LAND, facecolor="lightgray")
axE.coastlines(resolution="50m")

gl = axE.gridlines(crs=ccrs.PlateCarree())

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        p_value = Welch(ITCZ_E1[:, lat_i, lon_i], ITCZ_E4[:, lat_i, lon_i])
        if p_value <= 0.95:
            axE.scatter(
                lon[lon_i], lat[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

axE.set_title(r"e) ITCZ location (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

plt.tight_layout()
plt.savefig(directory_figures + "Figure_2_CD.pdf", dpi=300, bbox_inches="tight")
plt.show()
# %%
