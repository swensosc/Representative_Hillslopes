#!/usr/bin/env python
# coding: utf-8

import time
import argparse
import os
import numpy as np
import netCDF4 as netcdf4
from numpy.random import default_rng

from representative_hillslope import CalcGeoparamsGridcell

# Create representative hillslope geomorphic parameters
parser = argparse.ArgumentParser(description="Geomorphic parameter analysis")
parser.add_argument("cndx", help="chunk", nargs="?", type=int, default=0)
parser.add_argument("--overwrite", help="overwrite", action="store_true", default=False)
parser.add_argument(
    "-d", "--debug", help="print debugging info", action="store_true", default=False
)
parser.add_argument(
    "-t", "--timer", help="print timing info", action="store_true", default=False
)
parser.add_argument("--pt", help="location", nargs="?", type=int, default=0)
parser.add_argument(
    "--hillslope-form",
    help="hillslope form",
    type=str,
    default="Trapezoidal",
    choices=["Trapezoidal", "AnnularSection", "CircularSection", "TriangularSection"],
)
parser.add_argument(
    "--sfcfile",
    help="Surface dataset from which grid should be taken",
    default="surfdata_0.9x1.25_78pfts_CMIP6_simyr2000_c170824.nc",
)
parser.add_argument(
    "-o",
    "--output-dir",
    help="Directory where output file should be saved (default: current dir)",
    default=os.getcwd(),
)
parser.add_argument(
    "--use-multi-processing",
    action="store_true",
    dest="useMultiProcessing",
    help="Use multiple processors",
)

default_nchunks = 6
parser.add_argument(
    "--nchunks",
    type=int,
    default=default_nchunks,
    help=f"Number of chunks to split processing into (default: {default_nchunks})",
)

dem_source_default = "MERIT"
dem_data_path_default = os.path.join("MERIT", "data")
parser.add_argument(
    "--dem-source",
    "--dem_source",
    type=str,
    default=dem_source_default,
    help=f"DEM to use (default: {dem_source_default})",
)
parser.add_argument(
    "--dem-data-path",
    type=str,
    default=dem_data_path_default,
    help=f"Path to DEM source data (default: {dem_data_path_default})",
)

parser.add_argument(
    "--no-add-stream-channel-vars",
    action="store_false",
    dest="addStreamChannelVariables",
    help="Do not add stream channel variables",
)
parser.add_argument(
    "--no-detrend-elevation",
    action="store_false",
    dest="detrendElevation",
    help="Do not detrend elevation",
)

default_n_bins = 4
parser.add_argument(
    "--n-bins",
    type=int,
    default=default_n_bins,
    help=f"Number of elevation bins (default: {default_n_bins})",
)
default_n_aspect = 4
parser.add_argument(
    "--n-aspect",
    type=int,
    default=default_n_aspect,
    help=f"Number of aspect bins (ordered clockwise from N; default: {default_n_aspect})",
)

args = parser.parse_args()

# Check paths
if not os.path.exists(args.sfcfile):
    raise FileNotFoundError(f"sfcfile not found: {args.sfcfile}")
if not os.path.exists(args.dem_data_path):
    raise FileNotFoundError(f"dem_data_path not found: {args.dem_data_path}")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Check and process chunk settings
totalChunks = args.nchunks * args.nchunks
if args.cndx < 0 or args.cndx > totalChunks:
    raise RuntimeError("args.cndx must be 1-{:d}".format(totalChunks))
if args.cndx == 0 and args.pt < 1:
    raise RuntimeError("args.cndx = 0; select a pt with --pt")

print("Chunk ", args.cndx)
chunkLabel = "{:02d}".format(args.cndx)

doTimer = args.timer

if doTimer:
    stime = time.time()

# set maximum hillslope length [m]
maxHillslopeLength = 10 * 1e3

# set number of bins for spectra
nlambda = 30

# fill values
fill_value = -9999

# ensure stream network is consistent with flow directions
useConsistentChannelMask = True

# removeTailDTND is meant to remove effects of basin, where
# the algorithm follows an ad hoc path through the basin leading
# to large dtnd values in the basin
removeTailDTND = True

# instead try identifying basins and flagging those points before
# determining representative hillslopes
flagBasins = False

printFlush = True

makePlot = False

# Set parameters used to define hillslope discretization
dtr = np.pi / 180.0
re = 6.371e6
if args.n_aspect == 4:
    aspect_bins = [[315, 45], [45, 135], [135, 225], [225, 315]]
    aspect_labels = ["North", "East", "South", "West"]
    asp_name = ["north", "east", "south", "west"]
else:
    raise RuntimeError(f"Unhandled --n-aspect: {args.n_aspect}")
# number of total hillslope elements
ncolumns_per_gridcell = args.n_aspect * args.n_bins
nhillslope = args.n_aspect

# Define output file template
outfile_template = os.path.join(
    args.output_dir,
    "chunk_"
    + chunkLabel
    + "_HAND_"
    + str(args.n_bins)
    + f"_col_hillslope_geo_params_{args.hillslope_form}.nc",
)

# Select DEM source data
if args.dem_source == "MERIT":
    efile0 = os.path.join(args.dem_data_path, "elv_DirTag", "TileTag_elv.tif")
    outfile_template = outfile_template.replace(".nc", "_MERIT.nc")
    print("\ndem template files: ", efile0, "\n")
else:
    raise ValueError(f"Invalid setting for --dem-source: {args.dem_source}")

print(f"Output filename template: {outfile_template}")

f = netcdf4.Dataset(args.sfcfile, "r")
slon2d = np.asarray(
    f.variables["LONGXY"][
        :,
    ]
)
slat2d = np.asarray(
    f.variables["LATIXY"][
        :,
    ]
)
slon = np.squeeze(slon2d[0, :])
slat = np.squeeze(slat2d[:, 0])
sim = slon.size
sjm = slat.size
landmask = np.asarray(
    f.variables["PFTDATA_MASK"][
        :,
    ]
)
pct_natveg = np.asarray(
    f.variables["PCT_NATVEG"][
        :,
    ]
)
f.close()

landmask[pct_natveg <= 0] = 0

dlon = np.abs(slon[0] - slon[1])
dlat = np.abs(slat[0] - slat[1])

# limit maximum hillslope length to fraction of grid spacing
hsf = 0.25
maxHillslopeLength = np.min([maxHillslopeLength, hsf * re * dtr * dlat])
print("max hillslope length ", maxHillslopeLength)

# initialize new fields to be added to surface data file
hand = np.zeros((ncolumns_per_gridcell, sjm, sim))
dtnd = np.zeros((ncolumns_per_gridcell, sjm, sim))
area = np.zeros((ncolumns_per_gridcell, sjm, sim))
slope = np.zeros((ncolumns_per_gridcell, sjm, sim))
aspect = np.zeros((ncolumns_per_gridcell, sjm, sim))
width = np.zeros((ncolumns_per_gridcell, sjm, sim))
zbedrock = np.zeros((ncolumns_per_gridcell, sjm, sim))

pct_hillslope = np.zeros((nhillslope, sjm, sim))
hillslope_index = np.zeros((ncolumns_per_gridcell, sjm, sim))
column_index = np.zeros((ncolumns_per_gridcell, sjm, sim))
downhill_column_index = np.zeros((ncolumns_per_gridcell, sjm, sim))

nhillcolumns = np.zeros((sjm, sim))

if args.addStreamChannelVariables:
    wdepth = np.zeros((sjm, sim))
    wwidth = np.zeros((sjm, sim))
    wslope = np.zeros((sjm, sim))

chunk_mask = np.zeros((sjm, sim))

ptnum = args.pt
if ptnum == 0:
    checkSinglePoint = False
else:
    checkSinglePoint = True
if checkSinglePoint:
    if ptnum == 1:
        # colorado
        plon, plat = 254.0, 40

    makePlot = True

    kstart = np.argmin(np.abs(slon2d - plon) + np.abs(slat2d - plat))
    jstart, istart = np.unravel_index(kstart, slon2d.shape)
    plon, plat = slon[istart], slat[jstart]
    print("jstart,istart ", jstart, istart)
    print(slon[istart], slat[jstart])

    iend = istart + 1
    jend = jstart + 1
    verbose = True
else:
    istart, iend = 0, sim
    jstart, jend = 0, sjm
    verbose = args.debug

    nichunk = int(sim // args.nchunks)
    njchunk = int(sjm // args.nchunks)
    i = (args.cndx - 1) // args.nchunks
    j = np.mod((args.cndx - 1), args.nchunks)
    istart, iend = i * nichunk, min([(i + 1) * nichunk, sim])
    jstart, jend = j * njchunk, min([(j + 1) * njchunk, sjm])

    # adjust for remainder
    if (sim - iend) < nichunk:
        # print('adjusting i ',iend,sim,nichunk)
        iend = sim
    if (sjm - jend) < njchunk:
        # print('adjusting j ',jend,sjm,njchunk)
        jend = sjm

# Loop over points in domain
ji_pairs = []
for j in range(jstart, jend):
    for i in range(istart, iend):
        if landmask[j, i] == 1:
            ji_pairs.append([j, i])

n_points = len(ji_pairs)
print("number of points ", n_points, "\n")

# randomize point list to avoid multiple processes working on same point
randomizePointList = False
# randomizePointList = True
if randomizePointList:
    rng = default_rng()
    ji_pair_array = np.asarray(ji_pairs)
    rng.shuffle(ji_pair_array)
    ji_pairs = ji_pair_array.tolist()

# loop over point list
for index, k in enumerate(ji_pairs):
    j, i = k
    print(f"Beginning gridcell {j} {i} ({index+1}/{n_points})", flush=printFlush)
    CalcGeoparamsGridcell(
        [j, i],
        lon2d=slon2d,
        lat2d=slat2d,
        landmask=landmask,
        nhand_bins=args.n_bins,
        aspect_bins=aspect_bins,
        ncolumns_per_gridcell=ncolumns_per_gridcell,
        maxHillslopeLength=maxHillslopeLength,
        hillslope_form=args.hillslope_form,
        dem_file_template=efile0,
        detrendElevation=args.detrendElevation,
        nlambda=nlambda,
        dem_source=args.dem_source,
        flagBasins=flagBasins,
        outfile_template=outfile_template,
        overwrite=args.overwrite,
        printData=checkSinglePoint,
        verbose=verbose,
        useMultiProcessing=args.useMultiProcessing,
    )

if doTimer:
    etime = time.time()
    print("\nTime to complete script: {:.3f} seconds".format(etime - stime))
