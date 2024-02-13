#!/usr/bin/env python
# coding: utf-8

import sys 
import string 
import subprocess
import copy
import inspect
import time
import argparse
import numpy as np 
import netCDF4 as netcdf4 
import matplotlib 
import matplotlib.pyplot as plt
from numpy.random import default_rng

from representative_hillslope import CalcGeoparamsGridcell

# Create representative hillslope geomorphic parameters
parser = argparse.ArgumentParser(description='Geomorphic parameter analysis')
parser.add_argument("cndx", help="chunk", nargs='?',type=int,default=0)
parser.add_argument("-o", "--overwrite", help="overwrite", action="store_true",default=False)
parser.add_argument("-d", "--debug", help="print debugging info", action="store_true",default=False)
parser.add_argument("-t", "--timer", help="print timing info", action="store_true",default=False)
parser.add_argument("--pt", help="location", nargs='?',type=int,default=0)
parser.add_argument("--asp_ndx", help="aspect", nargs='?',type=int,default=0)
parser.add_argument("--form", help="hillslope form", nargs='?',type=int,default=0)
parser.add_argument("--pngfile", help="output file", nargs='?',type=str,default=None)
args = parser.parse_args()

cndx = args.cndx

# chunk data 
nChunks = 6
totalChunks = nChunks*nChunks
if cndx < 0 or cndx > totalChunks:
    print('cndx must be 1-{:d}'.format(totalChunks))
    stop
if cndx == 0 and args.pt < 1:
    print('cndx = 0; select a pt with --pt')
    stop

print('Chunk ', cndx)    
chunkLabel = '{:02d}'.format(cndx)

if type(args.pngfile) != type(None):
    usePngArg = True
    pngfile = args.pngfile
else:
    usePngArg = False
    pngfile = './geoparams.png'

doTimer = args.timer

if doTimer:
    stime = time.time()

addStreamChannelVariables = True
    
useMultiprocessing = False
#useMultiprocessing = True

detrendElevation = True

setMinToZero = True

filterElevation = True

applyOceanMask = False

# set maximum hillslope length [m]
maxHillslopeLength = 10 * 1e3

# select hillslope plan form to use
if args.form == 0:
    hillslope_form = 'CircularSection'
if args.form == 1:
    hillslope_form = 'TriangularSection'

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

#@
# Set parameters used to define hillslope discretization
dtr = np.pi/180.
re = 6.371e6
# number of elevation bins
nbins = 4
# number of aspect bins (ordered N, E, S, W)
naspect = 4 
aspect_bins = [[315,45],[45,135],[135,225],[225,315]]
aspect_labels = ['North','East','South','West']
asp_name = ['north','east','south','west']
# number of total hillslope elements
ncolumns_per_gridcell = naspect * nbins
nhillslope = naspect

# use MERIT dem
dem_source = 'MERIT'

# define regions from gridcells in surface data file
snum = 1
if snum == 1:
    sfcfile = '/fs/cgd/csm/inputdata/lnd/clm2/surfdata_map/surfdata_0.9x1.25_78pfts_CMIP6_simyr2000_c170824.nc'
    if cndx == 0:
        odir = './'
        outfile = odir + 'HAND_'+str(nbins)+'_col_hillslope_geo_params_section_quad_0.9x1.25_Global.nc'
    else:
        odir = './'
        outfile = odir + 'chunk_'+chunkLabel+'_HAND_'+str(nbins)+'_col_hillslope_geo_params_section_quad_0.9x1.25_Global.nc'
        
# Select DEM source data
if dem_source == 'MERIT':
    efile0 = '/project/tss02/swensosc/MERIT/data/elv_DirTag/TileTag_elv.tif'
    outfile = outfile.replace('.nc','_MERIT.nc')
    print('\ndem template files: ',efile0,'\n')

f = netcdf4.Dataset(sfcfile, 'r')
slon2d = np.asarray(f.variables['LONGXY'][:,])
slat2d = np.asarray(f.variables['LATIXY'][:,])
slon = np.squeeze(slon2d[0,:])                                 
slat = np.squeeze(slat2d[:,0])
sim  = slon.size
sjm  = slat.size
landmask = np.asarray(f.variables['PFTDATA_MASK'][:,])
pct_natveg = np.asarray(f.variables['PCT_NATVEG'][:,])
pct_lake = np.asarray(f.variables['PCT_LAKE'][:,])
f.close()

landmask[pct_natveg <= 0] = 0

dlon = np.abs(slon[0]-slon[1])
dlat = np.abs(slat[0]-slat[1])    

# limit maximum hillslope length to fraction of grid spacing
hsf = 0.25
maxHillslopeLength = np.min([maxHillslopeLength,hsf*re*dtr*dlat])
print('max hillslope length ', maxHillslopeLength)

# initialize new fields to be added to surface data file
hand   = np.zeros((ncolumns_per_gridcell,sjm,sim))
dtnd   = np.zeros((ncolumns_per_gridcell,sjm,sim))
area   = np.zeros((ncolumns_per_gridcell,sjm,sim))
slope  = np.zeros((ncolumns_per_gridcell,sjm,sim))
aspect = np.zeros((ncolumns_per_gridcell,sjm,sim))
width  = np.zeros((ncolumns_per_gridcell,sjm,sim))
zbedrock = np.zeros((ncolumns_per_gridcell,sjm,sim))

pct_hillslope   = np.zeros((nhillslope,sjm,sim))
hillslope_index = np.zeros((ncolumns_per_gridcell,sjm,sim))
column_index    = np.zeros((ncolumns_per_gridcell,sjm,sim))
downhill_column_index  = np.zeros((ncolumns_per_gridcell,sjm,sim))

nhillcolumns = np.zeros((sjm,sim))

if addStreamChannelVariables:
    wdepth = np.zeros((sjm,sim))
    wwidth = np.zeros((sjm,sim))
    wslope = np.zeros((sjm,sim))
            
chunk_mask   = np.zeros((sjm,sim))

#@
ptnum = args.pt
if ptnum == 0:
    checkSinglePoint = False
else:
    checkSinglePoint = True
if checkSinglePoint:
    if ptnum == 1:
        # colorado
        plon,plat = 254.,40
        
    makePlot = True
    
    kstart = np.argmin(np.abs(slon2d-plon)+np.abs(slat2d-plat))
    jstart,istart =np.unravel_index(kstart,slon2d.shape)
    plon,plat = slon[istart],slat[jstart]    
    print('jstart,istart ',jstart,istart)
    print(slon[istart],slat[jstart])
    
    iend = istart+1    
    jend = jstart+1
    verbose = True
else:
    istart, iend = 0, sim
    jstart, jend = 0, sjm
    verbose = args.debug

    nichunk = int(sim//nChunks)
    njchunk = int(sjm//nChunks)
    i = (cndx-1)//nChunks
    j = np.mod((cndx-1),nChunks)
    istart,iend = i*nichunk,min([(i+1)*nichunk,sim])
    jstart,jend = j*njchunk,min([(j+1)*njchunk,sjm])

    # adjust for remainder
    if (sim - iend) < nichunk:
        #print('adjusting i ',iend,sim,nichunk) 
        iend = sim
    if (sjm - jend) < njchunk:
        #print('adjusting j ',jend,sjm,njchunk)
        jend = sjm

#@
# Loop over points in domain
ji_pairs = []
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] == 1:
            ji_pairs.append([j,i])

#debug
ji_pairs = [[181,240]]

            
print('number of points ',len(ji_pairs),'\n')
    
# randomize point list to avoid multiple processes working on same point
randomizePointList = False
#randomizePointList = True
if randomizePointList:
    rng = default_rng()
    ji_pair_array = np.asarray(ji_pairs)
    rng.shuffle(ji_pair_array)
    ji_pairs = ji_pair_array.tolist()

# loop over point list
for k in ji_pairs:
    j,i = k
    print('Beginning gridcell ',j,i,flush=printFlush)
    x = CalcGeoparamsGridcell([j,i], \
                              lon2d=slon2d, \
                              lat2d=slat2d, \
                              landmask=landmask, \
                              nhand_bins=nbins, \
                              aspect_bins=aspect_bins, \
                              ncolumns_per_gridcell=ncolumns_per_gridcell, \
                              maxHillslopeLength=maxHillslopeLength,
                              hillslope_form=hillslope_form, \
                              dem_file_template=efile0, \
                              detrendElevation=detrendElevation, \
                              nlambda=nlambda, \
                              dem_source=dem_source, \
                              flagBasins=flagBasins, \
                              outfile_template=outfile, \
                              overwrite=args.overwrite, \
                              printData=checkSinglePoint, \
                              verbose=verbose)

if doTimer:
    etime = time.time()
    print('\nTime to complete script: {:.3f} seconds'.format(etime-stime))

    
