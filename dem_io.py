import sys 
import string 
import subprocess
import time
import argparse
import numpy as np 
import netCDF4 as netcdf4 
import pyproj
import rasterio
import gdal as gd

'''
routines related to reading DEM data

_is_integer_multiple:     test whether two floats are integer multiples
_north_or_south:          return 'n'/'s' label
_east_or_west:            return 'e'/'w' label
_get_MERIT_dem_filenames: return filenames required to span region
_get_ASTER_dem_filenames: return filenames required to span region

create_subregion_corner_lists: create four corner lists by subdividing input corner list
read_MERIT_dem_data:     read in DEM data for region
read_ASTER_dem_data:     read in DEM data for region

'''

def _is_integer_multiple(x,y,eps=1e-6):
    # is this just modulo?? test...
    if np.abs(np.round(x/y)-(x/y)) < eps:
        return True
    else:
        return False
                
def _north_or_south(lat):
    if lat >= 0:
        return 'n'
    else:
        return 's'
def _east_or_west(lon):
    if lon >= 0:
        return 'e'
    else:
        return 'w'

def create_subregion_corner_lists(corners,central_point,ensurePositive=True):
    clon,clat = central_point
    # split into 4 subregions, copy deepest list
    corner_list = []
    # ll
    dc = [corners[i].copy() for i in range(4)]
    dc[1][1] = clat
    dc[2][0] = clon
    dc[3] = [clon,clat]
    corner_list.append(dc)
    # ul
    dc = [corners[i].copy() for i in range(4)]
    dc[0][1] = clat
    dc[2] = [clon,clat]
    dc[3][0] = clon
    corner_list.append(dc)
    # lr
    dc = [corners[i].copy() for i in range(4)]
    dc[0][0] = clon
    dc[1] = [clon,clat]
    dc[3][1] = clat
    corner_list.append([[pt[0],pt[1]] for pt in dc])
    # ur
    dc = [corners[i].copy() for i in range(4)]
    dc[0] = [clon,clat]
    dc[1][0] = clon
    dc[2][1] = clat
    corner_list.append([pt for pt in dc])

    if ensurePositive:
        for corners in corner_list:
            for n in range(len(corners)):
                if corners[n][0] < 0:
                    corners[n][0] += 360
                    
    return corner_list

def _get_MERIT_dem_filenames(dem_file_template,corners):
    # dem_file_template is assumed to have form of:
    # 'my_path/elv_DirTag/TileTag_elv.tif'

    efiles = []

    # round to correct numbers that are just slightly less than integer
    sigfigs = 6
    ll_corner = [np.round(corners[0][0],sigfigs),np.round(corners[0][1],sigfigs)]
    ur_corner = [np.round(corners[-1][0],sigfigs),np.round(corners[-1][1],sigfigs)]
    
    lonmin, lonmax = int((ll_corner[0]//5)*5), int((ur_corner[0]//5)*5)
    latmin, latmax = int((ll_corner[1]//5)*5), int((ur_corner[1]//5)*5)
    
    # if right boundary is multiple of tile resolution, exclude it
    if (ur_corner[0]-lonmax) == 0.0:
        lnpad = 0
    else:
        lnpad = 1

    # if upper boundary is multiple of tile resolution, exclude it
    if (ur_corner[1]-latmax) == 0.0:
        ltpad = 0
    else:
        ltpad = 1

    # ensure lonmax > lonmin for regions spanning prime meridian
    if lonmax < lonmin:
        lonmax += 360
    
    nlon = lonmin + np.arange((lonmax - lonmin)//5 + lnpad)*5
    nlat = latmin + np.arange((latmax - latmin)//5 + ltpad)*5

    for lonc in nlon:
        for latc in nlat:
            tlon = int((lonc//5)*5)
            if tlon >= 180:
                tlon -= 360
            tlat = int((latc//5)*5)

            abstlon = abs(tlon)
            lonstr  = '{:03d}'.format(abstlon)
            lonstr = _east_or_west(tlon)+lonstr

            abstlat = abs(tlat)
            latstr  = '{:02d}'.format(abstlat)
            latstr = _north_or_south(tlat)+latstr

            tiletag = latstr+lonstr

            dir_tlon = (tlon//30)*30
            dir_tlat = (tlat//30)*30

            abstlon = abs(dir_tlon)
            lonstr  = '{:03d}'.format(abstlon)
            
            abstlat = abs(dir_tlat)
            latstr  = '{:02d}'.format(abstlat)
            
            dirtag = _north_or_south(tlat)+latstr \
                     +_east_or_west(tlon)+lonstr

            efile = dem_file_template.replace('DirTag',dirtag)
            efiles.append(efile.replace('TileTag',tiletag))

    # get unique values
    efiles = np.unique(np.asarray(efiles))
    numfiles = efiles.size

    # check that all files exist (call returns 0)
    # (corners may extend beyond existing dem tiles)
    emask = np.ones(efiles.size, dtype=bool)
    for n in range(efiles.size):
        geofile = efiles[n]
        command=['ls',geofile]
        file_exists=subprocess.run(command,capture_output=True).returncode
        if file_exists > 0:
            emask[n] = False
    efiles = efiles[emask]

    return efiles

def read_MERIT_dem_data(dem_file_template,corners,zeroFill=False):

    from geospatial_utils import arg_closest_point
    
    # Determine dem filenames
    demfiles = _get_MERIT_dem_filenames(dem_file_template,corners)
    
    if demfiles.size > 0:
        validDEM = True
    else:
        validDEM = False
        return {'validDEM':validDEM}

    for nfile in range(demfiles.size):
        meritfile  = demfiles[nfile]
        ds = gd.Open(meritfile)
        if nfile==0:
            crs = pyproj.Proj(ds.GetProjection(), preserve_units=True)
            # reorder geotransform to affine convention
            aff = [float(ds.GetGeoTransform()[i]) for i in [1,2,0,4,5,3]]
            affine = rasterio.Affine(*aff)
        
        # merit latitude is N->S
        merit_elev = ds.ReadAsArray()
        xs = ds.RasterXSize
        ys = ds.RasterYSize
        x  = ds.GetGeoTransform()
        x0, y0, dx, dy = x[0], x[3], x[1], x[5]
        mlon = (x0+0.5*dx) + dx*np.arange(xs)
        mlat = (y0+0.5*dy) + dy*np.arange(ys)
        dmlon = np.abs(mlon[0]-mlon[1])
        dmlat = np.abs(mlat[0]-mlat[1])

        # ensure zero is properly accounted for, so 0 is not set to 360
        less_than_zero = -1e-8
        mlon[mlon < less_than_zero] += 360
        
        # convert latitude to S->N
        mlat = np.flipud(mlat)
        merit_elev = np.flipud(merit_elev)

        fill_value = -9999
        if zeroFill:
            merit_elev[merit_elev <= fill_value] = 0

        if len(demfiles) > 1:
            if nfile==0:
                # for gridcells spanning greenwich
                dc0 = (corners[2][0] - corners[0][0])
                if dc0 < 0:
                    dc0 += 360

                nx = int(np.ceil(dc0/dmlon))
                # if integer multiple, add 1 to account for edge value
                if _is_integer_multiple(dc0,dmlon,eps=1e-8):
                    nx += 1
                x0 = int(np.floor(corners[0][0]/dmlon))*dmlon
                elon = x0 + np.arange(nx)*dmlon

                dc0 = (corners[1][1] - corners[0][1])
                ny = int(np.ceil(dc0/dmlat))
                if _is_integer_multiple(dc0,dmlat,eps=1e-8):
                    ny += 1
                y0 = int(np.floor(corners[0][1]/dmlat))*dmlat
                elat = y0 + np.arange(ny)*dmlat

                elev = np.zeros((ny,nx))

                lonmin,lonmax,latmin,latmax = np.min(elon),np.max(elon),np.min(elat),np.max(elat)
                if lonmin < 0:
                    lonmin+=360
                if lonmax < 0:
                    lonmax+=360
                if lonmin > 360:
                    lonmin-=360
                if lonmax > 360:
                    lonmax-=360

            # if gridcell spans zero longitude, shift coordinates
            if lonmin > lonmax:
                elon2 = np.where(elon <= 180,elon,elon - 360)
                mlon2 = np.where(mlon <= 180,mlon,mlon - 360)
                i1 = np.argmin(np.abs(np.min(elon2) - mlon2))
                i2 = np.argmin(np.abs(np.max(elon2) - mlon2))
                i3 = np.argmin(np.abs(np.min(mlon2[i1:i2+1]) - elon2))
                i4 = np.argmin(np.abs(np.max(mlon2[i1:i2+1]) - elon2))
            else:
                # use arg_closest_point() to compare in single precision
                i1 = arg_closest_point(np.min(elon), mlon)
                i2 = arg_closest_point(np.max(elon), mlon)
                i3 = arg_closest_point(np.min(mlon[i1:i2+1]), elon)
                i4 = arg_closest_point(np.max(mlon[i1:i2+1]), elon)

            j1 = arg_closest_point(np.min(elat), mlat)
            j2 = arg_closest_point(np.max(elat), mlat)
            j3 = arg_closest_point(np.min(mlat[j1:j2+1]), elat)
            j4 = arg_closest_point(np.max(mlat[j1:j2+1]), elat)
            
            elon[i3:i4+1] = mlon[i1:i2+1]
            elat[j3:j4+1] = mlat[j1:j2+1]
            ny,nx = mlat.size,mlon.size
            elev[j3:j4+1,i3:i4+1] = merit_elev[j1:j2+1,i1:i2+1]   

        else:
             # if gridcell spans zero longitude, shift coordinates
            if corners[0][0] > corners[3][0]:
                xind = np.where(np.logical_or(mlon >= corners[0][0],mlon < corners[3][0]))[0]
            else:
                xind = np.where(np.logical_and(mlon >= corners[0][0],mlon < corners[3][0]))[0]
            yind = np.where(np.logical_and(mlat >= corners[0][1],mlat < corners[3][1]))[0]

            if np.logical_and(xind.size > 0,yind.size > 0):
                i1, i2 = xind[0],xind[-1]
                j1, j2 = yind[0],yind[-1]

            elon = mlon[i1:i2+1]
            elat = mlat[j1:j2+1]
            ny,nx = mlat.size,mlon.size
            elev = merit_elev[j1:j2+1,i1:i2+1]
                
    # Adjust affine to represent actual elev bounds
    # x0,y0 should be top left pixel of raster
    dx, dy = affine.a, affine.e
    x0, y0 = np.min(elon)-0.5*dx, np.max(elat)-0.5*dy
    affine = rasterio.Affine(affine.a,affine.b,x0,affine.d,affine.e,y0)

    # to match affine, convert latitude back to N->S
    elat = np.flipud(elat)
    elev = np.flipud(elev)

    return {'elev':elev,'lon':elon,'lat':elat,'crs':crs,'affine':affine,'validDEM':validDEM}

def _get_ASTER_dem_filenames(dem_file_template,corners):
    # dem_file_template is assumed to have form of:
    # 'my_path/ASTGTMV003_TileTag_dem.nc'

    efiles = []

    # round to correct numbers that are just slightly less than integer
    sigfigs = 6
    ll_corner = [np.round(corners[0][0],sigfigs),np.round(corners[0][1],sigfigs)]
    ur_corner = [np.round(corners[-1][0],sigfigs),np.round(corners[-1][1],sigfigs)]
    
    lonmin, lonmax = int((ll_corner[0]//1)*1), int((ur_corner[0]//1)*1)
    latmin, latmax = int((ll_corner[1]//1)*1), int((ur_corner[1]//1)*1)
    
    # if right boundary is multiple of tile resolution, exclude it
    if (ur_corner[0]-lonmax) == 0.0:
        lnpad = 0
    else:
        lnpad = 1
    # if upper boundary is multiple of tile resolution, exclude it
    if (ur_corner[1]-latmax) == 0.0:
        ltpad = 0
    else:
        ltpad = 1

    # ensure lonmax > lonmin for regions spanning prime meridian
    if lonmax < lonmin:
        lonmax += 360
    
    nlon = lonmin + np.arange((lonmax - lonmin)//1 + lnpad)*1
    nlat = latmin + np.arange((latmax - latmin)//1 + ltpad)*1

    for lonc in nlon:
        for latc in nlat:
    
            tlon = int((lonc//1)*1)
            if tlon >= 180:
                tlon -= 360
            tlat = int((latc//1)*1)

            abstlon = abs(tlon)
            lonstr  = '{:03d}'.format(abstlon)
            lonstr = _east_or_west(tlon)+lonstr

            abstlat = abs(tlat)
            latstr  = '{:02d}'.format(abstlat)
            latstr = _north_or_south(tlat)+latstr

            tiletag = latstr+lonstr

            efiles.append(dem_file_template.replace('TileTag',tiletag.upper()))

    # get unique values
    efiles = np.unique(np.asarray(efiles))
    numfiles = efiles.size

    # check that all files exist (call returns 0)
    # (corners may extend beyond existing dem tiles)
    emask = np.ones(efiles.size, dtype=bool)
    for n in range(efiles.size):
        geofile = efiles[n]
        command=['ls',geofile]
        file_exists=subprocess.run(command,capture_output=True).returncode
        if file_exists > 0:
            emask[n] = False
    efiles = efiles[emask]

    return efiles

def read_ASTER_dem_data(dem_file_template,corners,zeroFill=False):

    from geospatial_utils import arg_closest_point

    # Determine dem filenames
    demfiles = _get_ASTER_dem_filenames(dem_file_template,corners)

    # Check for unneeded files (corner < 1 pixel from boundary)
    sigfigs = 6
    ll_corner = [np.round(corners[0][0],sigfigs),np.round(corners[0][1],sigfigs)]
    ur_corner = [np.round(corners[-1][0],sigfigs),np.round(corners[-1][1],sigfigs)]
    
    if demfiles.size > 0:
        validDEM = True
    else:
        validDEM = False
        return {'validDEM':validDEM}
        
    for nfile in range(demfiles.size):
        asterfile  = demfiles[nfile]
        f = netcdf4.Dataset(asterfile, 'r')
        # coordinates
        mlon = np.asarray(f.variables['lon'][:,])
        mlat = np.asarray(f.variables['lat'][:,])
        im  = mlon.size
        jm  = mlat.size
        aster_elev = np.asarray(f.variables['ASTER_GDEM_DEM'][:,],dtype=float)
        ys, xs = aster_elev.shape
        
        if nfile == 0:
            fill_value = f.variables['ASTER_GDEM_DEM'].getncattr('_FillValue')
            crs = pyproj.Proj(f.variables['crs'].spatial_ref, preserve_units=True)
            # reorder geotransform to affine convention
            aff = [float(f.variables['crs'].GeoTransform.split()[i]) for i in [1,2,0,4,5,3]]
            affine = rasterio.Affine(*aff)

        f.close()

        dmlon = np.abs(mlon[0]-mlon[1])
        dmlat = np.abs(mlat[0]-mlat[1])

        # ensure zero is properly accounted for, so 0 is not set to 360
        less_than_zero = -1e-8
        mlon[mlon < less_than_zero] += 360

        # convert latitude to S->N
        mlat = np.flipud(mlat)
        aster_elev = np.flipud(aster_elev)

        fill_value = -9999
        if zeroFill:
            aster_elev[aster_elev <= fill_value] = 0

        if len(demfiles) > 1:
            if nfile==0:
                dc0 = (corners[2][0] - corners[0][0])
                if dc0 < 0:
                    dc0 += 360
                
                nx = int(np.ceil(dc0/dmlon))
                # if integer multiple, add 1 to account for edge value
                if _is_integer_multiple(dc0,dmlon,eps=1e-8):
                    nx += 1
                x0 = int(np.floor(corners[0][0]/dmlon))*dmlon
                if x0 < corners[0][0]:
                    nx += 1
                elon = x0 + np.arange(nx)*dmlon

                dc0 = (corners[1][1] - corners[0][1])
                ny = int(np.ceil(dc0/dmlat))
                if _is_integer_multiple(dc0,dmlat,eps=1e-8):
                    ny += 1
                y0 = int(np.floor(corners[0][1]/dmlat))*dmlat
                if y0 < corners[0][1]:
                    ny += 1
                elat = y0 + np.arange(ny)*dmlat

                elev = np.zeros((ny,nx))

                lonmin,lonmax,latmin,latmax = np.min(elon),np.max(elon),np.min(elat),np.max(elat)

                if lonmin < 0:
                    lonmin+=360
                if lonmax < 0:
                    lonmax+=360
                if lonmin > 360:
                    lonmin-=360
                if lonmax > 360:
                    lonmax-=360

            # if gridcell spans zero longitude, shift coordinates
            if lonmin > lonmax:
                elon2 = np.where(elon <= 180,elon,elon - 360)
                mlon2 = np.where(mlon <= 180,mlon,mlon - 360)

                i1 = np.argmin(np.abs(np.min(elon2) - mlon2))
                i2 = np.argmin(np.abs(np.max(elon2) - mlon2))
                i3 = np.argmin(np.abs(np.min(mlon2[i1:i2+1]) - elon2))
                i4 = np.argmin(np.abs(np.max(mlon2[i1:i2+1]) - elon2))
            else:
                # use arg_closest_point() to compare in single precision
                i1 = arg_closest_point(np.min(elon), mlon)
                i2 = arg_closest_point(np.max(elon), mlon)
                i3 = arg_closest_point(np.min(mlon[i1:i2+1]), elon)
                i4 = arg_closest_point(np.max(mlon[i1:i2+1]), elon)

            j1 = arg_closest_point(np.min(elat), mlat)
            j2 = arg_closest_point(np.max(elat), mlat)
            j3 = arg_closest_point(np.min(mlat[j1:j2+1]), elat)
            j4 = arg_closest_point(np.max(mlat[j1:j2+1]), elat)

            elon[i3:i4+1] = mlon[i1:i2+1]
            elat[j3:j4+1] = mlat[j1:j2+1]
            ny,nx = mlat.size,mlon.size
            elev[j3:j4+1,i3:i4+1] = aster_elev[j1:j2+1,i1:i2+1]

        else:
            # if gridcell spans zero longitude, 
            if corners[0][0] > corners[3][0]:
                xind = np.where(np.logical_or(mlon >= corners[0][0],mlon < corners[3][0]))[0]
            else:
                xind = np.where(np.logical_and(mlon >= corners[0][0],mlon < corners[3][0]))[0]
            yind = np.where(np.logical_and(mlat >= corners[0][1],mlat < corners[3][1]))[0]

            if np.logical_and(xind.size > 0,yind.size > 0):
                i1, i2 = xind[0],xind[-1]
                j1, j2 = yind[0],yind[-1]

            elon = mlon[i1:i2+1]
            elat = mlat[j1:j2+1]
            ny,nx = mlat.size,mlon.size
            elev = aster_elev[j1:j2+1,i1:i2+1]
            
    # Adjust affine to represent actual elev bounds
    # x0,y0 should be top left pixel of raster
    dx, dy = affine.a, affine.e
    x0, y0 = np.min(elon)-0.5*dx, np.max(elat)-0.5*dy
    affine = rasterio.Affine(affine.a,affine.b,x0,affine.d,affine.e,y0)

    # to match affine, convert latitude back to N->S
    elat = np.flipud(elat)
    elev = np.flipud(elev)

    return {'elev':elev,'lon':elon,'lat':elat,'crs':crs,'affine':affine,'validDEM':validDEM}

