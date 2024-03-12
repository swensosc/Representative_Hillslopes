import subprocess
import numpy as np 
import netCDF4 as netcdf4 
import pyproj
import rasterio
from osgeo import gdal as gd

from geospatial_utils import arg_closest_point

'''
routines related to reading DEM data

_north_or_south:          return 'n'/'s' label
_east_or_west:            return 'e'/'w' label
_get_MERIT_dem_filenames: return filenames required to span region
_get_ASTER_dem_filenames: return filenames required to span region

create_subregion_corner_lists: create four corner lists by subdividing input corner list
read_MERIT_dem_data:     read in DEM data for region
read_ASTER_dem_data:     read in DEM data for region

'''

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

def _check_files_exist(dem_file_template, efiles):
    emask = np.ones(efiles.size, dtype=bool)
    for n in range(efiles.size):
        geofile = efiles[n]
        command=['ls',geofile]
        file_exists=subprocess.run(command,capture_output=True).returncode
        if file_exists > 0:
            emask[n] = False
    if not np.any(emask):
        print("All DEM files missing:")
        for file in efiles:
            print(f"   {file}")
        raise FileNotFoundError(f"No DEM files found matching template: {dem_file_template}")
    efiles = efiles[emask]
    return efiles

def _create_grid(corners, x0, y0, dmlon, dmlat, which_dem, tol):
    '''
    identify closest points to corners on DEM grid,
    ensuring that the region they define is larger than
    the region defined by corners
    '''

    # left side
    n0 = np.round((corners[0][0] - x0)/dmlon, tol)
    ex0 = x0 + np.floor(n0)*dmlon

    # ex0 should be < left edge, and within dmlon
    delta_lon = (corners[0][0]-ex0)
    if delta_lon > 360:
        delta_lon -= 360
    if np.round(delta_lon/dmlon, tol) > 1 or np.round(delta_lon/dmlon, tol) < 0:
        raise RuntimeError('ex0 ',ex0,corners[0][0],(corners[0][0]-ex0)/dmlon)

    # right side
    delta_lon = (corners[2][0] - ex0)
    # for gridcells spanning greenwich
    if delta_lon < 0:
        delta_lon += 360

    nx = np.ceil(delta_lon/dmlon).astype(int)

    delta_lon = ((ex0+nx*dmlon)-corners[2][0])
    if delta_lon > 360:
        delta_lon -= 360
    if np.round(delta_lon/dmlon, tol) > 1 or np.round(delta_lon/dmlon, tol) < 0:
        raise RuntimeError(ex0+nx*dmlon,corners[2][0])

    elon = ex0 + (np.arange(nx)+0.5)*dmlon
    if which_dem == "ASTER":
        elon[elon >= 360] -= 360
    elif which_dem != "MERIT":
        raise RuntimeError(f"Unrecognized DEM: {which_dem}")
            
    # bottom
    m0 = np.round((corners[0][1] - y0)/dmlat,tol)
    ey0 = y0 + np.floor(m0)*dmlat

    # ey0 should be < lower edge, and within dmlat
    if np.round((corners[0][1]-ey0)/dmlat, tol) > 1 or np.round((corners[0][1]-ey0)/dmlat, tol) < 0:
        raise RuntimeError('ey0 ',ey0,corners[0][1],(corners[0][1]-ey0)/dmlat)

    # top
    delta_lat = (corners[1][1] - ey0)
    ny = np.ceil(delta_lat/dmlat).astype(int)

    if np.round(((ey0+ny*dmlat)-corners[1][1])/dmlat, tol) > 1 or np.round(((ey0+ny*dmlat)-corners[1][1])/dmlat, tol) < 0:
        raise RuntimeError(ey0+ny*dmlat,corners[1][1])

    elat = ey0 + (np.arange(ny)+0.5)*dmlat

    # initialize output array
    elev = np.zeros((ny,nx))
    return elon,elat,elev

def _get_MERIT_dem_filenames(dem_file_template,corners):
    # dem_file_template is assumed to have form of:
    # 'my_path/elv_DirTag/TileTag_elv.tif'

    # tiles are 5 x 5 degree, directories contain 30 degree band
    mres = 5
    dres = 30
    sigfigs = 6

    # round to correct numbers that are just slightly less than integer
    ll_corner = [np.round(corners[0][0],sigfigs),np.round(corners[0][1],sigfigs)]
    ur_corner = [np.round(corners[-1][0],sigfigs),np.round(corners[-1][1],sigfigs)]
    
    lonmin, lonmax = int((ll_corner[0]//mres)*mres), int((ur_corner[0]//mres)*mres)
    latmin, latmax = int((ll_corner[1]//mres)*mres), int((ur_corner[1]//mres)*mres)
    
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
    
    nlon = lonmin + np.arange((lonmax - lonmin)//mres + lnpad)*mres
    nlat = latmin + np.arange((latmax - latmin)//mres + ltpad)*mres

    efiles = []
    for lonc in nlon:
        for latc in nlat:
            tlon = int((lonc//mres)*mres)
            if tlon >= 180:
                tlon -= 360
            tlat = int((latc//mres)*mres)

            abstlon = abs(tlon)
            lonstr  = '{:03d}'.format(abstlon)
            lonstr = _east_or_west(tlon)+lonstr

            abstlat = abs(tlat)
            latstr  = '{:02d}'.format(abstlat)
            latstr = _north_or_south(tlat)+latstr

            tiletag = latstr+lonstr

            dir_tlon = (tlon//dres)*dres
            dir_tlat = (tlat//dres)*dres

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
    efiles = _check_files_exist(dem_file_template, efiles)

    return efiles

def read_MERIT_dem_data(dem_file_template,corners,tol=10,zeroFill=False):

    # Determine dem filenames
    # MERIT filenames indicate lower left corner of tile
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
        # convert longitude to [0,360]
        if x0 < 0:
            x0 += 360

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

        # create grid that will be filled sequentially by dem files
        if nfile==0:
            elon, elat, elev = _create_grid(corners, x0, y0, dmlon, dmlat, "MERIT", tol)

        # locate dem tile within grid

        # use arg_closest_point() to compare in single precision
        i1 = arg_closest_point(elon[0],  mlon, angular=True)
        i2 = arg_closest_point(elon[-1], mlon, angular=True)
        i3 = arg_closest_point(mlon[i1], elon, angular=True)
        i4 = arg_closest_point(mlon[i2], elon, angular=True)

        j1 = arg_closest_point(elat[0],  mlat)
        j2 = arg_closest_point(elat[-1], mlat)
        j3 = arg_closest_point(mlat[j1], elat)
        j4 = arg_closest_point(mlat[j2], elat)

        if np.abs(np.mean(elon[i3:i4+1]-mlon[i1:i2+1])) > 1e-10:
            print(np.mean(elon[i3:i4+1]-mlon[i1:i2+1]))
            print(elon[i3:i4+1][:10])
            print(mlon[i1:i2+1][:10])
        if np.abs(np.mean(elat[j3:j4+1]-mlat[j1:j2+1])) > 1e-10:
            print(np.mean(elat[j3:j4+1]-mlat[j1:j2+1]))
            print(elat[j3:j4+1][:10])
            print(mlat[j1:j2+1][:10])

        elev[j3:j4+1,i3:i4+1] = merit_elev[j1:j2+1,i1:i2+1]

    # Adjust affine to represent actual elev bounds
    # x0,y0 should be top left pixel of raster
    dx, dy = affine.a, affine.e
    x0, y0 = elon[0]-0.5*np.abs(dx), elat[-1]+0.5*np.abs(dy)
    affine = rasterio.Affine(affine.a,affine.b,x0,affine.d,affine.e,y0)

    # for grids spanning greenwich
    elon[elon >= 360] -= 360
    # to match affine, convert latitude back to N->S
    elat = np.flipud(elat)
    elev = np.flipud(elev)

    return {'elev':elev,'lon':elon,'lat':elat,'crs':crs,'affine':affine,'validDEM':validDEM}

def _get_ASTER_dem_filenames(dem_file_template,corners):
    # dem_file_template is assumed to have form of:
    # 'my_path/ASTGTMV003_TileTag_dem.nc'

    # tiles are 1 x 1 degree
    ares = 1

    # round to correct numbers that are just slightly less than integer
    sigfigs = 6
    ll_corner = [np.round(corners[0][0],sigfigs),np.round(corners[0][1],sigfigs)]
    ur_corner = [np.round(corners[-1][0],sigfigs),np.round(corners[-1][1],sigfigs)]
    
    lonmin, lonmax = int((ll_corner[0]//ares)*ares), int((ur_corner[0]//ares)*ares)
    latmin, latmax = int((ll_corner[1]//ares)*ares), int((ur_corner[1]//ares)*ares)
    
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
    
    nlon = lonmin + np.arange((lonmax - lonmin)//ares + lnpad)*ares
    nlat = latmin + np.arange((latmax - latmin)//ares + ltpad)*ares

    efiles = []
    for lonc in nlon:
        for latc in nlat:
    
            tlon = int((lonc//ares)*ares)
            if tlon >= 180:
                tlon -= 360
            tlat = int((latc//ares)*ares)

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
    efiles = _check_files_exist(dem_file_template, efiles)

    return efiles

def read_ASTER_dem_data(dem_file_template,corners,tol=10,zeroFill=False):
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
        mlon = f.variables['lon'][:,]
        mlat = f.variables['lat'][:,]
        im  = mlon.size
        jm  = mlat.size
        aster_elev = f.variables['ASTER_GDEM_DEM'][:,].astype(float)
        ys, xs = aster_elev.shape
        # convert longitude to [0,360]
        # ensure zero is properly accounted for, so 0 is not set to 360
        less_than_zero = -1e-8
        mlon[mlon < less_than_zero] += 360

        if nfile == 0:
            fill_value = f.variables['ASTER_GDEM_DEM'].getncattr('_FillValue')
            crs = pyproj.Proj(f.variables['crs'].spatial_ref, preserve_units=True)
            # reorder geotransform to affine convention
            aff = [float(f.variables['crs'].GeoTransform.split()[i]) for i in [1,2,0,4,5,3]]
            affine = rasterio.Affine(*aff)

            x0, y0 = affine.c, affine.f
            # convert longitude to [0,360]
            if x0 < 0:
                x0 += 360

        f.close()

        dmlon = np.abs(mlon[0]-mlon[1])
        dmlat = np.abs(mlat[0]-mlat[1])

        # convert latitude to S->N
        mlat = np.flipud(mlat)
        aster_elev = np.flipud(aster_elev)

        fill_value = -9999
        if zeroFill:
            aster_elev[aster_elev <= fill_value] = 0

        # create grid that will be filled sequentially by dem files
        if nfile==0:
            elon, elat, elev = _create_grid(corners, x0, y0, dmlon, dmlat, "ASTER", tol)

        # locate dem tile within grid

        # use arg_closest_point() to compare in single precision
        i1 = arg_closest_point(elon[0],  mlon, angular=True)
        i2 = arg_closest_point(elon[-1], mlon, angular=True)
        i3 = arg_closest_point(mlon[i1], elon, angular=True)
        i4 = arg_closest_point(mlon[i2], elon, angular=True)

        j1 = arg_closest_point(elat[0],  mlat)
        j2 = arg_closest_point(elat[-1], mlat)
        j3 = arg_closest_point(mlat[j1], elat)
        j4 = arg_closest_point(mlat[j2], elat)

        if np.abs(np.mean(elon[i3:i4+1]-mlon[i1:i2+1])) > 1e-10:
            print(np.mean(elon[i3:i4+1]-mlon[i1:i2+1]))
            print(elon[i3:i4+1][:10])
            raise RuntimeError(mlon[i1:i2+1][:10])

        if np.abs(np.mean(elat[j3:j4+1]-mlat[j1:j2+1])) > 1e-10:
            print(np.mean(elat[j3:j4+1]-mlat[j1:j2+1]))
            print(elat[j3:j4+1][:10])
            raise RuntimeError(mlat[j1:j2+1][:10])

        elev[j3:j4+1,i3:i4+1] = aster_elev[j1:j2+1,i1:i2+1]
                
    # Adjust affine to represent actual elev bounds
    # x0,y0 should be top left pixel of raster
    dx, dy = affine.a, affine.e
    x0, y0 = elon[0]-0.5*np.abs(dx), elat[-1]+0.5*np.abs(dy)
    affine = rasterio.Affine(affine.a,affine.b,x0,affine.d,affine.e,y0)

    # for grids spanning greenwich
    elon[elon >= 360] -= 360
    # to match affine, convert latitude back to N->S
    elat = np.flipud(elat)
    elev = np.flipud(elev)

    return {'elev':elev,'lon':elon,'lat':elat,'crs':crs,'affine':affine,'validDEM':validDEM}
