import subprocess
import numpy as np
import netCDF4 as netcdf4
import pyproj
import rasterio
from osgeo import gdal as gd

from geospatial_utils import arg_closest_point, north_or_south, east_or_west
from rh_logging import info, warning, error, debug

"""
routines related to reading DEM data

utilities
_is_contained
_locate_point
_locate_point
_locate_edges
_locate_boundary
_north_or_south
_east_or_west

return filenames required to span region
_get_MERIT_dem_filenames
_get_ASTER_dem_filenames
_get_FAB_dem_filenames
_get_TNM_dem_filenames

create_subregion_corner_lists: create four corner lists by subdividing input corner list
read in DEM data for region
read_MERIT_dem_data
read_ASTER_dem_data
read_FAB_dem_data
read_TNM_dem_data

"""

less_than_zero = -1e-8        

def _is_contained(pt,bounds):
    # pt is [lon,lat]
    # bounds is [wlon,elon,slat,nlat]
    # check for bounds containing greenwich
    if np.logical_and(bounds[1]>360,pt[0]<360):
        l1 = np.logical_and(pt[0]>=(bounds[0]-360),pt[0]<=(bounds[1]-360))
    else:
        l1 = np.logical_and(pt[0]>=bounds[0],pt[0]<=bounds[1])
    l2 = np.logical_and(pt[1]>=bounds[2],pt[1]<=bounds[3])
    return (l1 and l2)

def _locate_point(pt,blon,blat):
    bounds = [np.min(blon),np.max(blon),np.min(blat),np.max(blat)]
    if _is_contained(pt,bounds):
        i = arg_closest_point(pt[0],blon,angular=True)
        j = arg_closest_point(pt[1],blat)
        return [0,j,i]
    else:
        return [-1,-1,-1]

def _locate_edges(pt,lon1,lat1,lon2,lat2):
    bounds1 = [np.min(lon1),np.max(lon1),np.min(lat1),np.max(lat1)]
    bounds2 = [np.min(lon2),np.max(lon2),np.min(lat2),np.max(lat2)]

    epts = []
    l1 = np.logical_and(bounds2[0]>=np.min(lon1),bounds2[0]<=np.max(lon1))
    if l1:
        elon = lon1[np.argmin(np.abs(bounds2[0]-lon1))]
        epts.append([elon,pt[1]])
    l1 = np.logical_and(bounds2[1]>=np.min(lon1),bounds2[1]<=np.max(lon1))
    if l1:
        elon = lon1[np.argmin(np.abs(bounds2[1]-lon1))]
        epts.append([elon,pt[1]])
    l1 = np.logical_and(bounds2[2]>=np.min(lat1),bounds2[2]<=np.max(lat1))
    if l1:
        elat = lat1[np.argmin(np.abs(bounds2[2]-lat1))]
        epts.append([pt[0],elat])
    l1 = np.logical_and(bounds2[3]>=np.min(lat1),bounds2[3]<=np.max(lat1))
    if l1:
        elat = lat1[np.argmin(np.abs(bounds2[3]-lat1))]
        epts.append([pt[0],elat])
    return epts

def _locate_overlap(lon1,lat1,lon2,lat2):
    if np.any([len(lon1.shape) > 1,
               len(lat1.shape) > 1,
               len(lon2.shape) > 1,
               len(lat2.shape) > 1]):
        raise RuntimeError('input arrays must be 1d')

    # assumes orthogonal / rectangular sides
    bounds1 = [np.min(lon1),np.max(lon1),np.min(lat1),np.max(lat1)]
    bounds2 = [np.min(lon2),np.max(lon2),np.min(lat2),np.max(lat2)]

    pts1 = [[bounds1[0],bounds1[2]],
            [bounds1[1],bounds1[2]],
            [bounds1[0],bounds1[3]],
            [bounds1[1],bounds1[3]]]
    pts2 = [[bounds2[0],bounds2[2]],
            [bounds2[1],bounds2[2]],
            [bounds2[0],bounds2[3]],
            [bounds2[1],bounds2[3]]]

    overlap_pts = []
    for pt in pts1:
        ei = _locate_point(pt,lon2,lat2)
        if ei[0]==0:
            j,i = ei[1:]
            opt = [lon2[i],lat2[j]]
            overlap_pts.append(opt)
            overlap_pts.extend(_locate_edges(opt,lon1,lat1,lon2,lat2))

    for pt in pts2:
        ei = _locate_point(pt,lon1,lat1)
        if ei[0]==0:
            j,i = ei[1:]
            opt2 = [lon1[i],lat1[j]]
            overlap_pts.append(opt2)
            overlap_pts.extend(_locate_edges(opt2,lon2,lat2,lon1,lat1))

    overlap_pts = np.unique(np.asarray(overlap_pts),axis=0)
    return overlap_pts.tolist()

def _north_or_south(lat):
    if lat >= 0:
        return "n"
    else:
        return "s"


def _east_or_west(lon):
    if lon >= 0:
        return "e"
    else:
        return "w"

def create_subregion_corner_lists(corners,central_point,ensurePositive=True):
    clon,clat = central_point
    # split into 4 subregions, copy deepest list
    corner_list = []
    # ll
    dc = [corners[i].copy() for i in range(4)]
    dc[1][1] = clat
    dc[2][0] = clon
    dc[3] = [clon, clat]
    corner_list.append(dc)
    # ul
    dc = [corners[i].copy() for i in range(4)]
    dc[0][1] = clat
    dc[2] = [clon, clat]
    dc[3][0] = clon
    corner_list.append(dc)
    # lr
    dc = [corners[i].copy() for i in range(4)]
    dc[0][0] = clon
    dc[1] = [clon, clat]
    dc[3][1] = clat
    corner_list.append([[pt[0], pt[1]] for pt in dc])
    # ur
    dc = [corners[i].copy() for i in range(4)]
    dc[0] = [clon, clat]
    dc[1][0] = clon
    dc[2][1] = clat
    corner_list.append([pt for pt in dc])

    if ensurePositive:
        for corners in corner_list:
            for n in range(len(corners)):
                if corners[n][0] < 0:
                    corners[n][0] += 360

    return corner_list


def _check_files_exist(dem_file_template, efiles, verbose=False):
    emask = np.ones(efiles.size, dtype=bool)
    for n in range(efiles.size):
        geofile = efiles[n]
        command = ["ls", geofile]
        file_exists = subprocess.run(command, capture_output=True).returncode
        if file_exists > 0:
            emask[n] = False
    if not np.any(emask):
        if verbose:
            error("All DEM files missing:")
            for file in efiles:
                error(f"   {file}")
            msg = f"No DEM files found matching template: {dem_file_template}"
            error(msg)
            #raise FileNotFoundError(msg)
        return np.asarray([])
    efiles = efiles[emask]
    return efiles

def _create_grid(corners, x0, y0, dmlon, dmlat, which_dem, tol):
    """
    identify closest points to corners on DEM grid,
    ensuring that the region they define is larger than
    the region defined by corners
    """

    # left side
    n0 = np.round((corners[0][0] - x0) / dmlon, tol)
    ex0 = x0 + np.floor(n0) * dmlon

    # ex0 should be < left edge, and within dmlon
    delta_lon = corners[0][0] - ex0
    if delta_lon > 360:
        delta_lon -= 360
    if np.round(delta_lon / dmlon, tol) > 1 or np.round(delta_lon / dmlon, tol) < 0:
        raise RuntimeError("ex0 ", ex0, corners[0][0], (corners[0][0] - ex0) / dmlon)

    # right side (subtract 1 pixel width from right edge)
    delta_lon = (corners[2][0] - dmlon) - ex0
    # for gridcells spanning greenwich
    if delta_lon < 0:
        delta_lon += 360

    nx = np.ceil(delta_lon / dmlon).astype(int)

    # update delta_lon for error check
    delta_lon = (ex0 + nx * dmlon) - (corners[2][0] - dmlon)
    if delta_lon > 360:
        delta_lon -= 360
    if np.round(delta_lon / dmlon, tol) > 1 or np.round(delta_lon / dmlon, tol) < 0:
        raise RuntimeError(ex0 + nx * dmlon, corners[2][0])

    elon = ex0 + (np.arange(nx)+0.5)*dmlon
    if which_dem in ["ASTER","FAB"]:
        elon[elon >= 360] -= 360
    elif which_dem not in ["MERIT","TNM"]:
        raise RuntimeError(f"Unrecognized DEM: {which_dem}")

    # bottom
    m0 = np.round((corners[0][1] - y0) / dmlat, tol)
    ey0 = y0 + np.floor(m0) * dmlat

    # ey0 should be < lower edge, and within dmlat
    delta_lat = (corners[0][1] - ey0) / dmlat
    if np.round(delta_lat, tol) > 1 or np.round(delta_lat, tol) < 0:
        raise RuntimeError("ey0 ", ey0, corners[0][1], (corners[0][1] - ey0) / dmlat)

    # top (subtract 1 pixel width from upper edge)
    delta_lat = (corners[1][1] - dmlat) - ey0
    ny = np.ceil(delta_lat / dmlat).astype(int)

    # update delta_lon for error check
    delta_lat = ((ey0 + ny * dmlat) - (corners[1][1] - dmlat)) / dmlat
    if np.round(delta_lat, tol) > 1 or np.round(delta_lat, tol) < 0:
        raise RuntimeError(ey0 + ny * dmlat, corners[1][1])

    elat = ey0 + (np.arange(ny) + 0.5) * dmlat

    # initialize output array
    elev = np.zeros((ny, nx))
    return elon, elat, elev


def _get_MERIT_dem_filenames(dem_file_template, corners):
    # dem_file_template is assumed to have form of:
    # 'my_path/elv_DirTag/TileTag_elv.tif'

    # tiles are 5 x 5 degree, directories contain 30 degree band
    mres = 5
    dres = 30
    sigfigs = 6

    # round to correct numbers that are just slightly less than integer
    ll_corner = [np.round(corners[0][0], sigfigs), np.round(corners[0][1], sigfigs)]
    ur_corner = [np.round(corners[-1][0], sigfigs), np.round(corners[-1][1], sigfigs)]

    lonmin, lonmax = int((ll_corner[0] // mres) * mres), int(
        (ur_corner[0] // mres) * mres
    )
    latmin, latmax = int((ll_corner[1] // mres) * mres), int(
        (ur_corner[1] // mres) * mres
    )

    # if right boundary is multiple of tile resolution, exclude it
    if (ur_corner[0] - lonmax) == 0.0:
        lnpad = 0
    else:
        lnpad = 1

    # if upper boundary is multiple of tile resolution, exclude it
    if (ur_corner[1] - latmax) == 0.0:
        ltpad = 0
    else:
        ltpad = 1

    # ensure lonmax > lonmin for regions spanning prime meridian
    if lonmax < lonmin:
        lonmax += 360

    nlon = lonmin + np.arange((lonmax - lonmin) // mres + lnpad) * mres
    nlat = latmin + np.arange((latmax - latmin) // mres + ltpad) * mres

    efiles = []
    for lonc in nlon:
        for latc in nlat:
            tlon = int((lonc // mres) * mres)
            if tlon >= 180:
                tlon -= 360
            tlat = int((latc // mres) * mres)

            abstlon = abs(tlon)
            lonstr  = '{:03d}'.format(abstlon)
            lonstr = east_or_west(tlon)+lonstr

            abstlat = abs(tlat)
            latstr  = '{:02d}'.format(abstlat)
            latstr = north_or_south(tlat)+latstr

            tiletag = latstr + lonstr

            dir_tlon = (tlon // dres) * dres
            dir_tlat = (tlat // dres) * dres

            abstlon = abs(dir_tlon)
            lonstr = "{:03d}".format(abstlon)

            abstlat = abs(dir_tlat)
            latstr = "{:02d}".format(abstlat)

            dirtag = north_or_south(tlat)+latstr \
                     +east_or_west(tlon)+lonstr

            efile = dem_file_template.replace("DirTag", dirtag)
            efiles.append(efile.replace("TileTag", tiletag))

    # get unique values
    efiles = np.unique(np.asarray(efiles))
    numfiles = efiles.size

    # check that all files exist (call returns 0)
    # (corners may extend beyond existing dem tiles)
    efiles = _check_files_exist(dem_file_template, efiles)

    return efiles


def read_MERIT_dem_data(dem_file_template, corners, tol=10, zeroFill=False):

    # Determine dem filenames
    # MERIT filenames indicate lower left corner of tile
    demfiles = _get_MERIT_dem_filenames(dem_file_template, corners)

    if demfiles.size > 0:
        validDEM = True
    else:
        validDEM = False
        return {"validDEM": validDEM}

    for nfile in range(demfiles.size):
        meritfile = demfiles[nfile]
        ds = gd.Open(meritfile)
        if nfile == 0:
            crs = pyproj.Proj(ds.GetProjection(), preserve_units=True)
            # reorder geotransform to affine convention
            aff = [float(ds.GetGeoTransform()[i]) for i in [1, 2, 0, 4, 5, 3]]
            affine = rasterio.Affine(*aff)

        # merit latitude is N->S
        merit_elev = ds.ReadAsArray()
        xs = ds.RasterXSize
        ys = ds.RasterYSize
        x = ds.GetGeoTransform()
        x0, y0, dx, dy = x[0], x[3], x[1], x[5]
        # convert longitude to [0,360]
        #if x0 < 0:
        if (x0 + 0.5 * dx) < less_than_zero: # check center of pixel
            x0 += 360
        # coordinates of center of pixel
        mlon = (x0 + 0.5 * dx) + dx * np.arange(xs)
        mlat = (y0 + 0.5 * dy) + dy * np.arange(ys)

        dmlon = np.abs(mlon[0] - mlon[1])
        dmlat = np.abs(mlat[0] - mlat[1])

        # ensure zero is properly accounted for, so 0 is not set to 360
        mlon[mlon < less_than_zero] += 360

        # convert latitude to S->N
        mlat = np.flipud(mlat)
        merit_elev = np.flipud(merit_elev)

        fill_value = -9999
        if zeroFill:
            merit_elev[merit_elev <= fill_value] = 0

        # create grid that will be filled sequentially by dem files
        if nfile == 0:
            elon, elat, elev = _create_grid(corners, x0, y0, dmlon, dmlat, "MERIT", tol)

        # locate dem tile within grid
        if 1==2:
            print('elon ',elon[0],elon[-1])
            print('mlon ',mlon[0],mlon[-1])
            print('elat ',elat[0],elat[-1])
            print('mlat ',mlat[0],mlat[-1])
            print(corners)
            print(demfiles)
        
        opts = np.asarray(_locate_overlap(elon,elat,mlon,mlat))
        obounds = [np.min(opts[:,0]),np.max(opts[:,0]),np.min(opts[:,1]),np.max(opts[:,1])]

        i1_dst,i2_dst = arg_closest_point(obounds[0],elon, angular=True),arg_closest_point(obounds[1],elon, angular=True)
        j1_dst,j2_dst = arg_closest_point(obounds[2],elat),arg_closest_point(obounds[3],elat)
        i1_src,i2_src = arg_closest_point(obounds[0],mlon, angular=True),arg_closest_point(obounds[1],mlon, angular=True)
        j1_src,j2_src = arg_closest_point(obounds[2],mlat),arg_closest_point(obounds[3],mlat)

        l1 = ((i2_src-i1_src)!=(i2_dst-i1_dst))
        l2 = ((j2_src-j1_src)!=(j2_dst-j1_dst))
        if (l1 or l2):
            msg = str('check overlap')
            error(msg)
            error(i1_dst,i2_dst,rlon.size,j1_dst,j2_dst,rlat.size)
            error(i2_src-i1_src,i2_dst-i1_dst)
            error(j2_src-j1_src,j2_dst-j1_dst)
            raise RuntimeError(msg)

        elev[j1_dst:j2_dst+1,i1_dst:i2_dst+1] = merit_elev[j1_src:j2_src+1,i1_src:i2_src+1]

    # Adjust affine to represent actual elev bounds
    # x0,y0 should be top left pixel of raster
    dx, dy = affine.a, affine.e
    x0, y0 = elon[0] - 0.5 * np.abs(dx), elat[-1] + 0.5 * np.abs(dy)
    affine = rasterio.Affine(affine.a, affine.b, x0, affine.d, affine.e, y0)

    # for grids spanning greenwich
    elon[elon >= 360] -= 360
    # to match affine, convert latitude back to N->S
    elat = np.flipud(elat)
    elev = np.flipud(elev)

    return {
        "elev": elev,
        "lon": elon,
        "lat": elat,
        "crs": crs,
        "affine": affine,
        "validDEM": validDEM,
    }

def _get_ASTER_dem_filenames(dem_file_template, corners):
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
    if (ur_corner[0] - lonmax) == 0.0:
        lnpad = 0
    else:
        lnpad = 1
    # if upper boundary is multiple of tile resolution, exclude it
    if (ur_corner[1] - latmax) == 0.0:
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
            tlat = int((latc // ares) * ares)

            abstlon = abs(tlon)
            lonstr  = '{:03d}'.format(abstlon)
            lonstr = east_or_west(tlon)+lonstr

            abstlat = abs(tlat)
            latstr  = '{:02d}'.format(abstlat)
            latstr = north_or_south(tlat)+latstr

            tiletag = latstr + lonstr

            efiles.append(dem_file_template.replace("TileTag", tiletag.upper()))

    # get unique values
    efiles = np.unique(np.asarray(efiles))
    numfiles = efiles.size

    # check that all files exist (call returns 0)
    # (corners may extend beyond existing dem tiles)
    efiles = _check_files_exist(dem_file_template, efiles)

    return efiles


def read_ASTER_dem_data(dem_file_template, corners, tol=10, zeroFill=False):
    # Determine dem filenames
    demfiles = _get_ASTER_dem_filenames(dem_file_template, corners)

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
        asterfile = demfiles[nfile]
        f = netcdf4.Dataset(asterfile, "r")
        # coordinates
        mlon = f.variables["lon"][
            :,
        ]
        mlat = f.variables["lat"][
            :,
        ]
        im = mlon.size
        jm = mlat.size
        aster_elev = f.variables["ASTER_GDEM_DEM"][
            :,
        ].astype(float)
        ys, xs = aster_elev.shape
        # convert longitude to [0,360]
        # ensure zero is properly accounted for, so 0 is not set to 360
        mlon[mlon < less_than_zero] += 360

        if nfile == 0:
            fill_value = f.variables["ASTER_GDEM_DEM"].getncattr("_FillValue")
            crs = pyproj.Proj(f.variables["crs"].spatial_ref, preserve_units=True)
            # reorder geotransform to affine convention
            aff = [
                float(f.variables["crs"].GeoTransform.split()[i])
                for i in [1, 2, 0, 4, 5, 3]
            ]
            affine = rasterio.Affine(*aff)

            x0, y0 = affine.c, affine.f
            # convert longitude to [0,360]
            if x0 < 0:
                x0 += 360

        f.close()

        dmlon = np.abs(mlon[0] - mlon[1])
        dmlat = np.abs(mlat[0] - mlat[1])

        # convert latitude to S->N
        mlat = np.flipud(mlat)
        aster_elev = np.flipud(aster_elev)

        fill_value = -9999
        if zeroFill:
            aster_elev[aster_elev <= fill_value] = 0

        # create grid that will be filled sequentially by dem files
        if nfile == 0:
            elon, elat, elev = _create_grid(corners, x0, y0, dmlon, dmlat, "ASTER", tol)

        # locate dem tile within grid
        opts = np.asarray(_locate_overlap(elon,elat,mlon,mlat))
        obounds = [np.min(opts[:,0]),np.max(opts[:,0]),np.min(opts[:,1]),np.max(opts[:,1])]

        i1_dst,i2_dst = arg_closest_point(obounds[0],elon, angular=True),arg_closest_point(obounds[1],elon, angular=True)
        j1_dst,j2_dst = arg_closest_point(obounds[2],elat),arg_closest_point(obounds[3],elat)
        i1_src,i2_src = arg_closest_point(obounds[0],mlon, angular=True),arg_closest_point(obounds[1],mlon, angular=True)
        j1_src,j2_src = arg_closest_point(obounds[2],mlat),arg_closest_point(obounds[3],mlat)

        l1 = ((i2_src-i1_src)!=(i2_dst-i1_dst))
        l2 = ((j2_src-j1_src)!=(j2_dst-j1_dst))
        if (l1 or l2):
            msg = str('check overlap')
            error(msg)
            error(i1_dst,i2_dst,rlon.size,j1_dst,j2_dst,rlat.size)
            error(i2_src-i1_src,i2_dst-i1_dst)
            error(j2_src-j1_src,j2_dst-j1_dst)
            raise RuntimeError(msg)

        elev[j1_dst:j2_dst+1,i1_dst:i2_dst+1] = aster_elev[j1_src:j2_src+1,i1_src:i2_src+1]

    # Adjust affine to represent actual elev bounds
    # x0,y0 should be top left pixel of raster
    dx, dy = affine.a, affine.e
    x0, y0 = elon[0] - 0.5 * np.abs(dx), elat[-1] + 0.5 * np.abs(dy)
    affine = rasterio.Affine(affine.a, affine.b, x0, affine.d, affine.e, y0)

    # for grids spanning greenwich
    elon[elon >= 360] -= 360
    # to match affine, convert latitude back to N->S
    elat = np.flipud(elat)
    elev = np.flipud(elev)

    return {
        "elev": elev,
        "lon": elon,
        "lat": elat,
        "crs": crs,
        "affine": affine,
        "validDEM": validDEM,
    }

def _get_FAB_dem_filenames(dem_file_template,corners):
    # dem_file_template is assumed to have form of:
        # 'my_path/data/TileTag_FABDEM_V1-2.tif'

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

            lonstr = f'{east_or_west(tlon)}{abs(tlon):03d}'
            latstr = f'{north_or_south(tlat)}{abs(tlat):02d}'
            tiletag = f'{latstr}{lonstr}'

            efiles.append(dem_file_template.replace('TileTag',tiletag.upper()))

    # get unique values
    efiles = np.unique(np.asarray(efiles))
    numfiles = efiles.size

    # check that all files exist (call returns 0)
    # (corners may extend beyond existing dem tiles)
    #efiles = _check_files_exist(dem_file_template, efiles)
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

def read_FAB_dem_data(dem_file_template,corners,tol=10,zeroFill=False):
    # Determine dem filenames
    demfiles = _get_FAB_dem_filenames(dem_file_template,corners)

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

        demfile  = demfiles[nfile]
        ds = gd.Open(demfile)
        if nfile==0:
            crs = pyproj.Proj(ds.GetProjection(), preserve_units=True)
            # reorder geotransform to affine convention
            aff = [float(ds.GetGeoTransform()[i]) for i in [1,2,0,4,5,3]]
            affine = rasterio.Affine(*aff)

        # latitude is N->S
        fab_dem_elev = ds.ReadAsArray()

        # check for nans, set to zero
        fab_dem_elev[np.isnan(fab_dem_elev)] = 0
        
        xs = ds.RasterXSize
        ys = ds.RasterYSize
        x  = ds.GetGeoTransform()
        x0, y0, dx, dy = x[0], x[3], x[1], x[5]
        # convert longitude to [0,360]
        #if x0 < 0:
        if (x0 + 0.5 * dx) < less_than_zero: # check center of pixel
            x0 += 360
        # coordinates of center of pixel
        mlon = (x0+0.5*dx) + dx*np.arange(xs)
        mlat = (y0+0.5*dy) + dy*np.arange(ys)

        dmlon = np.abs(mlon[0]-mlon[1])
        dmlat = np.abs(mlat[0]-mlat[1])

        # convert latitude to S->N
        mlat = np.flipud(mlat)
        fab_dem_elev = np.flipud(fab_dem_elev)

        fill_value = -9999
        if zeroFill:
            fab_dem_elev[fab_dem_elev <= fill_value] = 0

        # create grid that will be filled sequentially by dem files
        if nfile==0:
            elon, elat, elev = _create_grid(corners, x0, y0, dmlon, dmlat, "FAB", tol)

        # locate dem tile within grid
        opts = np.asarray(_locate_overlap(elon,elat,mlon,mlat))
        obounds = [np.min(opts[:,0]),np.max(opts[:,0]),np.min(opts[:,1]),np.max(opts[:,1])]

        i1_dst,i2_dst = arg_closest_point(obounds[0],elon, angular=True),arg_closest_point(obounds[1],elon, angular=True)
        j1_dst,j2_dst = arg_closest_point(obounds[2],elat),arg_closest_point(obounds[3],elat)
        i1_src,i2_src = arg_closest_point(obounds[0],mlon, angular=True),arg_closest_point(obounds[1],mlon, angular=True)
        j1_src,j2_src = arg_closest_point(obounds[2],mlat),arg_closest_point(obounds[3],mlat)

        l1 = ((i2_src-i1_src)!=(i2_dst-i1_dst))
        l2 = ((j2_src-j1_src)!=(j2_dst-j1_dst))
        if (l1 or l2):
            msg = str('check overlap')
            error(msg)
            error(i1_dst,i2_dst,rlon.size,j1_dst,j2_dst,rlat.size)
            error(i2_src-i1_src,i2_dst-i1_dst)
            error(j2_src-j1_src,j2_dst-j1_dst)
            raise RuntimeError(msg)

        elev[j1_dst:j2_dst+1,i1_dst:i2_dst+1] = fab_dem_elev[j1_src:j2_src+1,i1_src:i2_src+1]

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

def _get_TNM_dem_filenames(dem_file_template,corners):
    # dem_file_template is assumed to have form of:
    # 'my_path/USGS_13_n??w??_20210623.tif'
    # **note that TNM filename is wrt upper left corner**
    
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
    
    # shift lat bounds due to file naming convention
    latmin += ares
    latmax += ares
    
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

            lonstr = f'{_east_or_west(tlon)}{abs(tlon):03d}'
            latstr = f'{_north_or_south(tlat)}{abs(tlat):02d}'
            tiletag = f'{latstr}{lonstr}'

            efiles.append(dem_file_template.replace('TileTag',tiletag))

    # get unique values
    efiles = np.unique(np.asarray(efiles))
    numfiles = efiles.size

    # files are labeled with individual dates; get actual file names
    afiles = []
    for n in range(efiles.size):
        efile = efiles[n]
        command = f'ls {efile}'
        x = subprocess.run(command,capture_output=True,shell=True)
        file_exists = x.returncode
        if file_exists == 0:
            afiles.append(x.stdout.strip().decode())

    efiles = np.asarray(afiles)

    # check that all files exist (call returns 0)
    # (corners may extend beyond existing dem tiles)
    #efiles = check_files_exist(dem_file_template, efiles)
    emask = np.ones(efiles.size, dtype=bool)
    for n in range(efiles.size):
        geofile = efiles[n]
        command=['ls',geofile]
        file_exists=subprocess.run(command,capture_output=True).returncode
        if file_exists > 0:
            emask[n] = False
    efiles = efiles[emask]
    
    return efiles

def read_TNM_dem_data(dem_file_template,corners,tol=10,zeroFill=False):
    # Determine dem filenames
    demfiles = _get_TNM_dem_filenames(dem_file_template,corners)

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

        demfile  = demfiles[nfile]
        ds = gd.Open(demfile)
        if nfile==0:
            crs = pyproj.Proj(ds.GetProjection(), preserve_units=True)
            # reorder geotransform to affine convention
            aff = [float(ds.GetGeoTransform()[i]) for i in [1,2,0,4,5,3]]
            affine = rasterio.Affine(*aff)
            
        # latitude is N->S
        tnm_dem_elev = ds.ReadAsArray()

        # check for nans, set to zero
        tnm_dem_elev[np.isnan(tnm_dem_elev)] = 0

        xs = ds.RasterXSize
        ys = ds.RasterYSize
        x  = ds.GetGeoTransform()
        x0, y0, dx, dy = x[0], x[3], x[1], x[5]
        # convert longitude to [0,360]
        #if x0 < 0:
        if (x0 + 0.5 * dx) < less_than_zero: # check center of pixel
            x0 += 360
        # coordinates of center of pixel
        mlon = (x0+0.5*dx) + dx*np.arange(xs)
        mlat = (y0+0.5*dy) + dy*np.arange(ys)

        dmlon = np.abs(mlon[0]-mlon[1])
        dmlat = np.abs(mlat[0]-mlat[1])

        # convert latitude to S->N
        mlat = np.flipud(mlat)
        tnm_dem_elev = np.flipud(tnm_dem_elev)
        
        fill_value = -9999
        if zeroFill:
            tnm_dem_elev[tnm_dem_elev <= fill_value] = 0

        # create grid that will be filled sequentially by dem files
        if nfile==0:
            elon, elat, elev = _create_grid(corners, x0, y0, dmlon, dmlat, "TNM", tol)
            
        # locate dem tile within grid
        opts = np.asarray(_locate_overlap(elon,elat,mlon,mlat))
        obounds = [np.min(opts[:,0]),np.max(opts[:,0]),np.min(opts[:,1]),np.max(opts[:,1])]

        i1_dst,i2_dst = arg_closest_point(obounds[0],elon, angular=True),arg_closest_point(obounds[1],elon, angular=True)
        j1_dst,j2_dst = arg_closest_point(obounds[2],elat),arg_closest_point(obounds[3],elat)
        i1_src,i2_src = arg_closest_point(obounds[0],mlon, angular=True),arg_closest_point(obounds[1],mlon, angular=True)
        j1_src,j2_src = arg_closest_point(obounds[2],mlat),arg_closest_point(obounds[3],mlat)

        l1 = ((i2_src-i1_src)!=(i2_dst-i1_dst))
        l2 = ((j2_src-j1_src)!=(j2_dst-j1_dst))
        if (l1 or l2):
            msg = str('check overlap')
            error(msg)
            error(i1_dst,i2_dst,rlon.size,j1_dst,j2_dst,rlat.size)
            error(i2_src-i1_src,i2_dst-i1_dst)
            error(j2_src-j1_src,j2_dst-j1_dst)
            raise RuntimeError(msg)

        elev[j1_dst:j2_dst+1,i1_dst:i2_dst+1] = tnm_dem_elev[j1_src:j2_src+1,i1_src:i2_src+1]
                
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


