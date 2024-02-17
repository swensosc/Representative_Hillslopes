import sys 
import string 
import subprocess
import time
import argparse
import numpy as np 
import netCDF4 as netcdf4 

'''
routines for modifying geospatial data

smooth_2d_array:        smooth a 2d array
fit_planar_surface:     fit a planar surface to a 2d array
blend_edges:            blend the edges of a 2d array
arg_closest_point:      return the index of an array that is closest to a given value
calc_gradient:          calculate gradient
std_dev:                standard deviation
quadratic:              return a solution of a quadratic equation
_four_point_laplacian:  calculate laplacian using four neighboring points
_inside_indices_buffer: return indices excluding those in a buffer around array edges
_expand_mask_buffer:     expand a mask spatially

'''    

# Parameters

# degrees to radians
dtr = np.pi/180.
# earth radius [m]
re  = 6.371e6

# function definitions

def smooth_2d_array(elev,land_frac=1,scalar=1):
    hw = scalar/(land_frac**2*np.min(elev.shape))
    elev_fft = np.fft.rfft2(elev,norm='ortho')
    ny,nx = elev_fft.shape
    rowfreq = np.fft.fftfreq(elev.shape[0])
    colfreq = np.fft.rfftfreq(elev.shape[1])
    radialfreq = np.sqrt(np.tile(colfreq*colfreq,(ny,1)) \
                         +np.tile(rowfreq*rowfreq,(nx,1)).T)

    # smaller hw -> more smooth
    wl = np.exp(-radialfreq/hw)
    return np.fft.irfft2(wl*elev_fft,norm='ortho',s=elev.shape)

def fit_planar_surface(elev,elon,elat):
    elon2d = np.tile(elon,(elat.size,1))
    elat2d = np.tile(elat,(elon.size,1)).T
    ncoef=3
    g = np.zeros((elon2d.size,ncoef))
    g[:,0] = elat2d.flat
    g[:,1] = elon2d.flat
    g[:,2] = 1
    gtd = np.dot(np.transpose(g), elev.flat)
    gtg = np.dot(np.transpose(g), g)
    #  covm is the model covariance matrix
    covm = np.linalg.inv(gtg)
    #  coefs is the model parameter vector
    coefs=np.dot(covm, gtd)
    
    elev_planar = elat2d*coefs[0]+elon2d*coefs[1]+coefs[2]
    return elev_planar

def blend_edges(ifld,n=10):
    fld = np.copy(ifld)
    jm,im = fld.shape
    # j axis 
    tmp = np.zeros((jm,2*n))
    # begin at edges and work away
    for i in range(n):
        w = (n-i)
        ind = np.arange(-w,(w+1),1,dtype=int)
        # positive edge
        tmp[:,n+i] = np.sum(fld[:,ind+i],axis=1)/ind.size

        # negative edge
        tmp[:,n-(i+1)] = np.sum(fld[:,ind-(i+1)],axis=1)/ind.size

    # update fld values
    ind = np.arange(-n,n,1,dtype=int)
    fld[:,ind] = tmp

    # i axis 
    tmp = np.zeros((2*n,im))
    # begin at edges and work away
    for j in range(n):
        w = (n-j)
        ind = np.arange(-w,(w+1),1,dtype=int)
        # positive edge
        tmp[n+j,:] = np.sum(fld[ind+j,:],axis=0)/ind.size

        # negative edge
        tmp[n-(j+1),:] = np.sum(fld[ind-(j+1),:],axis=0)/ind.size

    # update fld values
    ind = np.arange(-n,n,1,dtype=int)
    fld[ind,:] = tmp

    return fld

def arg_closest_point(point,array,angular=False):
    # find closest value in an array using 32 bit precision
    if angular:
        # unit are degrees
        d = np.power(np.cos(dtr*np.float32(point))-np.cos(dtr*np.float32(array)),2) \
            +np.power(np.sin(dtr*np.float32(point))-np.sin(dtr*np.float32(array)),2)
        return np.argmin(np.abs(d))
    else:
        return np.argmin(np.abs(np.float32(point) - np.float32(array)))

def calc_gradient(z,lon,lat,method='Horn1981'):
    if method not in ['Horn1981','O1']:
        print('method must be either Horn1981 or O1')
        stop

    if method == 'O1':
        dzdy2,dzdx2 = np.gradient(z)

    if method == 'Horn1981':
        dzdy,dzdx = np.gradient(z)

        dzdy2, dzdx2 = np.zeros(dzdy.shape),np.zeros(dzdx.shape)
        # average [-1,0,0,1] gradient values at each point, in each direction
        # at edges, use 3 points instead of 4

        eind = np.asarray([0,0,1])
        dzdx2[0,:] = np.mean(dzdx[eind,:],axis=0)
        dzdy2[:,0] = np.mean(dzdy[:,eind],axis=1)
        eind = np.asarray([-2,-1,-1])
        dzdx2[-1,:] = np.mean(dzdx[eind,:],axis=0)
        dzdy2[:,-1] = np.mean(dzdy[:,eind],axis=1)
        ind = np.asarray([-1,0,0,1])
        for n in range(1,dzdx.shape[0]-1):
            dzdx2[n,:] = np.mean(dzdx[n+ind,:],axis=0)
        for n in range(1,dzdy.shape[1]-1):
            dzdy2[:,n] = np.mean(dzdy[:,n+ind],axis=1)
    
    # calculate spacing
    dx = re*dtr*np.abs(lon[0] - lon[1])
    dy = re*dtr*np.abs(lat[0] - lat[1])
    
    dx2d = dx*np.tile(np.cos(dtr*lat),(lon.size,1)).T
    dy2d = dy*np.ones((lat.size,lon.size))
  
    return [dzdx2/dx2d,dzdy2/dy2d]

def std_dev(x):
    return np.power(np.mean(np.power((x-np.mean(x)),2)),0.5)

def quadratic(coefs,root=0,verbose=False):
    ak, bk, ck = coefs
    if (bk**2-4*ak*ck) < 0:
        print('cannot solve quadratic with these values \
        {:.2f}  {:.2f}  {:.2f}'.format(ak,bk,ck))
        stop
    
    dm_roots = [(-bk + np.sqrt(bk**2-4*ak*ck))/(2*ak), \
                (-bk - np.sqrt(bk**2-4*ak*ck))/(2*ak)]
    if verbose:
        print('quadratic roots ',dm_roots)
    return dm_roots[root]

def _four_point_laplacian(mask):
    # mask is assumed to be 0-1 (used to multiply results)
    jm=mask.shape[0]
    im=mask.shape[1]

    laplacian = -4.0 * np.copy(mask)
    laplacian += (mask * np.roll(mask,1,axis=1) 
                  + mask * np.roll(mask,-1,axis=1))
    temp = np.roll(mask,1,axis=0)
    temp[0,:] = mask[1,:]
    laplacian += mask * temp
    temp = np.roll(mask,-1,axis=0)
    temp[jm-1,:] = mask[jm-2,:]
    laplacian += mask * temp

    return np.abs(laplacian)

def _inside_indices_buffer(data, buf=1, mask=None):
    # return indices of non-edge points (outside of buf)
    if mask is None:
        mask = np.array([]).astype(int)
    a = np.arange(data.size)
    offset = int(data.shape[1])

    top = []
    for i in range(buf):
        top.extend((i*offset + np.arange(offset)[buf:-buf]).tolist())
    top = np.array(top,dtype=int)
    
    bottom = data.size - 1 - top
    
    left = []
    for i in range(buf):
        left.extend(np.arange(i,data.size,offset))
    left = np.array(left,dtype=int)

    right = data.size - 1 - left
    
    exclude = np.unique(np.concatenate([top, left, right, bottom, mask]))
    inside = np.delete(a, exclude)

    return inside

def _expand_mask_buffer(mask,buf=1):
    omask = np.copy(mask)
    # this will use less memory by not accumulating all indices
    # prior to assigning mask points to 1
    inside = _inside_indices_buffer(mask,buf=buf)
    lmask = np.where(_four_point_laplacian(mask) > 0,1,0)
    ind = inside[(lmask.flat[inside] > 0)]
    offset = mask.shape[1]
    for k in range(-buf,buf+1):
        if k != 0:
            omask.flat[ind+k] = 1
        for j in range(buf):
            j1 = j+1
            # upper
            omask.flat[ind+k+j1*offset] = 1
            # lower
            omask.flat[ind+k-j1*offset] = 1
    
    return omask

def identify_basins(dem,basin_thresh=0.25,niter=30,buf=10):
    # create basin mask, 1 in basin, 0 outside of basin
    # flat areas often have large dtnd and small hand values
    # due to flowpaths in flooded/inflated part of dem
    imask = np.zeros(dem.shape)

    # find most common elevation value
    udem,ucnt = np.unique(dem,return_counts=True)
    umax = np.argmax(ucnt)
    dem_max_value = udem[umax]
    dem_max_fraction = ucnt[umax]/dem.size

    # use eps to catch roundoff values
    eps = 1e-2
    # if elevation is zero, assume open water and tighten tolerance
    if np.abs(dem_max_value) < eps:
        eps = 1e-6

    # flag areas with common elevation as a basin or open water
    if dem_max_fraction >= basin_thresh:
        imask[np.abs(dem-dem_max_value) < eps] = 1

    for n in range(niter):
        imask = _expand_mask_buffer(imask,buf=buf)

        # remove points each iteration
        imask[np.abs(dem-dem_max_value) >= eps] = 0

        # remove isolated points
        imask[_four_point_laplacian(imask) >= 3] = 0

    return imask
    
