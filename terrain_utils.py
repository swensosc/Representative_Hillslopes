import numpy as np
from scipy.stats import expon
from multiprocessing import Pool
from functools import partial

'''
calc_network_length:                   calculate length of stream network
set_aspect_to_hillslope_mean_parallel: assign mean value to all points in a catchment
set_aspect_to_hillslope_mean_serial:   assign mean value to all points in a catchment
TailIndex:                             return indices of tail of a distribution
SpecifyHandBounds:                     define bounds used to bin HAND distribution using aspect information
SpecifyHandBoundsNoAspect:             define bounds used to bin HAND distribution    
_calculate_hillslope_mean_aspect:       calculate the vector-mean aspect value
_four_point_laplacian:                  calculate the laplacian of a binary mask
_inside_indices_buffer:                 return indices of a 2d array excluding those within buffer of edges

'''    

# Parameters

# degrees to radians
dtr = np.pi/180.
# earth radius [m]
re  = 6.371e6

# function definitions

def calc_network_length(channel_ids,channel_coords,dem,fdir,lon,lat,\
                        latdir='south_to_north',verbose=False):
    dirmap=(64, 128, 1, 2, 4, 8, 16, 32)
    dir_to_index_dict  = {dirmap[n]:n for n in range(len(dirmap))}

    fjm,fim = dem.shape
    dmask = np.zeros((fjm,fim))

    # W -> E
    addi = [0,1,1,1,0,-1,-1,-1]
    # determine direction of latitude coordinates
    if latdir == 'south_to_north':
        # S -> N
        addj = [1,1,0,-1,-1,-1,0,1]
    if latdir == 'north_to_south':
        # N -> S
        addj = [-1,-1,0,1,1,1,0,-1]
        
    unique_ids = np.unique(channel_ids).astype(int)
    reach_length = []
    reach_elevation_difference = []
    for n in range(unique_ids.size):
        ind = np.where(unique_ids[n] == channel_ids)[0]
        # record network elevation and indices
        reach_elev = []
        reach_j = []
        reach_i = []
        for k in range(ind.size):
            ni = np.argmin(np.abs(channel_coords[ind[k],0] - lon))
            nj = np.argmin(np.abs(channel_coords[ind[k],1] - lat))
            reach_elev.append(dem[nj,ni])
            reach_j.append(nj)
            reach_i.append(ni)

        # locate highest elevation point in reach
        k1 = np.argmax(np.asarray(reach_elev))
        j0,i0 = reach_j[k1],reach_i[k1]

        # record index of each point in dmask and upstream neighbor
        if fdir[j0,i0] < 1:
            continue
        else:
            dmask_undx = [-1]
            dmask_ndx = [[j0,i0]]

        # follow gridded directions downstream
        cnt = 0
        while fdir[j0,i0] > 0 and cnt < fim*fjm:
            if dmask[j0,i0] >= 1:
                break

            dmask_undx.append([j0,i0])
            dmask[j0,i0] += 1
            ddir = fdir[j0,i0]
            m = dir_to_index_dict[ddir]
            i0 += addi[m]
            j0 += addj[m]

            check_i = np.logical_or(i0 < 0,i0 >= fim)
            check_j = np.logical_or(j0 < 0,j0 >= fjm)
            if check_i or check_j or (fdir[j0,i0] < 1):
                del dmask_undx[-1]
                break
            
            dmask_ndx.append([j0,i0])
            cnt += 1
            
        # sum elements of reach
        length = 0
        elevation_difference = 0
        for k in range(1,len(dmask_ndx)):
            j1,i1 = dmask_ndx[k]
            j2,i2 = dmask_undx[k]
            dlon = lon[i1] - lon[i2]
            dlat = lat[j1] - lat[j2]

            dist = np.power(np.sin(dtr*dlat/2),2) + np.cos(dtr*lat[j1]) \
                   * np.cos(dtr*lat[j2]) \
                   * np.power(np.sin(dtr*dlon/2),2)
            length += (re * 2 * np.arctan2(np.sqrt(dist),np.sqrt(1-dist)))

            if k==1:
                elevation_difference += dem[j2,i2]
            if k==(len(dmask_ndx)-1):
                elevation_difference -= dem[j1,i1]
            
        reach_length.append(length)
        reach_elevation_difference.append(elevation_difference)

    reach_length = np.asarray(reach_length)
    reach_elevation_difference = np.asarray(reach_elevation_difference)
    total_reach_length = np.sum(reach_length)
    reach_slopes = reach_elevation_difference[reach_length>0] \
                                 /reach_length[reach_length>0]
    
    # weight average slope by reach length
    w = reach_length[reach_length>0]
    mean_reach_slope   = np.sum(w*reach_slopes)/np.sum(w)

    if verbose:
        print('mean reach length and elevation difference ',np.mean(reach_length[reach_length>0]),np.mean(reach_elevation_difference[reach_length>0]))
        print('mean reach slope ',mean_reach_slope)
        
    return {'length':total_reach_length,'slope':mean_reach_slope}

def _calculate_hillslope_mean_aspect(did,drainage_id=None,aspect=None,hillslope=None,hillslope_types=None):
    # input arrays should be flattened
    out = []

    dind = np.where(drainage_id==did)[0]
    # type 4 (channels) will be combined with other types, so decrement number in n loop
    for n in hillslope_types[:hillslope_types.size-1]:
        # combine channels with other aspects 
        l2 = np.logical_or(hillslope[dind]==4,hillslope[dind]==n)
        ind = dind[np.where(l2)[0]]
        if ind.size > 0:
            mean_aspect = np.arctan2(np.mean(np.sin(dtr*aspect[ind])),np.mean(np.cos(dtr*aspect[ind]))) / dtr
            if mean_aspect < 0:
                mean_aspect += 360.
            out.append([did,n,mean_aspect,ind])

    return out         

def set_aspect_to_hillslope_mean_parallel(drainage_id,aspect,hillslope,npools=4,chunksize = 5e3):
    # input arrays are 2d
    l1 = np.logical_and(np.isfinite(drainage_id),drainage_id > 0)
    uid = np.unique(drainage_id[l1])
    hillslope_types = np.unique(hillslope[hillslope >0]).astype(int)

    aspect2d_catchment_mean = np.zeros(aspect.shape)

    if uid.size == 0:
        return aspect2d_catchment_mean

    # set up multiprocessing pool
    if npools == 0:
        pool1 = Pool()
    else:
        pool1 = Pool(npools)
        
    try:
        # chunk data to reduce cost of array searches (i.e. np.where)
        nchunks = int(np.max([1,int(uid.size//chunksize)]))
        chunksize = np.min([chunksize,uid.size-1])
        for n in range(nchunks):
            n1, n2 = int(n*chunksize),int((n+1)*chunksize)
            if n == nchunks-1:
                n2 = uid.size-1
            if n1==n2: # single drainage case
                cind = np.where(drainage_id.flat == uid[n1])[0]
            else:
                cind = np.where(np.logical_and(drainage_id.flat >= uid[n1],drainage_id.flat < uid[n2]))[0]

            x = pool1.map(partial(_calculate_hillslope_mean_aspect,drainage_id=drainage_id.flat[cind],aspect=aspect.flat[cind],hillslope=hillslope.flat[cind],hillslope_types=hillslope_types),uid[n1:n2+1])

            for x2 in x:
                for tmp in x2:
                    if len(tmp) > 0:
                        _, mean_aspect, ind = tmp[1:]
                        aspect2d_catchment_mean.flat[cind[ind]] = mean_aspect
    finally:
        pool1.close()
        pool1.join()
    
    return aspect2d_catchment_mean

def set_aspect_to_hillslope_mean_serial(drainage_id,aspect,hillslope,chunksize=5e2):
    l1 = np.logical_and(np.isfinite(drainage_id),drainage_id > 0)
    uid = np.unique(drainage_id[l1])
    hillslope_types = np.unique(hillslope[hillslope >0]).astype(int)

    aspect2d_catchment_mean = np.zeros(aspect.shape)

    # chunk data to reduce cost of array searches (i.e. np.where)
    nchunks = int(np.max([1,int(uid.size//chunksize)]))
    chunksize = np.min([chunksize,uid.size-1])
    for n in range(nchunks):
        n1, n2 = int(n*chunksize),int((n+1)*chunksize)
        if n == nchunks-1:
            n2 = uid.size-1
        if n1==n2: # single drainage case
            cind = np.where(drainage_id.flat == uid[n1])[0]
        else:
            cind = np.where(np.logical_and(drainage_id.flat >= uid[n1],drainage_id.flat < uid[n2]))[0]
        
        # search a subset of array in each chunk
        for did in uid[n1:n2+1]:
            dind = cind[np.where(drainage_id.flat[cind]==did)[0]]
            # type 4 (channels) will be combined with other types, so decrement number in n loop
            for n in hillslope_types[:hillslope_types.size-1]:
                # combine channels with other aspects 
                l2 = np.logical_or(hillslope.flat[dind]==4,hillslope.flat[dind]==n)
                ind = dind[np.where(l2)[0]]
                if ind.size > 0:
                    mean_aspect = np.arctan2(np.mean(np.sin(dtr*aspect.flat[ind])),np.mean(np.cos(dtr*aspect.flat[ind]))) / dtr
                    if mean_aspect < 0:
                        mean_aspect += 360.
                    aspect2d_catchment_mean.flat[ind] = mean_aspect
    
    return aspect2d_catchment_mean

def std_dev(x):
    return np.power(np.mean(np.power((x-np.mean(x)),2)),0.5)

def TailIndex(fdtnd, fhand,npdf_bins=5000,hval=0.05):
    # return indices of input arrays with tails removed
    std_dtnd = std_dev((fdtnd[fhand > 0]))
    fit_loc, fit_beta = expon.fit(fdtnd[fhand > 0]/std_dtnd)
    rv = expon(loc=fit_loc,scale=fit_beta)

    pbins = np.linspace(0,np.max(fdtnd),npdf_bins)
    rvpdf = rv.pdf(pbins/std_dtnd)
    r1 = np.argmin(np.abs(rvpdf - hval*np.max(rvpdf)))
    ind = np.where(fdtnd < pbins[r1])[0]   
    return ind

def SpecifyHandBounds(fhand,faspect,aspect_bins,bin1_max=2, \
                      BinMethod='fastsort'):
    '''
    Determine hand bounds from a (flattened) hand array such 
    that approximately equal areas are obtained, subject to the 
    constraint that the first bin is less than 2 meters.  In that 
    case, the area in the first bin may be smaller than the others.
    Currently, 4 hand bins are created.
    '''
    std_hand = std_dev(fhand[fhand > 0])
    # available methods: fit hand, explicit sum, fast sort

    if BinMethod == 'fithand':
        _, fit_beta = expon.fit(fhand[fhand > 0].flat/std_hand)
        xbin1 = np.min([-fit_beta*np.log(1/4),bin1_max/std_hand])
        x33 = -fit_beta*np.log(2/3) + xbin1
        x66 = -fit_beta*np.log(1/3) + xbin1
        hand_bin_bounds = [0,xbin1*std_hand,x33*std_hand,x66*std_hand,1e6]
    elif BinMethod == 'explicitsum':
        nhist = np.round(np.max(fhand)).astype(int)
        nhist = np.max([int(200),nhist])
        hbins = np.linspace(0,np.max(fhand),nhist+1)
        hind = np.where(np.logical_and(fhand >= hbins[0],fhand < hbins[1]))[0]
        # replace bins with smaller range if histogram skewed towards small values
        if (hind.size/fhand.size) > 0.5:
            hbin1 = hbins[1]
            hbins[:-1] = np.linspace(0,hbin1,nhist)

        histo_hand = np.zeros((nhist))
        for h in range(nhist):
            hind = np.where(np.logical_and(fhand >= hbins[h],fhand < hbins[h+1]))[0]
            histo_hand[h] = hind.size

        cum_histo_hand = np.zeros((nhist))
        for h in range(nhist):
            cum_histo_hand[h] = np.sum(histo_hand[:h+1])
        cum_histo_hand = cum_histo_hand/np.sum(histo_hand)
        b25  = hbins[np.argmin(np.abs(0.25 - cum_histo_hand))+1]
        # first bin must be <= bin1_max
        if b25 > bin1_max:
            b33  = hbins[np.argmin(np.abs(0.33 - cum_histo_hand))+1]
            b66  = hbins[np.argmin(np.abs(0.66 - cum_histo_hand))+1]
            if b33 == b66:
                # just shift b66 for now
                b66 = 2*b33-bin1_max
            hand_bin_bounds = [0,bin1_max,b33,b66,1e6]

        else:
            b50  = hbins[np.argmin(np.abs(0.50 - cum_histo_hand))+1]
            b75  = hbins[np.argmin(np.abs(0.75 - cum_histo_hand))+1]
            hand_bin_bounds = [0,b25,b50,b75,1e6]

    elif BinMethod == 'fastsort':
        quartiles = np.asarray([0.25,0.5,0.75,1.0])
        # if many zeros exist, both bins 0 and 1 may be equal to zero 
        hand_sorted = np.sort(fhand[fhand > 0])
        hand_bin_bounds = np.asarray([0]+[hand_sorted[int(quartiles[qi]*hand_sorted.size-1)] for qi in range(quartiles.size)])

        # first bin must be <= bin1_max unless too few
        # points present in bin1_max bin
        if hand_bin_bounds[1] > bin1_max:
            # ensure enough points exist in the lowland bin
            # for each aspect bin
            min_aspect_fraction = 0.01
            for asp_ndx in range(len(aspect_bins)):
                if asp_ndx == 0:
                    l1 = np.logical_or(faspect >= aspect_bins[asp_ndx][0],faspect < aspect_bins[asp_ndx][1])
                else:
                    l1 = np.logical_and(faspect >= aspect_bins[asp_ndx][0],faspect < aspect_bins[asp_ndx][1])

                hand_asp_sorted = np.sort(fhand[l1])
                if hand_asp_sorted.size > 0:
                    bmin = hand_asp_sorted[int(min_aspect_fraction*hand_asp_sorted.size-1)]
                else:
                    bmin = bin1_max

                if bmin > bin1_max:
                    print('Too few hand values < '+str(bin1_max)+'; setting lowest bin to '+str(bmin))
                    bin1_max = bmin

            tmp = hand_sorted[hand_sorted > bin1_max]

            if int(0.33*tmp.size-1) == -1:
                print('bad tmp ')
                print(tmp.size,bin1_max,hand_asp_sorted.size)
            
            b33 = tmp[int(0.33*tmp.size-1)]
            b66 = tmp[int(0.66*tmp.size-1)]
            if b33 == b66:
                # just shift b66 for now
                b66 = 2*b33-bin1_max
            hand_bin_bounds = [0,bin1_max,b33,b66,1e6]

    if (len(hand_bin_bounds) - 1) != 4:
        raise RuntimeError('bad hand bounds')
    return hand_bin_bounds
                
def SpecifyHandBoundsNoAspect(fhand,nbins=4):
    '''
    Determine hand bounds from a (flattened) hand array such 
    that approximately equal areas are obtained, subject to the 
    constraint that the first bin is less than 2 meters.  In that 
    case, the area in the first bin may be smaller than the others.
    Currently, 4 hand bins are created.
    '''
    # if many zeros exist, both bins 0 and 1 may be equal to zero
    hand_sorted = np.sort(fhand[fhand > 0])
    if nbins == 2:
        quartiles = np.asarray([0.5,1.0])
    if nbins == 3:
        quartiles = np.asarray([0.33,0.66,1.0])
    if nbins == 4:
        quartiles = np.asarray([0.25,0.5,0.75,1.0])
    
    hand_bin_bounds = np.asarray([0]+[hand_sorted[int(quartiles[qi]*hand_sorted.size-1)] for qi in range(quartiles.size)])

    if (len(hand_bin_bounds) - 1) != nbins:
        raise RuntimeError('bad hand bounds')
    
    return hand_bin_bounds
                
