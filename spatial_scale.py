import sys 
import string 
import subprocess
import time
import argparse
import numpy as np 
import netCDF4 as netcdf4 
from scipy import optimize,signal

'''
_fit_polynomial:         calculate polynomial coefficients
_synth_polynomial:       reconstruct polynomial from coefficients
_bin_amplitude_spectrum: bin amplitude spectrum into wavelength bins
_log_normal:             return a lognormal distribution
_fit_peak_lognormal:     fit a lognormal function to a peak
_gaussian_no_norm:       return a gaussian distribution
_fit_peak_gaussian:      fit a gaussian function to a peak
_LocatePeak:             locate a single peak in an array
IdentifySpatialScaleLaplacian:  identify peak amplitude of the DFT spectrum of the laplacian of a 2d array

'''    

# Parameters

# degrees to radians
dtr = np.pi/180.
# earth radius [m]
re  = 6.371e6

# Functions

def _fit_polynomial(x,y,ncoefs,weights=None):
    im = x.size
    if im < ncoefs:
        print('not enough data to fit '+str(ncoefs)+' coefficients')
        stop
        #return np.zeros((ncoefs),dtype=np.float64)
        
    coefs = np.zeros((ncoefs),dtype=np.float64)
    g = np.zeros((im,ncoefs),dtype=np.float64)
    for n in range(ncoefs):
        g[:,n] = np.power(x,n)

    if type(weights) == type(None):
        gtd = np.dot(np.transpose(g), y)
        gtg = np.dot(np.transpose(g), g)
    else:
        if y.size != weights.size:
            print('weights length must match data')
            stop
            
        gtd = np.dot(np.transpose(g), np.dot(np.diag(weights),y))
        gtg = np.dot(np.transpose(g), np.dot(np.diag(weights),g))

    covm = np.linalg.inv(gtg)
    coefs = np.dot(covm, gtd)

    return coefs

def _synth_polynomial(x,coefs):
    im = x.size
    ncoefs = coefs.size
    y = np.zeros((im),dtype=np.float64)
    for n in range(ncoefs):
        y+=coefs[n]*np.power(x,n)
    return y

def _bin_amplitude_spectrum(amp_fft,wavelength,nlambda=20):
    logLambda = np.zeros(wavelength.shape)
    logLambda[wavelength>0] = np.log10(wavelength[wavelength>0])
    
    lambda_bounds = np.linspace(0,np.max(logLambda),num=nlambda+1) 
    amp_1d = np.zeros((nlambda))
    lambda_1d = np.zeros((nlambda))
    # use > for lower bound to avoid zero values
    for n in range(nlambda):
        l1 = np.logical_and(logLambda > lambda_bounds[n], \
                            logLambda <= lambda_bounds[n+1])
        if np.any(l1):
            lambda_1d[n] = np.mean(wavelength[l1])
            amp_1d[n]    = np.mean(amp_fft[l1])
    ind = np.where(lambda_1d > 0)[0]
    return {'amp':amp_1d[ind],'lambda':lambda_1d[ind]}

def _log_normal(x,amp,sigma,mu,shift=0):
    # sigma widens distribution
    # mu widens and shifts peak away from zero
    # shift can be used to translate entire function
    f = np.zeros((x.size))
    if sigma > 0:
        f[x>shift] = amp*np.exp(-((np.log(x[x>shift]-shift)-mu)**2)/(2*(sigma**2)))
    return f

def _fit_peak_lognormal(x,y,verbose=False):
    # use scipy signal.find_peaks to locate a peak using a lognormal model
    meansig = np.mean(y)
    pheight = (meansig,None)
    pwidth = (0,0.75*x.size)
    pprom = (0.2*meansig,None)
    peaks, props = signal.find_peaks(y, height=pheight, threshold=None, distance=None, prominence=pprom, width=pwidth, wlen=None, rel_height=0.5, plateau_size=None)

    # if no peak found, try reducing prominence
    if peaks.size == 0:
        if verbose:
            print('no peaks found, reducing prominence')
        
        pprom = (0.1*meansig,None)
        peaks, props = signal.find_peaks(y, height=pheight, threshold=None, distance=None, prominence=pprom, width=pwidth, wlen=None, rel_height=0.5, plateau_size=None)

    # signal.find_peaks does not locate edge peaks, so add edge to test
    if peaks.size > 0:
        peaks = np.append(peaks,0)
        props['widths'] = np.append(props['widths'],np.max(props['widths']))
        
    if verbose:
        print('pheight ',pheight)
        print('pwidth ',pwidth)
        print('pprom ',pprom)
        print('props ',props)
    
    # fit a lognormal curve to each peak to quantify shape
    peak_sharp = []
    peak_coefs = []
    peak_gof   = []
    useIndividualWidths = True
    for ip in range(peaks.size):
        p = peaks[ip]

        # use width to create weights centered on each peak
        # and ensure a minimum number of points
        minw = 3
        if useIndividualWidths:
            # use individual widths
            pw = np.max([minw,int(0.5*props['widths'][ip])])
        else:
            # use the same width for each peak
            pw = np.max([minw,int(0.5*np.max(props['widths']))])
        
        i1,i2 = np.max([0,p-pw]),np.min([x.size-1,p+pw+1])

        # convert width to log value and create weights
        gsigma = np.mean([np.abs(x[p]- x[i1]),np.abs(x[i2]-x[p])])

        # p0 is optional; initial guesses
        amp = np.mean(y[i1:i2+1])
        center = x[p]
        sigma = gsigma
        mu = np.log(center)

        # One peak
        if verbose:
            print('\ninitial peak ',center)
            print('\ninitial amp,sigma,mu ',amp,sigma,mu)
        try:
            p0_1lognorm = [amp,sigma,mu]
            popt_1lognorm, pcov_1lognorm = optimize.curve_fit(_log_normal, x[i1:i2+1], y[i1:i2+1], p0=p0_1lognorm)

            # distance between initial center and fit
            # mode of log-normal is ~ exp(mu-sigma**2)
            ln_peak = np.exp(popt_1lognorm[2])
            pdist = np.abs(center - ln_peak)

            # if pdist larger than sigma, ignore peak
            if pdist > popt_1lognorm[1]:
                popt_1lognorm = [0,0,1]
        except:
            popt_1lognorm = [0,0,1]
        if verbose:
            print('popt_1lognorm: ',popt_1lognorm)

        peak_coefs.append(popt_1lognorm)
        peak_gof.append(np.mean(np.power(y[i1:i2+1]-_log_normal(x[i1:i2+1],*popt_1lognorm),2)))

        # exclude peaks that are below a height threshold
        if peak_gof[-1] < 1e6 and popt_1lognorm[0] > meansig:
            peak_sharp.append((x[-1]-x[0])/popt_1lognorm[1])
        else:
            peak_sharp.append(0)
        
    if verbose:
        print('peak sharp values ',peak_sharp)
        
    if len(peak_sharp) > 0:
        pmax = np.argmax(np.asarray(peak_sharp))
        psharp = peak_sharp[pmax]
        plognorm = peak_coefs[pmax]
        pgof = peak_gof[pmax]
    else:
        plognorm = [0,0,1]
        psharp = 0
        pgof = 0
        
    return {'coefs':plognorm,'psharp':psharp,'gof':pgof}

def _gaussian_no_norm(x,amp,cen,sigma):
    return amp*np.exp(-(x-cen)**2/(2*(sigma**2)))

def _fit_peak_gaussian(x,y,verbose=False):
    # use scipy signal.find_peaks to locate a peak using a gaussian model
    meansig = np.mean(y)
    pheight = (meansig,None)
    pwidth = (0,0.75*x.size)
    pprom = (0.2*meansig,None)
    peaks, props = signal.find_peaks(y, height=pheight, threshold=None, distance=None, prominence=pprom, width=pwidth, wlen=None, rel_height=0.5, plateau_size=None)

    # if no peak found, try reducing prominence
    if peaks.size == 0:
        if verbose:
            print('no peaks found, reducing prominence')
        
        pprom = (0.1*meansig,None)
        peaks, props = signal.find_peaks(y, height=pheight, threshold=None, distance=None, prominence=pprom, width=pwidth, wlen=None, rel_height=0.5, plateau_size=None)

    # signal.find_peaks does not locate edge peaks, so add edge to test
    if peaks.size > 0:
        peaks = np.append(peaks,0)
        props['widths'] = np.append(props['widths'],np.max(props['widths']))
        
    if verbose:
        print('pheight ',pheight)
        print('pwidth ',pwidth)
        print('pprom ',pprom)
        print('props ',props)
    
    # fit a gaussian to each peak to quantify shape
    peak_sharp = []
    peak_coefs = []
    peak_gof = []
    useIndividualWidths = True
    for ip in range(peaks.size):
        p = peaks[ip]

        # use width to create weights centered on each peak
        # and ensure a minimum number of points
        minw = 3
        if useIndividualWidths:
            # use individual widths
            pw = np.max([minw,int(0.5*props['widths'][ip])])
        else:
            # use the same width for each peak
            pw = np.max([minw,int(0.5*np.max(props['widths']))])
        
        i1,i2 = np.max([0,p-pw]),np.min([x.size-1,p+pw+1])

        # convert width to log value and create weights
        gsigma = 1*np.mean([np.abs(x[p]- x[i1]),np.abs(x[i2]-x[p])])

        # p0 is optional; initial guesses
        amp = np.mean(y[i1:i2+1])
        center = x[p]
        sigma = gsigma

        # One peak
        if verbose:
            print('\ninitial amp,center,sigma ',amp,center,sigma)
        try:
            p0_1gauss = [amp,center,sigma]
            popt_1gauss, pcov_1gauss = optimize.curve_fit(_gaussian_no_norm, x[i1:i2+1], y[i1:i2+1], p0=p0_1gauss)
            # distance between initial center and fit
            pdist = np.abs(center - popt_1gauss[1])

            # if pdist larger than sigma, ignore peak
            if pdist > popt_1gauss[2]:
                popt_1gauss = [0,0,1]
        except:
            popt_1gauss = [0,0,1]
        if verbose:
            print('popt_1gauss: ',popt_1gauss)

        peak_coefs.append(popt_1gauss)
        peak_gof.append(np.mean(np.power(y[i1:i2+1]-_gaussian_no_norm(x[i1:i2+1],*popt_1gauss),2)))

        # exclude peaks that are below a height threshold
        if peak_gof[-1] < 1e6 and popt_1gauss[0] > meansig:
            peak_sharp.append((x[-1]-x[0])/popt_1gauss[2])
        else:
            peak_sharp.append(0)
        
    if verbose:
        print('peak sharp values ',peak_sharp)
        
    if len(peak_sharp) > 0:
        pmax = np.argmax(np.asarray(peak_sharp))
        psharp = peak_sharp[pmax]
        pgauss = peak_coefs[pmax]
        pgof = peak_gof[pmax]
    else:
        pgauss = [0,0,1]
        psharp = 0
        pgof = 0
        
    return {'coefs':pgauss,'psharp':psharp,'gof':pgof}

def _LocatePeak(lambda_1d,ratio_var_to_lambda,maxWavelength=1e6,minWavelength=1,verbose=False):
    '''
    Fit ratio of variance to wavelength using linear and gaussian models.
    '''
    
    # minimum and maximum wavelengths to consider
    lmax = np.argmin(np.abs(lambda_1d - maxWavelength))

    lmin = np.argmin(np.abs(lambda_1d - minWavelength))

    # Fit different models to variance ratio
    # and compare goodness of fit
    if verbose:
        print('\nBeginning curve fitting')
        print('min max lambda ',minWavelength,maxWavelength)
        print('lambdamin lambdalmax ', lambda_1d[lmin], lambda_1d[lmax])
        print('lmin lmax ', lmin, lmax)
    
    ncurves = 3
    gof = np.zeros((ncurves))
    gof[:,] = 1e6

    # Linear
    logLambda = np.log10(lambda_1d)
    lcoefs = _fit_polynomial(logLambda[lmin:lmax+1],ratio_var_to_lambda[lmin:lmax+1],2)

    # Exponential
    ecoefs = _fit_polynomial(logLambda[lmin:lmax+1],np.log(ratio_var_to_lambda[lmin:lmax+1]),2)

    # Gaussian
    x = _fit_peak_gaussian(logLambda[lmin:lmax+1],ratio_var_to_lambda[lmin:lmax+1],verbose=verbose)
    pgauss    = x['coefs']
    psharp_ga = x['psharp']
    gof_ga = x['gof']

    # Lognormal
    x = _fit_peak_lognormal(logLambda[lmin:lmax+1],ratio_var_to_lambda[lmin:lmax+1],verbose=verbose)
    plognorm  = x['coefs']
    psharp_ln = x['psharp']
    gof_ln = x['gof']

    # calculate t-score for linear fit
    #scipy.stats.ttest_ind(a, b, equal_var=True)
    num = (1/(lmax-2))*np.sum(np.power(ratio_var_to_lambda[lmin:lmax+1]-_synth_polynomial(logLambda[lmin:lmax+1],lcoefs),2))
    den =np.sum(np.power(logLambda[lmin:lmax+1]-np.mean(logLambda[lmin:lmax+1]),2))
    se = np.sqrt(num/den)
    tscore = np.abs(lcoefs[1])/se
    if verbose:
        print('tscore ',tscore)                

    if verbose:
        if psharp_ga == 0:
            print('\npeak sharpness is zero; gaussian fit failed')
        elif psharp_ln == 0:
            print('\npeak sharpness is zero; lognormal fit failed')
        else:
            print('\npeak sharpness ',psharp_ga,psharp_ln)

    # Select best model
    psharp_threshold = 1.5
    tscore_threshold = 2
    model = 'None'
    # first check peaked distributions
    if psharp_ga >= psharp_threshold or psharp_ln >= psharp_threshold:
        if gof_ga < gof_ln:
            model = 'gaussian'
            spatialScale = np.min([10**pgauss[1],maxWavelength])
            spatialScale = np.max([spatialScale,minWavelength])
            selection = 1
            if verbose:
                print('\ngaussian selected')
        else:
            model = 'lognormal'
            ln_peak = np.exp(plognorm[2])
            spatialScale = np.min([10**ln_peak,maxWavelength])
            spatialScale = np.max([spatialScale,minWavelength])
            selection = 2
            if verbose:
                print('\nlognormal selected')
    else:
        # linear trend
        if tscore > tscore_threshold:  
            model = 'linear'
            if lcoefs[1] > 0:
                spatialScale = maxWavelength
                selection = 3
            else:
                spatialScale = minWavelength
                selection = 4
            if verbose:
                print('\nlinear selected ', selection)
        # if linear trend is not significant, select flat distribution
        else:
            model = 'flat'
            spatialScale = minWavelength
            selection = 5
            if verbose:
                print('\nflat selected')


    if model == 'None':
        print('No model selected')
        stop

    if verbose:
        print('\nEnd curve fitting')

    # set coefficients for output
    if model == 'gaussian':
        ocoefs = pgauss
    if model == 'lognormal':
        ocoefs = plognorm
    if model == 'linear':
        ocoefs = lcoefs
    if model == 'flat':
        ocoefs = [1]
            
    return {'model':model,'spatialScale':spatialScale,'selection':selection,'coefs':ocoefs}

def IdentifySpatialScaleLaplacian(corners, \
                                     maxHillslopeLength=0, \
                                     land_threshold=0.75, \
                                     min_land_elevation=0, \
                                     dem_file_template=None, \
                                     detrendElevation=False, \
                                     doBlendEdges=True, \
                                     nlambda=30, \
                                     dem_source='MERIT', \
                                     verbose=False):
    '''
    Identify the spatial scale at which the input DEM 
    exhibits the largest divergence/convergence of topographic gradient.
    '''

    from dem_io import read_MERIT_dem_data, read_ASTER_dem_data
    from geospatial_utils import fit_planar_surface, smooth_2d_array, blend_edges, calc_gradient_horn1981
    
    if maxHillslopeLength==0:
        print('maxHillslopeLength must be > 0')
        stop
    if type(dem_file_template)==type(None):
        print('no dem file template supplied')
        stop

    if dem_source not in ['MERIT','ASTER']:
        print('invalid dem source ', dem_source)
        stop

    if dem_source == 'MERIT':
        x = read_MERIT_dem_data(dem_file_template,corners,zeroFill=True)
    if dem_source == 'ASTER':
        x = read_ASTER_dem_data(dem_file_template,corners,zeroFill=True)
    validDEM = x['validDEM']

    if not validDEM:
        return {'validDEM':validDEM}
        
    elev,elon,elat = x['elev'],x['lon'],x['lat']
    ejm,eim = elev.shape

    # approximate resolution in m
    ares = np.abs(elat[0]-elat[1]) * (re*np.pi/180)
    maxWavelength = 2*maxHillslopeLength/ares

    # Create land/ocean mask
    # if land fraction is below a threshold,
    # remove smoothed elevation
    lmask = np.where(elev > min_land_elevation,1,0)            
    land_frac = np.sum(lmask)/lmask.size
    if verbose:
        print('approximate resolution in m: ',ares)
        print('max wavelength for hillslope, jm, im: ',maxWavelength,ejm,eim,'\n')
        print('land fraction ',land_frac)

    if land_frac == 0:
        return {'validDEM':False} 
    if land_frac <  land_threshold:
        if verbose:
            print('Removing smoothed elevation')
        sf = 0.75
        if elon[0] > elon[-1]: 
            # needed for points spanning prime meridian
            elon2 = np.where(elon < 180,elon,elon-360)
            elonc = np.mean(elon2)
            delonc = np.abs(elon2[0]-elon2[-1])
        else:
            elonc = np.mean(elon)
            delonc = np.abs(elon[0]-elon[-1])
            
        if elonc < 0:
            elonc += 360
        elatc = np.mean(elat)
        delatc = np.abs(elat[0]-elat[-1])

        corners = [[elonc-sf*delonc,elatc-sf*delatc],
                   [elonc-sf*delonc,elatc+sf*delatc],
                   [elonc+sf*delonc,elatc-sf*delatc],
                   [elonc+sf*delonc,elatc+sf*delatc]]
        
        # bound corners (assumes positive longitude)
        for n in range(len(corners)):
            if corners[n][0] < 0:
                corners[n][0] += 360
            if corners[n][0] > 360:
                corners[n][0] -= 360
                
        # Read in dem data spanning region defined by corners
        if dem_source == 'MERIT':
            x = read_MERIT_dem_data(dem_file_template,corners,zeroFill=True)
        if dem_source == 'ASTER':
            x = read_ASTER_dem_data(dem_file_template,corners,zeroFill=True)
        validDEM = x['validDEM']

        if validDEM:    
            selev,selon,selat = x['elev'],x['lon'],x['lat']
            sejm,seim = selev.shape

            i1 = np.argmin(np.abs(selon-elon[0]))
            i2 = np.argmin(np.abs(selon-elon[-1]))
            j1 = np.argmin(np.abs(selat-elat[0]))
            j2 = np.argmin(np.abs(selat-elat[-1]))

            smooth_elev = smooth_2d_array(selev,land_frac=land_frac)[j1:j2+1,i1:i2+1]
            elev -= smooth_elev

    # Remove a plane (de-trend elevation)
    if detrendElevation:
        elev_planar = fit_planar_surface(elev,elon,elat)
        elev -= elev_planar

    # blend edges to reduce high frequency leakage
    if doBlendEdges:
        win = int(np.min([ejm,eim])//33) # 3% from edge
        win = 7
        win = 4
        elev = blend_edges(elev,n=win)
        
        if verbose:
            print('Planar surface removed from elevation\n')

    # Calculate 2D DFTs (output is complex array)
    # first dft is real, giving complex result w/ N/2 coefs
    # 2nd dft is complex, with N coefs

    grad = calc_gradient_horn1981(elev,elon,elat)
    # get spectrum of divergence
    x = calc_gradient_horn1981(grad[0],elon,elat)
    laplac = x[0]
    x = calc_gradient_horn1981(grad[1],elon,elat)
    laplac += x[1]

    laplac_fft = np.fft.rfft2(laplac,norm='ortho')
    laplac_amp_fft = np.abs(laplac_fft)

    if verbose:
        print('DFTs calculated\n')

    # use appropriate (real/complex) routine for frequencies
    rowfreq = np.fft.fftfreq(ejm)
    colfreq = np.fft.rfftfreq(eim)

    ny,nx = laplac_fft.shape
    radialfreq = np.sqrt(np.tile(colfreq*colfreq,(ny,1)) \
                 +np.tile(rowfreq*rowfreq,(nx,1)).T)

    wavelength = np.zeros((ny,nx))
    wavelength[radialfreq > 0] = 1/radialfreq[radialfreq > 0]

    # set 0 frequency term to arbitrary value
    wavelength[0,0] = 2*np.max(wavelength)

    # Create logarithmically binned 1d amplitude spectra
    x = _bin_amplitude_spectrum(laplac_amp_fft,wavelength,nlambda=nlambda)
    lambda_1d,laplac_amp_1d = x['lambda'],x['amp']
    
    # fit curve in window around fit_peaks 
    x = _LocatePeak(lambda_1d,laplac_amp_1d,maxWavelength=maxWavelength,verbose=verbose)

    model = x['model']
    spatialScale = x['spatialScale']
    selection = x['selection']

    # now set minimum wavelength for spatial scale
    minWavelength = np.min(lambda_1d)
    spatialScale = np.max([spatialScale,minWavelength])

    if verbose:
        print('\nmodel, spatial scale, selection method: ', model, spatialScale, selection)

    return {'model':model,'spatialScale':spatialScale,'selection':selection,'res':ares,'lambda_1d':lambda_1d,'laplac_amp_1d':laplac_amp_1d,'validDEM':validDEM}

