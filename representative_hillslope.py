#!/usr/bin/env python
# coding: utf-8

import subprocess
import time
import sys
import numpy as np
import netCDF4 as netcdf4
import rasterio

from geospatial_utils import quadratic, arg_closest_point, identify_basins
from spatial_scale import IdentifySpatialScaleLaplacian
from dem_io import (
    create_subregion_corner_lists,
    read_MERIT_dem_data,
    read_ASTER_dem_data,
)
from terrain_utils import (
    SpecifyHandBounds,
    TailIndex,
    set_aspect_to_hillslope_mean_serial,
    set_aspect_to_hillslope_mean_parallel,
)
from rh_logging import info, warning, error, debug

sys.path.append("pysheds")
from pysheds.pgrid import Grid

"""
LandscapeCharacteristics: class for landscape terrain characteristics derived from digital elevation model.

CalcRepresentativeHillslopeForm: function calculating representative hillslope geometry.

CalcGeoparamsGridcell:               calculate representative hillslope geomorphic parameters for a gridcell

"""

# Parameters

# degrees to radians
dtr = np.pi / 180.0
# earth radius [m]
re = 6.371e6

# function definitions


def calc_width_parameters(
    dtnd, area, form="trapezoid", useAreaWeight=True, mindtnd=0, nhisto=10
):
    if form not in ["trapezoid", "annular"]:
        raise RuntimeError("form must be one of: ", ["trapezoid", "annular"])

    dtndbins = np.linspace(mindtnd, np.max(dtnd) + 1, nhisto + 1)
    binwidth = dtndbins[1:] - dtndbins[:-1]
    d = np.zeros((nhisto))
    A = np.zeros((nhisto))
    for k in range(nhisto):
        dind = np.where(dtnd >= dtndbins[k])[0]
        d[k] = dtndbins[k]
        A[k] = np.sum(area[dind])

    # add d=0, total area values
    if mindtnd > 0:
        d = np.asarray([0] + d.tolist())
        A = np.asarray([np.sum(area)] + A.tolist())

    if useAreaWeight:
        accum_coefs = _fit_polynomial(d, A, ncoefs=3, weights=A)
    else:
        accum_coefs = _fit_polynomial(d, A, ncoefs=3)

    if form == "trapezoid":
        slope = -accum_coefs[2]
        width = -accum_coefs[1]
        Atrap = accum_coefs[0]

        # if quadratic function has a region of positive slope
        # (possible near the tail, i.e. the large d values)
        # the implied width will be negative and the trapezoid
        # approximation will break down, so adjust base width to avoid this issue.
        if slope < 0:
            Atri = -(width**2) / (4 * slope)
            if Atri < Atrap:
                width = np.sqrt(-4 * slope * Atrap)

        return {"slope": slope, "width": width, "area": Atrap}

    if form == "annular":
        alpha = 2 * accum_coefs[2]
        rsec = -accum_coefs[1] / alpha
        Asec = accum_coefs[0]

        # adjust parameters to match actual area per hillslope
        Asum = np.sum(area)
        if Asec < Asum:
            eps = 1e-6
            alpha = 2 * Asum / (rsec**2) + eps
            Asec = (alpha / 2) * rsec**2

        return {"alpha": alpha, "radius": rsec, "area": Asec}


def _fit_polynomial(x, y, ncoefs, weights=None):
    im = x.size
    if im < ncoefs:
        raise RuntimeError("not enough data to fit " + str(ncoefs) + " coefficients")

    coefs = np.zeros((ncoefs), dtype=np.float64)
    g = np.zeros((im, ncoefs), dtype=np.float64)
    for n in range(ncoefs):
        g[:, n] = np.power(x, n)

    if type(weights) == type(None):
        gtd = np.dot(np.transpose(g), y)
        gtg = np.dot(np.transpose(g), g)
    else:
        if y.size != weights.size:
            raise RuntimeError("weights length must match data")

        gtd = np.dot(np.transpose(g), np.dot(np.diag(weights), y))
        gtg = np.dot(np.transpose(g), np.dot(np.diag(weights), g))

    covm = np.linalg.inv(gtg)
    coefs = np.dot(covm, gtd)

    return coefs


def _fit_trapezoid(area, dist_from_channel, applyWeights=True):
    # solve width and angle of a trapezoid given area per catchment and distance from channel

    nm = area.size

    # set up lsq matrix
    g = np.zeros((nm, 2))
    g[:, 0] = dist_from_channel
    g[:, 1] = -(dist_from_channel**2)
    d = np.zeros((nm))
    for n in range(nm):
        d[n] = area[n] / 2 + np.sum(area[:n])

    if applyWeights:
        weights = area
        gtd = np.dot(np.transpose(g), np.dot(np.diag(weights), d))
        gtg = np.dot(np.transpose(g), np.dot(np.diag(weights), g))
    else:
        gtd = np.dot(np.transpose(g), d)
        gtg = np.dot(np.transpose(g), g)

    #  covm is the model covariance matrix
    covm = np.linalg.inv(gtg)

    #  coefs is the model parameter vector
    coefs = np.dot(covm, gtd)

    w0, tana = coefs
    w = [w0]
    l = []
    d = [0]
    for n in range(nm):
        # length of section
        li = quadratic([-tana, w[n], -area[n]])
        l.append(li)
        # width at interface with next section
        wip1 = w[n] - np.sum(l) * tana
        w.append(wip1)
        d.append(np.sum(l))
    return [np.asarray(w), np.asarray(d)]


def CalcRepresentativeHillslopeForm(
    hillslope_fraction,
    column_index,
    area,
    dtnd,
    form="CircularSection",
    number_of_hillslopes=0,
    maxHillslopeLength=0,
):
    """
    calculate new distances and widths based on
    sections of hillslopes having a specified plan form
    node distances will be median distance of column
    width will be width at lower edge
    """

    if form not in ["CircularSection", "TriangularSection"]:
        raise RuntimeError("Invalid hillsope form")

    # angle of section
    alpha = 2.0 * np.pi * hillslope_fraction

    # number of valid bins
    nvbins = np.sum(np.where(column_index > 0, 1, 0))

    debug("hillslope fraction ", hillslope_fraction)
    debug("alpha ", alpha / dtr, " degrees")
    debug("number of valid bins ", nvbins)

    column_fraction = np.zeros(nvbins)
    for n in range(nvbins):
        column_fraction[n] = area[n] / np.sum(area)
        debug("column fraction ", n, column_fraction[n])

    # Hillslope assumed to occupy a section of a circle
    if form == "CircularSection":
        # calculate estimates of hill radius from relative
        # area and median distance
        nrad = []

        for n in range(nvbins):
            ufrac = np.sum(column_fraction[n + 1 : nvbins])
            ak = 1 - ufrac - column_fraction[n] / 2
            bk = -2 * dtnd[n]
            ck = dtnd[n] ** 2
            coefs = [ak, bk, ck]
            hill_radius = quadratic(coefs, root=0)
            debug("\nufrac ", n, np.sum(column_fraction), ufrac)
            debug("dmed ", dtnd[n])
            debug("hill_radius ", n, hill_radius)
            nrad.append(hill_radius)

        # weighted average of radii estimates
        mean_hill_radius = np.sum(nrad * column_fraction)

        # limit hillslope length
        if maxHillslopeLength > 0:
            mean_hill_radius = np.min([maxHillslopeLength, mean_hill_radius])

        # calculate estimates of hillslope area
        harea = hillslope_fraction * np.pi * mean_hill_radius**2
        narea = []
        for n in range(nvbins):
            narea.append(column_fraction[n] * harea)

        debug("hill radii ", nrad)
        debug("mean radius ", mean_hill_radius)
        debug("hill area ", harea)

        # calculate width and median distance to channel
        nwidth = []
        ndmed = []
        for n in range(nvbins):
            uarea = np.sum(narea[n:])
            rw = np.sqrt(uarea / (hillslope_fraction * np.pi))
            w = rw * alpha
            nwidth.append(w)

            rmed = np.sqrt((uarea - narea[n] / 2) / (hillslope_fraction * np.pi))
            ndmed.append(mean_hill_radius - rmed)

        return {
            "n_valid_bins": nvbins,
            "node_distance": ndmed,
            "edge_width": nwidth,
            "area": narea,
            "hill_length": mean_hill_radius,
        }

    if form == "TriangularSection":
        if hillslope_fraction > 135 / 360:
            raise RuntimeError(
                "Hillslope fraction too large for TriangularSection form: ",
                hillslope_fraction,
            )

        # calculate estimates of hill ridge-to-valley length
        nlength = []
        for n in range(nvbins):
            ufrac = np.sum(column_fraction[n + 1 : nvbins])
            ak = 1 - ufrac - column_fraction[n] / 2
            bk = -2 * dtnd[n]
            ck = dtnd[n] ** 2
            coefs = [ak, bk, ck]
            hill_length = quadratic(coefs, root=0)
            debug("\nufrac ", n, np.sum(column_fraction), ufrac)
            debug("dmed ", dtnd[n])
            debug("hill_length ", n, hill_length)
            nlength.append(hill_length)
        debug("\nhill lengths ", nlength)

        # weighted average of hillslope lengths
        mean_hill_length = np.sum(nlength * column_fraction)
        # limit hillslope length
        if maxHillslopeLength > 0:
            mean_hill_length = np.min([maxHillslopeLength, mean_hill_length])

        # total hillslope area
        harea = mean_hill_length**2 * np.tan(alpha / 2)

        debug("mean hillslope length ", mean_hill_length)
        debug("hillslope total area ", harea)

        # calculate estimates of hill column areas
        narea = []
        for n in range(nvbins):
            narea.append(column_fraction[n] * harea)

        # calculate width and median distance for mean hillslope
        nwidth = []
        ndmed = []
        for n in range(nvbins):
            uarea = np.sum(narea[n:])
            ledge = np.sqrt(uarea / np.tan(alpha / 2))
            w = 2 * uarea / ledge
            nwidth.append(w)
            uarea -= narea[n] / 2
            dmed = mean_hill_length - np.sqrt(uarea / np.tan(alpha / 2))
            ndmed.append(dmed)
            if n == 0:
                debug("mean_hill_length, ledge, dmed")
            debug(mean_hill_length, ledge, dmed)

        debug("base width {}\n".format(nwidth[0]))
        debug("new distances ", ndmed)
        debug("new widths ", nwidth)

        return {
            "n_valid_bins": nvbins,
            "node_distance": ndmed,
            "edge_width": nwidth,
            "area": narea,
            "hill_length": mean_hill_length,
        }


def CalcGeoparamsGridcell(
    ji,
    lon2d=None,
    lat2d=None,
    landmask=None,
    nhand_bins=None,
    aspect_bins=None,
    ncolumns_per_gridcell=None,
    maxHillslopeLength=None,
    dem_file_template=None,
    detrendElevation=None,
    nlambda=None,
    dem_source=None,
    outfile_template=None,
    overwrite=False,
    flagBasins=False,
    removeTailDTND=True,
    addStreamChannelVariables=True,
    hillslope_form=None,
    printData=False,
    useMultiProcessing=False,
):

    stime = time.time()
    j, i = ji
    debug("j i ", j, i)
    debug(lon2d[j, i], lat2d[j, i], "\n")

    outfile = outfile_template.replace(".nc", "_j_{:03d}_i_{:03d}.nc".format(j, i))
    debug(outfile)

    command = ["ls", outfile]
    file_exists = subprocess.run(command, capture_output=True).returncode

    if file_exists == 0 and not printData:
        if overwrite:
            debug(outfile, " exists; overwriting")
        else:
            debug(outfile, " exists; skipping")
            return

    # initialize new fields to be added to surface data file
    hand = np.zeros((ncolumns_per_gridcell))
    dtnd = np.zeros((ncolumns_per_gridcell))
    area = np.zeros((ncolumns_per_gridcell))
    slope = np.zeros((ncolumns_per_gridcell))
    aspect = np.zeros((ncolumns_per_gridcell))
    width = np.zeros((ncolumns_per_gridcell))
    zbedrock = np.zeros((ncolumns_per_gridcell))

    naspect = len(aspect_bins)
    pct_hillslope = np.zeros((naspect))
    # hillslope indices begin with 1 (oceans are 0)
    hillslope_index = np.zeros((ncolumns_per_gridcell))
    # column indices begin with 0
    column_index = np.zeros((ncolumns_per_gridcell))
    downhill_column_index = np.zeros((ncolumns_per_gridcell))

    # lowland index
    lowland_index = -9999

    chunk_mask = 0

    col_cnt = 1
    if landmask[j, i] == 1:
        chunk_mask = 1

        dlon = np.abs(lon2d[0, 0] - lon2d[0, 1])
        dlat = np.abs(lat2d[0, 0] - lat2d[1, 0])

        gsf = 0.5
        scorners = [
            [lon2d[j, i] - gsf * dlon, lat2d[j, i] - gsf * dlat],
            [lon2d[j, i] - gsf * dlon, lat2d[j, i] + gsf * dlat],
            [lon2d[j, i] + gsf * dlon, lat2d[j, i] - gsf * dlat],
            [lon2d[j, i] + gsf * dlon, lat2d[j, i] + gsf * dlat],
        ]

        # bound scorners
        for n in range(len(scorners)):
            if scorners[n][0] < 0:
                scorners[n][0] += 360
            if scorners[n][0] > 360:
                scorners[n][0] -= 360

        # Determine spatial scale at which to extract river network
        x = IdentifySpatialScaleLaplacian(
            scorners,
            maxHillslopeLength=maxHillslopeLength,
            land_threshold=0.75,
            min_land_elevation=0,
            dem_file_template=dem_file_template,
            detrendElevation=detrendElevation,
            nlambda=nlambda,
            dem_source=dem_source,
        )

        if not x["validDEM"]:
            debug("invalid dem ", j, i)
            return [-1]

        spatialScale = x["spatialScale"]
        model = x["model"]
        ares = x["res"]

        if model == "None":
            raise RuntimeError("No model selected")

        # Set accumulation threshold from spatial scale
        accum_thresh = 0.5 * (spatialScale**2)
        debug("\nSpatial scale, accum_thresh")
        debug(spatialScale, accum_thresh)

        # Set size of region used in catchment decomposition
        # us a larger region to resolve catchments that extend outside gridcell
        scale_in_meters = ares * spatialScale
        grid_spacing = np.abs(dlon * dtr * re)
        gs_ratio = grid_spacing / scale_in_meters
        # ad hoc increase in domain size
        sf = 1 + 4 * scale_in_meters / grid_spacing

        if gsf < 0.5:
            sf = sf * (gsf / 0.5)

        debug(
            "spatial scale in meters, grid spacing ", scale_in_meters, grid_spacing
        )
        debug("accum_thresh, grid scalar ", accum_thresh, sf)
        debug("ratio grid size to spatial scale ", gs_ratio)

        # Read in dem data for catchment decomposition
        dlonh = sf * dlon
        dlath = sf * dlat

        hcorners = [
            [lon2d[j, i] - 0.5 * dlonh, lat2d[j, i] - 0.5 * dlath],
            [lon2d[j, i] - 0.5 * dlonh, lat2d[j, i] + 0.5 * dlath],
            [lon2d[j, i] + 0.5 * dlonh, lat2d[j, i] - 0.5 * dlath],
            [lon2d[j, i] + 0.5 * dlonh, lat2d[j, i] + 0.5 * dlath],
        ]

        # bound hcorners
        for n in range(len(hcorners)):
            if hcorners[n][0] < 0:
                hcorners[n][0] += 360
            if hcorners[n][0] > 360:
                hcorners[n][0] -= 360

        # if gs_ratio is large, split into 4 subregions
        gs_thresh = 400
        if gs_ratio > gs_thresh:
            central_point = [lon2d[j, i], lat2d[j, i]]
            corner_list = create_subregion_corner_lists(hcorners, central_point)
        else:
            corner_list = [hcorners]

        # arrays to aggregate subregion parameters
        fhand_all = []
        fdtnd_all = []
        farea_all = []
        fslope_all = []
        faspect_all = []
        fdid_all = []
        fflood_all = []
        fhand_filter_all = []

        mean_stream_length = 0
        stream_number = 0
        mean_network_slope = 0

        nvalid_subregions = 0
        for nsub in range(len(corner_list)):
            debug("\nsubregion ", nsub + 1, " of ", len(corner_list), "\n")

            # Calculate landscape characteristics from dem
            lc = LandscapeCharacteristics()
            x = lc.CalcLandscapeCharacteristicsPysheds(
                corner_list[nsub],
                accum_thresh=accum_thresh,
                dem_file_template=dem_file_template,
                useMultiProcessing=useMultiProcessing,
                dem_source=dem_source,
                maskFlooded=False,
            )

            # if no valid data, skip to next subregion
            if x == -1:
                debug("no dem data ", lon2d[j, i], lat2d[j, i])
                continue

            # length will be summed, slope will be averaged
            mean_stream_length += lc.network_length
            stream_number += lc.nstreams
            mean_network_slope += lc.mean_network_slope

            # set corner values
            corners = corner_list[nsub]

            # bound corners
            for n in range(len(corners)):
                if corners[n][0] > 360:
                    corners[n][0] -= 360

            # pull arrays from center of grid object
            # pull arrays from center of grid object
            i1 = arg_closest_point(corners[0][0], lc.lon, angular=True)
            i2 = arg_closest_point(corners[3][0], lc.lon, angular=True)
            j1 = arg_closest_point(corners[0][1], lc.lat)
            j2 = arg_closest_point(corners[3][1], lc.lat)
            i1, i2 = np.sort([i1, i2])
            j1, j2 = np.sort([j1, j2])

            # extract geomorphic parameters of the gridcell
            fhand = lc.hand[j1:j2, i1:i2].flatten()
            fdtnd = lc.dtnd[j1:j2, i1:i2].flatten()
            faspect = lc.aspect[j1:j2, i1:i2].flatten()
            fslope = lc.slope[j1:j2, i1:i2].flatten()
            fdid = lc.drainage_id[j1:j2, i1:i2].flatten()
            farea = lc.area[j1:j2, i1:i2].flatten()
            fflood = lc.fflood[j1:j2, i1:i2].flatten()
            # filter to record modifications to arrays
            fhand_filter = np.arange(fhand.size, dtype=int)

            lat = lc.lat[j1:j2]
            lon = lc.lon[i1:i2]
            jm, im = lat.size, lon.size

            # identify basins and remove points (inputs are 2d arrays)
            if flagBasins:
                stimefb = time.time()
                basin_mask = identify_basins(lc.dem)[j1:j2, i1:i2]

                ind = np.where(basin_mask.flat == 0)[0]
                # set cutoff fraction
                non_flat_fraction = ind.size / fhand.size
                if (non_flat_fraction) > 0.01:
                    fhand = fhand[ind]
                    fdtnd = fdtnd[ind]
                    farea = farea[ind]
                    fslope = fslope[ind]
                    faspect = faspect[ind]
                    fdid = fdid[ind]
                    fflood = fflood[ind]
                    fhand_filter = fhand_filter[ind]
                    if printData and 1 == 2:
                        hand2d[basin_mask > 0] = 0
                        dtnd2d[basin_mask > 0] = 0
                else:
                    # if entire grid cell is a basin, skip
                    debug("subregion is too flat; skipping")
                    debug("non-flat fraction ", non_flat_fraction)
                    continue

                debug("Basins identified")
                debug("time: ", time.time() - stimefb, "\n")

            # add current array to previous data
            fhand_all.extend(fhand.tolist())
            fdtnd_all.extend(fdtnd.tolist())
            farea_all.extend(farea.tolist())
            fslope_all.extend(fslope.tolist())
            faspect_all.extend(faspect.tolist())
            fdid_all.extend(fdid.tolist())
            fflood_all.extend(fflood.tolist())
            fhand_filter_all.extend(fhand_filter.tolist())

            nvalid_subregions += 1

        # --  End of subregion loop  ---------------------------------
        fhand = np.asarray(fhand_all)
        fdtnd = np.asarray(fdtnd_all)
        farea = np.asarray(farea_all)
        fslope = np.asarray(fslope_all)
        faspect = np.asarray(faspect_all)
        fdid = np.asarray(fdid_all)
        fflood = np.asarray(fflood_all)
        fhand_filter = np.asarray(fhand_filter_all)

        # if no valid data, skip to next gridcell
        if nvalid_subregions == 0:
            debug("no subregions with valid dem data ", lon2d[j, i], lat2d[j, i])
            return [-1]

        # check for gridcells w/ no valid hand data
        # check for both all nans, or combination of zeros and nans
        hand_all_nans_check = not np.any(np.isfinite(fhand))
        hand_all_zeros_check = np.all(fhand[np.isfinite(fhand)] == 0.0)

        if fhand.size > 0:
            hand_coverage_fraction = (
                np.sum(np.where(fhand[np.isfinite(fhand)] > 0, 1, 0)) / fhand.size
            )
        else:
            hand_coverage_fraction = 0
        hand_insufficient_data = hand_coverage_fraction < 0.01

        if np.logical_or(hand_all_nans_check, hand_all_zeros_check):
            debug(lon2d[j, i], lat2d[j, i], "hand all nans or zeros, skipping...")
            return [-1]
        elif hand_insufficient_data:
            debug(
                    "fraction of region hand > 0 {:10.6f}, skipping...".format(
                        hand_coverage_fraction
                    )
                )
            return [-1]
        else:
            # continue processing data if some valid hand data exist
            # remove values where hand is nan
            # this leads to hillslope area < gridcell area in some places
            ind = np.where(np.isfinite(fhand))[0]
            nan_ratio = 1 - (ind.size / fhand.size)

            fhand = fhand[ind]
            fdtnd = fdtnd[ind]
            farea = farea[ind]
            fslope = fslope[ind]
            faspect = faspect[ind]
            fdid = fdid[ind]
            fflood = fflood[ind]
            fhand_filter = fhand_filter[ind]

        # eliminate tails of DTND distribution (large values can occur where dem is flooded/inflated)
        if removeTailDTND:
            ind = TailIndex(fdtnd, fhand)
            if ind.size > 0:
                fhand = fhand[ind]
                fdtnd = fdtnd[ind]
                farea = farea[ind]
                fslope = fslope[ind]
                faspect = faspect[ind]
                fdid = fdid[ind]
                fflood = fflood[ind]
                fhand_filter = fhand_filter[ind]

        # identify flooded regions in lowest hand bin
        hand_threshold = 2  # [meters]
        num_flooded_pts = np.sum((np.abs(fflood[fhand < hand_threshold]) > 0))
        if num_flooded_pts > 0:
            # exclude regions that have been flooded
            # by eliminating 95% of flooded values
            flood_thresh = 0
            for ft in np.linspace(0, 20, 50):
                if (
                    np.sum((np.abs(fflood[fhand < hand_threshold]) > ft))
                    / num_flooded_pts
                ) < 0.95:
                    flood_thresh = ft
                    break

            fhand = np.where(
                np.logical_and(np.abs(fflood) > flood_thresh, fhand < hand_threshold),
                -1,
                fhand,
            )

        # give minimum value for dtnd
        smallest_dtnd = 1.0  # [meters]
        fdtnd[fdtnd < smallest_dtnd] = smallest_dtnd

        debug("max value in fhand ", np.max(fhand))
        debug("max value in fdtnd ", np.max(fdtnd), "\n")

        # average channel slope
        mean_network_slope = mean_network_slope / nvalid_subregions

        # mean stream length
        mean_stream_length = mean_stream_length / stream_number

        debug("mean stream length ", mean_stream_length)
        debug("stream density ", stream_number * mean_stream_length / np.sum(farea))
        carea = np.asarray([np.sum(farea[fdid == i]) for i in np.unique(fdid)])
        debug(
            "mean catchment area ",
            np.mean(carea),
            np.sum(farea[np.isfinite(fhand)]) / stream_number,
        )
        debug("accum_thresh in m2 ", accum_thresh * ares * ares)

        # Determine hand bins such that approximately
        # equal areas are obtained, subject to a constraint on
        # the first bin's upper value
        hand_bin_bounds = SpecifyHandBounds(
            fhand, faspect, aspect_bins, bin1_max=2, BinMethod="fastsort"
        )

        debug("hand bin bounds ", hand_bin_bounds)

        # for each aspect, calculate hillslope elements
        hillslope_fraction = np.zeros((naspect))
        number_of_hillslopes = np.zeros((naspect))
        for asp_ndx in range(naspect):
            debug(
                    "----  Beginning aspect ",
                    asp_ndx + 1,
                    " of ",
                    naspect,
                    " --------------------------",
                )
            if asp_ndx == 0:
                aind = np.where(
                    np.logical_or(
                        faspect >= aspect_bins[asp_ndx][0],
                        faspect < aspect_bins[asp_ndx][1],
                    )
                )[0]
            else:
                aind = np.where(
                    np.logical_and(
                        faspect >= aspect_bins[asp_ndx][0],
                        faspect < aspect_bins[asp_ndx][1],
                    )
                )[0]

            if aind.size > 0:
                hillslope_fraction[asp_ndx] = np.sum(farea[aind]) / np.sum(farea)
                number_of_hillslopes[asp_ndx] = np.unique(fdid[aind]).size
                debug("hillslope fraction ", asp_ndx, hillslope_fraction)

                # use linear width hillslope models
                if hillslope_form == "Trapezoidal":
                    x = calc_width_parameters(
                        fdtnd[aind],
                        farea[aind] / number_of_hillslopes[asp_ndx],
                        form="trapezoid",
                        mindtnd=ares,
                        nhisto=10,
                    )
                    trap_slope = x["slope"]
                    trap_width = x["width"]
                    trap_area = x["area"]

                    # if quadratic function has a region of positive slope
                    # (possible near the tail, i.e. the large d values)
                    # the implied width will be negative and the trapezoid
                    # approximation will break down, so adjust base width to avoid this issue.

                    if trap_slope < 0:
                        Atri = -(trap_width**2) / (4 * trap_slope)
                        if Atri < trap_area:
                            trap_width = np.sqrt(-4 * trap_slope * trap_area)

                if hillslope_form == "AnnularSection":
                    x = calc_width_parameters(
                        fdtnd[aind],
                        farea[aind] / number_of_hillslopes[asp_ndx],
                        form="annular",
                        mindtnd=ares,
                        nhisto=10,
                    )
                    alpha = x["alpha"]
                    hill_length = x["radius"]
                    Asec = x["area"]
                    Asec = (alpha / 2) * hill_length**2

                    ann_area = np.sum(farea[aind]) / number_of_hillslopes[asp_ndx]
                    # adjust parameters to match actual area per hillslope
                    if Asec < ann_area:
                        eps = 1e-6
                        alpha = 2 * ann_area / (hill_length**2) + eps
                        Asec = (alpha / 2) * hill_length**2

                    ann_alpha[asp_ndx] = alpha
                    ann_hill_length[asp_ndx] = hill_length

                # calculate geomorphic parameters in each bin
                for n in range(nhand_bins):
                    b1 = hand_bin_bounds[n]
                    b2 = hand_bin_bounds[n + 1]
                    hind = np.logical_and(fhand[aind] >= b1, fhand[aind] < b2)
                    cind = np.where(hind)[0]

                    if cind.size > 0:
                        cind = aind[cind]
                        if np.mean(fhand[cind]) <= 0:
                            info(n, " all hand data are zero ")
                            continue

                        hand[asp_ndx * nhand_bins + n] = np.mean(fhand[cind])
                        # median distance
                        dtnd_sorted = np.sort(fdtnd[cind])
                        dtnd[asp_ndx * nhand_bins + n] = dtnd_sorted[
                            int(0.5 * dtnd_sorted.size - 1)
                        ]
                        area[asp_ndx * nhand_bins + n] = np.sum(farea[cind])
                        # exclude nans from calculation of mean slope
                        tmp = fslope[cind]
                        slope[asp_ndx * nhand_bins + n] = np.mean(tmp[np.isfinite(tmp)])

                        if hillslope_form == "Trapezoidal":
                            # preserve relative areas
                            area_fraction = np.sum(farea[cind]) / np.sum(farea[aind])
                            area[asp_ndx * nhand_bins + n] = trap_area * area_fraction

                            # lower edge widths
                            da = np.sum(
                                area[asp_ndx * nhand_bins : asp_ndx * nhand_bins + n]
                            )
                            le = quadratic([trap_slope, trap_width, -da])
                            we = trap_width + le * trap_slope * 2
                            width[asp_ndx * nhand_bins + n] = we

                            # median distances
                            da = (
                                np.sum(
                                    area[
                                        asp_ndx * nhand_bins : asp_ndx * nhand_bins
                                        + n
                                        + 1
                                    ]
                                )
                                - area[asp_ndx * nhand_bins + n] / 2
                            )
                            ld = quadratic([trap_slope, trap_width, -da])
                            dtnd[asp_ndx * nhand_bins + n] = ld

                        if hillslope_form == "AnnularSection":
                            # preserve relative areas
                            area_fraction = np.sum(farea[cind]) / np.sum(farea[aind])
                            area[asp_ndx * nhand_bins + n] = ann_area * area_fraction

                            # lower edge distances and widths
                            asum = np.sum(
                                area[asp_ndx * nhand_bins : asp_ndx * nhand_bins + n]
                            )
                            ri = np.sqrt((Asec - asum) * (2 / alpha))
                            width[asp_ndx * nhand_bins + n] = alpha * ri

                            # median distances and widths
                            asum = (
                                np.sum(
                                    area[
                                        asp_ndx * nhand_bins : asp_ndx * nhand_bins
                                        + n
                                        + 1
                                    ]
                                )
                                - area[asp_ndx * nhand_bins + n] / 2
                            )
                            ri = np.sqrt((Asec - asum) * (2 / alpha))
                            dtnd[asp_ndx * nhand_bins + n] = hill_length - ri

                        """
                        aspect needs to be averaged using circular 
                        (vector) mean rather than arithmatic mean
                        (to avoid cases of e.g. mean([355,5])->180, 
                        when it should be 0)
                        """

                        mean_aspect = (
                            np.arctan2(
                                np.mean(np.sin(dtr * faspect[cind])),
                                np.mean(np.cos(dtr * faspect[cind])),
                            )
                            / dtr
                        )
                        if mean_aspect < 0:
                            mean_aspect += 360.0
                        aspect[asp_ndx * nhand_bins + n] = mean_aspect

                        if not np.isfinite(mean_aspect):
                            warning("bad aspect: ", lon2d[j, i], lat2d[j, i], mean_aspect)

                        hillslope_index[asp_ndx * nhand_bins + n] = asp_ndx + 1
                        column_index[asp_ndx * nhand_bins + n] = col_cnt
                        if n == 0:
                            downhill_column_index[
                                asp_ndx * nhand_bins + n
                            ] = lowland_index
                        else:
                            downhill_column_index[asp_ndx * nhand_bins + n] = (
                                col_cnt - 1
                            )
                        col_cnt += 1

                        debug(
                                "chk h/d/a: ",
                                n,
                                hand[asp_ndx * nhand_bins + n],
                                dtnd[asp_ndx * nhand_bins + n],
                                area[asp_ndx * nhand_bins + n],
                            )

                # identify lowland column with first nonzero column index
                for n in range(nhand_bins):
                    if column_index[asp_ndx * nhand_bins + n] > 0:
                        if (
                            downhill_column_index[asp_ndx * nhand_bins + n]
                            > lowland_index
                        ):
                            downhill_column_index[
                                asp_ndx * nhand_bins + n
                            ] = lowland_index
                        break

            if printData:
                info("\n---------  final values aspect", asp_ndx + 1, "----------")
                info("area_all_columns ", np.sum(area[: ind.size]), np.sum(area[:]))
                info(
                    "height ",
                    hand[asp_ndx * nhand_bins : asp_ndx * nhand_bins + nhand_bins],
                )
                info(
                    "width ",
                    width[asp_ndx * nhand_bins : asp_ndx * nhand_bins + nhand_bins],
                )
                info(
                    "distance ",
                    dtnd[asp_ndx * nhand_bins : asp_ndx * nhand_bins + nhand_bins],
                )
                info(
                    "area ",
                    area[asp_ndx * nhand_bins : asp_ndx * nhand_bins + nhand_bins],
                )
                info(
                    "colndx ",
                    column_index[
                        asp_ndx * nhand_bins : asp_ndx * nhand_bins + nhand_bins
                    ],
                )
                info("max distance ", dtnd[asp_ndx * nhand_bins + nhand_bins - 1])
                info("")

        # --  Compress data  ----------------------
        ind = np.where(column_index[:] > 0)[0]
        if printData:
            info("\ncompressing data")
            info("hand ", hand[ind])
            info("dtnd ", dtnd[ind])
            info("slope ", slope[ind])
            info("col_ndx ", column_index[ind])
            info("dcol_ndx ", downhill_column_index[ind])
        nhillcolumns = ind.size
        hand[: ind.size] = hand[ind]
        dtnd[: ind.size] = dtnd[ind]
        area[: ind.size] = area[ind]
        slope[: ind.size] = slope[ind]
        aspect[: ind.size] = aspect[ind]
        width[: ind.size] = width[ind]
        # zbedrock[:ind.size]  = zbedrock[ind]
        hillslope_index[: ind.size] = hillslope_index[ind]
        column_index[: ind.size] = column_index[ind]
        downhill_column_index[: ind.size] = downhill_column_index[ind]

        harea = area[: ind.size]
        area_all_columns = np.sum(harea)
        hndx = hillslope_index[: ind.size]
        if area_all_columns > 0:
            for n in range(naspect):
                area_hillslope = np.sum(harea[hndx == (n + 1)])
                pct_hillslope[n] = 100 * (area_hillslope / area_all_columns)

        # set unused portion of ncolumns to zero
        if ind.size < ncolumns_per_gridcell:
            hand[ind.size :] = 0
            dtnd[ind.size :] = 0
            area[ind.size :] = 0
            slope[ind.size :] = 0
            aspect[ind.size :] = 0
            width[ind.size :] = 0
            # zbedrock[ind.size:] = 0
            hillslope_index[ind.size :] = 0
            column_index[ind.size :] = 0
            downhill_column_index[ind.size :] = 0

        # --  Remove hillslope data if not enough aspects represented
        if nhillcolumns > 0:
            # check number of hillslopes
            h_ndx = hillslope_index[:]
            nactual_hillslopes = np.unique(h_ndx[h_ndx > 0]).size
            min_number_of_hillslopes = 3
            if nactual_hillslopes < min_number_of_hillslopes:
                info("\nremoving hillslope parameters")
                debug("number of hillslopes ", nactual_hillslopes)
                debug(lon2d[j, i], lat2d[j, i])
                nhillcolumns = 0
                pct_hillslope[:] = 0
                hand[:] = 0
                dtnd[:] = 0
                area[:] = 0
                slope[:] = 0
                aspect[:] = 0
                width[:] = 0
                # zbedrock[:] = 0
                hillslope_index[:] = 0
                column_index[:] = 0
                downhill_column_index[:] = 0

        # use sectional hillslope models
        if hillslope_form in ["CircularSection", "TriangularSection"]:

            # calculate new distances and widths after initial n loop
            # sections of circular shaped hillslopes
            # grid_area is the larger grid used to
            # create catchments and stream network
            # farea is the area of the actual gridcell
            # and is smaller than grid_area
            # width will be width at lower edge

            new_num_hillslopes = 0

            mean_hill_length = np.zeros((naspect))
            for asp_ndx in range(naspect):
                debug("Calculating hillslope distance/width for aspect ", asp_ndx)

                aind = np.where(hillslope_index[:] == (asp_ndx + 1))[0]
                if aind.size > 0:

                    x = CalcRepresentativeHillslopeForm(
                        hillslope_fraction[asp_ndx],
                        column_index[aind],
                        area[aind],
                        dtnd[aind],
                        form=hillslope_form,
                        maxHillslopeLength=maxHillslopeLength,
                    )
                    nvbins = x["n_valid_bins"]
                    dtnd[aind[0:nvbins]] = x["node_distance"]
                    width[aind[0:nvbins]] = x["edge_width"]
                    area[aind[0:nvbins]] = x["area"]
                    mean_hill_length[asp_ndx] = x["hill_length"]
                    debug("column areas ", area[aind])
                    debug("total column area ", np.sum(area[aind]))
                    debug("lowland width ", width[aind[0]])
                    debug(
                            asp_ndx + 1,
                            " approximate number of hillslopes ",
                            hillslope_fraction[asp_ndx]
                            * np.sum(farea[np.isfinite(fhand)])
                            / np.sum(area[aind[0:nvbins]]),
                            "\n",
                        )

                    new_num_hillslopes += (
                        hillslope_fraction[asp_ndx]
                        * np.sum(farea[np.isfinite(fhand)])
                        / np.sum(area[aind[0:nvbins]])
                    )

            # DEBUGGING OUTPUT
            # compare Agrc/nstreams eff. radius to mean hill length
            # debug('mean_hill_lengths ',mean_hill_length)
            debug("mean_hill_length ", np.mean(mean_hill_length))
            # in this model, 4 catchments make up a feature
            debug(
                "Total area eff rad ",
                np.sqrt(
                    4 * np.sum(farea[np.isfinite(fhand)]) / stream_number / np.pi
                ),
            )
            debug(
                "area per stream ",
                np.sum(farea[np.isfinite(fhand)]) / stream_number,
            )
            debug("mean hillslope area ", 0.25 * np.sum(area))

        # Calculate stream geometry from hillslope parameters
        adepth, bdepth = 1e-3, 0.4
        awidth, bwidth = 1e-3, 0.6
        uharea = np.sum(area[:])
        wdepth = adepth * (uharea**bdepth)
        wwidth = awidth * (uharea**bwidth)
        wslope = mean_network_slope
        debug("\nstream channel: width, depth, slope")
        debug(wwidth, wdepth, wslope, "\n")

        # Write data to file
        if not printData:
            command = 'date "+%y%m%d"'
            timetag = (
                subprocess.Popen(command, stdout=subprocess.PIPE, shell="True")
                .communicate()[0]
                .strip()
                .decode()
            )

            w = netcdf4.Dataset(outfile, "w")
            info(f"outfile: {outfile}")
            w.creation_date = timetag

            w.createDimension("lsmlon", 1)
            w.createDimension("lsmlat", 1)
            w.createDimension("nhillslope", naspect)
            w.createDimension("nmaxhillcol", ncolumns_per_gridcell)

            olon = w.createVariable("longitude", float, ("lsmlon",))
            olon.units = "degrees"
            olon.long_name = "longitude"

            olat = w.createVariable("latitude", float, ("lsmlat",))
            olat.units = "degrees"
            olat.long_name = "latitude"

            olon2d = w.createVariable(
                "LONGXY",
                float,
                (
                    "lsmlat",
                    "lsmlon",
                ),
            )
            olon2d.units = "degrees"
            olon2d.long_name = "longitude - 2d"

            olat2d = w.createVariable(
                "LATIXY",
                float,
                (
                    "lsmlat",
                    "lsmlon",
                ),
            )
            olat2d.units = "degrees"
            olat2d.long_name = "latitude - 2d"

            ohand = w.createVariable(
                "hillslope_elevation",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            ohand.units = "m"
            ohand.long_name = "hillslope elevation above channel"

            odtnd = w.createVariable(
                "hillslope_distance",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            odtnd.units = "m"
            odtnd.long_name = "hillslope distance from channel"

            owidth = w.createVariable(
                "hillslope_width",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            owidth.units = "m"
            owidth.long_name = "hillslope width"

            oarea = w.createVariable(
                "hillslope_area",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            oarea.units = "m2"
            oarea.long_name = "hillslope area"

            oslop = w.createVariable(
                "hillslope_slope",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            oslop.units = "m/m"
            oslop.long_name = "hillslope slope"

            oasp = w.createVariable(
                "hillslope_aspect",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            oasp.units = "radians"
            oasp.long_name = "hillslope aspect (clockwise from North)"

            obed = w.createVariable(
                "hillslope_bedrock_depth",
                np.float64,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            obed.units = "meters"
            obed.long_name = "hillslope bedrock depth"

            onhill = w.createVariable(
                "nhillcolumns",
                np.int32,
                (
                    "lsmlat",
                    "lsmlon",
                ),
            )
            onhill.units = "unitless"
            onhill.long_name = "number of columns per landunit"

            opcthill = w.createVariable(
                "pct_hillslope",
                np.float64,
                (
                    "nhillslope",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            opcthill.units = "per cent"
            opcthill.long_name = "percent hillslope of landunit"

            ohillndx = w.createVariable(
                "hillslope_index",
                np.int32,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            ohillndx.units = "unitless"
            ohillndx.long_name = "hillslope_index"

            ocolndx = w.createVariable(
                "column_index",
                np.int32,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            ocolndx.units = "unitless"
            ocolndx.long_name = "column index"

            odcolndx = w.createVariable(
                "downhill_column_index",
                np.int32,
                (
                    "nmaxhillcol",
                    "lsmlat",
                    "lsmlon",
                ),
            )
            odcolndx.units = "unitless"
            odcolndx.long_name = "downhill column index"

            ocmask = w.createVariable(
                "chunk_mask",
                np.int32,
                (
                    "lsmlat",
                    "lsmlon",
                ),
            )
            ocmask.units = "unitless"
            ocmask.long_name = "chunk mask"

            olon[
                :,
            ] = lon2d[j, i]
            olat[
                :,
            ] = lat2d[j, i]
            olon2d[
                :,
            ] = lon2d[j, i]
            olat2d[
                :,
            ] = lat2d[j, i]

            ohand[
                :,
            ] = hand
            odtnd[
                :,
            ] = dtnd
            oarea[
                :,
            ] = area
            owidth[
                :,
            ] = width
            oslop[
                :,
            ] = slope
            obed[
                :,
            ] = zbedrock
            # aspect should be in radians on surface data file
            oasp[:,] = (
                aspect * dtr
            )
            opcthill[
                :,
            ] = pct_hillslope
            onhill[
                :,
            ] = np.int32(nhillcolumns)
            ohillndx[
                :,
            ] = hillslope_index.astype(np.int32)
            ocolndx[
                :,
            ] = column_index.astype(np.int32)
            odcolndx[
                :,
            ] = downhill_column_index.astype(np.int32)
            ocmask[
                :,
            ] = np.int32(chunk_mask)

            if addStreamChannelVariables:
                wdims = w["LONGXY"].dimensions
                odepth = w.createVariable("hillslope_stream_depth", float, wdims)
                owidth = w.createVariable("hillslope_stream_width", float, wdims)
                oslope = w.createVariable("hillslope_stream_slope", float, wdims)

                odepth.long_name = "stream channel bankfull depth"
                odepth.units = "m"

                owidth.long_name = "stream channel bankfull width"
                owidth.units = "m"

                oslope.long_name = "stream channel slope"
                oslope.units = "m/m"

                odepth[
                    :,
                ] = wdepth
                owidth[
                    :,
                ] = wwidth
                oslope[
                    :,
                ] = wslope

                w.close()
                debug(outfile + " created")

    etime = time.time()
    debug("time calc_geoparams ", etime - stime, " s")
    return


# Class definitions


class LandscapeCharacteristics(object):
    """
    Container class for landscape terrain characteristics derived from
    digital elevation model.

    Attributes
    ==========
    Scalars
    thresh:   accumulation threshold used to calculate stream network
    nreach:   number of channel reaches
    nstreams: number of streams (comprised of one or more reaches)

    Arrays
    hand:          height above nearest drainage [m]
    dtnd:          distance to nearest drainage  [m]
    area:          area of pixel [m2]
    slope:         mean slope of pixel [m/m]
    aspect:        direction of pixel face with respect to north [radians]
    drainage_id:   identifier for individual catchments
    fflood:        mask for areas of dem that are 'flooded'

    Methods
    =======
        ---------------
        Data Processing
        ---------------
        CalcLandscapeCharacteristicsPysheds
        Taking a DEM as input, use Pysheds to calculate stream
        network and derived quantities describing landscape
        characteristics

    """

    def __init__(self):

        # initialize arrays
        self.nreach = None
        self.nstreams = None
        self.thresh = None
        self.hand = None
        self.dtnd = None
        self.area = None
        self.width = None
        self.slope = None
        self.aspect = None
        self.drainage_id = None
        self.fflood = None

    def CalcLandscapeCharacteristicsPysheds(
        self,
        corners,
        accum_thresh=0,
        dem_file_template=None,
        fill_value=-9999,
        useConsistentChannelMask=True,
        useMultiProcessing=True,
        npools=4,
        dem_source="MERIT",
        maskFlooded=True,
        pshape=None,
    ):

        if accum_thresh == 0:
            raise RuntimeError("accumulation threshold must be > 0")
        if type(dem_file_template) == type(None):
            raise RuntimeError("no dem file template supplied")

        if dem_source == "MERIT":
            x = read_MERIT_dem_data(dem_file_template, corners, zeroFill=True)
        if dem_source == "ASTER":
            x = read_ASTER_dem_data(dem_file_template, corners, zeroFill=True)

        if not x["validDEM"]:
            return -1

        elev, elon, elat, ecrs, eaffine = (
            x["elev"],
            x["lon"],
            x["lat"],
            x["crs"],
            x["affine"],
        )
        ejm, eim = elev.shape

        self.dem = np.copy(elev)

        if type(pshape) != type(None):
            value = 1
            pmask = rasterio.features.rasterize(
                [(pshape, value)],
                out_shape=(elev.shape),
                fill=0,
                out=None,
                transform=eaffine,
            )
            elev[pmask == 0] = fill_value

        # mask out large regions of zero elevation
        # use eps to catch roundoff values and ignore land areas below sea level
        eps = 1e-6
        land_fraction = np.sum(np.where(np.abs(elev) > eps, 1, 0)) / elev.size
        min_land_fraction = 0.01
        if land_fraction <= min_land_fraction:
            warning("skipping; land fraction too small ", land_fraction)
            return -1

        # lf_thresh = 0.75
        # if land_fraction <= lf_thresh:
        #    elev[np.abs(elev) < eps] = fill_value
        basin_mask = identify_basins(elev)
        elev[basin_mask > 0] = fill_value

        # ---  Create pysheds Grid object  -----------------------------
        grid = Grid.from_array(
            data=elev,
            data_name="dem",
            affine=eaffine,
            shape=elev.shape,
            crs=ecrs,
            nodata=fill_value,
            metadata={},
        )

        # ---  Calculate geographic coordinates  -----------------------
        x = grid.affine
        debug("grid affine ", x.a, x.b, x.c, x.d, x.e, x.f)
        x0, y0, dx, dy = x.c, x.f, x.a, x.e
        ys, xs = grid.shape
        # lon/lat will be center of pixel
        lon = (x0 + 0.5 * dx) + dx * np.arange(xs)
        lon[lon > 360] -= 360
        lat = (y0 + 0.5 * dy) + dy * np.arange(ys)
        jm, im = lat.size, lon.size

        # ---  Fill depressions and resolve flats in the DEM  --------------
        grid.fill_depressions("dem", out_name="flooded_dem", nodata_in=fill_value)

        debug("depressions filled")

        # Ignore dems with only one point
        s1 = np.sum(np.where(grid.dem > 0, 1, 0))
        s2 = np.sum(np.where(grid.flooded_dem > 0, 1, 0))
        if np.logical_or(s1 <= 1, s2 <= 1):
            info("no dem, no flooded ", s1, s2)
            info("skipping")
            return -1

        # Resolve flats in DEM
        try:
            grid.resolve_flats(
                "flooded_dem", out_name="inflated_dem", nodata_in=fill_value
            )
            debug("flats resolved")
        except ValueError:
            warning("flats cannot be resolved")
            grid.add_gridded_data(grid.dem, "inflated_dem", nodata=fill_value)

        # Set flat areas to fill_value
        # identify flooded regions in lowest hand bin
        fflood = np.abs(np.asarray(grid.flooded_dem - grid.dem))
        num_flooded_pts = np.sum((fflood > 0))
        flat_mask = np.zeros(grid.dem.shape)
        if maskFlooded:
            if num_flooded_pts > 0:
                debug("total flooded fraction ", num_flooded_pts / grid.dem.size)
                # determine threshold for cells to be excluded
                flood_thresh = 0
                # fraction of flood mask to remove
                ffraction = 0.95
                # when frac_below_ft is greater than ffraction, save value
                for ft in np.linspace(0, 20, 50):
                    frac_below_ft = (
                        np.sum((np.abs(fflood[fflood > 0]) < ft)) / num_flooded_pts
                    )
                    if frac_below_ft > ffraction:
                        flood_thresh = ft
                        break
                debug("flood threshold ", flood_thresh)
                # exclude regions that have been flooded
                flat_mask = np.where(np.abs(fflood) > flood_thresh, 1, 0)
            else:
                debug("no flooded points")
                pass

        grid.dem[flat_mask > 0] = fill_value
        grid.flooded_dem[flat_mask > 0] = fill_value
        grid.inflated_dem[flat_mask > 0] = fill_value

        # ---  Define directional map  ---------------------------------

        """ Specify directional mapping, i.e. a list of integer values representing the following 
        cardinal and intercardinal directions (in order): [N, NE, E, SE, S, SW, W, NW]"""
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        dirnames = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        # ---  Compute flow directions from DEM
        # extract_profiles cannot handle missing data, so do not use nodata_out
        grid.flowdir(
            data="inflated_dem", out_name="dir", dirmap=dirmap, nodata_in=fill_value
        )
        debug("flow directions computed")

        # ---  Calculate flow accumulation using direction map

        # Calculate flow accumulation (input data is flow direction) #
        grid.accumulation(data="dir", dirmap=dirmap, out_name="acc")
        debug("accumulation calculated")

        grid.slope_aspect(dem="dem")
        debug("slope/aspect calculated")

        # --- check for flat areas (e.g. lakes) that were not considered flooded
        slope_threshold = 1e-5
        fflood = np.where(grid.view("slope") < slope_threshold, 1, 0)

        # ---  Calculate stream network
        dir_raster = grid.view("dir")
        acc_raster = grid.view("acc")

        # adjust threshold if max(accumulation) is smaller than accum_thresh
        if np.max(grid.acc) > accum_thresh:
            self.thresh = accum_thresh
        else:
            self.thresh = np.max(grid.acc) / 100
            debug("new thresh: ", self.thresh)

        # create mask of accumulation values above thresh, where dem is valid
        acc_mask = np.logical_and(
            (acc_raster > self.thresh), (grid.inflated_dem != fill_value)
        )

        try:
            branches = grid.extract_river_network(
                fdir=dir_raster, mask=acc_mask, dirmap=dirmap
            )
        except MemoryError:
            warning("Memory Error in extract_river_network, skipping")
            return -1

        network = branches[
            "features"
        ]  # list of features (<class 'geojson.feature.Feature'>)
        debug("stream network calculated")

        # ---  Create stream network id/coordinate arrays  --------------
        nstreams = len(network)
        debug("nstreams ", nstreams)
        nreach = 0
        for n in range(nstreams):
            nreach += len(network[n]["geometry"]["coordinates"])

        branch_id = np.zeros((nreach))
        branch_xy = np.zeros((nreach, 2))
        n = 0
        for branch in network:
            for pt in branch["geometry"]["coordinates"]:
                branch_id[n] = branch["id"]
                branch_xy[n] = pt
                n += 1

        # stream ids of each channel reach
        network = branch_xy
        stream_id = branch_id
        ustream_id = np.unique(stream_id).astype(int)
        self.nreach = stream_id.size
        self.nstreams = ustream_id.size

        self.network = network
        self.network_id = stream_id

        # determine direction of latitude coordinates
        if (lat[1] - lat[0]) > 0:
            # S -> N
            latdir = "south_to_north"
        else:
            # N -> S
            latdir = "north_to_south"

        try:
            x = grid.river_network_length_and_slope(
                fdir=dir_raster, mask=acc_mask, dirmap=dirmap
            )
        except MemoryError:
            warning("Memory Error in river_network_length_and_slope, skipping")
            return -1

        self.network_length = x["length"]
        self.mean_network_slope = x["slope"]

        # return all reach slopes and coordinates
        self.reach_slopes = x["reach_slopes"]
        self.reach_lengths = x["reach_lengths"]
        self.reach_lon = x["mlon"]
        self.reach_lat = x["mlat"]

        debug("\nnetwork length ", self.network_length)

        # add stream channel mask and initial stream channel id to grid object
        if useConsistentChannelMask:
            grid.create_channel_mask(fdir=dir_raster, mask=acc_mask, dirmap=dirmap)

            debug("channel mask and id created")

        else:
            # use channel network to define mask
            dmask = np.zeros((jm, im))
            # initialize drainage id of each pixel in the stream network and create a mask
            did = np.zeros((jm, im))
            # use clipped stream network
            for n in range(nreach):
                ni = np.argmin(np.abs(network[n, 0] - lon))
                nj = np.argmin(np.abs(network[n, 1] - lat))
                dmask[nj, ni] = 1
                did[nj, ni] = stream_id[n]

            grid.add_gridded_data(
                np.asarray(dmask), "channel_mask", affine=grid.affine, crs=grid.crs
            )
            grid.add_gridded_data(
                np.asarray(did), "channel_id", affine=grid.affine, crs=grid.crs
            )

        # calculate hand and new methods (distance to nearest drainage "dtnd" and angle wrt nearest drainage)
        # the compute_hand method also assigns a drainage id to all pixels
        # default nodata_out = NaN

        grid.compute_hand(
            fdir="dir",
            dem="inflated_dem",
            channel_mask="channel_mask",
            channel_id="channel_id",
            out_name="hand",
            dirmap=dirmap,
            nodata_in_dem=fill_value,
        )

        self.channel_mask = grid.channel_mask

        debug("hand calculated")

        # self.dem    = np.asarray(grid.view('dem'))
        self.hand = np.asarray(grid.view("hand"))
        self.dtnd = np.asarray(grid.view("dtnd"))
        self.aspect = np.asarray(grid.view("aspect"))
        self.slope = np.asarray(grid.view("slope"))
        self.aznd = np.asarray(grid.view("aznd"))
        self.drainage_id = np.asarray(grid.view("drainage_id"))
        self.fflood = fflood
        lon[lon >= 360] -= 360
        self.lon = lon
        self.lat = lat

        # calculate area
        farea = np.zeros((jm, im))
        phi = dtr * lon
        th = dtr * (90.0 - lat)
        dphi = np.abs(phi[1] - phi[0])
        dth = np.abs(th[0] - th[1])
        farea = np.tile(np.sin(th), (im, 1)).T
        self.area = farea * dth * dphi * np.power(re, 2)

        # hillslopes will be 1:headwater, 2:right bank, 3: left bank, 4: channel
        grid.compute_hillslope(
            fdir="dir", channel_mask="channel_mask", bank_mask="bank_mask"
        )
        self.hillslope = np.asarray(grid.view("hillslope"))

        # create hillslope masks
        hillmask = []
        for k1 in range(1, 4):
            hillmask.append(
                np.where(
                    np.logical_or(self.hillslope == 4, self.hillslope == k1),
                    self.drainage_id,
                    0,
                )
            )

        # convert aspect to hillslope mean values
        debug("averaging aspect across catchments")
        aspect2d_catchment_mean = np.zeros((jm, im))
        uid = np.unique(self.drainage_id[np.isfinite(self.drainage_id)])
        if useMultiProcessing:
            # parallel version
            stime = time.time()

            aspect2d_catchment_mean = set_aspect_to_hillslope_mean_parallel(
                self.drainage_id, self.aspect, self.hillslope, npools=npools
            )
            etime = time.time()
            debug(
                "\nTime to complete set_aspect_to_hillslope_mean_parallel: {:.3f} seconds".format(
                    etime - stime
                )
            )

        else:
            # serial version
            stime = time.time()
            aspect2d_catchment_mean = set_aspect_to_hillslope_mean_serial(
                self.drainage_id, self.aspect, self.hillslope
            )
            etime = time.time()
            debug(
                "\nTime to complete set_aspect_to_hillslope_mean_serial: {:.3f} seconds".format(
                    etime - stime
                )
            )

        # set aspect to catchment averaged values
        self.aspect = aspect2d_catchment_mean
        debug("aspect averaged over catchments")

        return 0
