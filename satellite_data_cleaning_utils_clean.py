import gdal_images
import glob
import numpy as np
import re
import os
import copy
from natsort import natsorted
import pandas as pd
import scipy
import scipy.stats.stats
import time
import random
import gdal
gdal.UseExceptions()
import warnings

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning


def normalize_all_data(imList, 
                       outPath, 
                       negativeNan=[16,44]):
    """
    Normalizes data from 0 to 1. Writes files out.

    Args:
        imList: list of file paths/names
        outPath: folder to write out normalized files
        negativeNan: to determine where negatives represent NAN and should be 
            excluded, NOT bands that have valid negatives (e.g., temperature) 
            that should just be normalized -- below first band index and 
            above second band index are considered nans

    Returns:
        outputs: list of output image file paths (imList)
    """
    #https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    dataset = gdal.Open(imList[0])
    nB = dataset.RasterCount

    # Get min and max for band over all images.
    minInBands = np.zeros((len(imList), nB))
    maxInBands = np.zeros((len(imList), nB))
    for i, imPath in enumerate(imList):
        dataset, imdata, bandNames, nC, nR, nB, xOrigin, yOrigin, pixelWidth, \
            pixelHeight = gdal_images.read_gdal_image(imPath)

        # Replace invalid negatives with nan's.
        whereNegative = np.where(imdata[:,:,:negativeNan[0]] < 0)
        imdata[whereNegative] = np.nan
        whereNegative = np.where(imdata[:,:,negativeNan[1]:] < 0)
        imdata[whereNegative[0], 
               whereNegative[1], 
               whereNegative[2]+negativeNan[1]] = np.nan

        # Find min/max (excluding nan's). 
        for band in range(nB):
            minInBands[i, band] = np.nanmin(imdata[:,:,band])
            maxInBands[i, band] = np.nanmax(imdata[:,:,band])

    minimumBands = np.nanmin(minInBands, axis=0)
    maximumBands = np.nanmax(maxInBands, axis=0)

    # Normalize.
    outputs = []
    for imPath in imList:
        index = int(re.search(r'\d+',
                              os.path.basename(imPath).split('.')[0]).group())

        # Read in original
        dataset, imdata, bandNames, nC, nR, nB, xOrigin, yOrigin, pixelWidth, \
            pixelHeight = gdal_images.read_gdal_image(imPath)

        whereNegative = np.where(imdata[:,:,:negativeNan[0]] < 0)
        imdata[whereNegative] = np.nan
        whereNegative = np.where(imdata[:,:,negativeNan[1]:] < 0)
        imdata[whereNegative[0], 
               whereNegative[1], 
               whereNegative[2]+negativeNan[1]] = np.nan

        imdataPrime = np.zeros((imdata.shape[0], 
                                imdata.shape[1], 
                                imdata.shape[2]))
        for band in range(nB):
            imdataPrime[:,:,band] = (imdata[:,:,band] - minimumBands[band]) / \
                                    (maximumBands[band]- minimumBands[band])
        
        outname = outPath+str(index)+'.tif'
        outputs.append(outname)
        gdal_images.write_gdal_image(outname, 
                                     dataset, 
                                     imdataPrime, 
                                     bandNames, 
                                     nC, 
                                     nR, 
                                     nB, 
                                     xOrigin, 
                                     yOrigin, 
                                     pixelWidth, 
                                     pixelHeight)
    return outputs

def imputation(imList, 
               imputedPath, 
               imputationMode='domain', 
               zeroBands=list(np.arange(5)) + list(np.arange(7,9)) + \
                   list(np.arange(14,16)) + list(np.arange(44,90)), 
               verbose=True):
    """
    Impute nan's. Write out imputed images.

    Args:
        imList: list of file paths/names
        imputedPath: folder to write out normalized files
        imputationMode: domain (fill with 0 or nearest neighbor) or mean (fill 
            with nanmean)
        zeroBands: crop/animal yield/population and can be assumed 0 if nan
        verbose: prints details out if True

    Returns:
        outputs: list of output image file paths (imList)
        bandNames: names of features
    """
    outputs = []
    for imPath in imList:
        index = int(re.search(r'\d+', 
                              os.path.basename(imPath).split('.')[0]).group())
        if verbose:
            print('image index', index)

        # Read
        dataset, imdata, bandNames, nC, nR, nB, xOrigin, yOrigin, pixelWidth, \
            pixelHeight = gdal_images.read_gdal_image(imPath)

        # Determine whether to impute (only impute if there is missing data)
        whereNan = np.where(np.isnan(imdata))
        if whereNan[0].shape[0] > 0:
            if verbose:
                print('imputation needed for im ', index)
            imdataPrime = copy.deepcopy(imdata)
            nanBands = np.unique(whereNan[2])
            if verbose:
                print(nanBands)

            # Impute by bands
            for nanBand in nanBands:
                if verbose:
                    print('band: ', nanBand)
                
                # Find where nan in band
                whereNanInBand = np.where(np.isnan(imdata[:, :, nanBand]))

                # If no data in that band (i.e., all nan), set equal to 0
                if len(whereNanInBand[0]) == imdata.shape[0] * imdata.shape[1]:
                    print('no data') #always notify of this
                    imdataPrime[whereNanInBand[0], 
                                whereNanInBand[1], 
                                nanBand] = 0

                # If data, proceed with imputation
                else:
                    if imputationMode == 'mean':
                        if verbose:
                            print('mean')
                        # Take mean of whole band, based on only this image
                        imdataPrime[whereNanInBand[0], 
                                    whereNanInBand[1], 
                                    nanBand] = \
                                        np.nanmean(imdata[:, :, nanBand])

                    else:
                        # Set zero bands to 0
                        if nanBand in zeroBands:
                            if verbose:
                                print('setting to 0')
                            imdataPrime[whereNanInBand[0], 
                                        whereNanInBand[1], 
                                        nanBand] = 0
                        
                        # Impute by linear (since we might not have a nearest 
                        # non-nan neighbor)
                        #https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
                        else:
                            toInterp = imdata[:,:,nanBand].astype(np.float64)
                            colX = np.arange(0, toInterp.shape[1])
                            rowY = np.arange(0, toInterp.shape[0])
                            
                            # Mask invalid values
                            array = np.ma.masked_invalid(toInterp)
                            xx, yy = np.meshgrid(colX, rowY)
                            # Get only the valid values
                            x1 = xx[~array.mask]
                            y1 = yy[~array.mask]
                            newarr = array[~array.mask]
                            interpolatedArr = scipy.interpolate.griddata(
                                (x1, y1), 
                                newarr.ravel(),
                                (xx, yy),
                                method='nearest')
                            imdataPrime[:,:,nanBand] = \
                                interpolatedArr.astype(np.float32)

                # Double check this band is good
                whereNanInBand2 = np.where(np.isnan(imdataPrime[:,:,nanBand]))
                if whereNanInBand2[0].shape[0] != 0:
                    print('num nans after interp, should be 0: ', \
                        whereNanInBand2[0].shape[0])

        else:
            # If no imputation, just copy
            imdataPrime = copy.deepcopy(imdata)

        # Write
        outname = imputedPath+str(index)+'.tif'
        outputs.append(outname)
        gdal_images.write_gdal_image(outname, 
                                     dataset, 
                                     imdataPrime, 
                                     bandNames, 
                                     nC, 
                                     nR, 
                                     nB, 
                                     xOrigin, 
                                     yOrigin, 
                                     pixelWidth, 
                                     pixelHeight)
        
    return outputs, bandNames


def find_all_zero_bands(imList):
    """
    Find zero bands throughout all images in imList (e.g., if 0 in all but 1 
    band 5 in provided images, band 5 is kept).

    Args:
        imList: list of ims (e.g., natsorted(glob.glob('path/*.tif')))

    Returns:
        list of bands with all 0 to remove
    """
    dataset = gdal.Open(imList[0])
    nB = dataset.RasterCount

    uniqueInBands = [copy.deepcopy([]) for i in range(nB)]
    for imPath in imList:
        dataset = gdal.Open(imPath)
        nC = dataset.RasterXSize
        nR = dataset.RasterYSize
        nB = dataset.RasterCount
        for band in range(nB):
            data = dataset.GetRasterBand(band+1)
            uniqueInBands[band] += list(np.unique(data.ReadAsArray(0, 
                                                                   0, 
                                                                   nC, 
                                                                   nR)))

    toRemove = []
    for i, listi in enumerate(uniqueInBands):
        listi = np.asarray(listi)
        uniqueValues = list(np.unique(listi[~np.isnan(listi)]))
        if uniqueValues == [0.0]:
            toRemove.append(i)

    return toRemove


def remove_bands_feature_selection(toRemove, 
                                   imList, 
                                   featureSelectedPath):
    """
    Remove provided bands in all provided images, and write out.

    Args:
        toRemove: list of bands to remove
        imList: list of ims (e.g., natsorted(glob.glob('path/*.tif')))
        featureSelectedPath: folder to write out feature selected files
    
    Returns:
        outputs: list of output image file paths (imList)
    """
    outputs = []
    for imPath in imList:
        index = int(re.search(r'\d+', 
                              os.path.basename(imPath).split('.')[0]).group())
        dataset, imdata, bandNames, nC, nR, nB, xOrigin, yOrigin, pixelWidth, \
            pixelHeight = gdal_images.read_gdal_image(imPath)
        
        imdataPrime = np.delete(imdata, toRemove, axis=2)
        reverse = sorted(toRemove)[::-1]
        for i in reverse:
            del bandNames[i]
        nB = imdataPrime.shape[2]

        outname = featureSelectedPath+str(index)+'.tif'
        outputs.append(outname)
        gdal_images.write_gdal_image(outname, 
                                     dataset, 
                                     imdataPrime, 
                                     bandNames, 
                                     nC, 
                                     nR, 
                                     nB, 
                                     xOrigin, 
                                     yOrigin, 
                                     pixelWidth, 
                                     pixelHeight)
    return outputs


def correlation(im1, im2):
    """
    Flatten images & compute Pearson Correlation Coefficient between 2 
    resulting vectors b/c fastest.

    Args:
        im1: 2D np array containing image
        im2: 2D np array containing image

    Returns:
        correlation coefficient between im1 & im2
    """
    
    corr = np.corrcoef(im1.flatten(), im2.flatten())
    return corr[0,1]


def find_correlation(imList, 
                     outcome=None, 
                     labelIndex=0,
                     verbose=False):
    """
    Find correlations.

    Args:
        imList: list of ims (e.g., natsorted(glob.glob('path/*.tif')))
        outcome: labelList if you want to compute correlation between outcomes
          (MND label) and features, or None if you want to do between features
        labelIndex: in which band is the MND (or other desired label) stored in
          labelList ims
        verbose: prints details out if True

    Returns:
        if outcome=None, matrix representing correlation between features in 
        current index; if outcome != None, row representing correlation between
        feature in current index and outcome; NOTE: flips it! to prepare for 
        distance metric, where same will be 0 distance
    """
    dataset = gdal.Open(imList[0])
    nB = dataset.RasterCount

    # Determine whether this is b/w features or b/w outcome & features
    if outcome is not None:
        if verbose:
            print('wrt outcomes')
        overall = np.zeros((1, nB))
    else:
        if verbose:
            print('wrt features')
        overall = np.zeros((nB, nB))

    # Read all images (and outcomes), accumulate data for calculations
    if outcome is not None:
        outList = []
    imAccum = []
    for i, imPath in enumerate(imList):
        index = int(re.search(r'\d+', 
                              os.path.basename(imPath).split('.')[0]).group())
        
        # Read im
        dataset, imdata, bandNames, _, _, nB, _, _, _, _ = \
            gdal_images.read_gdal_image(imPath)
        if outcome is not None:
            _, labeldata, _, _, _, _, _, _, _, _ = \
                gdal_images.read_gdal_image(outcome[i])
            labeldata = labeldata[:, :, labelIndex]
            outList.append(labeldata)

        
        # if index == 11:
        #     imdata = imdata[1:, :, :]
        imAccum.append(imdata)

    # Calculate mean over multiple images, so correlations are b/w overall bands
    meanImData = np.mean(imAccum, axis=0)
    if outcome is not None:
        meanOutData = np.mean(outList, axis=0)

    # Check correlation between each feature
    # Correlation is symmetrical, so only need to do 1 pair once, 
    # don't need to do diagonal
    #https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix
    if outcome is None:
        loop = np.triu_indices(meanImData.shape[2])
        for f, feature in enumerate(loop[0]):
            overall[feature, loop[1][f]] = \
                correlation(meanImData[:,:,feature], 
                            meanImData[:,:,loop[1][f]])
        
        # Fill in full matrix
        i_lower = np.tril_indices(overall.shape[0], -1)
        overall[i_lower] = overall.T[i_lower]

    # Check correlation between feature and label
    else:
        for feature in range(imdata.shape[2]):
            overall[0, feature] = correlation(meanImData[:,:,feature], 
                                              meanOutData)
    
    return (overall * -1) + 1


def group_feature_selection(overall, 
                            K, 
                            max_iter=10000, 
                            init='k-medoids++'):
    """
    Run k-medoids clustering with correlation.

    https://stackoverflow.com/questions/62215324/sklearn-kmedoids-returns-empty-clusters
    https://scikit-learn-extra.readthedocs.io/en/latest/_modules/sklearn_extra/cluster/_k_medoids.html#KMedoids
    https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html

    Args:
        overall: matrix representing correlation between features in current 
          index
        K: # clusters
        max_iter: maximum number of iterations when fitting
        init: medoid initialization method ('random', 'heuristic', or 
          'k-medoids++' [default], see below KMedoids documentation for more 
          details)
        

    Returns:
        groups: list corresponding to each medoid, with inner lists of feature 
          indices in the medoid
        centers: medoid feature indices
    """
    model = KMedoids(n_clusters = K, 
                     metric = 'precomputed', 
                     max_iter = max_iter, 
                     init = init, 
                     random_state = 0).fit(overall)
    labels = model.labels_
    centers = model.medoid_indices_
    
    # For each label, get indices in labels -- these correspond to bands 
    # (e.g., where medoid label is 0 gives bands with label 0)
    groups = []
    for l in np.unique(labels): 
        whereLabels = list(np.where(labels == l)[0])
        groups.append(whereLabels)

    # Return groups.
    return groups, centers
    

def combine_groups(imList, outPath, groups, mode='center', centers=None):
    """
    Combine groups by one of three modes.

    Args:
        imList: list of ims (e.g., natsorted(glob.glob('path/*.tif')))
        outPath: folder to write out feature selected files
        groups: list corresponding to each medoid, with inner lists of feature 
          indices in the medoid
        mode: mean, random, or center to group by mean, by randomly selecting 1
          of the group, or to select the medoid center
        centers: medoid feature indices

    Returns:
        outputs: list of output image file paths (imList)
    """
    outputs = []
    for imPath in imList:
        dataset, imdata, bandNames, nC, nR, nB, xOrigin, yOrigin, pixelWidth, \
            pixelHeight = gdal_images.read_gdal_image(imPath)
        newBandNames = []
        for gi, g in enumerate(groups):
            currNames = [bandNames[x] for x in g]
            comboName = '_'.join(currNames)

            if mode == 'mean':
                if gi == 0:
                    newImData = np.mean(imdata[:,:,g], axis=2)
                else:
                    newImData = np.dstack((newImData, 
                                           np.mean(imdata[:,:,g], axis=2)))
                newBandNames.append(comboName)

            elif mode == 'random':
                random.seed(4)
                toUse = random.choice(g)
                
                # Make sure not to naively choose empty band.
                if np.unique(imdata[:,:,toUse]).shape[0] < 2:
                    random.seed(5)
                    toUse = random.choice(g)
                
                newBandNames.append(bandNames[toUse]+'__'+comboName)
                if gi == 0:
                    newImData = imdata[:,:,toUse]
                else:
                    newImData = np.dstack((newImData, imdata[:,:,toUse]))
                    
            else:
                # Find which center is in this group.
                toUseL = list(set(g) & set(centers))
                if len(toUseL) != 1:
                    raise ValueError('Too many intersections.')
                toUse = toUseL[0]

                newBandNames.append(bandNames[toUse]+'__'+comboName)
                if gi == 0:
                    newImData = imdata[:,:,toUse]
                else:
                    newImData = np.dstack((newImData, imdata[:,:,toUse]))


        index = int(re.search(r'\d+', 
                              os.path.basename(imPath).split('.')[0]).group())
        outname = outPath+str(index)+'.tif'
        outputs.append(outname)
        gdal_images.write_gdal_image(outname, 
                                     dataset, 
                                     newImData, 
                                     newBandNames, 
                                     nC, 
                                     nR, 
                                     len(groups), 
                                     xOrigin, 
                                     yOrigin, 
                                     pixelWidth, 
                                     pixelHeight)
    return outputs


def prepare_lr_data(imList, labelList, labelIndex=0, regionIndex=1):
    """
    FOR ONE METHOD & MND: Remove pixels without data, flatten so features are 
    in columns, pixels with data in rows, combine with labels in last columns,
    in order to make it easy to run logistic regression and other 
    training/prediction.

    NOTE: assumes labelIndex == 0 when saving final lrdata's, please edit if 
    not true! See NOTE below, inline.

    Args:
        imList: list of ims (e.g., natsorted(glob.glob('path/*.tif')))
        labelList: list of label ims, same format as imList
        labelIndex: in which band is the MND (or other desired label) stored in
          labelList ims [if not 0, please read NOTE above and below]
        regionIndex: in which band is the region index stored in labelList ims

    Returns:
        lrdataList: list of final data array for each region
        bandNames: column labels
        reportedRegions: indices of regions, in order
    """
    # Determine size of lrdata -- num features + region, label, im
    dataset = gdal.Open(imList[0])
    nB = dataset.RasterCount
    dataset = gdal.Open(labelList[0])
    nL = dataset.RasterCount
    lrdata = np.zeros((1, nB+1+nL))  # +1 for classification

    for i, im in enumerate(imList):
        # Read in both image and label.
        dataset, labelData, _, _, _, _, _, _, _, _ = \
            gdal_images.read_gdal_image(labelList[i])  # should be mnd
        dataset, imdata, bandNames, _, _, nB, _, _, _, _ = \
            gdal_images.read_gdal_image(im)

        sliceImData = np.where(labelData[:,:,labelIndex] > 0)
        # - 1 because labels add 1 to denote where there is data, in order 
        # to include negative samples
        labelData[:,:,labelIndex] = labelData[:,:,labelIndex] - 1
        numGTZero = sliceImData[0].shape[0]
        if numGTZero > 0:
            sImData = imdata[sliceImData].reshape(numGTZero, -1)
            sliceLabelData = labelData[sliceImData].reshape(numGTZero, -1)
            sliceClassData = np.where(sliceLabelData[:,labelIndex] > 0, 
                                      1, 
                                      0).reshape(numGTZero, 1)
            toAdd = np.hstack((sImData, sliceClassData, sliceLabelData))
            lrdata = np.vstack((lrdata, toAdd))

    # Split by region.
    # NOTE: [1:] to remove initial row of zeros, +1 for classification, 
    # :len(bandNames)+2 because assuming everything after class & reg columns 
    # is unnecessary, so just saving class & reg
    lrdataList = []
    lrdata = lrdata[1:] 
    regions = lrdata[:, len(bandNames)+1+regionIndex]
    reportedRegions = [] 
    for r in np.unique(regions):
        reportedRegions.append(r)
        whereRegion = lrdata[regions == r]
        if labelIndex != 0:
            print('WARNING: the below line of code may fail')
        lrdataList.append(whereRegion[:, :len(bandNames)+2])

    return lrdataList, bandNames+['Classification', 'Regression'], reportedRegions


def run_satellite_cleaning(fullImList, 
                           outPath, 
                           labelLists, 
                           regionDict={2: 'SE', 3: 'SW', 4: 'WCO', 5: 'CP'}, 
                           runExpert=None):
    """
    Run all of the above.

    Args:
        fullImList: list of initial input ims (e.g., glob.glob)
        outPath: folder in which to save all intermediate files
        labelLists: list of folders containing MND labels
        regionDict: dictionary mapping region indices to strings
        runExpert: leave this as None if simply running the pipeline and not 
          comparing; if comparing with expert selection, should be list in 
          which first element is path to CSV file containing expert selections,
          second element is indices of expert center choices
    """
    # Normalize.
    rawImList = normalize_all_data(fullImList, outPath+'all_rasters_norm/')

    # Imputation.
    imputedImList, bandNames = imputation(rawImList, 
                                          outPath+'final_imgs_imputed_domain/',
                                          imputationMode='domain')

    # Feature selection (remove zeros, then k-medoids, then center).
    toRemove = find_all_zero_bands(imputedImList)
    fsImList = remove_bands_feature_selection(toRemove, 
                                imputedImList, 
                                outPath+'final_imgs_feature_selection_zeros/')
    overall = find_correlation(fsImList)
    groups, centers = group_feature_selection(overall, 
                                              21, 
                                              max_iter=100000, 
                                              init='k-medoids++')
    autoList = combine_groups(fsImList, 
                        outPath+'final_imgs_feature_selection_kmedoids_center/',
                        groups, 
                        mode='center', 
                        centers=centers)

    # Expert feature selection, if applicable.
    if runExpert is not None:
        expertCategories = pd.read_csv(runExpert[0], header=[0], index_col=0)
        cat = expertCategories['Category'].to_numpy()
        groupsEx = []
        for i in range(expertCategories['Category'].max()):
            groupsEx.append(list(np.where(cat == i)[0]))
        
        expertList = combine_groups(fsImList, 
                            outPath+'final_imgs_feature_selection_center/', 
                            groupsEx, 
                            mode='center', 
                            centers=runExpert[1])

    # Compute correlation between each band and ground truth MND (Fe here).
    outcomeCorrelation = find_correlation(imputedImList, 
        outcome=natsorted(glob.glob(labelLists[0]+'/*.tif')))
    print('Iron:', bandNames)
    print(outcomeCorrelation)

    # Prepare LR data.
    for mnd in labelLists:
        mndTitle = mnd.split('/')[-1]

        # Auto
        lrdataListAuto, bandNamesAuto, reportedRegionsAuto = \
            prepare_lr_data(autoList, natsorted(glob.glob(mnd+'/*.tif')))

        # Remove 0
        lrdataList0, bandNames0, reportedRegions0 = prepare_lr_data(fsImList, 
                                          natsorted(glob.glob(mnd+'/*.tif')))

        # Expert
        if runExpert is not None:
            lrdataListEx, bandNamesEx, reportedRegionsEx = \
                prepare_lr_data(expertList, natsorted(glob.glob(mnd+'/*.tif')))
            if reportedRegionsEx != reportedRegionsAuto:
                raise ValueError('Expert regions do not align')

        if reportedRegionsAuto == reportedRegions0:
            for ir, region in enumerate(reportedRegionsAuto):
                lrdf = pd.DataFrame(lrdataListAuto[ir], columns=bandNamesAuto)
                lrdf.to_csv(outPath+'auto_'+mndTitle+'_'+regionDict[region]+'.csv')

                lrdf = pd.DataFrame(lrdataList0[ir], columns=bandNames0)
                lrdf.to_csv(outPath+'remove0_'+mndTitle+'_'+regionDict[region]+'.csv')

                if runExpert is not None:
                    lrdf = pd.DataFrame(lrdataListEx[ir], columns=bandNamesEx)
                    lrdf.to_csv(outPath+'expert_'+mndTitle+'_'+regionDict[region]+'.csv')
        else:
            raise ValueError('Auto and 0 regions do not align')


# K-medoids clustering
# From sklearn_extra.cluster. Required small tweak for our k-medoids algorithm
# Changes marked by #### EDIT
# Authors: Timo Erkkilä <timo.erkkila@gmail.com>
#          Antti Lehmussola <antti.lehmussola@gmail.com>
#          Kornel Kiełczewski <kornel.mail@gmail.com>
#          Zane Dufour <zane.dufour@gmail.com>
# License: BSD 3 clause


class KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-medoids clustering.

    Read more in the :ref:`User Guide <k_medoids>`.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances
        metric can be 'precomputed', the user must then feed the fit method
        with a precomputed kernel matrix and not the design matrix X.

    method : {'alternate', 'pam'}, default: 'alternate'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.

    init : {'random', 'heuristic', 'k-medoids++', 'build'}, optional, default: 'build'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 'k-medoids++'
        follows an approach based on k-means++_, and in general, gives initial
        medoids which are more separated than those generated by the other methods.
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.

        .. _k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import KMedoids
    >>> import numpy as np

    >>> X = np.asarray([[1, 2], [1, 4], [1, 0],
    ...                 [4, 2], [4, 4], [4, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmedoids.predict([[0,0], [4,4]])
    array([0, 1])
    >>> kmedoids.cluster_centers_
    array([[1, 2],
           [4, 2]])
    >>> kmedoids.inertia_
    8.0

    See scikit-learn-extra/examples/plot_kmedoids_digits.py for examples
    of KMedoids with various distance metrics.

    References
    ----------
    Maranzana, F.E., 1963. On the location of supply points to minimize
      transportation costs. IBM Systems Journal, 2(2), pp.129-135.
    Park, H.S.and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
      clustering.  Expert systems with applications, 36(2), pp.3336-3341.

    See also
    --------

    KMeans
        The KMeans algorithm minimizes the within-cluster sum-of-squares
        criterion. It scales well to large number of samples.

    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        method="alternate",
        init="heuristic",
        max_iter=300,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state


    def _check_nonnegative_int(self, value, desc, strict=True):
        """Validates if value is a valid integer > 0"""
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "%s should be a nonnegative integer. "
                "%s was given" % (desc, value)
            )

    def _check_init_args(self):
        """Validates the input arguments. """

        # Check n_clusters and max_iter
        self._check_nonnegative_int(self.n_clusters, "n_clusters")
        self._check_nonnegative_int(self.max_iter, "max_iter", False)

        # Check init
        init_methods = ["random", "heuristic", "k-medoids++", "build"]
        #### EDIT: Commented out compared to original.
        # if self.init not in init_methods:
        #     raise ValueError(
        #         "init needs to be one of "
        #         + "the following: "
        #         + "%s" % init_methods
        #     )

    def fit(self, X, y=None):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features), \
                or (n_samples, n_samples) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        random_state_ = check_random_state(self.random_state)

        self._check_init_args()
        X = check_array(X, accept_sparse=["csr", "csc"])
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "The number of medoids (%d) must be less "
                "than the number of samples %d."
                % (self.n_clusters, X.shape[0])
            )

        D = pairwise_distances(X, metric=self.metric)
        medoid_idxs = self._initialize_medoids(
            D, self.n_clusters, random_state_
        )
        labels = None

        if self.method == "pam":
            # Compute the distance to the first and second closest points
            # among medoids.
            Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]

        # Continue the algorithm as long as
        # the medoids keep changing and the maximum number
        # of iterations is not exceeded

        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[medoid_idxs, :], axis=0)
            #### EDIT: Added print compared to original.
            print(labels)

            if self.method == "alternate":
                # Update medoids with the new cluster indices
                self._update_medoid_idxs_in_place(D, labels, medoid_idxs)
            elif self.method == "pam":
                not_medoid_idxs = np.delete(np.arange(len(D)), medoid_idxs)
                optimal_swap = _compute_optimal_swap(
                    D,
                    medoid_idxs.astype(np.intc),
                    not_medoid_idxs.astype(np.intc),
                    Djs,
                    Ejs,
                    self.n_clusters,
                )
                if optimal_swap is not None:
                    i, j, _ = optimal_swap
                    medoid_idxs[medoid_idxs == i] = j

                    # update Djs and Ejs with new medoids
                    Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]
            else:
                raise ValueError(
                    f"method={self.method} is not supported. Supported methods "
                    f"are 'pam' and 'alternate'."
                )

            if np.all(old_medoid_idxs == medoid_idxs):
                break
            elif self.n_iter_ == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

        # Set the resulting instance variables.
        if self.metric == "precomputed":
            self.cluster_centers_ = None
        else:
            self.cluster_centers_ = X[medoid_idxs]

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = np.argmin(D[medoid_idxs, :], axis=0)
        self.medoid_indices_ = medoid_idxs
        self.inertia_ = self._compute_inertia(self.transform(X))

        # Return self to enable method chaining
        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs):
        """In-place update of the medoid indices"""

        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_cost(self, D, medoid_idxs):
        """ Compute the cose for a given configuration of the medoids"""
        return self._compute_inertia(D[:, medoid_idxs])

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(X, accept_sparse=["csr", "csc"])

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")

            Y = self.cluster_centers_
            return pairwise_distances(X, Y=Y, metric=self.metric)

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, accept_sparse=["csr", "csc"])

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            return pairwise_distances_argmin(
                X, Y=self.cluster_centers_, metric=self.metric
            )

    def _compute_inertia(self, distances):
        """Compute inertia of new samples. Inertia is defined as the sum of the
        sample distances to closest cluster centers.

        Parameters
        ----------
        distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
            Distances to cluster centers.

        Returns
        -------
        Sum of sample distances to closest cluster centers.
        """

        # Define inertia as the sum of the sample-distances
        # to closest cluster centers
        inertia = np.sum(np.min(distances, axis=1))

        return inertia

    def _initialize_medoids(self, D, n_clusters, random_state_):
        """Select initial mediods when beginning clustering."""

        #### EDIT: Added np.ndarray, direct initialization.
        if type(self.init) is np.ndarray:
            medoids = self.init
        elif self.init == "random":  # Random initialization
            # Pick random k medoids as the initial ones.
            medoids = random_state_.choice(len(D), n_clusters)
        elif self.init == "k-medoids++":
            medoids = self._kpp_init(D, n_clusters, random_state_)
        elif self.init == "heuristic":  # Initialization by heuristic
            # Pick K first data points that have the smallest sum distance
            # to every other point. These are the initial medoids.
            medoids = np.argpartition(np.sum(D, axis=1), n_clusters - 1)[
                :n_clusters
            ]
        elif self.init == "build":  # Build initialization
            medoids = _build(D, n_clusters).astype(np.int64)
        else:
            raise ValueError(f"init value '{self.init}' not recognized")

        return medoids

    # Copied from sklearn.cluster.k_means_._k_init
    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        """Init n_clusters seeds with a method similar to k-means++

        Parameters
        -----------
        D : array, shape (n_samples, n_samples)
            The distance matrix we will use to select medoid indices.

        n_clusters : integer
            The number of seeds to choose

        random_state : RandomState
            The generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-medoid clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        n_samples, _ = D.shape

        centers = np.empty(n_clusters, dtype=int)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        center_id = random_state_.randint(n_samples)
        centers[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = D[centers[0], :] ** 2
        current_pot = closest_dist_sq.sum()

        # pick the remaining n_clusters-1 points
        for cluster_index in range(1, n_clusters):
            rand_vals = (
                random_state_.random_sample(n_local_trials) * current_pot
            )
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_dist_sq), rand_vals
            )

            # Compute distances to center candidates
            distance_to_candidates = D[candidate_ids, :] ** 2

            # Decide which candidate is the best
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = np.minimum(
                    closest_dist_sq, distance_to_candidates[trial]
                )
                new_pot = new_dist_sq.sum()

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[cluster_index] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers
