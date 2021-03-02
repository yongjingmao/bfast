'''
Created on Apr 19, 2018

@author: fgieseke
'''

import gc
import time
from datetime import datetime

import pandas
import numpy
import pyopencl
import pyopencl.array as pycl_array

from bfast.base import BFASTMonitorBase
from bfast.monitor.utils import compute_end_history, compute_lam, map_indices

from .bfastfinal import bfastfinal

#################################################
# ## Remember to compile with, e.g.,
# ## $ gpu-pyopencl --library bfastfuth.fut
#################################################

class BFASTMonitorOpenCL(BFASTMonitorBase):

    """ Parallel BFASTMonitor implementation optimized for GPUs. The
    interface follows the one of the corresponding R package,
    see: https://cran.r-project.org/web/packages/bfast

    Parameters
    ----------

    start_monitor : datetime
        A datetime object specifying the start of
        the monitoring phase.
    freq : int, default 365
        The frequency for the seasonal model
    k : int, default 3
        The number of harmonic terms
    hfrac : float, default 0.25
        Float in the interval (0,1) specifying the
        bandwidth relative to the sample size in
        the MOSUM/ME monitoring processes.
    trend : bool, default True
        Whether a tend offset term shall be used or not
    level : float, default 0.05
        Significance level of the monitoring (and ROC,
        if selected) procedure, i.e., probability of
        type I error.
    detailed_results : bool, default False
        If detailed results should be returned or not
    find_magnitudes : bool, default True
        If magnitudes should be calculated or not
    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=output)
    platform_id : int, default 0
        Specifies the OpenCL platform id
    device_id int, default 0
        Specified the OpenCL device id

    Examples
    --------

    Notes
    -----

    """

    def __init__(self,
                 start_monitor,
                 freq=365,
                 k=3,
                 hfrac=0.25,
                 trend=True,
                 level=0.05,
                 period=10,
                 detailed_results=False,
                 find_magnitudes=True,
                 verbose=0,
                 platform_id=0,
                 device_id=0
                 ):

        k_valid = list(range(3, 11))

        if k not in k_valid:
            raise Exception("Current implementation can only handle the following values for k: {}".format(k_valid))

        super(BFASTMonitorOpenCL, self).__init__(start_monitor,
                                       freq,
                                       k=k,
                                       hfrac=hfrac,
                                       trend=trend,
                                       level=level,
                                       period=period,
                                       verbose=verbose)

        self.detailed_results = detailed_results
        self.find_magnitudes = find_magnitudes
        self.platform_id = platform_id
        self.device_id = device_id

        # initialize device
        self._init_device(platform_id, device_id)

        self.futobj = bfastfinal(command_queue=self.queue,
                                 interactive=False,
                                 default_tile_size=8,
                                 default_reg_tile_size=3,
                                 sizes=self._get_futhark_params())

    def _init_device(self, platform_id, device_id):
        """ Initializes the device.
        """

        try:
            platforms = pyopencl.get_platforms()
            devices = platforms[platform_id].get_devices()
        except Exception as e:
            raise Exception("Could not access device '{}' on platform with '{}': {}".format(str(platform_id), str(device_id), str(e)))

        self.device = devices[device_id]
        self.ctx = pyopencl.Context(devices=[self.device])
        self.queue = pyopencl.CommandQueue(self.ctx)

    def _print_device_info(self):
        """ Prints information about the current device.
        """

        if self.verbose > 0:
            print("=================================================================================")
            print("Device id: " + str(self.device_id))
            print("Device name: " + str(self.device.name()))
            print("---------------------------------------------------------------------------------")
            print("Attributes:\n")
            attributes = self.device.get_attributes()
            for (key, value) in attributes.iteritems():
                print("\t%s:%s" % (str(key), str(value)))
            print("=================================================================================")


    def _get_futhark_params(self):
        sizes = {
            "main.suff_intra_par_11":128,
            "main.suff_intra_par_13":128,
            "main.suff_intra_par_24":235,
            "main.suff_intra_par_29":113,
            "main.suff_intra_par_34":122,
            "main.suff_outer_par_16":2000000000,
            "main.suff_outer_par_17":892448,
            "main.suff_outer_par_18":2000000000,
            "main.suff_outer_par_19":892448,
            "main.suff_outer_par_20":2000000000,
            "main.suff_outer_par_21":26215660,
            "main.suff_outer_par_28":2000000000,
            "main.suff_outer_par_31":111556,
            "main.suff_outer_par_6":2000000000,
            "main.suff_outer_par_7":2000000000,
            "main.suff_outer_par_8":7139584,
            "mainMagnitude.suff_intra_par_11":128,
            "mainMagnitude.suff_intra_par_13":2000000000,
            "mainMagnitude.suff_intra_par_24":235,
            "mainMagnitude.suff_intra_par_29":113,
            "mainMagnitude.suff_intra_par_37":122,
            "mainMagnitude.suff_outer_par_16":2000000000,
            "mainMagnitude.suff_outer_par_17":892448,
            "mainMagnitude.suff_outer_par_18":111556,
            "mainMagnitude.suff_outer_par_19":2000000000,
            "mainMagnitude.suff_outer_par_20":2000000000,
            "mainMagnitude.suff_outer_par_21":26215660,
            "mainMagnitude.suff_outer_par_28":2000000000,
            "mainMagnitude.suff_outer_par_31":111556,
            "mainMagnitude.suff_outer_par_6":2000000000,
            "mainMagnitude.suff_outer_par_7":2000000000,
            "mainMagnitude.suff_outer_par_8":7139584
        }
        return sizes


    def fit(self, data, dates, n_chunks=None, nan_value=0):
        """ Fits the BFAST model for the ndarray 'data'

        Parameters
        ----------
        data: ndarray of shape (N, W, H),
            where N is the number of time
            series points per pixel and W
            and H the width and the height
            of the image, respectively.
        dates : list of datetime objects
            Specifies the dates of the elements
            in data indexed by the first axis
            n_chunks : int or None, default None
        nan_value : int, default 0
            Specified the NaN value used in
            the array data

        Returns
        -------
        self : instance of BFASTGPU
            The object itself
        """

        # TODO: crop dates here, crop data here
        #dates = self.dates
        self._timers = {}
        self._timers['chunk_normalization'] = 0
        self._timers['chunk_fit'] = 0
        self._timers['initialization'] = 0
        self._timers['transfer_host_gpu'] = 0
        self._timers['preprocessing'] = 0
        self._timers['kernel'] = 0
        self._timers['transfer_gpu_host'] = 0

        # fit single or fit several models, one per chunk
        if n_chunks is None:
            results = self._fit_single(data, dates, nan_value)
        else:
            results = self._fit_chunks(data, dates, n_chunks=n_chunks, nan_value=nan_value)

        if self.detailed_results:
            self.y_pred = results['y_pred']
            self.mosum = results['mosum']
            self.bounds = results['bounds']

        self.breaks = results['breaks']
        self.means = results['means']
        self.valids = results['valids']

        if self.find_magnitudes or self.detailed_results:
            self.magnitudes = results['magnitudes']

        return self

    def get_timers(self):
        """ Returns runtime measurements for the
        different phases of the fitting process.

        Returns
        -------
        dict : An array containing the runtimes
            for the different phases.
        """

        return self._timers

    def _fit_chunks(self, data, dates, n_chunks=10, nan_value=0):
        data_chunks = numpy.array_split(data, n_chunks, axis=1)

        results = []

        for chunk_idx in range(n_chunks):

            start_chunk = time.time()
            if self.verbose > 0:
                print("Processing chunk index {}/{}".format(chunk_idx + 1, n_chunks))

            # get chunk and reshape it
            start = time.time()
            data = data_chunks[chunk_idx]
            data = numpy.asarray(data, order='C')
            end = time.time()
            if self.verbose > 0:
                self._timers['chunk_normalization'] += end - start
                print("- runtime for chunk normalization:\t\t\t{}".format(end - start))

            # fit model to chunk
            start = time.time()
            res = self._fit_single(data, dates, nan_value)
            end = time.time()
            if self.verbose > 0:
                self._timers['chunk_fit'] += end - start
                print("- runtime for fitting chunk:\t\t\t\t{}".format(end - start))

            end_chunk = time.time()
            if self.verbose > 0:
                print("- all computations have taken:\t\t\t\t{}".format(end_chunk - start_chunk))

            results.append(res)

        return self._merge_results(results)

    def _fit_single(self, data, dates, nan_value):
        oshape = data.shape

        # (1) preprocessing
        y_cl, mapped_indices_cl = self._fit_single_preprocess(data, dates, nan_value)

        # (2) execute kernel
        results = self._fit_single_kernel(y_cl, mapped_indices_cl)

        # (3) copy data back from device to host
        self._fit_single_postprocessing(results, oshape)

        return results

    def _fit_single_preprocess(self, data, dates, nan_value):
        start = time.time()
        mapped_indices = map_indices(dates).astype(numpy.int32)
        N = data.shape[0]
        self.n = compute_end_history(dates, self.start_monitor)
        self.lam = compute_lam(N, self.hfrac, self.level, self.period)
        end = time.time()

        if self.verbose > 0:
            self._timers['initialization'] += end - start
            print("--- runtime for data initialization:\t\t{}".format(end - start))

        # (1) copy data from host to device
        start = time.time()
        data_cl = pycl_array.to_device(self.queue, data)
        mapped_indices_cl = pycl_array.to_device(self.queue, mapped_indices)
        end = time.time()
        if self.verbose > 0:
            self._timers['transfer_host_gpu'] += end - start
            print("--- runtime for data transfer (host->device):\t{}".format(end - start))

        start = time.time()
        data_cl = self.futobj.remove_nans(nan_value, data_cl)
        y_cl = self.futobj.reshapeTransp(data_cl)
        end = time.time()
        if self.verbose > 0:
            self._timers['preprocessing'] += end - start
            print("--- runtime for data preprocessing:\t\t{}".format(end - start))

        return y_cl, mapped_indices_cl

    def _fit_single_kernel(self, y_cl, mapped_indices_cl):
        start = time.time()
        if self.trend:
            trend = 1
        else:
            trend = 0

        if self.detailed_results:
            MO_first, \
                Ns, \
                ns, \
                sigmas, \
                mosum, \
                mosum_nn, \
                bounds, \
                breaks, \
                means, \
                magnitudes, \
                y_error, \
                y_pred = self.futobj.mainDetailed(trend,
                                                  self.k,
                                                  self.n,
                                                  self.freq,
                                                  self.hfrac,
                                                  self.lam,
                                                  mapped_indices_cl, y_cl)
        elif self.find_magnitudes:
            Ns, \
                breaks, \
                means, \
                magnitudes = self.futobj.mainMagnitude(trend,
                                                       self.k,
                                                       self.n,
                                                       self.freq,
                                                       self.hfrac,
                                                       self.lam,
                                                       mapped_indices_cl, y_cl)
        else:
            Ns, breaks, means = self.futobj.main(trend,
                                                 self.k,
                                                 self.n,
                                                 self.freq,
                                                 self.hfrac,
                                                 self.lam,
                                                 mapped_indices_cl, y_cl)

        end = time.time()

        if self.verbose > 0:
            self._timers['kernel'] += end - start
            print("--- runtime for kernel execution:\t\t{}".format(end - start))

        results = {}
        if self.detailed_results:
            results['mosum'] = mosum
            results['bounds'] = bounds
            results['y_pred'] = y_pred
        results['breaks'] = breaks
        results['means'] = means
        results['valids'] = Ns

        if self.find_magnitudes or self.detailed_results:
            results['magnitudes'] = magnitudes

        return results

    def _fit_single_postprocessing(self, results, oshape):
        start = time.time()
        if self.detailed_results:
            mosum = results['mosum'].get().T
            mosum = mosum.reshape((mosum.shape[0], oshape[1], oshape[2]))
            results['mosum'] = mosum
            results['bounds'] = results['bounds'].get()
            results['y_pred'] = results['y_pred'].get().T.reshape(oshape)

        results['breaks'] = results['breaks'].get().reshape(oshape[1:])
        results['means'] = results['means'].get().reshape(oshape[1:])
        results['valids'] = results['valids'].get().T.reshape(oshape[1:])

        if self.find_magnitudes or self.detailed_results:
            results['magnitudes'] = results['magnitudes'].get().reshape(oshape[1:])

        end = time.time()

        if self.verbose > 0:
            self._timers['transfer_gpu_host'] += end - start
            print("--- runtime for data transfer (device->host):\t{}".format(end - start))

    def _merge_results(self, results):
        final_results = {}

        while(len(results) > 0):
            if self.verbose > 0:
                print("Length of list to be merged: {}".format(len(results)))
            self.__append_results(results.pop(0), final_results)
            gc.collect()

        return final_results

    def __append_results(self, res, results):
        if 'breaks' in results.keys():
            results['breaks'] = numpy.concatenate([results['breaks'], res['breaks']], axis=0)
        else:
            results['breaks'] = res['breaks']

        if 'means' in results.keys():
            results['means'] = numpy.concatenate([results['means'], res['means']], axis=0)
        else:
            results['means'] = res['means']

        if 'valids' in results.keys():
            results['valids'] = numpy.concatenate([results['valids'], res['valids']], axis=0)
        else:
            results['valids'] = res['valids']

        if self.find_magnitudes or self.detailed_results:
            if 'magnitudes' in results.keys():
                results['magnitudes'] = numpy.concatenate([results['magnitudes'], res['magnitudes']], axis=0)
            else:
                results['magnitudes'] = res['magnitudes']

        if self.detailed_results:

            if not 'bounds' in results.keys():
                results['bounds'] = res['bounds']

            if 'y_pred' in results.keys():
                results['y_pred'] = numpy.concatenate([results['y_pred'], res['y_pred']], axis=1)
            else:
                results['y_pred'] = res['y_pred']

            if 'mosum' in results.keys():
                results['mosum'] = numpy.concatenate([results['mosum'], res['mosum']], axis=1)
            else:
                results['mosum'] = res['mosum']

        return results
