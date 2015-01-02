import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import os
import time


class AgeModel:
    """
    Class for processing age data and making a simple Monte Carlo age model.

    Assumes date errors have a Gaussian distribution.

    """

    def __init__(self, data, num_sigmaunits, total_length,
                 length_units):
        """
        Initialize AgeModel object.

        Initializes several properties without checking for
            age reversals.  self.find_reversals() should be used next

        Parameters
        ----------
        data : list of 1d Numpy arrays
            Use load_worksheet() and load_data in sample_record_class.py
            Order should be depths, dates, date errors
        num_sigmaunits : int
            [1|2].  Whether the date errors are in 1 or 2 sigma.
        total_length : float
            The total length of the record.  Depths should be given as
            distances from top (i.e., depth increases with age),
            but if they are given as distances from bottom
            this will be used to convert data to the correct format
        length_units : string
            The units of total_length, for example 'mm'

        """

        data_array = np.empty((data[0].size, 0))

        for d in data:
            d = np.atleast_2d(d).T
            data_array = np.concatenate((data_array, d), axis=1)

        self.original_data = data_array

        self.total_length = total_length
        self.length_units = length_units

        # Check that depths increase with age
        if not self.original_data[0][0] < self.original_data[-1][0]:
            raise ValueError('z must be reported as depth from top!')

        if num_sigmaunits == 2:
            self.original_data[:,2] = self.original_data[:,2] / 2.0

        self.dates = self.original_data[:,1]
        self.dating_errors = self.original_data[:,2]
        self.depths = self.original_data[:,0]
        self.date_count = self.dates.size



    def reset_to_originaldata(self):
        """
        Reset self.dates, self.depths, self.dating_errors after modification
            to the original data read in from datafile.

        """

        self.dates = self.original_data[:,1]
        self.dating_errors = self.original_data[:,2]
        self.depths = self.original_data[:,0]
        self.date_count = self.dates.size



    def print_currentdata(self):
        """
        Print the current depths, dates, errors in a numbered list.

        """

        # Print header
        print "\n\tDepth\tAge\t\t2s Error\n"
        numlist = range(0, self.date_count)

        for n, d, a, e in zip(numlist, self.depths, self.dates,
                              self.dating_errors*2):

            print n, '.\t', d, '\t', a, '\t', e



    def print_modeldata(self):
        """
        Print the current depths, dates, errors in a numbered list.

        """

        # Print header
        print "\n\tDepth\tAge\t\t2s Error\n"
        numlist = range(0, self.date_count)

        for n, d, a, e in zip(numlist, self.depths, self.model_median,
                              self.model_error):

            print n, '.\t', d, '\t', a, '\t', e



    def check_monotonicity(self):
        """
        Inspects dating information for tractable and intractable
            reversals and prompts the user to do something about it.

        Shows a plot with 2s error bars colored according to reversal status.

        """

        # Set loop condition
        needs_work = True

        while needs_work:

            # Initialize
            younger = self.dates[:-1]
            youngest = younger - (self.dating_errors[:-1]*2)

            older = self.dates[1:]
            oldest = older + (self.dating_errors[1:]*2)

            # Look for reversals, grab indices of that date and the older on
            reversal = np.atleast_2d(np.nonzero(younger > older)).T
            reversal = np.concatenate((reversal, reversal+1), axis=1)

            # Do the same for intractable reversals
            bad_rev = np.atleast_2d(np.nonzero(youngest > oldest)).T
            bad_rev = np.concatenate((bad_rev, bad_rev+1), axis=1)

            # Find, delete regular reversals that are also intractable
            in_both = np.array(np.all((reversal[:,None,:] == bad_rev[None,:,:]),
                                axis=-1).nonzero()).T
            reversal = np.delete(reversal, in_both[:,0], 0)

            # Plot ages, if reversals are present
            if reversal.size + bad_rev.size > 0:

                # Set axis parameters
                fig, ax = plt.subplots(1,1, figsize=(10,10))
                ax.invert_yaxis()
                ax.tick_params(labelsize=16, pad=15)
                ax.set_xlabel('Years B.P.', fontsize=20, labelpad=20)
                ax.set_ylabel('Depth in Sample (' +str(self.length_units) +
                              ')', fontsize=20, labelpad=20)
                ax.xaxis.set_tick_params(which='major', length=10, width=1)
                ax.yaxis.set_tick_params(which='major', length=10, width=1)

                # Plot all ages and errors in black
                ax.errorbar(self.dates, self.depths, lw=3, color='black',
                            marker='o', xerr=self.dating_errors*2, elinewidth=3,
                            ecolor='black')

                colors = ['orange', 'orange', 'red', 'red']
                colored_errorbars = [reversal[:,0], reversal[:,1],
                                     bad_rev[:,0], bad_rev[:,1]]

                # Over that, plot regular reversals in orange and
                # intractable in red
                for c, ce in zip(colors, colored_errorbars):

                    ax.errorbar(self.dates[ce], self.depths[ce],
                                xerr=self.dating_errors[ce]*2, elinewidth=3,
                                color='none', ecolor=c, marker='o')

                plt.show()

                # Print the current dating table
                self.print_currentdata()
                edit_table = True

                # User may accept dating table if there are no
                # intractable reversals
                if bad_rev.size == 0:
                    print '\nNo intractable age reversals present. \n'
                    edit_more = raw_input('Accept age data table for ' +
                                          'modelling [Y/N]?\t')

                    if edit_more[0].lower() == 'y':
                        edit_table = False
                        needs_work = False

                # User edits table to get rid of intractable reversals,
                # either by enlarging indicated errorbars or removing dates
                if edit_table:

                    largerr = raw_input('\nEnter the numbers of the dating '+
                                        'errors you wish to enlarge:\t')
                    if len(largerr) > 0:
                        largerr = largerr.replace(',', ' ').split()
                        largerr = [int(i) for i in largerr]

                        for i in largerr:
                            self.dating_errors[i] = self.dating_errors[i] * 1.5

                    remove_dates = raw_input('\nEnter the numbers of the dates '+
                                             'you wish to remove:\t')
                    if len(remove_dates) > 0:

                        remove_dates = remove_dates.replace(',', ' ').split()
                        remove_dates = np.asarray(remove_dates, dtype=int)

                        self.dates = np.delete(self.dates, remove_dates)
                        self.dating_errors = np.delete(self.dating_errors,
                                                       remove_dates)
                        self.depths = np.delete(self.depths, remove_dates)
                        self.date_count = self.dates.size


                    # Updated plot and dating table are displayed
                    # User may proceed with current data table or
                    # revert to the original data
                    print '\nUpdated Age Plot and Data Table'

                    ax.errorbar(self.dates, self.depths, lw=3, color='black',
                                marker='o', xerr=self.dating_errors*2,
                                elinewidth=3, ecolor='black')

                    plt.show()

                    self.print_currentdata()

                    keepnew = raw_input('\nEnter `k` to keep and check '+
                                        'monotonicity of new age data table '+
                                        'or `r` to revert to original data:\t')

                    if keepnew[0].lower() == 'r':
                        self.reset_to_originaldata()

            else:
                needs_work = False

        # Prints model-ready dating table
        print "\nHere is the age data ready for modelling:"
        self.print_currentdata()



    def montycarlo(self, successful_sims, minutes_torun=2):
        """
        Takes age data and errors and does cool stuff

        Parameters
        ----------
        successful_sims : int
            The number of monotonic age model realizations to gather

        Keyword Arguments
        -----------------
        minutes_torun : int or float
            The maximum time to spend gathering monotonic age model
            realizations.  May be reached before successful_sims, and if so,
            will stop the simulation.

        """

        # Initialize
        good_realiz = np.empty((self.date_count, 0))
        all_realiz = np.empty((self.date_count, 0))

        success_count = 0
        simulation_count = 0

        max_runtime = 60.0 * minutes_torun

        start_time = time.time()

        run_simulation = True

        while run_simulation:

            simulation = []

            for date, error in zip(self.dates, self.dating_errors):
                simulation.append(random.gauss(date, error))

            simulation = (np.atleast_2d(np.asarray(simulation)
                                         ).reshape(self.date_count, 1))

            monotonicity_test = simulation[1:] - simulation[:-1]

            if not (monotonicity_test < 0.0).any():
                good_realiz = np.concatenate((good_realiz, simulation), axis=1)
                success_count += 1

            all_realiz = np.concatenate((all_realiz, simulation), axis=1)
            simulation_count += 1

            # Try ending the simulation
            if success_count == successful_sims:
                print 'Target number of monotonic simulations acquired.\n'
                run_simulation = False

            else:
                current_time = time.time()
                runtime = current_time - start_time

                if runtime > max_runtime:
                    run_simulation = False
                    print 'Simulation timed out.\n'


        self.success_count = success_count
        self.simulation_count = simulation_count

        self.model_median = np.median(good_realiz, axis=1)
        self.model_error = stats.nanstd(good_realiz, axis=1)*2

        self.good_simulations = good_realiz
        self.all_simulations = all_realiz



    def view_montycarlo_results(self, xlabel, ylabel,
                                figsize=(15,15), tick_dim=(10,2),
                                tick_direction='in'):
        """
        View a plot that has the current depth-age data, all montycarlo
            simulations, the monotonic simulations, and the final
            median value of the monotonic simulations

        Parameters
        ----------
        xlabel : string
            X-axis title
        ylabel : string
            Y-axis title

        Keyword Arguments
        -----------------
        figsize : tuple of ints
            Default (15,15).  The dimensions of the figure.
        tick_dim : tuple of ints
            Default (10,2).  The length, width of the tick marks.
        tick_direction
            Default 'in'.  ['in'|'out'|'inout'].  Sets the ticks to the
            inside, outside, or crossing the plot frame.

        Returns
        -------
        fig, ax : matplotlib figure, plot
            Plot of the results of the montycarlo() simulations.

        """

        # Initialize
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.invert_yaxis()

        # Plot all the simulations that occurred in orange
        for i in range(0, self.simulation_count):
            ax.plot(self.all_simulations[:,i], self.depths, color='orange')

        # Plot only the monotonic simulations in black
        for i in range(0, self.success_count):
            ax.plot(self.good_simulations[:,i], self.depths, color='black')

        # Plot the selected dates/depths in a bold line
        ax.plot(self.dates, self.depths, marker='o', lw=5, markersize=10)

        # Plot the median values of the monotonic simulations in a bold line
        ax.plot(self.model_median, self.depths, marker='d', lw=5, markersize=10)

        # Format plot ticks and labels
        ax.xaxis.set_tick_params(length=tick_dim[0], width=tick_dim[1],
                                 direction=tick_direction)

        ax.tick_params(labelsize=16, pad=15)

        ax.set_ylabel(ylabel, fontsize=20, labelpad=20)
        ax.set_xlabel(xlabel, fontsize=20, labelpad=20)

        return fig, ax



    def interpolate_agemodel(self, interpolation):
        """
        Get ages from sample depths.

        Parameters
        ----------
        interpolation : string
            ['cubic'|'linear'].  Interpolation method between age poitns.
            Cubic interpolation is one that enforces monotonicity.

        """

        # Aquire interpolation equations
        if interpolation is 'linear':

            self.interpolation = 'linear'
            self.interp_eq = interp1d(self.depths, self.model_median, axis=0,
                                      bounds_error=False, kind='linear')

            self.interp_eq_err_low = interp1d(self.depths, (self.model_median -
                                               self.model_error), axis=0,
                                              bounds_error=False, kind='linear')

            self.interp_eq_err_hi = interp1d(self.depths, (self.model_median +
                                              self.model_error), axis=0,
                                             bounds_error=False, kind='linear')

        if interpolation is 'cubic':

            self.interpolation = 'cubic'
            self.interp_eq = PchipInterpolator(self.depths, self.model_median,
                                               extrapolate=True)

            self.interp_eq_err_low = PchipInterpolator(self.depths,
                                                       (self.model_median -
                                                        self.model_error),
                                                       extrapolate=True)

            self.interp_eq_err_hi = PchipInterpolator(self.depths,
                                                      (self.model_median +
                                                       self.model_error),
                                                      extrapolate=True)



    def get_subsammple_ages(self, sample_depths, depth_isrange=False,
                            interpolation='linear', usecurrent_interp=True,
                            return_asrow=True):
        """
        Get ages of subsamples from age model and subsample depths.

        Parameters
        ----------
        sample_depths : sequence of floats
            The positions of stable isotope or trace element drill subsample
            sites.  Can input sequence of sample positions (ideal for irregular
            spacings), or input [start, inclusive stop, sample interva] and set
            depth_isrange to True.

        Keyword Arguments
        -----------------
        depth_isrange : Boolean
            Default False.  [True|False].
            True: sample_depths are given as [start, stop, interval]
            False: sample_depths are a complete list of depths to use
        interpolation : string
            Default 'linear'.  ['cubic'|'linear'].
            Interpolation method between age points.
            Cubic interpolation is one that enforces monotonicity.
            If given interpolation doesn't match current one, or current one
            does not exist, or usecurrent_interp is False, then new
            interpolation equations will be calculated.
        usecurrent_interp : Boolean
            Default True.  [True|False].  If the desired interpolation matches
            the current interpolation and this is set to False,
            current interpolation equations will be overwritten
            with new equations

        """

        # Initialize sample_depths
        if depth_isrange:
            sample_depths = np.arange(sample_depths[0], sample_depths[1] +
                                      sample_depths[2], sample_depths[2])

        self.sample_depths = sample_depths

        # Check that appropriate interpolation equations oare available

        if not hasattr(self, 'interpolation'):
            self.interpolate_agemodel(interpolation)

        if self.interpolation != interpolation and not usecurrent_interp:
            self.interpolate_agemodel(interpolation)

        self.sample_ages = self.interp_eq(self.sample_depths)
        self.sample_herr = self.interp_eq_err_hi(self.sample_depths)
        self.sample_lerr = self.interp_eq_err_low(self.sample_depths)
        self.sample_err = self.sample_herr-self.sample_ages

        print ('***DISCLAIMER: Any automatically extrapolated ages will '
               'be crap, and therefore we recommend you do it yourself.***')

        if return_asrow:
            return self.sample_ages, self.sample_err
        else:
            # Return self.sample_ages as a column for copy-pasta into excel
            return np.concatenate((np.atleast_2d(self.sample_depths).T,
                                   np.atleast_2d(self.sample_ages).T,
                                   np.atleast_2d(self.sample_err).T),axis=1)



    def oxcal_format(self, z_upperbound=None, z_lowerbound=None):
        """
        Output oxcal_format.  Depths will be converted into height using
            total length given in __init__.

        """

        if z_upperbound is not None:
            upper_boundary = str(self.total_length - z_upperbound)
        else:
            upper_boundary = str(self.total_length)

        if z_lowerbound is not None:
            lower_boundary = str(self.total_length - z_lowerbound)
        else:
            lower_boundary = 0

        print ('  Options(){BCAD=FALSE;};\n'
               '  Plot()\n'
               '  {\n'
               '    P_Sequence("",0.1,"",U(-2,2))\n'
               '    {\n'
               '      Boundary(){z=' + lower_boundary + ';};')

        for date in range(0, self.date_count)[::-1]:
            ht = self.total_length - self.depths[date]
            age = self.dates[date]
            err = self.dating_errors[date]

            print ('      Date("",N(calBP('+ str(age) + '),' +
                   str(err) + ')){z=' + str(ht) + ';};')

        print ('      Boundary(){z=' + upper_boundary + ';};\n'
               '    };\n'
               '  };')