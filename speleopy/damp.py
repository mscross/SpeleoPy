from __future__ import division, print_function, absolute_import
import numpy as np
import random
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator


class AgeModel(object):
    """
    Make a simple Monte Carlo age model and get subsample ages.
    Does not model across hiatuses or find hiatuses.

    """

    def __init__(self, z, dates, errors, length, is_2sigma=True,
                 is_depth=True):
        """
        Initialize age model.

        Parameters
        ----------
        depths : (M) 1D ndarray of floats
            The depth of each age control point.  In order of increasing depth.
        dates : (M) 1D ndarray of floats
            The age control points, in order of increasing age.
        errors : (M) 1D ndarray of floats
            The errors associated with each age control point.
            In 1 or 2 sigma.
        length : float
            The total length of the record
        is_2sigma : Boolean
            Default True, ``errors`` are in two sigma.
            Assumes Gaussian distribution of error.
            Set ``False`` if error in 1 sigma.
        is_depth : Boolean
            Default ``True``, ``z`` indicates depths rather than height.

        """
        # Set 2 sigma errors
        self.errors = errors
        if not is_2sigma:
            self.errors = errors * 2

        # Set depths
        if not is_depth:
            self.depths = length - z
            self.heights = z
        else:
            self.depths = z
            self.heights = length - z

        self.dates = dates
        self.length = length
        self.reversal = {}
        self.intractable = {}
        self.warning_color = 'orange'
        self.good_color = 'black'
        self.bad_color = 'red'

    def _print_header(self):
        """
        Print  header for viewing age control, model, or subsample data.

        """

        print("\n\tDepth\tHeight\tAge\t\t2s Error\n")

    def print_agecontrol_dates(self):
        """
        Print the current sequence of age control points.

        """

        self._print_header()

        for n, (d, h, a, e) in enumerate(zip(self.depths, self.heights,
                                             self.dates, self.errors)):

            print(n, '.\t', d, '\t', h, '\t', a, '\t', e)

    def print_model_dates(self):
        """
        Print the modelled age control points.

        """

        self._print_header()

        try:
            for n, (d, h, a, e) in enumerate(zip(self.depths, self.heights,
                                                 self.model_median,
                                                 self.model_error)):
                print(n, '.\t', d, '\t', h, '\t', a, '\t', e)
        except AttributeError:
            print('Model not yet calculated')

    def print_subsample_dates(self):
        """
        Print the subsample depths, ages, and age errors

        """

        self._print_header()

        try:
            for n, (d, h, a, e) in enumerate(zip(self.subsample_depths,
                                                 self.subsample_heights,
                                                 self.subsample_ages,
                                                 self.subsample_err)):
                print(n, '.\t', d, '\t', h, '\t', a, '\t', e)
        except AttributeError:
            print('Subsample ages not yet interpolated')

    def check_monotonicity(self):
        """
        Check if the ages are monotonically increasing with depth.
        Distinguishes between tractable (errorbars overlap) and intractable
        (errorbars do not overlap) age reversals.

        """

        younger = self.dates[:-1]

        # Convert errors, dates, depths, heights to arrays if necessary
        try:
            youngest = younger - self.errors[:-1]

        except TypeError:
            self.dates = np.array(self.dates)
            self.errors = np.array(self.errors)
            self.depths = np.array(self.depths)
            self.heights = np.array(self.heights)
            younger = np.array(younger)
            youngest = younger - self.errors[:-1]

        older = self.dates[1:]
        oldest = older + self.errors[1:]

        reversal = np.nonzero(younger > older)[0]
        intractable = np.nonzero(youngest > oldest)[0]

        reversal = set(np.union1d(reversal, reversal + 1))
        self.intractable = set(np.union1d(intractable, intractable + 1))

        # Separate tractable and intractable reversals
        self.reversal = reversal - self.intractable

    def adjust_errors(self, adjust_by, inds):
        """
        Multiply the indicated errors by a given factor
        to improve tractability.

        Parameters
        ----------
        adjust_by : int or float or sequence of length (M)
            The factor(s) by which the indicated errors will be multiplied
        inds : int or sequence of ints of length (M)
            Indices corresponding to the errors requiring adjustment

        """

        self.errors[inds] = self.errors[inds] * adjust_by

    def delete_dates(self, inds):
        """
        Remove dates from the age model and update monotonicity results.

        Parameters
        ----------
        inds : int or sequence of ints
            The indices of the dates to delete.  Can be easily found with
            ``self.print_agecontrol_dates()``

        """

        self.dates = np.delete(self.dates, inds)
        self.depths = np.delete(self.depths, inds)
        self.heights = np.delete(self.heights, inds)
        self.errors = np.delete(self.errors, inds)

        self.check_monotonicity()

    def view_monotonicity(self, ax, use_depth=True, **ekwargs):
        """
        View a plot of the monotonicity test results.  Intractable reversals
        are shown in red, tractable in orange, and monotonically increasing
        dates in black.

        Parameters
        ----------
        ax : matplotlib Axes object
            The Axes on which to plot the montonicity test results.
        use_depth : Boolean
            Default ``True``, plot on depth scale.  If ``False``, plot on
            height scale.
        **ekwargs
            Any Axes.errorbar keyword arguments

        """
        z = self.depths
        if not use_depth:
            z = self.heights

        ax.errorbar(self.dates, z, xerr=self.errors,
                    color=self.good_color, ecolor=self.good_color, **ekwargs)

        for i in self.reversal:
            ax.errorbar(self.dates[i], z[i], xerr=self.errors[i],
                        color=self.warning_color, ecolor=self.warning_color,
                        **ekwargs)

        for i in self.intractable:
            ax.errorbar(self.dates[i], z[i], xerr=self.errors[i],
                        color=self.bad_color, ecolor=self.bad_color,
                        **ekwargs)

    def monty_carlo(self, successful_sims, max_sims):
        """
        Perform Monte Carlo age modelling.  Model median and error results are
        determined by successful age model results.  Successful age models have
        ages increasing monotonically with depth.

        Monte Carlo simulations stop when either the number of successes or
        the maximum number of simulations is reached.

        Parameters
        ----------
        successful_sims : int
            The number of successes after which the model will quit and report
            success.  If ``max_sims`` is reached first, then
            failure will be reported.
        max_sims : int
            The maximum number of simulations to run.  The model quits before
            this number is reached if ``successful_sims`` is reached first.

        """

        # Initialize empty simulation arrays
        good_sims = np.empty((self.dates.size, successful_sims))
        all_sims = np.empty((self.dates.size, max_sims))

        sims = 0
        good = 0
        good_inds = []
        # random.gauss takes 1 sigma
        errors = self.errors / 2

        # Quit when max or number of successes is hit
        while sims < max_sims and good < successful_sims:
            # Have to simulate each date separately
            simulation = []
            for date, error in zip(self.dates, errors):
                simulation.append(random.gauss(date, error))
            simulation = np.array(simulation)

            # Put simulation in array
            all_sims[:, sims] = simulation
            # Test
            monotonicity_test = simulation[1:] - simulation[:-1]

            # Check that there are no negative results
            if not (monotonicity_test < 0.0).any():
                # Place in 'good' array, record position in 'all' array
                good_sims[:, good] = simulation
                good_inds.append(sims)
                good += 1

            sims += 1

        # Get rid of empty portions of arrays, if any
        self.good_sims = np.delete(good_sims, slice(good, successful_sims),
                                   axis=1)
        all_sims = np.delete(all_sims, slice(sims, max_sims), axis=1)

        # Remove successful simulations from 'all'
        self.bad_sims = np.delete(all_sims, good_inds, axis=1)

        # Calculate model median and error
        self.model_median = np.median(self.good_sims, axis=1)
        self.model_error = np.nanstd(self.good_sims, axis=1) * 2

        # Record simulation counts and report
        self.sim_count = (good, self.bad_sims.shape[1])

        if good == successful_sims:
            print('Age modelling complete')
        else:
            print('Age modelling unsuccessful')

    def view_agemodel(self, ax, badsim_step=10, view_simulations=True,
                      original_marker='o', model_marker='d',
                      original_color='blue', model_color='green', zorder=20,
                      use_depth=True, **ekwargs):
        """
        View a plot of the age model results.

        Parameters
        ----------
        ax : matplotlib Axes object
            The Axes on which to plot the age modelling results
        badsim_step : int
            Default 10.  Plotting all unsuccessful simulations in a large
            Monte Carlo model run has significant time and memory costs.
            This parameter will result in every nth unsuccessful simulation
            plotted.
        view_simulations : Boolean
            Default True.  Turns on/off the simulation plotting, leaving just
            the original age control points and the model results.
        original_marker : string
            Default 'o'.  The marker for the age control points.
        model_marker : string
            Default 'd'.  The marker for the model points.
        original_color : string or tuple of floats
            Default 'blue'.  The marker, line, and errorbar color for the
            age control points.
        model_color : string or tuple of floats
            Default 'green'.  The marker, line, and errorbar color for the
            model results.
        zorder : int
            Default 20.  Zorder of the age control points and model results.
        use_depth : Boolean
            Default ``True``, plot on depth scale.  If ``False``, plot on
            height scale.
        **ekwargs
            Any Axes.errorbar keyword.

        """

        z = self.depths
        if not use_depth:
            z = self.heights

        # Plot simulations
        if view_simulations:
            for i in range(0, self.sim_count[1], badsim_step):
                ax.plot(self.bad_sims[:, i], z, color=self.warning_color)

            for i in range(0, self.sim_count[0]):
                ax.plot(self.good_sims[:, i], z, color=self.good_color)

        # Plot original age control points and profile
        ax.errorbar(self.dates, z, xerr=self.errors,
                    marker=original_marker, zorder=zorder,
                    color=original_color, ecolor=original_color, **ekwargs)

        # Plot model results
        ax.errorbar(self.model_median, z, xerr=self.model_error,
                    marker=model_marker, zorder=zorder + 2,
                    color=model_color, ecolor=model_color, **ekwargs)

    def set_linear_interpolation_eq(self):
        """
        Calculate the linear interpolation equation for the model results.

        """

        kwargs = {'axis': 0,
                  'bounds_error': False,
                  'kind': 'linear'}

        self.interpolationeq = interp1d(self.depths, self.model_median,
                                        **kwargs)
        self.interpolationeq_young = interp1d(self.depths, self.model_median -
                                              self.model_error, **kwargs)
        self.interpolationeq_old = interp1d(self.depths, self.model_median +
                                            self.model_error, **kwargs)

    def set_cubic_interpolation_eq(self):
        """
        Calculate the monotonic cubic interpolation equation for the
        model results.

        """

        kwargs = {'extrapolate': True}

        self.interpolationeq = PchipInterpolator(self.depths,
                                                 self.model_median, **kwargs)
        self.interpolationeq_young = PchipInterpolator(self.depths,
                                                       self.model_median -
                                                       self.model_error,
                                                       **kwargs)
        self.interpolationeq_old = PchipInterpolator(self.depths,
                                                     self.model_median +
                                                     self.model_error,
                                                     **kwargs)

    def set_subsample_ages(self, subsample_z, is_depth=True):
        """
        Use the interpolation equation to find subsample ages from z.

        Parameters
        ----------
        subsample_z : sequence of ints or floats
            The height or depth of the subsamples.
        is_depth : Boolean
            Default ``True``, indicates ``subsample_z`` represents depth
            rather than height.

        """

        if not hasattr(self, 'interpolationeq'):
            raise AttributeError('Model interpolation equation not found.')

        # Set both subsample depth and height attributes
        if is_depth:
            self.subsample_depths = subsample_z
            self.subsample_heights = self.length - subsample_z
        else:
            self.subsample_depths = self.length - subsample_z
            self.subsample_heights = subsample_z

        # Use depths to make interpolations
        self.subsample_ages = self.interpolationeq(self.subsample_depths)
        self.subsample_young = self.interpolationeq_young(
            self.subsample_depths)
        self.subsample_old = self.interpolationeq_old(self.subsample_depths)
        self.subsample_err = self.subsample_old - self.subsample_ages

    def view_subsample_ages(self, ax, use_depth=True,
                            errors_or_betweenx='errors', **kwargs):
        """
        View a plot of the modelled subsample ages.  Plotted as either a series
        of points with errorbars, or as a between value plot.

        Parameters
        ----------
        ax : matplotlib Axes object
            The Axes on which to plot the subsample ages.
        use_depth : Boolean
            Default ``True``, plot on depth scale.  If ``False``, plot on
            height scale.
        errors_or_betweenx : 'string'
            Default 'errors'.  Plot series of points with errorbars.  If
            anything else, plot as betweenx.
        **kwargs
            Any Axes.errorbar or Axes.fill_betweenx keyword argument

        """

        z = self.depths
        if not use_depth:
            z = self.heights

        if errors_or_betweenx is 'errors':
            ax.errorbar(self.subsample_ages, z,
                        xerr=self.subsample_err, **kwargs)
        else:
            ax.fill_betweenx(z, self.subsample_young,
                             self.subsample_old, **kwargs)
            # Plot age line
            ax.plot(self.subsample_ages, z,
                    color=self.good_color, marker=None)

    def write_oxcal_format(self, z_olderbound, z_youngerbound, output_file,
                           is_depth=True):
        """
        Output age control points into a text file, the contents of which
        can be copy-pasted into an OxCal P_Sequence model setup.

        Parameters
        ----------
        z_olderbound : int or float
            The depth or height of the older boundary.
        z_youngerbound : int or float
            The depth or height of the younger boundary.
        output_file : string
            Full or relative path to destination text file.
        is_depth : Boolean
            Default ``True``, ``z_olderbound`` and ``z_youngerbound``
            represent depths instead of heights.

        """

        if is_depth:
            olderbound = self.length - z_olderbound
            youngerbound = self.length - z_youngerbound
        else:
            olderbound = z_olderbound
            youngerbound = z_youngerbound

        oxcal = ['  Options(){BCAD=FALSE;};\n',
                 '  Plot()\n',
                 '  {\n',
                 '    P_Sequence("",0.1,"",U(-2,2))\n',
                 '    {\n',
                 '      Boundary(){z=' + str(olderbound) + ';};\n']

        for d, e, h in zip(self.dates, self.errors, self.heights):
            oxcal.extend(['      Date("",N(calBP(' + str(d) + '),' +
                          str(e) + ')){z=' + str(h) + ';};\n'])

        oxcal.extend(['      Boundary(){z=' + str(youngerbound) + ';};\n',
                      '    };\n',
                      '  };\n'])

        with open(output_file, 'w') as oxcalfile:
            oxcalfile.writelines(oxcal)
