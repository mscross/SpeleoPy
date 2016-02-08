from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator


class AgeModel(object):
    """
    Make a simple Monte Carlo age model and get subsample ages.
    Does not find hiatuses; each growth phase should be a separate age model.
    """

    def __init__(self, z, dates, errors, length, is_2sigma=True,
                 is_depth=True):
        """
        Initialize age model.

        Parameters
        ----------
        z : (M) 1D ndarray of floats
            The height or depth of each age control point.
            In order of increasing depth or decreasing height.
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

        if len(errors * 2) > len(errors):
            errors = np.array(errors)
            z = np.array(z)
            dates = np.array(dates)

        if not is_depth:
            self.depths = length - z
            self.heights = z
        else:
            self.heights = length - z
            self.depths = z

        self.errors = errors
        if not is_2sigma:
            self.errors *= 2

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

        print("\n \tDepth\tHeight\tAge\t2s Error\n")

    def print_agecontrol_dates(self):
        """
        Print the current sequence of age control points.
        """

        self._print_header()

        for n, (d, h, a, e) in enumerate(zip(self.depths, self.heights,
                                             self.dates, self.errors)):
            print(n, '.\t', d, '\t', h, '\t', a, '\t', e)

    def print_monotonicity_results(self):
        """
        Like ``print_agecontrol_dates()``, but with reversal information.
        Blanks indicate dates are perfectly in stratigraphic order.
        """
        print("\n \tDepth\tHeight\tAge\t2s Error\tInformation\n")

        for n, (d, h, a, e) in enumerate(zip(self.depths, self.heights,
                                             self.dates, self.errors)):
            if n in self.intractable:
                x = 'Complete reversal'
            elif n in self.reversal:
                x = 'In order within error'
            else:
                x = ''

            print(n, '.\t', d, '\t', h, '\t', a, '\t', e, '\t', x)

    def print_model_dates(self, print_extrapolated=False):
        """
        Print the modelled age control points.

        Parameters
        ----------
        print_extrapolated : Boolean
            Default False.  If True, includes the extrapolated point(s) at
            the top and/or bottom of the model.
        """

        self._print_header()

        med, dep, err, hts = self._extrap_checker(print_extrapolated)

        try:
            for n, (d, h, a, e) in enumerate(zip(dep, hts, med, err)):
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

        self.errors[inds] *= adjust_by

        self.check_monotonicity()

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

    def view_monotonicity(self, ax, use_depth=True, lw=1, **ekwargs):
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

        good = None
        ok = None
        bad = None

        ax.plot(self.dates, z, lw=lw, color='k')

        good = ax.errorbar(self.dates, z, xerr=self.errors,
                           color=self.good_color, ecolor=self.good_color,
                           linestyle='none', **ekwargs)

        for i in self.reversal:
            ok = ax.errorbar(self.dates[i], z[i], xerr=self.errors[i],
                             color=self.warning_color,
                             ecolor=self.warning_color, linestyle='none',
                             **ekwargs)

        for i in self.intractable:
            bad = ax.errorbar(self.dates[i], z[i], xerr=self.errors[i],
                              color=self.bad_color, ecolor=self.bad_color,
                              linestyle='none', **ekwargs)

        return good, ok, bad

    def monte_carlo(self, successful_sims, max_sims, chunk_size=1000):
        """
        Alternate name for ``self.monty_carlo()``
        """
        self.monte_carlo.__doc__ += self.monty_carlo.__doc__

        return self.monty_carlo(successful_sims, max_sims,
                                chunk_size=chunk_size)

    def monty_carlo(self, successful_sims, max_sims, chunk_size=1000):
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
        chunk_size : int
            Default 1000, the number of simulations to batch.  For most
            runs, 1000 will be optimal in terms of memory and speed.
        """

        # Initialize
        good_sims = []
        bad_sims = []
        max_loops = int(np.round(max_sims / chunk_size))
        good_count = 0
        errors = self.errors / 2

        for i in range(max_loops):
            good_count += self._simulator(chunk_size, good_sims, bad_sims,
                                          errors)
            if good_count >= successful_sims:
                break

        self.good_sims = np.hstack(good_sims)
        self.bad_sims = np.hstack(bad_sims)

        self.model_median = np.median(self.good_sims, axis=1)
        self.model_error = np.nanstd(self.good_sims, axis=1) * 2

        self.sim_count = (good_count, self.bad_sims.shape[1])

    def _simulator(self, chunk_size, good_sims, bad_sims, errors):
        """
        Where actual random dates are chosen

        Parameters
        ----------
        chunk_size : int
            The number of simulations to batch
        good_sims : list of ndarrays of floats
            Contains all monotonic simulations
        bad_sims : list of ndarrays of floats
            Contains all non-monotonic simulations
        errors : ndarray of floats
            The 1 sigma dating errors
        """
        sims = np.zeros((self.dates.size, chunk_size))
        sims = np.random.normal(size=sims.shape)
        sims *= errors[:, np.newaxis].repeat(chunk_size, axis=1)
        sims += self.dates[:, np.newaxis].repeat(chunk_size, axis=1)

        is_bad = np.diff(sims, axis=0).min(axis=0) < 0

        good_sims.append(sims[:, ~is_bad])
        bad_sims.append(sims[:, is_bad])

        good_count = good_sims[-1].shape[1]

        return good_count

    def view_agemodel(self, ax, badsim_step=10, view_simulations=True,
                      view_results=True, view_original=True,
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

        original = None
        model = None

        # Plot simulations
        if view_simulations:
            for i in range(0, self.sim_count[1], badsim_step):
                ax.plot(self.bad_sims[:, i], z, color=self.warning_color)

            for i in range(0, self.sim_count[0]):
                ax.plot(self.good_sims[:, i], z, color=self.good_color)

        if view_original:
            # Plot original age control points and profile
            original = ax.errorbar(self.dates, z, xerr=self.errors,
                                   marker=original_marker, zorder=zorder,
                                   color=original_color, ecolor=original_color,
                                   **ekwargs)
        if view_results:
            # Plot model results
            model = ax.errorbar(self.model_median, z, xerr=self.model_error,
                                marker=model_marker, zorder=zorder + 2,
                                color=model_color, ecolor=model_color,
                                **ekwargs)

        return original, model

    def set_linear_interpolation_eq(self, use_extrapolated=False):
        """
        Calculate the linear interpolation equation for the model results.

        Parameters
        ----------
        use_extrapolated : Boolean
            Default False.  Indicates whether to use the model results, or the
            model results with extrapolated top/bottom dates.
        """

        med, z, err, _ = self._extrap_checker(use_extrapolated)

        kwargs = {'axis': 0,
                  'bounds_error': False,
                  'kind': 'linear'}

        self.interpolationeq = interp1d(z, med, **kwargs)
        self.interpolationeq_young = interp1d(z, med - err, **kwargs)
        self.interpolationeq_old = interp1d(z, med + err, **kwargs)

    def set_cubic_interpolation_eq(self, use_extrapolated=False):
        """
        Calculate the monotonic cubic interpolation equation for the
        model results.

        Parameters
        ----------
        use_extrapolated : Boolean
            Default False.  Indicates whether to use the model results, or the
            model results with extrapolated top/bottom dates.
        """

        med, z, err, _ = self._extrap_checker(use_extrapolated)

        self.interpolationeq = PchipInterpolator(z, med)
        self.interpolationeq_young = PchipInterpolator(z, med - err)
        self.interpolationeq_old = PchipInterpolator(z, med + err)

    def _extrap_checker(self, use_extrapolated):
        """
        Checks for availability of model results and extrapolated results;
        if available returns the indicated attributes

        Parameters
        ----------
        use_extrapolated : Boolean
            Default False.  Indicates whether to return the model results, or
            the model results with extrapolated top/bottom dates.

        Returns
        -------
        median, depths, error, heights
            May be the extrapolated versions, or the original versions.
        """

        if use_extrapolated:
            if not hasattr(self, 'model_median_extrap'):
                raise AttributeError('use ``self.extrapolate()`` first!')
            return (self.model_median_extrap, self.depths_extrap,
                    self.model_error_extrap, self.heights_extrap)

        else:
            if not hasattr(self, 'model_median'):
                raise AttributeError('First perform age modelling!')
            return (self.model_median, self.depths, self.model_error,
                    self.heights)

    def extrapolate(self, young_z, old_z, young_p0=0, young_p1=(1, 2),
                    old_p0=-1, old_p1=(-2, -3), is_depth=True):
        """
        Extrapolate beyond the age control points to get a top and/or bottom
        point at the limit(s) of the subsamples.  Extrapolation consists of
        drawing a line between indicated points, and extending it outward to
        given z value(s).

        Parameters
        ----------
        young_z : int or float
            The depth or height of the youngest point to extrapolate to.
            If None, will not extrapolate younger than the set of age control
            points, and younger subsamples will have Nans for ages
        old_z : int or float
            The depth or height of the oldest point to extrapolate to.
            If None, will not extrapolate older than the set of age control
            points, and older subsamples will have Nans for ages
        young_p0 : int or tuple of ints
            Default 0.  Age control point index or indices to get first point
            for the younger extrapolation.  If tuple is provided, the first
            point will be the midpoint of two age control points.
        young_p1 : int or tuple of ints
            Default (1, 2).  Age control point index or indices to get first
            point for the younger extrapolation.  If tuple is provided, the
            first point will be the midpoint of two age control points.
        old_p0 : int or tuple of ints
            Default -1.  Age control point index or indices to get first point
            for the older extrapolation.  If tuple is provided, the first
            point will be the midpoint of two age control points.
        young_p1 : int or tuple of ints
            Default (-2, -3).  Age control point index or indices to get first
            point for the older extrapolation.  If tuple is provided, the
            first point will be the midpoint of two age control points.
        is_depth : Boolean
            Default True.  Indicates whether ``old_z``, ``young_z`` are depth
            or height.
        """
        self.model_median_extrap = self.model_median
        self.model_error_extrap = self.model_error
        self.depths_extrap = self.depths

        if young_z is not None:
            if not is_depth:
                young_z = self.length - young_z

            young_p0x, young_p0y, young_p0xerr = self._try_midpoint(young_p0)
            young_p1x, young_p1y, young_p1xerr = self._try_midpoint(young_p1)

            m_young, b_young, xerr_young = self._line(young_p0x, young_p0y,
                                                      young_p0xerr, young_p1x,
                                                      young_p1y, young_p1xerr)

            young_age = (young_z - b_young) / m_young

            self.model_median_extrap = np.concatenate((
                np.array([young_age]), self.model_median_extrap))
            self.model_error_extrap = np.concatenate((
                np.array([xerr_young]), self.model_error_extrap))
            self.depths_extrap = np.concatenate((
                np.array([young_z]), self.depths_extrap))

        if old_z is not None:
            if not is_depth:
                old_z = self.length - old_z

            old_p0x, old_p0y, old_p0xerr = self._try_midpoint(old_p0)
            old_p1x, old_p1y, old_p1xerr = self._try_midpoint(old_p1)

            m_old, b_old, xerr_old = self._line(old_p0x, old_p0y, old_p0xerr,
                                                old_p1x, old_p1y, old_p1xerr)

            old_age = (old_z - b_old) / m_old

            self.model_median_extrap = np.concatenate((
                self.model_median_extrap, np.array([old_age])))
            self.model_error_extrap = np.concatenate((
                self.model_error_extrap, np.array([xerr_old])))
            self.depths_extrap = np.concatenate((
                self.depths_extrap, np.array([old_z])))

        self.heights_extrap = self.length - self.depths_extrap

    def _line(self, p0x, p0y, p0xerr, p1x, p1y, p1xerr):
        """
        Get components of a line between p0 and p1.

        Parameters
        ----------
        p0x, p1x : floats
            The x values (ages) of the points
        p0y, p1y : floats
            The y values (depths) of the points
        p0xerr, p1xerr : floats
            The errors on the x values (ages)

        Returns
        -------
        m : float
            Slope of line between p0 and p1
        b : float
            Y-intercept of line
        xerr : float
            Error on new x (age) value
        """
        m = (p1y - p0y) / (p1x - p0x)
        b = p1y - (m * p1x)
        xerr = self._addsub_error(p0xerr, p1xerr)

        return m, b, xerr

    def _try_midpoint(self, item):
        """
        Distinguish between tuples and ints, then either get the indicated
        values or the midpoints/combinations of the indicated values.

        Parameters
        ----------
        item : int or tuple of ints
            The index of an age control point or a pair of indices.

        Returns
        -------
        Returns results of ``self._not_midpoint()`` or ``self._midpoint()``
        """
        try:
            item[0]
        except TypeError:
            return self._not_midpoint(item)
        else:
            return self._midpoint(item)

    def _midpoint(self, item):
        """
        Find the midpoint between the indicated age control points,
        get the quadratically combined age error.

        Parameters
        ----------
        item : tuple of ints
            The indices of two age control points

        Returns
        -------
        px : float
            The x value (age) of the midpoint
        py : float
            The y value (depth) of the midpoint
        pxerr : float
            The error on px
        """
        p0_ind = item[0]
        p1_ind = item[1]

        p0x = self.model_median[p0_ind]
        p1x = self.model_median[p1_ind]

        p0y = self.depths[p0_ind]
        p1y = self.depths[p1_ind]

        p0xerr = self.model_error[p0_ind]
        p1xerr = self.model_error[p1_ind]

        px = (p1x + p0x) / 2
        py = (p1y + p0y) / 2
        pxerr = self._addsub_error(p0xerr, p1xerr)

        return px, py, pxerr

    def _addsub_error(self, p0xerr, p1xerr):
        """
        Get the quadratically combined error.  Absolute errors, add/subtract

        Parameters
        ----------
        p0xerr, p1xerr : floats
            The errors to combine

        Returns
        -------
        Returns the square root of the sum of the squares of the errors.
        """
        return np.sqrt(p0xerr ** 2 + p1xerr ** 2)

    def _not_midpoint(self, item):
        """
        Retrieves model median, depth, and error

        Parameters
        ----------
        item : int
            Index of modeled age control point

        Returns
        -------
        px : float
            Model median age at ``item``
        py : float
            Depth at ``item``
        pxerr : float
            Error on age at ``item``
        """
        px = self.model_median[item]
        py = self.depths[item]
        pxerr = self.model_error[item]

        return px, py, pxerr

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

        try:
            self.length - subsample_z
        except TypeError:
            subsample_z = np.array(subsample_z)

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

        z = self.subsample_depths
        if not use_depth:
            z = self.subsample_heights

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

        for d, e, h in zip(self.dates[::-1], self.errors[::-1],
                           self.heights[::-1]):
            oxcal.extend(['      Date("",N(calBP(' + str(d) + '),' +
                          str(e) + ')){z=' + str(h) + ';};\n'])

        oxcal.extend(['      Boundary(){z=' + str(youngerbound) + ';};\n',
                      '    };\n',
                      '  };\n'])

        with open(output_file, 'w') as oxcalfile:
            oxcalfile.writelines(oxcal)
