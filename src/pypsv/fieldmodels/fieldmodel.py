from abc import abstractmethod, ABCMeta


class FieldModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        # should set self._l_max, self._knots, self._coeffs, self._n_samples
        pass

    @abstractmethod
    def __call__(self, t):
        # should return coefficients for a given epoch or list of epochs
        pass

    @property
    def l_max(self):
        """ int : The maximum spherical harmonic degree
        """
        return int(self._l_max)

    @l_max.setter
    def l_max(self, val):
        raise RuntimeError("L max is fixed and can not be set!")

    @property
    def n_coeffs(self):
        """ int : The number of coefficients per knot and sample
        """
        return int(self.l_max * (self.l_max + 2))

    @n_coeffs.setter
    def n_coeffs(self, val):
        raise RuntimeError(
            "Number of coefficients is fixed and can not be set!"
        )

    @property
    def n_samples(self):
        """ int : The number of samples per coefficient and knot
        """
        return self._n_samples

    @n_samples.setter
    def n_samples(self, val):
        raise RuntimeError(
            "Number of samples is fixed and can not be set!"
        )

    @property
    def knots(self):
        """ array : Knots of the model
        """
        return self._knots

    @knots.setter
    def knots(self, val):
        raise RuntimeError(
            "Knots are fixed and can not be set!"
        )

    @property
    def coeffs(self):
        """ array : SH coefficients of the model
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, val):
        raise RuntimeError(
            "Coefficients are fixed and can not be set!"
        )
