"""Priors for ISP net."""
import torch
from typing import Union, Tuple, Optional, Callable
from sbi.utils import RestrictionEstimator as RE


class _Prior_ISPnet_6params:
    """Prior for 6 parameters of inhibitory synaptic plasticity network."""

    def __init__(
        self,
        lower_lim: Optional[list] = None,
        upper_lim: Optional[list] = None
    ):
        """
        Set up prior for 6 parameters of ISPnet.

        Args:
            num_samples: number of samples.
            lower_lim: lower bound of uniform distribution for each parameter.
            upper_lim: upper bound of uniform distribution for each parameter.
            return_numpy: if True, return thetas as numpy arrays. Otherwise,
                return torch tensors.
        """
        # prior
        if lower_lim is None:
            self.lower_lim = torch.ones((1, 6)) * (-2)
        else:
            assert len(lower_lim) == 6
            self.lower_lim = torch.FloatTensor(lower_lim).reshape(1, 6)

        if upper_lim is None:
            self.upper_lim = torch.ones((1, 6)) * 2
        else:
            assert len(upper_lim) == 6
            self.upper_lim = torch.FloatTensor(upper_lim).reshape(1, 6)

    def sample(
        self, num_samples: Union[torch.Size, Tuple[int, ...]] = torch.Size([1, ])
               ):
        """
        Forward pass to sample from prior.

        Args:
            num_samples: number of samples from prior.
        """
        sz = torch.Size([num_samples[0], 6])
        theta = self.lower_lim + torch.rand(sz) * (
            self.upper_lim - self.lower_lim
        )
        return theta


class _Prior_bg_IF_EEEIIEII_6pPol:
    """Prior for 24 parameters network + 1 param of input rates."""

    def __init__(
        self,
        lower_lim: Optional[list] = None,
        upper_lim: Optional[list] = None,
    ):
        """
        Set up prior for 25 parameters (4 6pPol connections + 1 input rate).

        Args:
            num_samples: number of samples.
            lower_lim: lower bound of uniform distribution for each parameter.
            upper_lim: upper bound of uniform distribution for each parameter.

        """
        if lower_lim is None:
            self.lower_lim = torch.ones((1, 25)) * (-2)
            self.lower_lim[1, 24] = 5
        else:
            assert len(lower_lim) == 25
            self.lower_lim = torch.FloatTensor(lower_lim).reshape(1, 25)

        if upper_lim is None:
            self.upper_lim = torch.ones((1, 25)) * 2
            self.lower_lim[1, 24] = 15
        else:
            assert len(upper_lim) == 25
            self.upper_lim = torch.FloatTensor(upper_lim).reshape(1, 25)

    def sample(
        self, num_samples: Union[torch.Size, Tuple[int, ...]] = torch.Size([1, ])
               ):
        """
        Forward pass to sample from prior.

        Args:
            num_samples: number of samples from prior.
        """
        sz = torch.Size([num_samples[0], 25])
        theta = self.lower_lim + torch.rand(sz) * (
            self.upper_lim - self.lower_lim
        )
        return theta


class RestrictedPrior:
    """Wrapper for classifier-based domain restriction of all priors."""

    def __init__(
        self,
        prior_class: str,
        prior_class_args: dict,
        return_numpy: Optional[bool] = True,
        restrict_prior: Optional[bool] = False,
        thetas_for_classifier: Optional[torch.tensor] = None,
        x_for_classifier: Optional[torch.tensor] = None,
        classifier_args: Optional[dict] = dict()
                 ) -> None:
        """
        Set up wrapper for restricted prior.

        This class modifies the functioning of the input prior class. With the default
        settings, this class merely returns samples from the original prior. With
        restrict_prior=True, we train a classifier, and use it to rejection sample
        the input prior only for those parameters that would give "valid" simulations.
        See sbi.utils.RestrictionEstimator docs for more information.

        Args:
            prior_class: name of implemented prior class from `synapsbi.prior`.
            prior_class_args: dictionary of keyword arguments for prior.
            return_numpy: if True, return thetas as numpy arrays. Otherwise,
                return torch tensors.
            restrict_prior: if True, additionally train a classifier to reject prior
                samples that give rise to bad simulations.
            thetas_for_classifier: parameters (regressors) for training classifier.
            x_for_classifier: observations (NaN values are "bad/invalid") for training
                classifier.
            classifier_args: dictionary of keyword arguments for classifier. See
                sbi.utils.RestrictionEstimator docs for more information.
        """
        # check if prior class is implemented
        assert(callable(eval(prior_class)))

        self.prior = eval(prior_class)(**prior_class_args)
        self.return_numpy = return_numpy

        # restricted prior
        self.restrict_prior = restrict_prior
        self.tr_thetas, self.tr_x = thetas_for_classifier, x_for_classifier

        # train restriction estimator
        if self.restrict_prior:
            self.restriction_estimator = RE(prior=self.prior, **classifier_args)
            self.restriction_estimator.append_simulations(self.tr_thetas, self.tr_x)
            self.restriction_estimator.train()
            self.restricted_prior = self.restriction_estimator.restrict_prior(
                allowed_false_negatives=0.01)

    def _return_numpy(sampler):
        def _return_transformed_samples(self, num_samples):
            if self.return_numpy:
                return sampler(self, num_samples).data.numpy()
            else:
                return sampler(self, num_samples)
        return _return_transformed_samples

    @_return_numpy
    def sample(
        self, num_samples: Union[torch.Size, Tuple[int, ...]] = torch.Size([1, ])
                ):
        """
        Forward pass to sample from prior.

        Args:
            num_samples: number of samples from prior.
        """
        if self.restrict_prior:
            return self.restricted_prior.sample(num_samples)
        else:
            return self.prior.sample(num_samples)
        
             
class _Prior_bg_IF_EEEIIEII_6pPol_noInput:
    """Prior for 24 parameters network"""

    def __init__(
        self,
        lower_lim: Optional[list] = None,
        upper_lim: Optional[list] = None,
    ):
        """
        Set up prior for 24 parameters (4 6pPol connections).

        Args:
            num_samples: number of samples.
            lower_lim: lower bound of uniform distribution for each parameter.
            upper_lim: upper bound of uniform distribution for each parameter.

        """
        if lower_lim is None:
            self.lower_lim = torch.ones((1, 24)) * (-2)
        else:
            assert len(lower_lim) == 24
            self.lower_lim = torch.FloatTensor(lower_lim).reshape(1, 24)

        if upper_lim is None:
            self.upper_lim = torch.ones((1, 24)) * 2
        else:
            assert len(upper_lim) == 24
            self.upper_lim = torch.FloatTensor(upper_lim).reshape(1, 24)

    def sample(
        self, num_samples: Union[torch.Size, Tuple[int, ...]] = torch.Size([1, ])
               ):
        """
        Forward pass to sample from prior.

        Args:
            num_samples: number of samples from prior.
        """
        sz = torch.Size([num_samples[0], 24])
        theta = self.lower_lim + torch.rand(sz) * (
            self.upper_lim - self.lower_lim
        )
        return theta
    

class _Prior_bg_CVAIF_EEIE_T4wvceciMLP_noInput:
    """
    Prior for 22 params for 2 rules (eta, Wpre, Wpost). No input rate there (nuisance param, added elsewhere) 
    TOTAL 22
    """

    def __init__(
        self,
        lower_lim: Optional[list] = None,
        upper_lim: Optional[list] = None,
    ):
        """
        Set up prior for 22 parameters (2 T4wvceciMLP connections).

        Args:
            num_samples: number of samples.
            lower_lim: lower bound of uniform distribution for each parameter.
            upper_lim: upper bound of uniform distribution for each parameter.

        """
        if lower_lim is None:
            self.lower_lim = torch.ones((1, 22)) * (-2)
        else:
            assert len(lower_lim) == 22
            self.lower_lim = torch.FloatTensor(lower_lim).reshape(1, 22)

        if upper_lim is None:
            self.upper_lim = torch.ones((1, 22)) * 2
        else:
            assert len(upper_lim) == 22
            self.upper_lim = torch.FloatTensor(upper_lim).reshape(1, 22)

    def sample(
        self, num_samples: Union[torch.Size, Tuple[int, ...]] = torch.Size([1, ])
               ):
        """
        Forward pass to sample from prior.

        Args:
            num_samples: number of samples from prior.
        """
        sz = torch.Size([num_samples[0], 22])
        theta = self.lower_lim + torch.rand(sz) * (
            self.upper_lim - self.lower_lim
        )
        return theta


class _Prior_bg_CVAIF_EEIE_T4wvceciMLP:
    """
    Prior for 22 params for 2 rules (eta, Wpre, Wpost) + input rate there (nuisance param) 
    TOTAL 23
    """
    
    def __init__(
        self,
        lower_lim: Optional[list] = None,
        upper_lim: Optional[list] = None,
    ):
        """
        Set up prior for 23 parameters (2 T4wvceciMLP connections + 1 input rate).

        Args:
            num_samples: number of samples.
            lower_lim: lower bound of uniform distribution for each parameter.
            upper_lim: upper bound of uniform distribution for each parameter.

        """
        if lower_lim is None:
            self.lower_lim = torch.ones((1, 23)) * (-1)
            self.lower_lim[1, 22] = 5
        else:
            assert len(lower_lim) == 23
            self.lower_lim = torch.FloatTensor(lower_lim).reshape(1, 23)

        if upper_lim is None:
            self.upper_lim = torch.ones((1, 23)) * 1
            self.lower_lim[1, 22] = 15
        else:
            assert len(upper_lim) == 23
            self.upper_lim = torch.FloatTensor(upper_lim).reshape(1, 23)

    def sample(
        self, num_samples: Union[torch.Size, Tuple[int, ...]] = torch.Size([1, ])
               ):
        """
        Forward pass to sample from prior.

        Args:
            num_samples: number of samples from prior.
        """
        sz = torch.Size([num_samples[0], 23])
        theta = self.lower_lim + torch.rand(sz) * (
            self.upper_lim - self.lower_lim
        )
        return theta