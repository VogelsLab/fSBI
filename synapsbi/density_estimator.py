"""Code for density estimation."""
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from sbi.inference import SNPE
from sbi.utils import BoxUniform
import torch
from typing import Callable, Optional, List
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pickle


def _get_posterior(thetas: torch.tensor, obs: torch.tensor, prior) -> Callable:
    inference = SNPE(prior=prior, density_estimator="nsf")
    inference = inference.append_simulations(thetas, obs)
    density_estimator = inference.train(training_batch_size=1000)
    return inference.build_posterior(density_estimator)


class MakePosterior:
    """SBI posterior."""

    def __init__(
        self,
        prior: Callable = None,
        theta_dim: int = 1,
        low_lim: List = -2.0,
        up_lim: List = 2.0,
        num_ensemble: int = 10,
    ):
        """
        Make SBI posteriors.

        Args:
            prior: prior for SNPE. If None, defaults to BoxUniform prior.
            theta_dim: number of parameter dimensions
            low_lim: lower limit for uniform distribution
            up_lim: upper limit for uniform distribution
            num_ensemble: number of posteriors in ensemble.
        """
        if prior is None:
            if type(low_lim) == float:
                low = torch.FloatTensor([low_lim] * theta_dim)
            else:
                low = torch.FloatTensor(low_lim)
            if type(up_lim) == float:
                high = torch.FloatTensor([up_lim] * theta_dim)
            else:
                high = torch.FloatTensor(up_lim)

            self.prior = BoxUniform(low=low, high=high)
        else:
            self.prior = prior
        self.theta_dim = theta_dim
        self.num_ensemble = num_ensemble

    def get_ensemble_posterior(
        self,
        thetas: torch.tensor,
        obs: torch.tensor,
        prior: Optional[Callable] = None,
        n_jobs: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Callable:
        """
        Make ensemble posterior.

        Args:
            thetas: simulator parameters for training.
            obs: simulations for training  corresponding to thetas.
            prior: prior for density estimator.
            n_jobs: number of training jobs to launch in parallel.
            save_path: path to save trained posterior.
        """
        if n_jobs is None:
            n_jobs = self.num_ensemble

        print("prior will be put to none for now anyway. we do rejection sampling after the fitting instead")

        with parallel_backend("threading", n_jobs=n_jobs):
            posteriors = Parallel()(
                delayed(_get_posterior)(thetas, obs, prior)
                for _ in range(self.num_ensemble)
            )
            ensemble = NeuralPosteriorEnsemble(posteriors)

        setattr(self, "posterior", ensemble)

        if save_path is not None:
            self._save_trained_posterior(save_path)

        return ensemble

    def get_post_samples(
        self,
        default_x: torch.tensor,
        num_samples: int = 1,
        posterior: Optional[Callable] = None,
        bounds: Optional[List] = None,
        save_path: Optional[str] = None,
    ) -> torch.tensor:
        """
        Get posterior samples.

        Args:
            default_x: list of observations to condition posterior.
            num_samples: number of samples per default_x.
            bounds: [ [low_lim_theta1, ..., low_lim_thetaN], [up_lim_theta1, ..., up_lim_thetaN] ] for rejection sampling, None if rejection sampling not desired
            posterior: trained density estimator. If None, defaults to class
                attribute "posterior".
            save_path: path to save posterior samples.
        """
        assert default_x.dim() > 1
        if posterior is None:
            assert hasattr(self, "posterior")
            posterior = self.posterior

        post_thetas = []

        for x in default_x:
            thetas = posterior.sample((num_samples,),
                                      x=x,
                                      show_progress_bars=False)
            post_thetas.append(thetas)
        post_thetas = torch.cat(post_thetas, 0)

        if save_path is not None:
            self._save_to_file(save_path, post_thetas, default_x)

        return post_thetas
    
    def rsample(
         self,
         default_x: torch.tensor,
         bounds: List
    ) -> torch.tensor:
        """
        Get posterior samples.

        Args:
            default_x: list of observations to condition posterior.
            bounds:[ torch tensor([low_lim_theta1, ..., low_lim_thetaN]), torch tensor([up_lim_theta1, ..., up_lim_thetaN]) ] for rejection sampling, None if rejection sampling not desired
        """
        
        assert default_x.dim() > 1
        assert hasattr(self, "posterior")
        
        post_thetas = []
        n_thetas = len(bounds["low"])
        
        for x in default_x:
            done = False
            while not done:
                thetas = self.posterior.sample((1,),x=x,show_progress_bars=False)
                # Reject samples that are outside prior bounds
                if torch.all( torch.logical_and(bounds["low"] < thetas, thetas < bounds["high"]) ):
                    done = True
                    post_thetas.append(thetas)
        post_thetas = torch.cat(post_thetas, 0)
        return post_thetas

    def _save_to_file(self, save_path, thetas, default_x):
        if save_path[-3:] == "npz":
            np.savez(save_path, thetas=thetas, default_x=default_x)
        else:
            raise NotImplementedError

    def _save_trained_posterior(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
