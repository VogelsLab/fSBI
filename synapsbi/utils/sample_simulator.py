"""Utils for sampling from the simulator."""
import hashlib


def _make_unique_samples(num_samples, prior=None, thetas=None, saved_seeds=[]):
    seeds = []
    if prior is None:
        assert thetas is not None
    else:
        thetas = prior.sample(num_samples)
    new_thetas = []
    for th in thetas:
        str_th = str(th).encode()
        seed = hashlib.md5(str_th).hexdigest()
        if seed not in saved_seeds:
            seeds.append(seed)
            new_thetas.append(th)
    return new_thetas, seeds


def _forward(inputs):
    simulator = inputs.pop("simulator")

    fwd = simulator.sample(**inputs)
    return fwd
