import warnings
import numpy as np

def subsample_iflarger(data, **kwargs):
    if data.X.shape[0] > kwargs['num']:
        return subsample(data,**kwargs)
    return data.copy() if kwargs.get('copy', False) else data


def subsample(data,num=1000, seed=None, copy = False):
    np.random.seed(seed)
    obs_indices = np.random.choice(data.n_obs, size=num, replace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r=  data[obs_indices]
        if copy:
            r = r.copy()
        r.obs_names_make_unique()
    return r


