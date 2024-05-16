import pathlib
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils
from cedalion.datasets import get_multisubject_fingertapping_snirf_paths
import numpy as np
import xarray as xr
import pint
import matplotlib.pyplot as p

import pandas as pd


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score

xr.set_options(display_max_rows=3, display_values_threshold=50)
np.set_printoptions(precision=4)




vfc = pathlib.Path('../fnirs_data/vfc_high_density/')
fnames = sorted(list(vfc.rglob("*.snirf")))[::-1]
subjects  = [f"sub-{i:02d}" for i in range(1, len(fnames)+1)]

# store data of different subjects in a dictionary
data = {} 
for subject,fname in zip(subjects, fnames):
    print(fname)
    
    elements = cedalion.io.read_snirf(fname)

    amp = elements[0].data[0]
    stim = elements[0].stim # pandas Dataframe
    geo3d = elements[0].geo3d
    # cedalion registers an accessor (attribute .cd ) on pandas DataFrames
    # stim.cd.rename_events( {
    #     "1.0" : "control",
    #     "2.0" : "Tapping/Left",
    #     "3.0" : "Tapping/Right"
    # })
    # TODO: Talk to Eike whether this is needed
    
    dpf = xr.DataArray([6, 6], dims="wavelength", coords={"wavelength" : amp.wavelength})
    
    data[subject] = xr.Dataset(
        data_vars = {
            "amp" : amp,
            "od"  : - np.log( amp / amp.mean("time") ),
            "geo" : geo3d,
            "conc": cedalion.nirs.beer_lambert(amp, geo3d, dpf)
        },
        attrs={"stim" : stim}, # store stimulus data in attrs
        coords={"subject" : subject} # add the subject label as a coordinate
    )
