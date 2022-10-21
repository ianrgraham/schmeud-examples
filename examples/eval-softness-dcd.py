import MDAnalysis as mda
import MDAnalysis.topology
import MDAnalysis.coordinates
import freud
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import pickle

import schmeud
from schmeud._schmeud import dynamics, ml

import sys
sys.argv[1]


# get trajectory reader
traj = mda.coordinates.DCD.DCDReader('eval.dcd')

# make freud box and get type array
initial_ts = traj[0]
box = freud.box.Box.from_matrix(initial_ts.triclinic_dimensions)
type_id = np.zeros(len(initial_ts.positions), dtype=np.uint8)

# extract positions into 3D np.array
data = []
print(traj.n_frames)
for ts in traj.trajectory[::10]: 
    data.append(ts.positions)
data = np.array(data).astype(np.float32)

# NOTE do some window averaging if you want as a preprocessing step

# compute phop
phop = dynamics.p_hop(data, 11)

output = []

types = 1
mus = np.linspace(0.4, 3.0, 27, dtype=np.float32)
mu_spread = 3

# select highest and lowest phop values within thresholds
dynamic_indices = schmeud.ml.group_hard_soft_by_cutoffs(phop, distance=100, hard_distance=100, sub_slice=slice(0, len(phop), 1))

# load the pipeline
with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

for idx, (pos, phop_i) in tqdm(enumerate(zip(data, phop))):

    nlist_query = freud.locality.AABBQuery.from_system((box, pos))

    # need r_min so you ignore yourself (since the indices of the query points are not necessarily the same as the indices of the points in the system)
    nlist = nlist_query.query(
        pos, 
        {'r_min': 0.1, 'r_max': 3.0}
    ).toNeighborList()
    
    sf = ml.radial_sf_snap_generic_nlist(
        nlist.query_point_indices,
        nlist.point_indices,
        nlist.neighbor_counts,
        nlist.segments,
        nlist.distances,
        type_id,
        types,
        mus,
        mu_spread,
    )

    softness = pipeline.decision_function(sf)

    output.append([idx, phop_i, sf, softness])


df = pd.DataFrame(output, columns=['frame', 'phop', 'sf', 'softness'])

# explode (unwrap) the lists at each frame index
df = df.set_index(['frame']).apply(pd.Series.explode).reset_index()

df.to_feather("eval.feather")
