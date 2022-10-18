import MDAnalysis as mda
import MDAnalysis.topology
import MDAnalysis.coordinates
import freud
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

import schmeud
from schmeud._schmeud import dynamics, ml


# get trajectory reader
traj = mda.coordinates.DCD.DCDReader('training.dcd')

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

# for idx, (pos, phop) in tqdm(enumerate(zip(data, phop))):

#     # simple conditional filter to get hard and soft indices
#     soft_indices = np.ravel(np.argwhere(phop > 0.2))
#     hard_indices = np.ravel(np.argwhere(phop < 0.022))

for idx, soft_indices, hard_indices in tqdm(dynamic_indices):

    soft_indices = np.array(soft_indices, dtype=np.uint32)
    hard_indices = np.array(hard_indices, dtype=np.uint32)

    pos = data[idx]
    phop_i = phop[idx]

    # now = time.time()
    nlist_query = freud.locality.AABBQuery.from_system((box, pos))

    query_pos = np.concatenate([pos[soft_indices], pos[hard_indices]])
    # need r_min so you ignore yourself (since the indices of the query points are not necessarily the same as the indices of the points in the system)
    nlist = nlist_query.query(
        query_pos, 
        {'r_min': 0.1, 'r_max': 3.0}
    ).toNeighborList()
    # nlist_time = time.time() - now
    
    # now = time.time()
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
    # sf_time = time.time() - now
    # print(nlist_time, sf_time)

    out_phop = np.concatenate([phop_i[soft_indices], phop_i[hard_indices]])
    out_rearrang = np.concatenate([np.ones_like(soft_indices), np.zeros_like(hard_indices)])

    output.append([idx, out_phop, out_rearrang, sf])


df = pd.DataFrame(output, columns=['frame', 'phop', 'rearrang', 'sf'])

# explode (unwrap) the lists at each frame index
df = df.set_index(['frame']).apply(pd.Series.explode).reset_index()

# arrow is a very fast storage format, and is used by both feather and parquet
# must have pyarrow installed to use
# df.to_feather("training.feather")
# or
# df.to_parquet("training.parquet")

# training step

# load dataframe
# df = pd.read_feather("training.feather")

counts = df.value_counts("rearrang").min()
choose = min(counts, 10_000)
print(choose)
df = df.groupby("rearrang").sample(n=choose)

Xs = list(df["sf"])
ys = list(df["rearrang"])

pipeline, (acc, conf_mat) = schmeud.ml.train_hyperplane_pipeline(Xs, ys)


softness = pipeline.decision_function(Xs)