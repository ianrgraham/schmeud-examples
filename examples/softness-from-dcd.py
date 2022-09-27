import MDAnalysis as mda
import MDAnalysis.topology
import MDAnalysis.coordinates
import freud
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

from schmeud._schmeud import dynamics, ml


# get trajectory reader
traj = MDAnalysis.coordinates.DCD.DCDReader('training.dcd')

# make freud box and get type array
initial_ts = traj[0]
box = freud.box.Box.from_matrix(initial_ts.triclinic_dimensions)
type_id = np.zeros(len(initial_ts.positions), dtype=np.uint8)

# extract positions into 3D np.array
data = []
for ts in traj.trajectory[:200]:
    data.append(ts.positions)
data = np.array(data).astype(np.float32)

# compute phop
phop = dynamics.p_hop(data, 101)

output = []

types = 1
mus = np.linspace(0.4, 3.0, 27, dtype=np.float32)
mu_spread = 3

for pos, phop in tqdm(zip(data, phop)):

    # simple conditional filter to get hard and soft indices
    soft_indices = np.ravel(np.argwhere(phop > 0.2))
    hard_indices = np.ravel(np.argwhere(phop < 0.022))

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

    out_phop = np.concatenate([phop[soft_indices], phop[hard_indices]])
    out_rearrang = np.concatenate([np.ones_like(soft_indices), np.zeros_like(hard_indices)])

    output.append([out_phop, out_rearrang, sf])


df = pd.DataFrame(output, columns=['phop', 'rearrang', 'sf'])

# explode (unwrap) the lists at each index
df = df.apply(pd.Series.explode).reset_index()

# arrow is a very fast storage format, and is used by both feather and parquet
# must have pyarrow installed to use
df.to_feather("training.feather")
# or
# df.to_parquet("training.parquet")

print(df.head())