import MDAnalysis as mda
import MDAnalysis.topology
import MDAnalysis.coordinates
import mdtraj
import freud
import time
import numpy as np

from schmeud._schmeud import dynamics, ml

# reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')
# u = mda.Universe('input.data', 'input.dump', atom_style='id type mol x y z')

# parse = MDAnalysis.topology.LAMMPSParser.DATAParser('input.data')
# top = parse.parse()
# print(top.n_atoms)
now = time.time()
u = MDAnalysis.coordinates.LAMMPS.DumpReader('input.dump')
print(u.n_frames)
data = []
for ts in u.trajectory[:1000]:
    data.append(ts.positions)
data = np.array(data).astype(np.float32)
print(time.time() - now)
print(data.shape)
now = time.time()
phop = dynamics.p_hop(data, 101)
print(time.time() - now)

# print(phop)

# t = mdtraj.load_lammpstrj('input.dump', top='input_trj.psf', frame=0)

# print(t.n_frames)
# rdf = freud.density.RDF(100, 4.0)

# for frame in reader:
#     rdf.compute(system=frame, reset=False)