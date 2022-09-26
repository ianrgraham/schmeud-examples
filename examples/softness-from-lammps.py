import MDAnalysis as mda
import mdtraj
import freud

# reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')
u = mda.Universe('input.data', 'input.dump', atom_style='id mol x y z')
# t = mdtraj.load_lammpstrj('input.dump', top='input_trj.pdb')

# print(t.n_frames)
# rdf = freud.density.RDF(100, 4.0)

# for frame in reader:
#     rdf.compute(system=frame, reset=False)