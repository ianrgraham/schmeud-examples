import MDAnalysis as mda
import MDAnalysis.topology
import MDAnalysis.coordinates
import mdtraj
import freud

# reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')
# u = mda.Universe('input.data', 'input.dump', atom_style='id type mol x y z')

# parse = MDAnalysis.topology.LAMMPSParser.DATAParser('input.data')
# top = parse.parse()
# print(top.n_atoms)
u = MDAnalysis.coordinates.LAMMPS.DumpReader('input.dump')
print(u.n_frames)
for ts in u.trajectory:
    print(ts.positions[40:42])
    break
# t = mdtraj.load_lammpstrj('input.dump', top='input_trj.psf', frame=0)

# print(t.n_frames)
# rdf = freud.density.RDF(100, 4.0)

# for frame in reader:
#     rdf.compute(system=frame, reset=False)