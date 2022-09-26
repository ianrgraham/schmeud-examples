import MDAnalysis
import freud

reader = MDAnalysis.coordinates.DCD.DCDReader('trajectory.dcd')

rdf = freud.density.RDF(100, 4.0)

for frame in reader:
    rdf.compute(system=frame, reset=False)