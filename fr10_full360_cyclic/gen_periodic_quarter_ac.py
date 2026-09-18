from pathlib import Path
import json, math
import gmsh

HERE = Path(__file__).resolve().parent
MESH = HERE / 'meshes'
MESH.mkdir(exist_ok=True)
cfg = json.loads((HERE / 'configs' / 'fr10_full360_cyclic.json').read_text())
R = float(cfg['geometry']['acoustic_radius_mm'])
near = float(cfg['numerics']['acoustic_mesh_near_mm'])
far = float(cfg['numerics']['acoustic_mesh_far_mm'])

gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.model.add('acoustic_periodic_quarter')
o = gmsh.model.occ
# 90-degree sector of the upper half-space; the two radial planes are cyclic-periodic partners.
o.addSphere(0, 0, 0, R, angle1=0, angle2=math.pi/2, angle3=math.pi/2)
o.synchronize()
# Rotation x-axis radial plane -> y-axis radial plane by +90 deg about z.
T = [0,-1,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,1]
gmsh.model.mesh.setPeriodic(2, [4], [3], T)

f = gmsh.model.mesh.field.add('Ball')
for k, v in [('XCenter',0), ('YCenter',0), ('ZCenter',0), ('Radius',70), ('VIn',near), ('VOut',far)]:
    gmsh.model.mesh.field.setNumber(f, k, v)
gmsh.model.mesh.field.setAsBackgroundMesh(f)
gmsh.option.setNumber('Mesh.MeshSizeMin', near)
gmsh.option.setNumber('Mesh.MeshSizeMax', far)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
gmsh.model.mesh.generate(3)
out = MESH / 'acoustic_base_quarter.msh'
gmsh.write(str(out))
print('wrote', out)
print('nodes', len(gmsh.model.mesh.getNodes()[0]), 'elements3D', sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]))
try:
    slave, master, node_tags, node_tags_master, affine = gmsh.model.mesh.getPeriodicNodes(2, 4, True)
    print('periodic slave entity', slave, 'master entity', master, 'node pairs', len(node_tags))
except Exception as e:
    print('periodic node API warning:', e)
gmsh.finalize()
