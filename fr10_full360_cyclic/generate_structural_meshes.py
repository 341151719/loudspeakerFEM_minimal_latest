from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import meshio, gmsh

HERE=Path(__file__).resolve().parent; MESH=HERE/'meshes';MESH.mkdir(exist_ok=True)


def _hex6(a,b,c,d,e,f,g,h):
    return [[a,b,c,g],[a,c,d,g],[a,d,h,g],[a,h,e,g],[a,e,f,g],[a,f,b,g]]

def grid_to_tets(P):
    nt,ns,nq=P.shape[0]-1,P.shape[1]-1,P.shape[2]-1; nodes=P.reshape(-1,3)
    def idx(i,j,k): return (i*(ns+1)+j)*(nq+1)+k
    t=[]
    for i in range(nt):
      for j in range(ns):
       for k in range(nq):
        a=idx(i,j,k);b=idx(i+1,j,k);c=idx(i+1,j+1,k);d=idx(i,j+1,k);e=idx(i,j,k+1);f=idx(i+1,j,k+1);g=idx(i+1,j+1,k+1);h=idx(i,j+1,k+1)
        t.extend(_hex6(a,b,c,d,e,f,g,h))
    return nodes,np.asarray(t,np.int32)

def axisym_shell(rvals,zvals,thickness,ntheta=16):
    rvals=np.asarray(rvals,float);zvals=np.asarray(zvals,float);assert len(rvals)==len(zvals)
    th=np.linspace(0,math.pi/2,ntheta+1); ns=len(rvals)-1
    # meridional front normal n=(-dz,dr) / ds, chosen with +z component for increasing r
    dr=np.gradient(rvals);dz=np.gradient(zvals);L=np.hypot(dr,dz);nr=-dz/L;nz=dr/L
    P=np.empty((ntheta+1,ns+1,2,3),float)
    for i,a in enumerate(th):
      ca,sa=math.cos(a),math.sin(a)
      for j,(r,z) in enumerate(zip(rvals,zvals)):
        for q,sgn in enumerate((-0.5,0.5)):
          ro=r+sgn*thickness*nr[j];zo=z+sgn*thickness*nz[j];P[i,j,q]=(ro*ca,ro*sa,zo)
    return grid_to_tets(P)

def annular(ri,ro,z0,z1,ntheta=16,nr=2,nz=4,rvals=None,zvals=None):
    rr=np.asarray(rvals,float) if rvals is not None else np.linspace(ri,ro,nr+1);zz=np.asarray(zvals,float) if zvals is not None else np.linspace(z0,z1,nz+1);th=np.linspace(0,math.pi/2,ntheta+1)
    P=np.empty((ntheta+1,len(rr),len(zz),3),float)
    for i,a in enumerate(th):
      ca,sa=math.cos(a),math.sin(a)
      for j,r in enumerate(rr):
       for k,z in enumerate(zz):P[i,j,k]=(r*ca,r*sa,z)
    return grid_to_tets(P)

def write_structural(cfg):
    g=cfg['geometry'];nt=int(cfg['numerics']['structural_ntheta_quarter_base']); mr=float(cfg['numerics'].get('structural_meridional_refine',1.0))
    ni=lambda n:max(2,int(round(n*mr)))
    # mm. Front-side geometry is explicitly curved; thickness is normal to the meridional midsurface.
    ri,ro=10.30,40.05;zi,zo=-15.20,-2.60
    rcone=np.unique(np.r_[np.linspace(ri,17.50,ni(5)),np.linspace(17.50,ro,ni(10))]); zcone=zi+(rcone-ri)/(ro-ri)*(zo-zi)
    rdc=np.linspace(1.20,17.50,ni(9)); zedge=zi+(17.50-ri)/(ro-ri)*(zo-zi); zdc=-7.05+(rdc-1.20)/(17.50-1.20)*(zedge+7.05)
    rs=np.linspace(40.05,49.75,ni(8)); ss=(rs-40.05)/(49.75-40.05); zs=zo+(float(g['surround_outer_z_mm'])-zo)*ss+float(g['surround_roll_height_mm'])*np.sin(np.pi*ss)
    rsp=np.linspace(10.00,31.50,ni(11));sp=(rsp-10)/(31.5-10); zsp=-21.50+float(g['spider_corrugation_amp_mm'])*np.sin(6*np.pi*sp)
    former_z=np.unique(np.r_[np.linspace(-31.5,-30.75,2),np.linspace(-30.75,-24.75,5),np.linspace(-24.75,-21.5,3),np.linspace(-21.5,-15.2,5),[-14.75]])
    specs={
      'surround':axisym_shell(rs,zs,0.45,nt),
      'cone':axisym_shell(rcone,zcone,0.30,nt),
      'dustcap':axisym_shell(rdc,zdc,0.28,nt),
      'spider':axisym_shell(rsp,zsp,0.30,nt),
      'former':annular(9.88,10.00,-31.50,-14.75,ntheta=nt,nr=1,zvals=former_z),
      'coil':annular(10.00,10.75,-30.75,-24.75,ntheta=nt,nr=2,nz=6),
      'neck_glue':annular(10.00,10.30,-15.25,-14.70,ntheta=nt,nr=1,zvals=[-15.25,-15.20,-14.75,-14.70])}
    rows=[]
    for name,(p,t) in specs.items():
        path=MESH/f'{name}.msh';meshio.write_points_cells(path,p,[('tetra',t)],file_format='gmsh22',binary=False);rows.append((name,len(p),len(t)))
    return rows

def write_acoustic(cfg):
    R=float(cfg['geometry']['acoustic_radius_mm']);near=float(cfg['numerics']['acoustic_mesh_near_mm']);far=float(cfg['numerics']['acoustic_mesh_far_mm'])
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('acoustic_base_quarter');occ=gmsh.model.occ
    tag=occ.addSphere(0,0,0,R,angle1=0,angle2=math.pi/2,angle3=math.pi/2);occ.synchronize();gmsh.model.addPhysicalGroup(3,[tag],1)
    fld=gmsh.model.mesh.field.add('Ball')
    for k,v in [('XCenter',0),('YCenter',0),('ZCenter',0),('Radius',70),('VIn',near),('VOut',far)]:gmsh.model.mesh.field.setNumber(fld,k,v)
    gmsh.model.mesh.field.setAsBackgroundMesh(fld);gmsh.option.setNumber('Mesh.MeshSizeMin',near);gmsh.option.setNumber('Mesh.MeshSizeMax',far);gmsh.option.setNumber('Mesh.Algorithm3D',1);gmsh.option.setNumber('Mesh.ElementOrder',1);gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
    gmsh.model.mesh.generate(3);path=MESH/'acoustic_base_quarter.msh';gmsh.write(str(path));nn=len(gmsh.model.mesh.getNodes()[0]);ne=sum(len(x) for x in gmsh.model.mesh.getElements(3)[1]);gmsh.finalize();return nn,ne

if __name__=='__main__':
    cfg=json.loads((HERE/'configs/fr10_full360_cyclic.json').read_text())
    print('structural', write_structural(cfg))
    print('NOTE: generate periodic acoustic mesh separately with gen_periodic_quarter_ac.py')
