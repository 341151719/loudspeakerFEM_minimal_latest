from pathlib import Path
from collections import defaultdict
import numpy as np,meshio,json

def flatten(m):
 phys=m.cell_data_dict.get('gmsh:physical',{});tris=[];tt=[];lines=[];lt=[];ti=li=0
 for cb in m.cells:
  if cb.type=='triangle':
   a=np.asarray(cb.data,int);tris.append(a);tt.append(np.asarray(phys['triangle'][ti:ti+len(a)],int));ti+=len(a)
  elif cb.type=='line':
   a=np.asarray(cb.data,int);lines.append(a);lt.append(np.asarray(phys['line'][li:li+len(a)],int));li+=len(a)
 return np.vstack(tris),np.concatenate(tt),np.vstack(lines),np.concatenate(lt)
def ek(a,b):return (a,b) if a<b else (b,a)
parent=Path('/mnt/data/fig5_work/global_p2_mesh.msh');m=meshio.read(parent);pts=np.asarray(m.points,float);tri,tags,lines,lt=flatten(m);ne=len(tri)
edgeall=defaultdict(list)
for i,(a,b,c) in enumerate(tri):
 for e in [ek(int(a),int(b)),ek(int(b),int(c)),ek(int(c),int(a))]:edgeall[e].append(i)
marked=set()
for e,adj in edgeall.items():
 for i in adj:
  if tags[i] not in (6,23):continue
  if len(adj)==1 or any(tags[j]!=tags[i] for j in adj):marked.add(i)
front=set(marked)
for _ in range(1):
 nxt=set()
 for i in front:
  a,b,c=tri[i]
  for e in [ek(int(a),int(b)),ek(int(b),int(c)),ek(int(c),int(a))]:
   for j in edgeall[e]:
    if tags[j]==tags[i] and j not in marked:nxt.add(j)
 marked|=nxt;front=nxt
split=set()
for i in marked:
 a,b,c=tri[i];split.update([ek(int(a),int(b)),ek(int(b),int(c)),ek(int(c),int(a))])
sorted_edges=sorted(split);mid={e:len(pts)+i for i,e in enumerate(sorted_edges)};newpts=np.vstack([pts,np.array([(pts[a]+pts[b])/2 for a,b in sorted_edges])])
newtri=[];newtags=[];parents=[]
for ie,(a,b,c) in enumerate(tri):
 es=[ek(int(a),int(b)),ek(int(b),int(c)),ek(int(c),int(a))];s=[e in split for e in es];n=sum(s);m0=mid.get(es[0]);m1=mid.get(es[1]);m2=mid.get(es[2])
 if n==0:kids=[[a,b,c]]
 elif n==1:
  if s[0]:kids=[[a,m0,c],[m0,b,c]]
  elif s[1]:kids=[[b,m1,a],[m1,c,a]]
  else:kids=[[c,m2,b],[m2,a,b]]
 elif n==2:
  if s[0] and s[1]:kids=[[b,m1,m0],[a,m0,c],[m0,m1,c]]
  elif s[1] and s[2]:kids=[[c,m2,m1],[b,m1,a],[m1,m2,a]]
  else:kids=[[a,m0,m2],[c,m2,b],[m2,m0,b]]
 else:kids=[[a,m0,m2],[m0,b,m1],[m2,m1,c],[m0,m1,m2]]
 newtri.extend(kids);newtags.extend([int(tags[ie])]*len(kids));parents.extend([ie]*len(kids))
newlines=[];newlt=[]
for (a,b),t in zip(lines,lt):
 e=ek(int(a),int(b))
 if e in split:
  mm=mid[e];newlines.extend([[a,mm],[mm,b]]);newlt.extend([int(t),int(t)])
 else:newlines.append([a,b]);newlt.append(int(t))
out=Path('/mnt/data/fig5_work/skin_p2_mesh.msh');meshio.write(out,meshio.Mesh(newpts,[('line',np.asarray(newlines,int)),('triangle',np.asarray(newtri,int))],cell_data={'gmsh:physical':[np.asarray(newlt,int),np.asarray(newtags,int)],'gmsh:geometrical':[np.asarray(newlt,int),2000+np.asarray(newtags,int)]},field_data=m.field_data),file_format='gmsh22',binary=True)
# project static
v=meshio.read('/mnt/data/fig5_work/global_p2_static_projected.vtu');A0=np.asarray(v.point_data['A_phi_Wb_per_m']);A=np.empty(len(newpts));A[:len(pts)]=A0
for e,i in mid.items():A[i]=.5*(A0[e[0]]+A0[e[1]])
cd=v.cell_data_dict;names=['B_r_T','B_z_T','B_norm_T','H_norm_A_m','mu_r'];cell={n:np.asarray(cd[n]['triangle'])[np.asarray(parents)] for n in names}
sv=Path('/mnt/data/fig5_work/skin_p2_static.vtu');meshio.write(sv,meshio.Mesh(newpts,[('triangle',np.asarray(newtri,int))],point_data={'A_phi_Wb_per_m':A},cell_data={**{n:[x] for n,x in cell.items()},'domain':[np.asarray(newtags,int)]}),binary=True)
# stats
p=newpts[:,:2];T=np.asarray(newtri,int);tg=np.asarray(newtags,int);E=np.stack([np.linalg.norm(p[T[:,0]]-p[T[:,1]],axis=1),np.linalg.norm(p[T[:,1]]-p[T[:,2]],axis=1),np.linalg.norm(p[T[:,2]]-p[T[:,0]],axis=1)],1)
st={'parent_nodes':len(pts),'parent_triangles':len(tri),'child_nodes':len(newpts),'child_triangles':len(T),'marked_triangles':len(marked),'split_edges':len(split),'domains':{}}
for d in [6,23]:st['domains'][str(d)]={'triangles':int((tg==d).sum()),'edge_mm_percentiles':np.percentile(E[tg==d]*1e3,[0,10,50,90,100]).tolist()}
Path('/mnt/data/fig5_work/skin_p2_mesh.json').write_text(json.dumps(st,indent=2));print(json.dumps(st,indent=2))
