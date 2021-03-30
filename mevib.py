# -*- coding: utf-8 -*-
''' 
 FITE 2A - ARTS ET METIERS 
 Utilities for MEVIB Course
 Start with runfile('mevib.py')
 
 Copyright 2019-2021, Eric Monteiro, Etienne Balmes
 xxx IPython / external xxx
'''
from matplotlib import cm
import matplotlib.pyplot as plot
import matplotlib.animation as animation
import mpl_toolkits.mplot3d as m3d
from numpy.linalg import svd as svd
import scipy.linalg as linalg
from scipy.linalg import block_diag
from scipy.optimize import root
import scipy.io as sio
import numpy as np
import warnings

#------------------------------------------------------------------------------
# DEFINITION OF CLASSES
#------------------------------------------------------------------------------
class model:
    def __init__(self, Node=[], Elt=[], pl=[], il=[]):
        self.Node=Node
        self.Elt=Elt
        self.pl=pl
        self.il=il
        
class res:
    def __init__(self, DOF=[], Def=[], data=[],TR=[]):
        self.DOF=DOF
        self.Def=Def
        self.data=data
        self.TR=TR

#------------------------------------------------------------------------------
# DEFINITION OF MATHS FUNCTIONS 
#------------------------------------------------------------------------------
def feeig(K,M=[],norm='M'):
    ''' Compute eigenvalues and eigenvectors '''
    om2, vecs = linalg.eig(K,M) if len(M)>0 else linalg.eig(K)
    om=np.sqrt(om2);om=om.real;idx=np.argsort(om)
    val=om[idx];phi=vecs[:,idx]
    #norm to mass
    if norm=='M':
     gm=np.sqrt(np.diag(phi.T @ M @ phi)); 
     fact=np.reshape(np.tile(1./gm,gm.size),[gm.size,gm.size])
     phi=np.multiply(phi,fact)
    #output
    return [val.real, phi]  


def cosd(x):
    ''' Compute cosinus in degree '''
    return np.round(np.cos(x*np.pi/180.),12)

def sind(x):
    ''' Compute sinus in degree '''
    return np.round(np.sin(x*np.pi/180.),12)

#------------------------------------------------------------------------------
# DEFINITION OF FUNCTIONS 
#------------------------------------------------------------------------------        
def compute_section(param):
    ''' Compute BEAM section properties '''
    if not('section' in param.keys()):
        param['section']='rect'
        
    if param['section']=='rect':
        param['S'] = param['b']*param['h']
        param['I'] = param['b']*param['h']**3/12.0
        
    return param

  

def quad_seg(n=1):
    '''  Quadrature points for segment in [-1;1] '''
    if n==2:
        xg=np.array([[-np.sqrt(3)/3.,1],[np.sqrt(3)/3.,1]])
    elif n==3:
        xg=np.array([[-np.sqrt(3./5.),5./9.],[0.,8./9.],[np.sqrt(3./5.),5./9.]])  
    elif n==4:
        xg=np.array([[-0.339981043584856,0.652145154862546],[0.339981043584856,0.652145154862546],
                     [-0.861136311594053,0.347854845137454],[0.861136311594053,0.347854845137454]])
    elif n==50:
        xg=np.concatenate( np.linspace(0,1,50).T,np.ones[n,1]);
        
    else: #Center point
        n=1; xg=np.array([0.,2.],ndmin=2)
    NdN=np.concatenate((np.array([(1.-xg[:,0])/2.,(1.+xg[:,0])/2.]), #N
                       np.array([[-1],[1]])*0.5*np.ones([1,n]))      #dN
                      ).T 
    return [xg,NdN]
    
def plot_bode(xyplot,n=0):
    ''' Plot Bode Diagrams  ''' 
    plot.figure()
    plot.loglog(xyplot['X'],np.abs(xyplot['Y'][:,n]))
    plot.xlabel(xyplot['Xlabel'])
    plot.ylabel(xyplot['Ylabel'][n])
    plot.grid()
    plot.show()
    
def plot2D(xyplot,style='-',xscale='linear',yscale='linear',xlim=[],ylim=[],):
    ''' Plot 2D curves  ''' 
    plot.figure();plot.plot(xyplot['X'],xyplot['Y'],style)
    plot.xlabel(xyplot['Xlabel']);plot.ylabel(xyplot['Ylabel'])
    plot.grid();plot.xscale(xscale);plot.yscale(yscale);
    if 'legend' in xyplot.keys(): plot.legend(xyplot['legend'])
    if len(xlim)>0: plot.xlim(xlim)
    if len(ylim)>0: plot.ylim(ylim)
    plot.show()
    
# animation 
def anim1D_mode(mo1,vals,vecs,idmode=1, num=100, fact=[]):
    
    #get model for visualization
    [EGroup,nGroup,ElemP] = getegroup(mo1.Elt)
    jGroup=0  #for jGroup in xrange(nGroup):
    eltind=range(EGroup[jGroup]+1,EGroup[jGroup+1])
    if ElemP[jGroup]=='beam2': npts=2
    if ElemP[jGroup]=='beam3': npts=3
    Elt=np.array(mo1.Elt[eltind,0:npts]-1,dtype=int)  
    mybox=np.array([np.min(mo1.Node[:,4:7],axis=0),np.max(mo1.Node[:,4:7],axis=0)])
    dim1=np.diff(mybox,axis=0);XY=mo1.Node[:,4:6];uv=np.zeros([XY.shape[0],2])
    #parameters to animate modes
    wy=vals[idmode-1] if vals[idmode-1]>1e-8 else 1.
    freq1=wy/(2.0*np.pi);t1=np.linspace(0,1,num)/freq1;     
    if type(fact) is list: fact=1.0/np.max(np.abs(vecs[0::2,idmode-1])); 
    y=fact*(vecs[:,idmode-1].reshape((vecs.shape[0],1)))*np.sin(wy*t1);     
    #init figure
    fig=plot.figure();line1=plot.plot(mo1.Node[Elt.T,4],mo1.Node[Elt.T,5],'b*-');plot.ylim([-1.,1.])        
    def init_anim():      
        plot.plot(mo1.Node[Elt.T,4],mo1.Node[Elt.T,5],'r*-')
        plot.title('Animation du mode '+str(idmode)+' de frequence '+'%.2f'%freq1+' Hz') 
        return line1
    
    def run_anim(i2):       
        uv[:,1]=y[0::2,i2]
        XY1=XY+uv
        for i3 in range(len(line1)):
         line1[i3].set_data(XY1[Elt[i3,:],0],XY1[Elt[i3,:],1])
        return line1
    #run animation
    ani1 = animation.FuncAnimation(fig, run_anim, init_func=init_anim, frames=num, blit=False, interval=50, repeat=True)
    plot.show()
    return ani1
    
def importSDT(filename=''):
    #get model from matlab
    poppy=sio.loadmat(filename,struct_as_record=False)
    for key1 in poppy.keys():
        if key1.startswith('mo'): 
            mo1=poppy[key1][0,0]    
        
    return mo1

def getegroup(elt,jGroup=[]):
    ''' Get elements groups '''
    in1=np.where(np.isinf(elt[:,0]))[0];nGroup=in1.size;
    if nGroup==0: warnings.warn('No element group in elt')
    EGroup=np.append(in1,elt.shape[0]);
    ElemP=[''.join([chr(int(s)) for s in elt[j1,1::] if s>32 and s< 127]) for j1 in in1]
    return [EGroup,nGroup,ElemP]

def stack_get(mo1=[],type1='',name1=''):
    
    for j1 in range(mo1.Stack.shape[0]):
        if mo1.Stack[j1][0][0]==type1:
            if mo1.Stack[j1][1][0]==name1:
                return mo1.Stack[j1][2][0,0]

def plotmesh(mo1=[]):
    
    sel=stack_get(mo1,'info','Sel')
    
    cmap1=cm.get_cmap('jet')    
    mesh1 = m3d.art3d.Line3DCollection(sel.vert0[sel.f2-1,:],cmap=cmap1)
    mesh1.set_alpha(0.95);mesh1.set_color('grey')
        
    fig=plot.figure();axes = m3d.Axes3D(fig)    
    axes = fig.add_subplot(111, projection='3d')
    axes.add_collection3d(mesh1)
    
    axes.set_xlim3d(np.min(mo1.Node[:,4]),np.max(mo1.Node[:,4]))
    axes.set_ylim3d(np.min(mo1.Node[:,5]),np.max(mo1.Node[:,5]))
    axes.set_zlim3d(np.min(mo1.Node[:,6]),np.max(mo1.Node[:,6]))
    
    axes.set_xlim3d(-25.,25.);axes.set_ylim3d(-120.,120.);axes.set_zlim3d(-30.,30.)    
    axes.view_init(20.,-150.);axes.dist=10. ;axes.set_axis_off()
    
    plot.show()  
    

def vtk_type(ElemP):
    ''' VTK correspondance '''
    npts=0;etyp=0
    #if ElemP=='rigid': npts=2;etyp=3
    if ElemP=='tetra4': npts=4;etyp=10
    if ElemP=='hexa8': npts=8;etyp=12    
    if ElemP=='tetra10': npts=10;etyp=24
    if ElemP=='hexa20': npts=20;etyp=25
    
    return [npts,etyp]        
    
def writePARAVIEW(mo1=[],TR=[],filename='results.vtk'):
    ''' Write file for PAraview '''
    # header    
    fid1=open(filename,'w')
    fid1.write('# vtk DataFile Version 2.0 \nPoppy \nASCII\n')   
    fid1.write('DATASET UNSTRUCTURED_GRID \n')   

    # nodes              
    fid1.write('POINTS '+str(mo1.Node.shape[0])+' float \n')    #nodes header
    for j1 in range(mo1.Node.shape[0]):
        str1 = '{:+9.4f}   {:+9.4f}   {:+9.4f}  \n'.format(mo1.Node[j1,4],mo1.Node[j1,5],mo1.Node[j1,6])
        fid1.write(str1)
    fid1.write('\n')
    
    # element count
    [EGroup,nGroup,ElemP] = getegroup(mo1.Elt)
    ncount=[0,0,[]]
    for jGroup in range(nGroup):
        eltind=range(EGroup[jGroup]+1,EGroup[jGroup+1]);
        [npts,etyp]=vtk_type(ElemP[jGroup])        
        if npts>0: ncount[0]+=len(eltind);ncount[1]+=(npts+1)*len(eltind);ncount[2].append(jGroup)    
    fid1.write('CELLS '+str(ncount[0])+' '+str(ncount[1])+' \n')    #elements header
    
    # elements cell
    allelt=np.zeros(ncount[0]);c=0;
    for jGroup in ncount[2]:
        eltind=range(EGroup[jGroup]+1,EGroup[jGroup+1]);[npts,etyp]=vtk_type(ElemP[jGroup])  
        for j1 in eltind:
            fid1.write(str(npts) +''.join(' {:6.0f} '.format(mo1.Elt[j1,j2]-1) for j2 in range(npts))+' \n')
            allelt[c]=etyp;c+=1
    fid1.write('\n')
    
    # elements type
    fid1.write('CELL_TYPES '+str(ncount[0])+' \n')    #elements header    
    for j1 in range(ncount[0]):
        fid1.write('{:2d} \n'.format(etyp))        
    fid1.write('\n')
    
    # displacements
    if len(TR)>0:
        in1=np.floor(TR['DOF']).astype(int)
        in2=np.round((TR['DOF']-in1)*100).astype(int)   
        inx=np.where(in2==1)[0];iny=np.where(in2==2)[0];inz=np.where(in2==3)[0]; 
        ux=TR['val'][inx,:];uy=TR['val'][iny,:];uz=TR['val'][inz,:];
        
        fid1.write('POINT_DATA {:6d} \n'.format(len(inx)))
        for j1 in range(TR['val'].shape[1]):
            fid1.write('VECTORS mode_'+str(j1)+' float  \n')    #elements header 
            for j2 in range(len(inx)):
                fid1.write('{:+9.4f}   {:+9.4f}   {:+9.4f}  \n'.format(ux[j2,j1],uy[j2,j1],uz[j2,j1] ))
            fid1.write('\n')
    
    fid1.close()   
    
    
    