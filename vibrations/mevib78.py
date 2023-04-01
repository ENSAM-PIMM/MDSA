# -*- coding: utf-8 -*-
"""
ED78 VIBRATIONS - FITE 2A - ARTS ET METIERS 
 Start with runfile('mevib78.py')
Copyright 2019, Eric Monteiro, Etienne Balmes
"""

import mevib as mv 
import feplot
import numpy as np
from scipy.linalg import block_diag


#%% assembly
def matrix_assembly(mo2, add_m=0.0, MROT=np.identity(12)):
 ''' matrix assembly function  '''
 
 #init matrices
 M=np.zeros((13,13)); K=np.zeros((13,13)); Ks=np.zeros((13,13));
 DOF=np.vstack((mo2.dMKu[0][6:12], mo2.dMKf[0][3], mo2.dMKf[0][6:12]))
 
 #upperarm
 in1=np.array(range(6),ndmin=2);            #DOF in M,K  
 in2=np.array(range(6,12),ndmin=2);         #DOF in Mu,Ku
 M[in1.T,in1]=mo2.dMKu[1][in2.T,in2]; K[in1.T,in1]=mo2.dMKu[2][in2.T,in2]
 
 #forearm
 in1=np.array([np.hstack((np.arange(3),6,4,5,np.arange(7,13)))]);  #DOF in M,K
 M[in1.T,in1] = M[in1.T,in1] + MROT @ mo2.dMKf[1] @ MROT.T 
 K[in1.T,in1] = K[in1.T,in1] + MROT @ mo2.dMKf[2] @ MROT.T 
 
 #join
 in1=np.array([[3,6]]) #DOF in M,K
 Ks[in1.T,in1]=np.array([[1,-1],[-1,1]])
 
 #add concentrated mass on elbow (unit system : Millimeter, Newton)
 in1=np.array([np.arange(3)]);M[in1.T,in1]=M[in1.T,in1]+add_m*np.eye(3);
 
 #output
 return M, K, Ks


#%% load poppy mesh
def load_poppy(filename='poppyse1.mat', pflag=1):
 ''' Import Poppy model '''
 
 #import from MATLAB/SDT
 mo1 = feplot.importSDT(filename)
  
 # get superelements
 up = feplot.stack_get(mo1,'SE','up')     #upperarm (shoulder -> elbow)  
 fo = feplot.stack_get(mo1,'SE','fore')   #forearm  (elbow ->  hand)

 # build full poppy
 mo2 = feplot.model()
 mo2.Node = np.vstack((up.Node,fo.Node))
 mo2.Elt = np.vstack((up.Elt,fo.Elt))
 mo2.dMKu = (up.DOF, up.K[0,0], up.K[0,1]) 
 mo2.dMKf = (fo.DOF, fo.K[0,0], fo.K[0,1]) 
 
 # correct names def
 uTR=up.TR[0,0];uTR.__dict__['Def']=uTR.__dict__.pop('def');   
 fTR=fo.TR[0,0];fTR.__dict__['Def']=fTR.__dict__.pop('def');

 # prepare restitution
 n1=uTR.Def.shape[0];n2=fTR.Def.shape[0];
 mo2.TR=dict([ ('Def',
    np.vstack( (np.hstack((uTR.Def[:,6:12],np.zeros((n1,7)))), # upper arm
     np.hstack((fTR.Def[:,:3],np.zeros((n2,1)),fTR.Def[:,4:6],np.array(fTR.Def[:,3],ndmin=2).T,fTR.Def[:,6:12] ))))), # Shapes in forearm
     ('DOF',np.vstack((uTR.DOF,fTR.DOF))) ])

 #export for paraview
 if pflag:
   feplot.writePARAVIEW(mo2,[],filename='poppy_mesh1.vtk')
  
 #output
 return mo2


#------------------------------------------------------------------------------
#%% Q0: visualization 
#------------------------------------------------------------------------------
mo2=load_poppy(pflag=1) 


#------------------------------------------------------------------------------
#%%  Q1: modes 
#------------------------------------------------------------------------------
def q1(mo2=[], kj=0., pflag=1):
  
 if type(mo2)==list:
    mo2 = load_poppy(pflag=0) 
  
 # compute modes
 M,K,Ks = matrix_assembly(mo2)
 [w,phi] = mv.feeig(K+kj*Ks, M); f = w/(2.*np.pi)
 #def1=res(DOF=DOF,Def=p,data=f.reshape((13,1)))
 mo2.TR['val'] = mo2.TR['Def'] @ phi
 if pflag:
  feplot.writePARAVIEW(mo2,mo2.TR,filename='results1.vtk')

 return f, phi


#------------------------------------------------------------------------------
#%% Q2: effect of orientation  
#------------------------------------------------------------------------------
def q2(mo2=[], kj=0., add_m=0., Nt=20, pflag=0):
    
 if type(mo2)==list:
    mo2 = load_poppy(pflag=0)
    
 theta=np.linspace(0,1,Nt)*90.
 xyplot=dict([('X',theta), ('Xlabel','Angle'),  ('Ylabel','Frequency'), 
             ('Y',np.zeros((theta.size,mo2.TR['Def'].shape[1]))) ])

 for j1 in range(len(theta)):
    a=theta[j1]
    ROT = np.array([[1,0,0],[0,mv.cosd(a),mv.sind(a)],[0,-mv.sind(a),mv.cosd(a)]])
    MROT = block_diag(ROT,ROT,ROT,ROT)
    M,K,Ks = matrix_assembly(mo2, add_m, MROT)    
    #compute modes
    [w,phi] = mv.feeig(K+kj*Ks,M); f = w/(2.*np.pi)
    mo2.TR['val'] = mo2.TR['Def'] @ phi; xyplot['Y'][j1,:]=f
    if pflag:
     feplot.writePARAVIEW(mo2,mo2.TR,filename='results_TH{:2d}.vtk'.format(int(a)))
    
 mv.plot2D(xyplot,yscale='log',ylim=[3.0e0,1.0e3])
 
 return xyplot


#------------------------------------------------------------------------------
#%% Q3: effect of stiffness
#------------------------------------------------------------------------------
def q3(mo2=[], add_m=0., Nk=20, pflag=0):

 if type(mo2)==list:
    mo2 = load_poppy(pflag=0)    
    
 kj=np.logspace(2,6,Nk); M,K,Ks = matrix_assembly(mo2, add_m) 
 xyplot=dict([('X',np.log10(kj)), ('Xlabel','log(ks)'),('Ylabel','Frequency'),
             ('Y',np.zeros((kj.size,mo2.TR['Def'].shape[1]))) ])  
 
 for j1 in range(kj.size):
    [w,phi]=mv.feeig(K+kj[j1]*Ks,M); f = w/(2.*np.pi)
    mo2.TR['val'] = mo2.TR['Def'] @ phi; xyplot['Y'][j1,:]=f
    if pflag:
     feplot.writePARAVIEW(mo2,mo2.TR,filename='results_KS{:2d}.vtk'.format(int(kj[j1])))
        
 mv.plot2D(xyplot,yscale='log')    

 return xyplot

#------------------------------------------------------------------------------
#%% Q3d: Rayleigh ratio
#------------------------------------------------------------------------------ 
def q3d(mo2=[], idmode=1, add_m=0.0):

 if type(mo2)==list:
    mo2 = load_poppy(pflag=0)   
  
 M,K,Ks = matrix_assembly(mo2, add_m)
 [w,p] = mv.feeig(K+100*Ks,M); X1=p[:,idmode-1:idmode]; #Shape of first mode
 [w1,p1] = mv.feeig(K+1000*Ks,M); f1=w1/(2.*np.pi)
 fr=np.sqrt(X1.T @ (K+1000*Ks) @ X1)/ (X1.T @ M @ X1) / (2.*np.pi)

 print(f1[0],fr[0,0],np.abs(f1[0]/fr[0,0]-1.)*100.)


#------------------------------------------------------------------------------
#%% Q3e: Rigid harm
#------------------------------------------------------------------------------ 
def q3e(mo2=[], kj=1.e3, add_m=0.0):

 if type(mo2)==list:
    mo2 = load_poppy(pflag=0) 
 
 M,K,Ks = matrix_assembly(mo2, add_m)  
 [w,p] = mv.feeig(K+kj*Ks,M); f=w/(2.*np.pi);
 [w1,p1] = mv.feeig(K*100+kj*Ks,M); f1=w1/(2.*np.pi);
 
 print('f  =',f[:4],'\nfr =',f1[:4],'\ndf =',(f1[:4]/f[:4]-1)*100.)    
    

#------------------------------------------------------------------------------
#%% Q3f: Adding damping
#------------------------------------------------------------------------------ 
def q3f(mo2=[], nmode=2, kj=1.e3, add_m=0.0):

 if type(mo2)==list:
    mo2 = load_poppy(pflag=0) 
 
 cs=np.logspace(-4,8,5000);
 xyplot=dict([('X',np.zeros((cs.size,nmode))), ('Xlabel','Real'),
             ('Y',np.zeros((cs.size,nmode))),('Ylabel','Imag') ]) 
 
 M,K,Ks = matrix_assembly(mo2, add_m); Mm1=np.linalg.inv(M)
 for j1 in range(cs.size):
    A = np.vstack(( np.hstack((np.zeros(K.shape), np.eye(K.shape[0]) )),
                    np.hstack((-Mm1 @ (K+kj*Ks) , -Mm1 @ (cs[j1]*Ks) )) ))
    [w,p]=np.linalg.eig(A)
    w=w[np.imag(w)>=0];idx=np.argsort(np.abs(w));f=w[idx]/2/np.pi;
    xyplot['X'][j1,:]=np.real(f[:nmode]);xyplot['Y'][j1,:]=np.imag(f[:nmode])

 mv.plot2D(xyplot,style='o',xlim=[np.min(xyplot['X']),0.],ylim=[0., np.max(xyplot['Y'])]) 

 
#------------------------------------------------------------------------------
#%% Q4: Forced Response
#------------------------------------------------------------------------------      
def q4(mo2=[], kj=1.e3, add_m=430.0e-6):
    
 if type(mo2)==list:
    mo2 = load_poppy(pflag=0)    
    
 xi=0.01;freq=np.linspace(0,300,5000);w=freq*2.*np.pi;
 M,K,Ks = matrix_assembly(mo2,add_m); in1=9;out1=[9,3,10];
 [wk,pk]=mv.feeig(K+kj*Ks,M);  
 
 xyplot=dict([('X',freq), ('Xlabel','Frequency'),
             ('Y',np.zeros((freq.size,2))),('Ylabel',['Amplitude','Moment']) ]) 
 
 for j1 in range(len(w)):
    d1 = 1./(wk**2-w[j1]**2 + 2.j*xi*wk*w[j1] );
    h1 = pk @ np.diag(d1) @ pk.T
    xyplot['Y'][j1,0]=np.abs(h1[3,3]);xyplot['Y'][j1,1]=np.abs(h1[3,10])

 mv.plot2D(xyplot,yscale='log') 
#manque calcul de moment