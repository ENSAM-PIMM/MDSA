# -*- coding: utf-8 -*-
"""
ARTS ET METIERS - PGE2A - VIBRATIONS - ED3/4
    Modes propres et couplage de sous-systèmes 
    
Contributed by E. Monteiro and E. Balmes
Copyright (c) 2018-2024 by ENSAM, All Rights Reserved.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from mevib import cosd,sind,feeig,plot2D,plotFourier,plotFreq 
from feplot import model,anim1D_mode

#%%
def build_model():
 mo1=model(); typ1=[ord(s) for s in 'beam2'];typ1.insert(0,np.inf) 
 mo1.Node=np.array([[1,0.,0.,0.,1.,0.,0.],[2,0.,0.,0.,0.,1.,0.],[3,0.,0.,0.,1.,1.,0],
      [4,0.,0.,0.,2.,2.,0.],[5,0.,0.,0.,3.,2.,0.],[6,0.,0.,0.,2.,3.,0]]) 
 mo1.Elt=np.zeros((6,6));mo1.Elt[0,:]=typ1 
 mo1.Elt=np.array([typ1,[1,3,0,0,0,0],[2,3,0,0,0,0],[3,4,0,0,0,0],[4,5,0,0,0,0],[4,6,0,0,0,0]])
 mo1.DOF=np.array([3.01,3.02,4.01,4.02])
 return mo1

#------------------------------------------------------------------------------
#
#   Part 1: Coupling
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#%%   Q1: equations
#------------------------------------------------------------------------------
def q1(pa=[], pflag=True):
 # parameters 4 DOF system
 if type(pa)==list:
  pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
  
 # Compute matrices of 4 DOF system
 c=cosd(pa['a']);s=sind(pa['a']); M=np.eye(4)*pa['m']

 K=np.array([[pa['k1']+pa['k3']*c**2, pa['k3']*c*s, -pa['k3']*c**2, -pa['k3']*c*s],
             [pa['k3']*c*s, pa['k1']+pa['k3']*s**2, -pa['k3']*c*s, -pa['k3']*s**2],
             [-pa['k3']*c**2, -pa['k3']*c*s, pa['k2']+pa['k3']*c**2, pa['k3']*c*s],
             [-pa['k3']*c*s, -pa['k3']*s**2, pa['k3']*c*s, pa['k2']+pa['k3']*s**2]])
  # Question if you have time : write k3 stiffnes contribution 
  # using an observation matrix that is in the form  K=C^T k3 C
 if pflag: 
  print("M= ",M);print(' ');print("K= ",K)
 else:
  return M,K


#------------------------------------------------------------------------------
#%%   Q2: modes
#------------------------------------------------------------------------------
def q2(pa=[], idmode=1, pflag=True):
 # parameters 4 DOF system
 if type(pa)==list:
  pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
 # compute modes
 M,K = q1(pa, pflag=False);
 om1, phi1 = feeig(K, M, norm='M')  
 # run anim
 if pflag:
  print("omega=", om1);print(' ');print("phi= ", phi1)
  mo1=build_model(); fig1=anim1D_mode(mo1,om1,phi1,idmode)
  return fig1
 else:
  return om1,phi1


#------------------------------------------------------------------------------
#%%   Question 3: coupling force
#------------------------------------------------------------------------------
def q3():
 # parameters 4 DOF system 
 pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
 # compute modes
 om1, phi1 = q2(pa, pflag=False)

 # compute force observation matrix
 c=cosd(pa['a']);s=sind(pa['a']);
 C=pa['k3']*np.array([-c,-s,c,s]); 
 f3= C @ phi1
 print("F3 =",f3);
  
 # Computation strategies
 B=np.array([1,0,0,0]); 
 M,K = q1(pa, pflag=False)

 #q3c : write formula forced response
 y1 = lambda s : C @ ( linalg.solve(complex(0.,s)**2*M+K,B) )  # Ecrire la formule
 # Write formula spectral response
 y2 = lambda s :( C @ phi1) @ (np.diag(1/(complex(0.,s)**2+om1**2))) @ ( phi1.T @ B )
 # 4 edit code to add Rayleigh damping in the code below
 xyplot=dict([('X',np.linspace(0,50,1000)), ('Xlabel','w'), ('Ylabel','F3') ])
 xyplot['Y']=np.zeros((1000,2))
 for j1 in range(1000):
   xyplot['Y'][j1,:]=[y1(xyplot['X'][j1]).real, y2(xyplot['X'][j1]).real ]
 plotFreq(xyplot,gf=5)  
   
#------------------------------------------------------------------------------
#%%   Q4: damping
#------------------------------------------------------------------------------
def q4():
 # parameters 4 DOF system 
 pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
 # compute modes
 om1, phi1 = q2(pa, pflag=False); M,K = q1(pa, pflag=False)
 # init data structure
 bmax = 4./om1[-1]
 xyplot=dict([('X',np.linspace(0,bmax,int(np.ceil(bmax/1.e-4)))),  
             ('Xlabel','Beta'), ('Y',''), ('Ylabel','Damping ratio'),
             ('legend',np.arange(phi1.size)+1) ])
 # loop over values
 xyplot['Y']=np.zeros((len(xyplot['X']),om1.size))
 for j1 in range(om1.size):
   xyplot['Y'][:,j1]=xyplot['X']*om1[j1]/2.
 plot2D(xyplot,gf=4)
 
 # compute transfer
 w=np.linspace(0,50,500)
 xyplot2=dict([('X',w),('Xlabel','w'), ('Y',''), ('Ylabel','h41')  ])
 xyplot2['Y']=np.zeros((len(xyplot2['X']),1))
 for j1 in range(om1.size):
   xyplot2['Y'][:,0]+= (
      phi1[1,j1]*phi1[1,j1] #Formula for this and correction
      /(om1[j1]**2-w**2) # Formula for this and correction
     )
 plotFreq(xyplot2,gf=5)

 # Q4e loss factor usage
 pa2=dict([('m',0.),('L',1.),('a',45.), 
           ('k1',0.0),('k2',0),('k3',1000.)])
 M2,K2 = q1(pa2,pflag=False)
 dK=.1* phi1.T @ K2 @ phi1
 v=np.diag(dK)/om1/om1/2 # What is this ?
 [ld2,psi]=linalg.eig(-phi1.T@(K+0.1j*K2)@phi1);ld=np.sqrt(ld2)
 print(np.real(ld)/np.abs(ld)) #What is this ? why are two values close to 0 ?
 

 # additional (future) questions
 # - rewrite  res['phi'][3,j1] with an observation matrix 
 # - observe resultant load


 
#------------------------------------------------------------------------------
#
#   Part 2: Modification raideurs et/ou masses
#
#------------------------------------------------------------------------------  
  
#------------------------------------------------------------------------------
#%%   Q6: Modification de raideur
#------------------------------------------------------------------------------  
def q6(cas=2, rate=0.05):
 # parameters 4 DOF system 
 pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
          ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
          ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
          ])
 # compute modes for initial model
 om1, phi1 = q2(pa, pflag=False); M,K = q1(pa, pflag=False) 
 # compute added stiffness
 c=cosd(pa['a']);s=sind(pa['a']);z4=[0.,0.,0.,0.]
 if cas==1:
   idmode=1; dK = np.array([ [c**2,c*s,0.,0.],[c*s,s**2,0.,0.],z4,z4])
 elif cas==2:
   idmode=1; dK = np.array([ [s**2,-c*s,0.,0.],[-c*s,c**2,0.,0.],z4,z4])
 elif cas==3:
   idmode=3; dK = np.array([ z4,z4,[0.,0.,s**2,-c*s],[0.,0.,-c*s,c**2]])
 elif cas==4:
   idmode=4; dK = np.array([ z4,z4,[0.,0.,c**2,c*s],[0.,0.,c*s,s**2]])
 # compute modes for modified model
 ki=((1+rate)**2-1)*om1[idmode-1]**2/(phi1[:,idmode-1].T@dK@phi1[:,idmode-1])
 om2, phi2 = feeig(K+ki*dK, M)
 # print
 print(' ');print('k = ', ki)
 print("omega_new= ", om2);print("omega_ini=", om1);print("ratio = ", om2/om1)
 print(' ');print(' ')
 print('phi_ini =', phi1);print(' ');print('phi_new =', phi2)
  
  
#------------------------------------------------------------------------------
#%%   Q7: Modification de masse
#------------------------------------------------------------------------------
def q7(m=-2.1):
 # parameters 4 DOF system 
 pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
           ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
           ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
           ])
 # compute modes for initial model
 om1, phi1 = q2(pa, pflag=False); M,K = q1(pa, pflag=False) 
 # compute modes for modified model
 dM = np.zeros((4,4)); dM[0,0]=m
 om2, phi2 = feeig(K, M+dM) 
 # print
 print("omega_new= ", om2);print("omega_ini=", om1);print("ratio = ", om2/om1)
 print(' ');print(' ')
 print('phi_ini =', phi1);print(' ');print('phi_new =', phi2)   
    
  
  
  
  
  