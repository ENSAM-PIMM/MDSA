# -*- coding: utf-8 -*-
"""
ARTS ET METIERS - PGE2A - VIBRATIONS - ED5/6
    Modes de flexion d'une poutre Euler-Bernoulli
         analytique et numérique

 Start with runfile('mevib56.py')
 
Contributed by E. Monteiro and E. Balmes
Copyright (c) 2018-2024 by ENSAM, All Rights Reserved.
"""

from mevib import compute_section,plot_bode,plot2D,quad_seg,feeig
import numpy as np
from numpy import cos,cosh,sin,sinh 
from scipy.optimize import root
from feplot import model,anim1D_mode

#------------------------------------------------------------------------------
#%%  Geometry
#------------------------------------------------------------------------------
def beam_geo(mout=0):
 # define geometry of the beam
 pa=dict([('L',1),('h',.01),('b',.01),('E',200.0e9),('rho',7800.),('section','rect'),('Ne',40) ])
 pa=compute_section(pa)
 if mout==0:
   return pa
 elif mout==1:
  #init model for visualization
  mo1=model(); typ1=[ord(s) for s in 'beam2'];typ1.insert(0,np.inf) 
  mo1.Node=np.array([[1,0.,0.,0.,0.,0.,0.],[2,0.,0.,0.,pa['L'],0.,0.]]) 
  mo1.Elt=np.array([typ1,[1,2,0,0,0,0]]); mo1.DOF=np.array([1.02,1.06,2.02,2.06])
  return pa,mo1
 else:
   #init model for visualization
   mo1=model();mo1.Node=np.zeros((pa['Ne']+1,7));mo1.Elt=np.zeros((pa['Ne']+1,6))
   mo1.Node[:,0]=range(1,pa['Ne']+2); mo1.Node[:,4]=np.linspace(0,1,pa['Ne']+1)*pa['L']
   typ1=[ord(s) for s in 'beam2'];typ1.insert(0,np.inf);mo1.Elt[0,:]=typ1
   mo1.Elt[1:,0]=range(1,pa['Ne']+1);mo1.Elt[1:,1]=range(2,pa['Ne']+2)
   mo1.DOF=np.zeros((2*(pa['Ne']+1)));mo1.DOF[0::2]=np.arange(1,pa['Ne']+2)+.02
   mo1.DOF[1::2]=np.arange(1,pa['Ne']+2)+.06
   return pa,mo1


#------------------------------------------------------------------------------
#%%  Q2 Response to harmonic excitation
#------------------------------------------------------------------------------
def q2(pa=[], iplot=0):
  #geometry
  if type(pa)==list: pa=beam_geo()
  #init variables  
  pa['freq']=np.linspace(1,1000,1001)
  xyplot=dict([('X',pa['freq']), ('Xlabel','Frequency (Hz)'), 
             ('Y',np.zeros((pa['freq'].size,3))),('Ylabel',['Displacement','Moment','Determinant']) ])
  #define useful parameters
  EI = pa['E']*pa['I']; rsEI=pa['rho']*pa['S']/EI; L= pa['L']
  #sweep over frequencies
  for j1 in range(len(xyplot['X'])):
    k = (rsEI*(xyplot['X'][j1]*2.*np.pi)**2)**0.25
    #compute boundary conditions
    u0 = np.array([ cos(k*0.), sin(k*0.), cosh(k*0.), sinh(k*0.)])
    r0 = np.array([-sin(k*0.), cos(k*0.), sinh(k*0.), cosh(k*0.)])*k
    mL = np.array([ -cos(k*L), -sin(k*L),  cosh(k*L),  sinh(k*L)])*EI*k**2
    tL = np.array([  sin(k*L), -cos(k*L),  sinh(k*L),  cosh(k*L)])*-EI*k**3
    #compute coefficients
    M, F = np.array([u0, r0, mL, tL]), np.array([0,0,1,0])
    coef = np.linalg.solve(M, F)
    #compute physical quantities
    uL = np.array([ cos(k*L), sin(k*L), cosh(k*L), sinh(k*L)])
    m0 = np.array([ -cos(k*0.), -sin(k*0.),  cosh(k*0.),  sinh(k*0.)])*EI*k**2
    xyplot['Y'][j1,0:2]=np.dot(np.array([uL,m0]),coef)
    xyplot['Y'][j1,2]=np.linalg.det(M)
  #plot transfer (use 0,1,2 to select output)
  plot_bode(xyplot,iplot)


#------------------------------------------------------------------------------
#%%  Q4  Get analytical modes
#------------------------------------------------------------------------------
def q4(pa=[], kini=6.): 
  #geometry
  if type(pa)==list: pa=beam_geo()
  EI = pa['E']*pa['I']; L=pa['L']
  #define matrix determinant   
  det1=lambda x: 1.+cos(x)*cosh(x)
  #find root of determinant
  sol1=root(det1,kini);k1=sol1['x'][0]/L; #seek roots using scipy.optimize.root
  #find coefficients for given root 
  u0 = np.array([ cos(k1*0.), sin(k1*0.), cosh(k1*0.), sinh(k1*0.)])
  r0 = np.array([-sin(k1*0.), cos(k1*0.), sinh(k1*0.), cosh(k1*0.)])*k1
  mL = np.array([ -cos(k1*L), -sin(k1*L),  cosh(k1*L),  sinh(k1*L)])*EI*k1**2
  tL = np.array([  sin(k1*L), -cos(k1*L),  sinh(k1*L),  cosh(k1*L)])*-EI*k1**3
  M=np.array([u0, r0, mL, tL]);u,s,v=np.linalg.svd(M);coef = v[-1,:];#np.dot(M,coef)
  #plot
  xyplot=dict([('X',np.linspace(0.,L,100)), ('Xlabel','x (m)'), ('Y',[]), ('Ylabel','Displacement') ])
  xyplot['Y']= coef[0]*cos(k1*xyplot['X'])+coef[1]*sin(k1*xyplot['X'])+coef[2]*cosh(k1*xyplot['X'])+  \
     coef[3]*sinh(k1*xyplot['X'])
  plot2D(xyplot)


#------------------------------------------------------------------------------
#%% Q5c compute shape functions
#------------------------------------------------------------------------------
def q5c(pa=[], nout=0):
  #geometry
  if type(pa)==list: pa=beam_geo()
  #shape functions
  TdT = lambda x : np.array([np.power(x,[3,2,1,0]), np.multiply(np.power(x,[2,1,0,0]),[3,2,1,0])])
  N = np.linalg.inv(np.concatenate((TdT(0),TdT(pa['L']))))
  if nout==0:
    xyplot=dict([('X',np.linspace(0,pa['L'],100)), ('Xlabel','x'), ('Ylabel','Shape Function') ])
    xyplot['Y']=np.transpose(np.array([np.polyval(N[:,j1],xyplot['X']) for j1 in range(4)]))
    plot2D(xyplot)
  else:
    return N


#------------------------------------------------------------------------------
#%% Q5f compute element matrices by analytical integration [0;L]
#------------------------------------------------------------------------------
def q5f(pa=[]):
  #geometry
  if type(pa)==list: pa=beam_geo()
  N=q5c(pa, nout=1)
  #
  Ke, Me = np.zeros((4,4)), np.zeros((4,4))
  for j1 in range(4):
    for j2 in range(4):
        coef=np.polyint(np.polymul(N[:,j1],N[:,j2]))
        Me[j1,j2]=np.polyval(coef,pa['L'])-np.polyval(coef,0)
        coef=np.polyint(np.polymul(np.polyder(N[:,j1],2),np.polyder(N[:,j2],2)))
        Ke[j1,j2]=np.polyval(coef,pa['L'])-np.polyval(coef,0)
  #display matrices
  print('Me_a : ',Me*420./pa['L']);print('Ke_a : ',Ke*pa['L']**3)


#------------------------------------------------------------------------------
#%% Q5g compute element matrices by numerical integration [0;L]
#------------------------------------------------------------------------------
def q5g(pa=[], ng=4, nout=0):
  #geometry
  if type(pa)==list: pa=beam_geo()
  N=q5c(pa, nout=1)
  #get quadrature points [-1;1]
  [GS,NdN]=quad_seg(ng)
  #compute quadrature points [0;L]
  xi=[[0],[pa['L']]];
  xg= NdN[:,0:2] @ xi # N_i x_i
  Jac=NdN[:,2:4] @ xi # ???
  #compute shape functions & derivatives at quadrature points
  Nval=np.array([np.polyval(N[:,j2],xg) for j2 in range(4)])
  Bval=np.array([np.polyval(np.polyder(N[:,j2],2),xg) for j2 in range(4)])
  #build matrices
  K, M = np.zeros((4,4)), np.zeros((4,4))
  for j1 in range(GS.shape[0]):      
    M=M+Nval[:,j1].T*Nval[:,j1]*GS[j1,1]*Jac[j1] 
    K=K+Bval[:,j1].T*Bval[:,j1]*GS[j1,1]*Jac[j1]
  #display matrices
  if nout==0: print('Me_n : ',M*420./pa['L']);print('Ke_n : ',K*pa['L']**3)
  #export matrices
  return M,K


#------------------------------------------------------------------------------
#%% Q6a compute modes
#------------------------------------------------------------------------------
def q6a(pa=[],idmode=3, ng=4, nout=0):
  #geometry
  if type(pa)==list: pa,mo1=beam_geo(mout=1)
  M,K = q5g(pa, ng, nout=2) 
  #compute modes for single free/free element
  om1, phi1 = feeig(K,M)
  #export or view
  if nout==0:
    #view modes (use 1,2,3,4 to select output)
    fig1=anim1D_mode(mo1,om1,phi1,idmode) # problem with values outside nodes not shown
    return fig1
  else:
    return phi1


#------------------------------------------------------------------------------
#%% Q6b display moment and shear
#------------------------------------------------------------------------------
def q6b(pa=[], idmode=2, ng=4):
  #geometry
  if type(pa)==list: pa=beam_geo()
  N=q5c(pa, nout=1); phi1=q6a(pa,ng=ng,nout=1)
  #
  xg=np.linspace(0,pa['L'],50);
  cDisp=np.array([np.polyval(N[:,j2],xg) for j2 in range(4)]).T
  cMoment=np.array([np.polyval(np.polyder(N[:,j2],2),xg) for j2 in range(4)]).T
  cShear =np.array([-np.polyval(np.polyder(N[:,j2],3),xg) for j2 in range(4)]).T
  # why skip two modes ?
  xyplot=dict([('X',xg), ('Xlabel','x'), ('Ylabel','Out'),('legend',['Disp','Force','Moment']) ])
  xyplot['Y']=np.array([cDisp @ phi1[:,idmode-1], cShear @ phi1[:,idmode-1], cMoment @ phi1[:,idmode-1] ]).T# cShear @ vecs[:,0].T
  plot2D(xyplot)


#------------------------------------------------------------------------------
#%% Q6c compute modes for variable number of elements
#------------------------------------------------------------------------------
def q6c(pa=[],idmode=1, ng=4): 
  #geometry
  if type(pa)==list: pa,mo1=beam_geo(mout=2)
  #build element matrices
  if pa['Ne']==1:
    Me,Ke = q5g(pa, ng=ng, nout=2)
  else:
    pa2=pa.copy();pa2['L']=pa['L']/pa['Ne']
    Me,Ke = q5g(pa2, ng=ng, nout=2)
  #build global matrices
  n_dof=2*pa['Ne']+2;
  K, M = np.zeros((n_dof,n_dof)), np.zeros((n_dof,n_dof))
  for j1 in range(pa['Ne']):
    ddl=range(2*j1,2*j1+4);id_ddl=np.ix_(ddl,ddl);
    K[id_ddl]=K[id_ddl]+Ke;M[id_ddl]=M[id_ddl]+Me
  #compute modes free-free
  omf2, phif2 = feeig(K,M)
  #compute modes ?? what boundary conditions
  vals, vecs1 = feeig(K[2::,2::],M[2::,2::])
  vecs=np.zeros([M.shape[0],vecs1.shape[1]],order='F');vecs[2:,:]=vecs1
  #view modes 
  #fig1=anim1D_mode(mo1,omf2,phif2,idmode)
  fig1=anim1D_mode(mo1,vals,vecs,idmode)
  return fig1



'''
#%% compute moment and force at root
fg=np.dot(K-M*vals[0]**2,vecs[:,0]) 
# only two non zero values force and moment at root
fg[0:6]
'''














