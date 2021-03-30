# -*- coding: utf-8 -*-
"""
ED56 VIBRATIONS - FITE 2A - ARTS ET METIERS 
 Start with runfile('mevib56.py')
Copyright 2019-2021, Eric Monteiro, Etienne Balmes
"""

from mevib import * # Import all functions with no root
import numpy as np


# pa : parameter dictionnary 
pa=dict([('L',1),('h',.01),('b',.01),('E',200.0e9),('rho',7800.),('section','rect'),('Ne',40) ])
pa=compute_section(pa)


#------------------------------------------------------------------------------
#%% Q2 Response to harmonic excitation
#------------------------------------------------------------------------------
pa['freq']=np.linspace(1,1000,1001)
xyplot=dict([('X',pa['freq']), ('Xlabel','Frequency (Hz)'), 
             ('Y',np.zeros((pa['freq'].size,3))),('Ylabel',['Displacement','Moment','Determinant']) ])
#define useful parameters
rsEI=pa['rho']*pa['S']/pa['E']/pa['I']
L= pa['L']; EI = pa['E']*pa['I'];
#sweep over frequencies
for j1 in range(len(xyplot['X'])):
    k = (rsEI*(xyplot['X'][j1]*2.*np.pi)**2)**0.25
    #compute boundary conditions
    u0 = np.array([ np.cos(k*0.), np.sin(k*0.), np.cosh(k*0.), np.sinh(k*0.)])
    r0 = np.array([-np.sin(k*0.), np.cos(k*0.), np.sinh(k*0.), np.cosh(k*0.)])*k
    mL = np.array([ -np.cos(k*L), -np.sin(k*L),  np.cosh(k*L),  np.sinh(k*L)])*EI*k**2
    tL = np.array([  np.sin(k*L), -np.cos(k*L),  np.sinh(k*L),  np.cosh(k*L)])*-EI*k**3
    #compute coefficients
    M, F = np.array([u0, r0, mL, tL]), np.array([0,0,1,0])
    coef = np.linalg.solve(M, F)
    #compute physical quantities
    uL = np.array([ np.cos(k*L), np.sin(k*L), np.cosh(k*L), np.sinh(k*L)])
    m0 = np.array([ -np.cos(k*0.), -np.sin(k*0.),  np.cosh(k*0.),  np.sinh(k*0.)])*EI*k**2
    xyplot['Y'][j1,0:2]=np.dot(np.array([uL,m0]),coef)
    xyplot['Y'][j1,2]=np.linalg.det(M)
#plot transfer (use 0,1,2 to select output)
plot_bode(xyplot,0)


#------------------------------------------------------------------------------
#%% Q4 : Analytical modes
#------------------------------------------------------------------------------
det1=lambda x: 1.+np.cos(x)*np.cosh(x)
sol1=root(det1,6.);k1=sol1['x'][0]/L; #seek roots using scipy.optimize.root
u0 = np.array([ np.cos(k1*0.), np.sin(k1*0.), np.cosh(k1*0.), np.sinh(k1*0.)])
r0 = np.array([-np.sin(k1*0.), np.cos(k1*0.), np.sinh(k1*0.), np.cosh(k1*0.)])*k1
mL = np.array([ -np.cos(k1*L), -np.sin(k1*L),  np.cosh(k1*L),  np.sinh(k1*L)])*EI*k1**2
tL = np.array([  np.sin(k1*L), -np.cos(k1*L),  np.sinh(k1*L),  np.cosh(k1*L)])*-EI*k1**3
M=np.array([u0, r0, mL, tL]);u,s,v=svd(M);coef = v[-1,:];#np.dot(M,coef)
xyplot=dict([('X',np.linspace(0.,L,100)), ('Xlabel','x (m)'), ('Y',[]), ('Ylabel','Displacement') ])
xyplot['Y']= coef[0]*np.cos(k1*xyplot['X'])+coef[1]*np.sin(k1*xyplot['X'])+coef[2]*np.cosh(k1*xyplot['X'])+coef[3]*np.sinh(k1*xyplot['X'])
plot2D(xyplot)





#------------------------------------------------------------------------------
#%% Q5 compute shape functions
#------------------------------------------------------------------------------
TdT = lambda x : np.array([np.power(x,[3,2,1,0]), np.multiply(np.power(x,[2,1,0,0]),[3,2,1,0])])
N = np.linalg.inv(np.concatenate((TdT(0),TdT(pa['L']))))

xyplot=dict([('X',np.linspace(0,pa['L'],100)), ('Xlabel','x'), ('Ylabel','Shape Function') ])
xyplot['Y']=np.transpose(np.array([np.polyval(N[:,j1],xyplot['X']) for j1 in range(4)]))
plot2D(xyplot)





#%% compute element matrices by analytical integration [0;L]
Ke, Me = np.zeros((4,4)), np.zeros((4,4))
for j1 in range(4):
    for j2 in range(4):
        coef=np.polyint(np.polymul(N[:,j1],N[:,j2]))
        Me[j1,j2]=np.polyval(coef,pa['L'])-np.polyval(coef,0)
        coef=np.polyint(np.polymul(np.polyder(N[:,j1],2),np.polyder(N[:,j2],2)))
        Ke[j1,j2]=np.polyval(coef,pa['L'])-np.polyval(coef,0)

print(Me*420./pa['L']);print(Ke*pa['L']**3)


#%% compute element matrices by numerical integration [0;L]
#get quadrature points [-1;1]
[GS,NdN]=quad_seg(4)
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

print(M*420./pa['L']);print(K*pa['L']**3)


#%% Q6a compute modes
#init model for visualization
mo1=model()
mo1.Node=np.array([[1,0.,0.,0.,0.,0.,0.],[2,0.,0.,0.,pa['L'],0.,0.]])
typ1=[ord(s) for s in 'beam2'];typ1.insert(0,np.inf)
mo1.Elt=np.array([typ1,[1,2,0,0,0,0]])
#compute modes for single free/free element
om1, phi1 = feeig(K,M)
#view modes (use 1,2,3,4 to select output)
anim1D_mode(mo1,om1,phi1,idmode=3) # problem with values outside nodes not shown

#%% Q6b display moment and shear

xg=np.linspace(0,pa['L'],50);
cDisp=np.array([np.polyval(N[:,j2],xg) for j2 in range(4)]).T
cMoment=np.array([np.polyval(np.polyder(N[:,j2],2),xg) for j2 in range(4)]).T
cShear =np.array([-np.polyval(np.polyder(N[:,j2],3),xg) for j2 in range(4)]).T

# why skip two modes ?
xyplot=dict([('X',xg), ('Xlabel','x'), ('Ylabel','Out'),('legend',['Disp','Force','Moment']) ])
xyplot['Y']=np.array([cDisp @ phi1[:,2], cShear @ phi1[:,2], cMoment @ phi1[:,2] ]).T# cShear @ vecs[:,0].T
plot2D(xyplot)

#%% Q6c compute modes for variable number of elements
N = np.linalg.inv(np.concatenate((TdT(0),TdT(pa['L']/pa['Ne']))))
#get quadrature points [-1;1]
[GS,NdN]=quad_seg(4)
#compute quadrature points [0;Le]
xg =np.dot(NdN[:,0:2],np.array([ [0],[pa['L']/pa['Ne']] ]))
Jac=np.dot(NdN[:,2:4],np.array([ [0],[pa['L']/pa['Ne']] ]))
#compute shape functions & derivatives at quadrature points
Nval=np.array([np.polyval(N[:,j2],xg) for j2 in range(4)])
Bval=np.array([np.polyval(np.polyder(N[:,j2],2),xg) for j2 in range(4)])
#build element matrices
Ke, Me = np.zeros((4,4)), np.zeros((4,4))
for j1 in range(GS.shape[0]):      
    Me=Me+Nval[:,j1].T*Nval[:,j1]*GS[j1,1]*Jac[j1] 
    Ke=Ke+Bval[:,j1].T*Bval[:,j1]*GS[j1,1]*Jac[j1]
#build global matrices
if pa['Ne']>1:
    n_ddl=2*pa['Ne']+2
    K, M = np.zeros((n_ddl,n_ddl)), np.zeros((n_ddl,n_ddl))
    for j1 in range(pa['Ne']):
        ddl=range(2*j1,2*j1+4);id_ddl=np.ix_(ddl,ddl);
        K[id_ddl]=K[id_ddl]+Ke;M[id_ddl]=M[id_ddl]+Me
else:
    M=Me;K=Ke;
#init model for visualization
mo1=model();mo1.Node=np.zeros((pa['Ne']+1,7));mo1.Elt=np.zeros((pa['Ne']+1,6))
mo1.Node[:,0]=range(1,pa['Ne']+2); mo1.Node[:,4]=np.linspace(0,1,pa['Ne']+1)*pa['L']
typ1=[ord(s) for s in 'beam2'];typ1.insert(0,np.inf);mo1.Elt[0,:]=typ1
mo1.Elt[1:,0]=range(1,pa['Ne']+1);mo1.Elt[1:,1]=range(2,pa['Ne']+2)
#compute modes free-free
omf2, phif2 = feeig(K,M)
[omf2[2],om1[2]]

#compute modes ?? what boundary conditions
vals, vecs1 = feeig(K[2::,2::],M[2::,2::])
vecs=np.zeros([vecs1.shape[0]+2,vecs1.shape[1]],order='F');vecs[2:,:]=vecs1
#view modes (use 1,2,3,4,5,6,... to select output)
anim1D_mode(mo1,vals,vecs,idmode=1)


#%% compute moment and force at root
fg=np.dot(K-M*vals[0]**2,vecs[:,0]) 
# only two non zero values force and moment at root
fg[0:6]















