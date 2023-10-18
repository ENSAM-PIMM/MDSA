# -*- coding: utf-8 -*-
"""
ARTS ET METIERS - PGE2A - VIBRATIONS - ED1/2
    OSCILLATEUR LINEAIRE A 1 et 2 DDL 
    
Start with 
  cd  YourOwnPath 
  from mevib12 import *
  q2d()  # or another questions

Contributed by E. Monteiro and E. Balmes
Copyright (c) 2018-2024 by ENSAM, All Rights Reserved.
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib as matplotlib
import matplotlib.pyplot as plot

import scipy.linalg as linalg
from mevib import feeig,plot2D,plotFourier,plotFreq 
# on change import importlib; importlib.reload(mevib)


#------------------------------------------------------------------------------
#
#   Part 0: predefined functions
#
#------------------------------------------------------------------------------
#%% Free Response analytic formula (default w0=1,xi=0.5)
def FR_formula(t, w0=1., xi=0.5, u0=5.0, v0=1.0):
  if xi<1.: # UnderDamped 1DOF System
      wd=w0*np.sqrt(1-xi**2)
      return np.exp(-xi*w0*t)*(u0*np.cos(wd*t)+(v0+xi*w0*u0)/wd*np.sin(wd*t))      
  elif xi==1. : # Critically Damped 1DOF System 
      return np.exp(-w0*t)*(u0+(v0+w0*u0)*t)
  elif xi>1.: # Overdamped 1DOF System
      ws=w0*np.sqrt(xi**2-1)
      return np.exp(-xi*w0*t)*(u0*np.cosh(ws*t)+(v0+xi*w0*u0)/ws*np.sinh(ws*t))    

#%%  Compute 1 DOF parameters
def compute_param_1DOF(pa):
    w0=np.sqrt(pa['k']/pa['m']);              # Natural frequency
    xi=pa['c']/(2.*np.sqrt(pa['k']*pa['m']))  # Damping ratio
    return w0, xi
    
#%%  Free Response of 1 DOF System 
def FR_1DOF(pa, t=[]):
    #compute parameters
    w0, xi =compute_param_1DOF(pa)
    #check time discretisation
    if len(t)==0:t0=2.*np.pi/w0;t=np.linspace(0,10.*t0,200)    
    #select cases
    return FR_formula(t,w0,xi,pa['u0'],pa['v0'])


#%%  Q2 Free Response : integrate with ODE, display time and frequency 
def FR_ODE(pa):

    if not("A") in pa:
        pa['A']=np.array([[0,1],[-pa['k']/pa['m'],-pa['c']/pa['m']]]);
    if not("x0" in pa):        pa['x0']=[0,1];
        
    odeFun=lambda t,x:pa['A'].dot(x)
    t_eval=np.linspace(0,pa['Tend'],int(np.ceil(pa['Tend']/pa['dt'])))
    sol=integrate.solve_ivp(odeFun,[0,pa['Tend']],pa['x0'],t_eval=t_eval,method='RK45')
    
    C1=dict([('X',sol.t), ('Xlabel','Time (s)'), 
             ('Y',np.transpose(sol.y)), ('Ylabel','Displacement (m)') ])
    if "cobs" in pa: C1['Y']=np.transpose(pa['cobs'].dot(sol.y));

    if not("fmax" in pa):        pa['fmax']=1e10;
    if not("tlim" in pa):        pa['tlim']=[];
    plot2D(C1,gf=20,xlim=pa['tlim'])
    plotFourier(C1,gf=2,fmax=pa['fmax'])
    print('Time in figure 20, frequency in figure 2')
    
#------------------------------------------------------------------------------
#
#   Part 1: system with 1 degree of freedom
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#%%  Q2 State space ODE integration
#------------------------------------------------------------------------------    
def q2d():
 pa=dict([('m',1.),('c',0.2),('k',100),  #1DOF 
         ('F0',200.0),('a',70.),        #sinusoidal excitation
         ('u0',1.), ('v0',1.),          #initial conditions
         ('Tend',30.), ('dt',0.05)      #plot limit and time discretisation
         ])
 FR_ODE(pa)
 # What is the effect of playing with dt and Tend ? you can do it with
 # pa['dt']=.05;pa['Tend']=5;FR_ODE(pa)
   
 
#------------------------------------------------------------------------------
#%%  Q3 Modal coordinates transfer
#------------------------------------------------------------------------------  
def q3():
 #init plot data structure
 pa=dict([('m',1.),('c',.4),('k',400.),   #1DOF 
         ('u0',1.), ('v0',1.),          #initial conditions
         ('Tend',10.), ('dt',0.1)       #plot limit and time discretisation
         ])
 w0, xi=compute_param_1DOF(pa)
 xy=dict([('X',np.linspace(0,w0*2,100)),  
             ('Xlabel','Frequency (rad/s)'), ('Y',''), ('Ylabel','Displacement (m)') ])
 #compute forced response
 s=xy['X']*complex(0.,1); 
 H=1/(pa['m']*s*s+pa['c']*s+pa['k']); xy['Y']=H 

 #plot Free Response in figure(1)  
 plotFreq(xy,gf=1)


#------------------------------------------------------------------------------
#%%  Q4 Effect of damping on time domain Free Response
#------------------------------------------------------------------------------  
def q4t():
    
 pa=dict([('m',1.),('c',0.5),('k',1.),   #1DOF 
         ('u0',1.), ('v0',0.),          #initial conditions
         ('Tend',10.), ('dt',0.1)       #plot limit and time discretisation
         ])
 w0, xi = compute_param_1DOF(pa)
 val_xi = [0, 0.01, .7, 1.0, 4.0]
 #init data structure
 xyplot=dict([('X',np.linspace(0,pa['Tend'],int(np.ceil(pa['Tend']/pa['dt'])))),  
             ('Xlabel','Time (s)'), ('Y',''), ('Ylabel','Displacement (m)'),
             ('legend',val_xi) ])
 #loop over values
 xyplot['Y']=np.zeros((len(xyplot['X']),len(val_xi)))
 for j1 in range(len(val_xi)): # response for multiple damping
    pa['c']=val_xi[j1]*(2*np.sqrt(pa['k']*pa['m'])) # c= xi * c_crit
    xyplot['Y'][:,j1]=FR_1DOF(pa, xyplot['X'] )  
 #plot Free Response  
 plot2D(xyplot,gf=20)


#------------------------------------------------------------------------------
#%%  Q4 Effect of damping and frequency on Forced Response
#------------------------------------------------------------------------------ 
def q4(): 
 pa=dict([('m',1.),('c',0.5),('k',1.),   #1DOF 
         ('u0',1.), ('v0',1.),          #initial conditions
         ('Tend',10.), ('dt',0.1)       #plot limit and time discretisation
         ])   
 w0, xi =compute_param_1DOF(pa);
 val_xi = [0.1, 0.3, 1/np.sqrt(2), 1.0, 4.0]
 val_f = np.linspace(.1,1.0e1,100);

 #compute Forced Response frequency domain
 s=val_f*complex(0.,1); 
 xy=dict([('X',val_f/w0),('Xlabel','a/w0'), ('Y',[]), 
          ('Ylabel','Amplitude (m)'), ('legend',val_xi) ])
 xy['Y']=np.zeros((len(xy['X']),len(val_xi)),dtype=np.complex_)
 for j1 in range(len(val_xi)):
   pa['c']=val_xi[j1]*(2*np.sqrt(pa['k']*pa['m'])) # c= xi * c_crit
   H=1/(pa['m']*s*s+pa['c']*s+pa['k']);   
   #if (j1==1): xy['Y']=H; else: xy['Y'][:,j1]=H.T
   xy['Y'][:,j1]=H.T
 plotFreq(xy,gf=4,xscale='log');print("Amplitude phase in figure(4)")


#------------------------------------------------------------------------------
#
#   Part 2: system with 2 degrees of freedom
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#%%  Q7b Compute modes
#------------------------------------------------------------------------------
def q7b(nout=0):    
 #parameters 2 DOF system
 pb=dict([('m',1500.),('a',1.2), ('b',1.4),     #2 DOF
         ('ka',[]), ('kb',[]), ('k0',40e3),   #stiffness  
         ('u0',[1., 0.]), ('v0',[1.,0.]),   #initial conditions
         ('balanced',1)])
 #Compute matrices of 2 DOF system
 if pb['balanced']==1:
    pb['ka']=(pb['a']+pb['b'])/pb['a']*pb['k0'];
    pb['kb']=(pb['a']+pb['b'])/pb['b']*pb['k0'];
 #Build matrices   
 M=np.array([[pb['m'],0],[0.,pb['m']*(pb['a']+pb['b'])**2/2.]]);
 K=np.array([ [pb['ka']+pb['kb'], -pb['a']*pb['ka']+pb['b']*pb['kb']],
            [-pb['a']*pb['ka']+pb['b']*pb['kb'], pb['ka']*pb['a']**2+pb['kb']*pb['b']**2] ])
 if nout==0: print("M= ",M);print(' ');print("K= ",K)
 # determine modes  (are these mass normalized ? see code in mevib.py)
 vals, vecs = feeig(K,M)
 if nout==0: print("Eigenvalues (Hz): ",vals/2/np.pi)
 else:  return pb,vals,vecs,M,K


#------------------------------------------------------------------------------
#%%  Q7d Forced response
#------------------------------------------------------------------------------
def q7d(nout=0):
 # On paper : normalize modes, express transfer, add modal damping
 [pb,vals,vecs,M,K]=q7b(nout=1);
 #%% add damping
 C=1e-3*K
 if nout==0: print("C= ",C)
 
 pb['A']= np.block([[np.zeros((2,2)),np.eye(2)],
      [linalg.solve(M,-K),linalg.solve(M,-C) ]])
 pb['B']=np.block([[np.zeros((2,1))],[linalg.solve(M,np.array([[1.],[0.]]))]]);
 pb['Tend']=200;pb['dt']=1e-2;pb['x0']=[1,.1,0,0]; 
 pb['cobs']=np.array([[1,1,0,0]]);  
 pb['fmax']=30/2/np.pi;

 if nout==0: 
   plot.figure(num=20);plot.clf(); FR_ODE(pb)
 else: 
   return pb,vals,vecs,M,K


#------------------------------------------------------------------------------
#%%  Q7e Modal amplitudes in time and frequency domain
#------------------------------------------------------------------------------
def q7e():
 #build matrices and compute modes
 [pb,vals,vecs,M,K]=q7d(nout=1);
 #
 un1=vecs[:,0].T@M@vecs[:,0]
 un2=vecs[:,1].T@M@vecs[:,1]
 un3=vecs[:,0].T@M@vecs[:,1]
 print('? = ',un1,un2,un3) #what are these values== 
 pb['cobs']=np.array([np.concatenate((vecs[:,0].T@M,np.array([0,0])),axis=0),
   np.concatenate((vecs[:,1].T@M,np.array([0,0])),axis=0)
   ]);  
 pb['fmax']=30/2/np.pi;pb['tlim']=np.array([0,10])
 FR_ODE(pb) #Analyze time (figure(20)) and frequency (figure(2))
 
 
matplotlib.use('qt5agg')#Force modern backend
#Force inline plots in the console use    :   %matplotlib inline 
#Force separate window in the console use :  %matplotlib qt

q2d()  # run question 2D
    
print("Run other questions using q3(), q4(), q4t()")
    
    
