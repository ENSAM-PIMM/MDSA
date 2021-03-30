# -*- coding: utf-8 -*-
"""
ARTS ET METIERS - FITE 2A - VIBRATIONS - ED3/4
    Modes propres et couplage de sous-systèmes  
    
Start with runfile('mevib34.py')

Contributed by E. Monteiro and E. Balmes
"""

import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation
import scipy.linalg as linalg
import scipy.sparse as sparse

#------------------------------------------------------------------------------
#
#   Part 0: predefined functions
#
#------------------------------------------------------------------------------
#%% Basics
cosd = lambda x : np.cos(np.radians(x))
sind = lambda x : np.sin(np.radians(x)) 

#%%  Compute eigenvalues and eigenvectors (multiple DOF)
def KMeig(K,M=[]):
    omega2, phi = linalg.eig(K,M) if len(M)>0 else linalg.eig(K)
    
    # What is the purpose of this ? 
    #print(phi) 
    muj=np.diag(phi.transpose().dot(M.dot(phi)))
    phi=phi.dot(np.diag(np.power(muj,-.5)))

    om=np.real(np.sqrt(omega2));idx=np.argsort(om)
    return dict([("w",om[idx]),("phi",phi[:,idx])]) 

#%% Animate modes
def anim_mode(res, idmode=2, nframe=50, ncycle=10, fact=1e-1):  
 # model 
 XY=np.matrix([ [1.,1.], [2.,2.], [0.,1.], [1.,0.], [3.,2.], [2.,3.] ])
 Elt=np.matrix([[0,1],[0,2],[0,3],[1,4],[1,5]]).T
 uv= np.zeros([6,2]) 
 # time evolution
 freq1 = res['w'][idmode-1]/(2.0*np.pi)
 if freq1>1e-6:
   dt=2.0*np.pi/res['w'][idmode-1]/nframe
   f1 = lambda t : np.sin(res['w'][idmode-1]*t)
 else:
   dt=2.0*np.pi/nframe
   f1 = lambda t : np.sin(t) 
 # init figure
 fig = plot.figure()
 line1=plot.plot(XY[Elt,0],XY[Elt,1],'b*-')
 
 def init_anim():
    plot.plot(XY[Elt,0],XY[Elt,1],'r*-')
    plot.title('Animation du mode '+str(idmode)+' de frequence '+'%.2f'%freq1+' Hz') 
    return line1

 def run_anim(i2):
    q=res['phi'][:,idmode-1]*f1(i2*dt)*fact  
    uv[0:2,0:2]=q.reshape([2,2])
    XY1=XY+uv
    for i3 in range(len(line1)):
     line1[i3].set_data(XY1[Elt[:,i3],0],XY1[Elt[:,i3],1])
    return line1

 ani1 = animation.FuncAnimation(fig, run_anim, init_func=init_anim, frames=ncycle*nframe, blit=True, interval=20., repeat=False)
 plot.show()
 return ani1

#%%  Plot 2D figures 
def plot2D(xyplot,style='-',xscale='linear',yscale='linear',xlim=[],ylim=[],gf=1,clf=0):
    plot.figure(num=gf);plot.clf();
    plot.plot(xyplot['X'],xyplot['Y'],style)
    plot.xlabel(xyplot['Xlabel']);plot.ylabel(xyplot['Ylabel'])   
    if 'legend' in xyplot.keys(): plot.legend(xyplot['legend'])   
    plot.grid();plot.xscale(xscale);plot.yscale(yscale);
    if len(xlim)>0: plot.xlim(xlim)
    if len(ylim)>0: plot.ylim(ylim)
    plot.show() 
#%%  Plot Bode 
def plotFreq(xy,style='-',xscale='linear',yscale='log',xlim=[],ylim=[],gf=1,clf=0):

    f2=plot.figure(num=gf);f2.clf()
    ax=f2.subplots(1,2,True,False,1)    
    ax[0].plot(xy['X'],abs(xy['Y']),style)
    ax[0].set_xlabel(xy['Xlabel']);ax[0].set_ylabel(xy['Ylabel'])   
    if 'legend' in xy.keys(): plot.legend(xy['legend'])   
    ax[0].grid();ax[0].set_xscale(xscale);ax[0].set_yscale(yscale);
    if len(xlim)>0: plot.xlim(xlim)
    if len(ylim)>0: plot.ylim(ylim)
    
    ax[1].plot(xy['X'],np.angle(xy['Y'],deg=True),style);ax[1].grid();

    plot.show()    

#------------------------------------------------------------------------------
#
#   Part 1: Coupling
#
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#%%   Question 1: equations
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
  return dict([("K",K),("M",M)])


#------------------------------------------------------------------------------
#%%   Question 2: modes
#------------------------------------------------------------------------------
def q2(pa=[], pflag=True):
 # parameters 4 DOF system
 if type(pa)==list:
  pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
 # compute modes
 mo1 = q1(pa, pflag=False)
 res = KMeig(mo1['K'], mo1['M'])  
 # run anim
 if pflag:
  print("omega=", res['w']);print(' ');print("phi= ", res['phi'])
  anim_mode(res, idmode=1, nframe=50, ncycle=10, fact=1.e-1)
 else:
  return res


#------------------------------------------------------------------------------
#%%   Question 3: coupling force
#------------------------------------------------------------------------------
def q3():
 # parameters 4 DOF system 
 pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
 #%% compute modes
 res = q2(pa, pflag=False);phi=res['phi'];om=res['w'];

 
 # compute modes
 res = q2(pa, pflag=False)
 # compute force

 #%% compute force observation matrix
 c=cosd(pa['a']);s=sind(pa['a']);
 C=pa['k3']*np.array([-c,-s,c,s]); 
 f3= C @ res['phi']
 print("F3 =",f3);
  
 #%% Computation strategies
 B=np.array([1,0,0,0]); 
 mo1= q1(pa, pflag=False); K=mo1['K'];M=mo1['M'];

 #q3c : write formula forced response
 s=2.j ;  Z=s**2*M+K
 y = C.dot ( linalg.solve(Z,B) )  # Ecrire la formule
 # Write formula spectral response
 y2=( C @ phi) @ (np.diag(1/(s**2+om**2))) @ ( phi.T @ B )
 
 # 4 edit code to add Rayleigh damping in the code below
 
#------------------------------------------------------------------------------
#%%   Question 4: damping
#------------------------------------------------------------------------------
def q4():
 # parameters 4 DOF system 
 pa=dict([('m',10.),('L',1.),('a',45.),           #mass, length, angle 
         ('k1',1000.0),('k2',10000.),('k3',1000.),  #spring stiffness
         ('Tend',10.), ('dt',0.1)               #plot limit and time discretisation
         ])
 # compute modes
 res = q2(pa, pflag=False); phi=res['phi'];mo1 = q1(pa, pflag=False)
 # init data structure
 bmax = 4./res['w'][-1]
 xyplot=dict([('X',np.linspace(0,bmax,int(np.ceil(bmax/1.e-4)))),  
             ('Xlabel','Beta'), ('Y',''), ('Ylabel','Damping ratio'),
             ('legend',np.arange(res['phi'].size)+1) ])
 # loop over values
 xyplot['Y']=np.zeros((len(xyplot['X']),res['w'].size))
 for j1 in range(res['w'].size):
   xyplot['Y'][:,j1]=xyplot['X']*res['w'][j1]/2.
 plot2D(xyplot,gf=4)
 
 # compute transfer
 w=np.linspace(0,50,500)
 xyplot2=dict([('X',w),('Xlabel','w'), ('Y',''), ('Ylabel','h41')  ])
 xyplot2['Y']=np.zeros((len(xyplot2['X']),1))
 for j1 in range(res['w'].size):
   xyplot2['Y'][:,0]+= (
      res['phi'][1,j1]*res['phi'][1,j1] #Formula for this and correction
      /(res['w'][j1]**2-w**2) # Formula for this and correction
     )
 plotFreq(xyplot2,gf=5)

 # Q4e loss factor usage
 mo2 = q1(dict([('m',0.),('L',1.),('a',45.), 
                ('k1',0.0),('k2',0),('k3',1000.)]),pflag=False)
 dK=.1* phi.T @ mo2['K'] @ phi
 v=np.diag(dK)/res['w']/res['w']/2 # What is this ?
 [ld2,psi]=linalg.eig(phi.T@(mo1['K']+.1j*mo2['K'])@phi);ld=np.sqrt(ld2)
 

 # additional (future) questions
 # - rewrite  res['phi'][3,j1] with an observation matrix 
 # - observe resultant load


 
#------------------------------------------------------------------------------
#
#   Part 2: Modification
#
#------------------------------------------------------------------------------  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  