# -*- coding: utf-8 -*-
''' 
 FITE 2A - ARTS ET METIERS 
 Utilities for MEVIB Course
  
Contributed by E. Monteiro and E. Balmes
Copyright (c) 2018-2021 by ENSAM, All Rights Reserved.
'''

import matplotlib.pyplot as plot
import scipy.linalg as linalg
import numpy as np


#------------------------------------------------------------------------------
# DEFINITION OF MATHS FUNCTIONS 
#------------------------------------------------------------------------------
def feeig(K,M=[],norm='M'):
    ''' Compute eigenvalues and eigenvectors '''
    om2, vecs = linalg.eig(K,M) if len(M)>0 else linalg.eig(K)
    om=np.sqrt(om2);om=om.real;idx=np.argsort(om)
    val=om[idx];phi=vecs[:,idx]
    #Use mass normalization
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
    plot.show();RaiseFigure()
    


#%%  Plot 2D figures 
def plot2D(xyplot,style='-',xscale='linear',yscale='linear',xlim=[],ylim=[],gf=1,clf=0):
    ''' Plot 2D curves  ''' 
    f2=plot.figure(num=gf);f2.clf()
    plot.plot(xyplot['X'],xyplot['Y'],style)
    plot.xlabel(xyplot['Xlabel']);plot.ylabel(xyplot['Ylabel'])  
 
    if ('legend' in xyplot.keys()): plot.legend(xyplot['legend'])   
    plot.grid(); ax=plot.gca();ax.set_xscale(xscale); ax.set_yscale(yscale)
    if len(xlim)>0: ax.set_xlim(xlim)
    if len(ylim)>0: ax.set_ylim(ylim)
    plot.show();RaiseFigure()


#%%  Plot Bode from frequency response
def plotFreq(xy,style='-',xscale='linear',yscale='log',xlim=[],ylim=[],gf=1,clf=0):

    f2=plot.figure(num=gf);f2.clf()
    ax=f2.subplots(1,2)    
    ax[0].plot(xy['X'],abs(xy['Y']),style)
    ax[0].set_xlabel(xy['Xlabel']);ax[0].set_ylabel(xy['Ylabel'])   
    if 'legend' in xy.keys(): plot.legend(xy['legend'])   
    ax[0].grid();ax[0].set_xscale(xscale);ax[0].set_yscale(yscale);
    if len(xlim)>0: plot.xlim(xlim)
    if len(ylim)>0: plot.ylim(ylim)
    
    ax[1].plot(xy['X'],np.angle(xy['Y'],deg=True),style);ax[1].grid();

    plot.show();RaiseFigure()    


#%%  Compute Fourier transform and plot
def plotFourier(xy,tmin=0,tmax=1e10,gf=2,fmax=0):
   t=xy['X']; y=np.array(xy['Y']);
   indt=np.logical_and(t>=tmin,t<tmax);indt=indt.reshape(indt.shape[0])
   t=t[indt];y=y[indt,]; t=t-t[0]; 
   
   Y=np.fft.fft(y.T);Y=Y.T; # Y.shape
   f=np.arange(0.,(y.shape[0]),1,'double')/y.shape[0]/(t[1]-t[0])
   indf=f<min(fmax,f[len(f)-1]/2); f=f[indf];Y=Y[indf]
   f2=plot.figure(num=2);f2.clf()
   ax=f2.subplots(1,2)    
   ax[0].semilogy(f*2*np.pi,np.abs(Y),':.')
   ax[0].set_xlabel('Fréquence (rad/s)'); ax[0].set_ylabel('Amplitude')
   ax[0].grid()
   ax[1].plot(f*2*np.pi,np.angle(Y,deg=True));ax[1].grid();
   ax[1].set_yscale('linear');
   ax[1].set_xlabel('Fréquence (rad/s)');ax[1].set_ylabel('Phase')
   #ax[1].grid()
   plot.show();RaiseFigure()
   

def RaiseFigure(gf=[]):
   #figure(figsize=(8, 6), dpi=80)
   #plot.gcf()
   cfm = plot.get_current_fig_manager();
   if hasattr(cfm,'window'):   
    cfm.window.activateWindow()
    cfm.window.raise_()
    