# -*- coding: mbcs -*- 

''' 
 ARTS ET METIERS - PGE GIM2 - PLASTICITY
 
 SIMULATION PORTIQUE
  (with contact)
 
 Contributed by M. Guskov, N. Ranc and E. Monteiro 
 Copyright (c) 2018-2023 by ENSAM Paris, All Rights Reserved.		
'''

from abaqus import *
from abaqusConstants import *
from caeModules import *

Mdb()

#---------------------------------------------------------------------------
#%% PARAMETERS (unit: MM-MPa)
#---------------------------------------------------------------------------
#choice of the model (1-beam; 2-solid2D; 3-solid3D)
idim=1
#name, dimensions of the sample [X,Y,Z], thickness, radius of cylinder,  
#size of elements, linear/quadratic interpolation (False/True), plane strain/stress in 2D ('strain'/'stress'), radius of fillet
param=dict([('name','sample'),('dim',[300.,200.,11.]), ('thickness',3.), ('rad',2.), 
  ('selt',2.),('quad',True), ('type2d','stress'), ('radF',5.) ])
#name, density, Young modulus, Poisson ratio, initial yield stress, linear hardening coefficient 
mat=dict([('name','steel'),('dens',7850.e-12),('E',2.1e5),('nu',0.3),('Re',400.),('E1',1.0e3)])
#load direction ('vertical'/'horizontal'), imposed displacement, 
#specific case (0-None; 1-BC; 2-distance; 3-angle), value for specific case (1-stiffness; 2-distance; 3-angle)
simu=dict([('load','vertical'),('dmax',30.), ('case',0), ('val',25)])




#---------------------------------------------------------------------------
#%% MODELING
#---------------------------------------------------------------------------
def check_input(param,simu):
    ''' check input data '''
    if simu['case']==1: #stiffness
      simu['val']=abs(simu['val'])
    elif simu['case']==2: #distance
      max1 = param['dim'][1]*0.75 if simu['load']=='horizontal' else 0.75*param['dim'][0]/2.0
      if simu['val']<0.: simu['val']=0.
      if simu['val']>max1: simu['val']=max1
      if simu['val']<1.e-8: simu['case']=0
    elif simu['case']==3: #angle
      if simu['val']<0.: simu['val']=0.
      if simu['val']>45.: simu['val']=45.
      simu['val']=deg2rad(simu['val']) 
    return simu
	
def create_material(mat):
    ''' create materials '''
    mdb.models['Model-1'].Material(name=mat['name'])
    mdb.models['Model-1'].materials[mat['name']].Density(table=((mat['dens'], ), ))
    mdb.models['Model-1'].materials[mat['name']].Elastic(table=((mat['E'], mat['nu']), ))
    mdb.models['Model-1'].materials[mat['name']].Plastic(table=((mat['Re'], 0.0), (mat['Re']+10.0*mat['E1'],10.0)))

def deg2rad(x):
    ''' convert degrees to radians '''
    return x*pi/180.0
	  
def sketch1D(param):
	''' 1D sketch '''
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',sheetSize=2.*max(param['dim']))
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=STANDALONE)
	s1.Line(point1=(-param['dim'][0]/2., 0.0), point2=(-param['dim'][0]/2., param['dim'][1]))
	s1.Line(point1=(-param['dim'][0]/2., param['dim'][1]), point2=(0., param['dim'][1]))
	s1.Line(point1=(0., param['dim'][1]),point2=(param['dim'][0]/2., param['dim'][1]))
	s1.Line(point1=(param['dim'][0]/2., param['dim'][1]),point2=(param['dim'][0]/2., 0.)) 
	
	le1=s1.geometry.keys();lv1=[[0,1,-2,1],[2,3,-1,-2]]; 
	for v1 in lv1:
	  e1=g[le1[v1[0]]];e2=g[le1[v1[1]]];xy=e1.getVertices()[1].coords	  
	  s1.FilletByRadius(radius=param['radF'], curve1=e1, nearPoint1=(xy[0],xy[1]), curve2=e2, nearPoint2=(xy[0],xy[1]))
	
	return s1
	
def sketch2D(param):
	''' 2D sketch '''
	s1=sketch1D(param)
	s1.Line(point1=(param['dim'][0]/2., 0.),point2=(param['dim'][0]/2.-param['thickness'], 0.)) 
	s1.Line(point1=(param['dim'][0]/2.-param['thickness'], 0.),point2=(param['dim'][0]/2.-param['thickness'], param['dim'][1]-param['thickness'])) 
	s1.Line(point1=(param['dim'][0]/2.-param['thickness'], param['dim'][1]-param['thickness']),point2=(0.0, param['dim'][1]-param['thickness'])) 
	s1.Line(point1=(0.0, param['dim'][1]-param['thickness']),point2=(-param['dim'][0]/2.+param['thickness'], param['dim'][1]-param['thickness'])) 
	s1.Line(point1=(-param['dim'][0]/2.+param['thickness'], param['dim'][1]-param['thickness']),point2=(-param['dim'][0]/2.+param['thickness'], 0.)) 
	s1.Line(point1=(-param['dim'][0]/2.+param['thickness'], 0.),point2=(-param['dim'][0]/2., 0.)) 
	
	le1=s1.geometry.keys();lv1=[[7,8,-2,-1],[9,10,1,-2]]; #,[3,4,2,-1],[4,5,1,2],[8,9,2,-1],[9,0,1,2]
	for v1 in lv1:
	  e1=s1.geometry[le1[v1[0]]];e2=s1.geometry[le1[v1[1]]];xy=e1.getVertices()[1].coords	  
	  s1.FilletByRadius(radius=param['radF'], curve1=e1, nearPoint1=(xy[0],xy[1]), curve2=e2, nearPoint2=(xy[0],xy[1]))
	
	return s1
	
def sketch2D_cyl(xc,yc,tc,rc):
	''' draw cylinder '''
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2.*abs(rc))
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints; s1.setPrimaryObject(option=STANDALONE)
	if tc<>0:
	 s1.ArcByCenterEnds(center=(xc-rc, yc), point1=(xc-rc, yc-rc), point2=(xc, yc), direction=COUNTERCLOCKWISE)
	 s1.ArcByCenterEnds(center=(xc-rc, yc), point1=(xc, yc), point2=(xc-rc, yc+rc), direction=COUNTERCLOCKWISE)
	else:
	 s1.ArcByCenterEnds(center=(xc, yc+rc), point1=(xc-rc, yc+rc), point2=(xc, yc), direction=COUNTERCLOCKWISE)
	 s1.ArcByCenterEnds(center=(xc, yc+rc), point1=(xc, yc), point2=(xc+rc, yc+rc), direction=COUNTERCLOCKWISE)
	return s1

def cylinder2D(xc,yc,tc,rc):
	''' contact elements '''
	s1=sketch2D_cyl(xc,yc,tc,rc)
	p1 = mdb.models['Model-1'].Part(name='cylinder', dimensionality=TWO_D_PLANAR, type=DISCRETE_RIGID_SURFACE)
	p1 = mdb.models['Model-1'].parts['cylinder']
	p1.BaseWire(sketch=s1);s1.unsetPrimaryObject()
	p1 = mdb.models['Model-1'].parts['cylinder']
	session.viewports['Viewport: 1'].setValues(displayedObject=p1)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#specific set and surface
	p1.Surface(side2Edges=p1.edges.getByBoundingBox(), name='cylinder')	
	p1.ReferencePoint(point=p1.InterestingPoint(edge=p1.edges[0], rule=CENTER))
	p1.Set(referencePoints=(p1.referencePoints.values()[0],), name='ref_load')
	
	#mesh 
	elemType1 = mesh.ElemType(elemCode=R2D2, elemLibrary=STANDARD)
	p1.setElementType(regions=(p1.surfaces['cylinder'].edges,), elemTypes=(elemType1, ))
	p1.seedPart(size=param['rad']/5., deviationFactor=0.01, minSizeFactor=0.1)
	p1.generateMesh()
	
	return p1

def cylinder3D(xc,yc,tc,rc):
	''' contact elements '''
	s1=sketch2D_cyl(xc,yc,tc,rc)
	p1 = mdb.models['Model-1'].Part(name='cylinder', dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
	p1 = mdb.models['Model-1'].parts['cylinder']
	p1.BaseShellExtrude(sketch=s1, depth=param['dim'][2]/2.)
	s1.unsetPrimaryObject()
	p1 = mdb.models['Model-1'].parts['cylinder']
	session.viewports['Viewport: 1'].setValues(displayedObject=p1)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#specific set and surface
	p1.Surface(side1Faces=p1.faces.getByBoundingBox(), name='cylinder')	
	p1.ReferencePoint(point=p1.InterestingPoint(edge=p1.edges.getByBoundingBox(zMax=1.e-3)[0], rule=CENTER))
	p1.Set(referencePoints=(p1.referencePoints.values()[0],), name='ref_load')
	
	#mesh 
	elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=STANDARD)
	elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=STANDARD)
	p1.setElementType(regions=(p1.surfaces['cylinder'].faces,), elemTypes=(elemType1, elemType2 ))
	p1.seedPart(size=param['rad']/5., deviationFactor=0.01, minSizeFactor=0.1)
	p1.generateMesh()
	
	return p1
	
def beam_geo(param,mat,simu):
	''' beam geometry'''
	#beam model	
	s1=sketch1D(param)
	p = mdb.models['Model-1'].Part(name=param['name'], dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts[param['name']]
	p.BaseWire(sketch=s1);	s1.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts[param['name']]
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#partition
	if simu['case']==2:
	 if simu['load']=='horizontal':
	   e1=p.edges.getByBoundingBox(xMin=param['dim'][0]*0.499)	 
 	   p.PartitionEdgeByPoint(edge=e1[0], point=(param['dim'][0]/2.,param['dim'][1]-simu['val'],0.))
	   p1=cylinder2D(param['dim'][0]/2.,param['dim'][1]-simu['val'],45,param['rad'])
	 else:
	   e1=p.edges.getByBoundingBox(yMin=param['dim'][1]*0.9, xMin=0.0)	 
 	   p.PartitionEdgeByPoint(edge=e1[0], point=(simu['val'],param['dim'][1],0.))
	   p1=cylinder2D(simu['val'],param['dim'][1],0,param['rad'])
	else:
	 if simu['load']=='horizontal':
	  p1=cylinder2D(param['dim'][0]/2.,param['dim'][1]-param['radF'],45,param['rad'])
	 else:
	  p1=cylinder2D(0.,param['dim'][1],0,param['rad'])
			
	#specific set
	p = mdb.models['Model-1'].parts[param['name']];v=p.vertices
	v1 = v.getByBoundingBox(yMax=1.e-5);p.Set(vertices=v1, name='base');
	v2 = v.getByBoundingSphere(center=(0.0, param['dim'][1], 0.0),radius=1.e-4);p.Set(vertices=v2, name='middle');
	if simu['load']=='horizontal':v2 = v.getByBoundingSphere(center=(param['dim'][0]/2., param['dim'][1]-param['radF'], 0.0),radius=1.e-4)
	p.Set(vertices=v2, name='trav');p.Surface(name='top', side1Edges=(p.edges.getByBoundingBox(),))
	
	#beam section
	mdb.models['Model-1'].RectangularProfile(name='prof1', a=param['dim'][2], b=param['thickness'])
	mdb.models['Model-1'].BeamSection(name='sec', integration=DURING_ANALYSIS, 
	   poissonRatio=mat['nu'], profile='prof1', material=mat['name'], temperatureVar=LINEAR, consistentMassMatrix=False)
	f = p.edges.getByBoundingBox();	region1 = regionToolset.Region(edges=f)
	p.SectionAssignment(region=region1, sectionName='sec', offset=0.0, 
          offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
	p.assignBeamSectionOrientation(region=region1, method=N1_COSINES, n1=(0.0, 0.0, -1.0))	
	
	#beam mesh
	r1 = p.edges.getByBoundingBox(); p.Set(name='all', edges=r1)
	if param['quad']:
	 #quadratic beam
	 elemType1 = mesh.ElemType(elemCode=B22, elemLibrary=STANDARD)
	else:
	 #linear beam
	 elemType1 = mesh.ElemType(elemCode=B21, elemLibrary=STANDARD) 
	create_mesh(param,r1,(elemType1, ))	 
	#	
	return p

def geo2D(param,mat,simu):
	''' 2D geometry'''	
	# 2D model	
	s1=sketch2D(param)
	p = mdb.models['Model-1'].Part(name=param['name'], dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts[param['name']]
	p.BaseShell(sketch=s1); s1.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts[param['name']]
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#partition
	p.PartitionFaceByShortestPath(faces=p.faces.getByBoundingBox(), point1=(0.,0.,0.), point2=(0.,param['dim'][1],0.))
	if simu['case']==2:
	 if simu['load']=='horizontal':
	   yc=param['dim'][1]-param['thickness']-param['radF']-simu['val'];p1=(0.,yc,0.);p2=(param['dim'][0],yc,0.);	 
	   cylinder2D(param['dim'][0]/2.-param['thickness']*1.002,yc,45,param['rad'])	   
	 else:
	   p1=(simu['val'],0.,0.);p2=(simu['val'],param['dim'][1],0.);
	   cylinder2D(simu['val'],param['dim'][1],0,param['rad'])
	 p.PartitionFaceByShortestPath(faces=p.faces.getByBoundingBox(), point1=p1, point2=p2)
	else:
	 if simu['load']=='horizontal':
	  cylinder2D(param['dim'][0]/2.-param['thickness']*1.005,param['dim'][1]-param['thickness']-param['radF'],45,param['rad'])
	 else:
	  cylinder2D(0.,param['dim'][1],0,param['rad'])
	  
	#specific set
	p = mdb.models['Model-1'].parts[param['name']];e=p.edges;v = p.vertices;
	v1 = v.getByBoundingBox(xMin=-1.e-4,xMax=1.e-4);p.Set(vertices=v1, name='middle');
	e1 = e.getByBoundingBox(yMax=1.e-4);p.Set(edges=e1, name='base');
	v2 = v.getByBoundingSphere(center=(0.0, param['dim'][1]-param['thickness'], 0.0),radius=1.e-4);
	if simu['load']=='horizontal': v2 = v.getByBoundingBox(xMin=param['dim'][0]/2.-1.01*param['thickness'], xMax=param['dim'][0]/2.-0.99*param['thickness'], yMin=param['dim'][1]/2., yMax=param['dim'][1]-1.01*param['thickness']);	
	p.Set(vertices=v2, name='trav');p.Surface(name='top', side1Edges=(p.edges.getByBoundingBox(),))
	
	#2D section 
	mdb.models['Model-1'].HomogeneousSolidSection(name='sec', material=mat['name'], thickness=param['dim'][2])
	p = mdb.models['Model-1'].parts[param['name']]
	f = p.faces.getByBoundingBox();	region1 = regionToolset.Region(faces=f)
	p.SectionAssignment(region=region1, sectionName='sec', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
		
	#2D mesh
	r1 = p.faces.getByBoundingBox();p.Set(name='all', faces=r1)
	p.setMeshControls(regions=r1, elemShape=TRI)
	if param['quad']:	  
	 #quadratic
	 if param['type2d']=='strain':
	  #plane strain
	  elemType1 = mesh.ElemType(elemCode=CPE8R, elemLibrary=STANDARD)
	  elemType2 = mesh.ElemType(elemCode=CPE6M, elemLibrary=STANDARD)
	 else:	 
	  #plane stress
	  elemType1 = mesh.ElemType(elemCode=CPS8R, elemLibrary=STANDARD)
	  elemType2 = mesh.ElemType(elemCode=CPS6M, elemLibrary=STANDARD)
	else:
	 #linear 
	 if param['type2d']=='strain':
	  #plane strain
	  elemType1 = mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
	  elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
	 else:	 
	  #plane stress
	  elemType1 = mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
	  elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
	create_mesh(param,r1,(elemType1, elemType2))
    #	
	return p

def geo3D(param,mat,simu):
	''' 3D geometry'''
	#3D model	
	s1=sketch2D(param)
	p = mdb.models['Model-1'].Part(name=param['name'], dimensionality=THREE_D, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts[param['name']]
	p.BaseSolidExtrude(sketch=s1, depth=param['dim'][2]/2.)
	s1.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts[param['name']]
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#partition
	p.PartitionCellByPlaneThreePoints(point1=(0.,0.,0.), point2=(0.,1.,0.), point3=(0.,0.,1.),cells=p.cells.getByBoundingBox())
	if simu['case']==2:
	 if simu['load']=='horizontal':
	   yc=param['dim'][1]-param['thickness']-param['radF']-simu['val'];p1=(0.,yc,0.);p2=(0.,yc,1.);p3=(1.,yc,0.)	
	   cylinder3D(param['dim'][0]/2.-param['thickness'],yc,45,param['rad'])		   
	 else:
	   p1=(simu['val'],0.,0.);p2=(simu['val'],1.,0.);p3=(simu['val'],0.,1.)	
	   cylinder3D(simu['val'],param['dim'][1],0,param['rad'])
	 p.PartitionCellByPlaneThreePoints(cells=p.cells.getByBoundingBox(), point1=p1, point2=p2, point3=p3)
	else:
	 if simu['load']=='horizontal':
	  cylinder3D(param['dim'][0]/2.-param['thickness'],param['dim'][1]-param['thickness']-param['radF'],45,param['rad'])
	 else:
	  cylinder3D(0.,param['dim'][1],0,param['rad'])

	
	#specific set
	f = p.faces.getByBoundingBox(xMin=-1.e-4, xMax=1.e-4);p.Set(faces=f, name='middle');
	f = p.faces.getByBoundingBox(yMax=1.e-4);p.Set(faces=f, name='base');
	f = p.faces.getByBoundingBox(zMax=1.e-3);p.Set(faces=f, name='z0')
	e1 = p.edges.getByBoundingBox(xMin=-1.e-4, xMax=1.e-4, yMax=param['dim'][1]-0.5*param['thickness']);
	if simu['load']=='horizontal': e1 = p.edges.getByBoundingBox(xMin=param['dim'][0]/2.-1.01*param['thickness'], xMax=param['dim'][0]/2.-0.99*param['thickness'], yMin=param['dim'][1]/2., yMax=param['dim'][1]-1.01*param['thickness'])
	p.Set(edges=e1, name='trav');p.Surface(name='top', side12Faces=(p.faces.getByBoundingBox()))
	
	#3D section
	mdb.models['Model-1'].HomogeneousSolidSection(name='sec', material=mat['name'], thickness=None)
	f = p.cells.getByBoundingBox();	region1 = regionToolset.Region(cells=f)
	p.SectionAssignment(region=region1, sectionName='sec', offset=0.0,  
         offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
		 
	#3D mesh
	r1 = p.cells.getByBoundingBox();p.Set(name='all', cells=r1);
	if param['quad']:
	 #quadratic 
	 elemType1 = mesh.ElemType(elemCode=C3D20, elemLibrary=STANDARD)
	 elemType2 = mesh.ElemType(elemCode=C3D15, elemLibrary=STANDARD)
	 elemType3 = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD)
	else:
	 #linear
	 elemType1 = mesh.ElemType(elemCode=C3D8, elemLibrary=STANDARD)
	 elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
	 elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
	create_mesh(param,r1,(elemType1, elemType2, elemType3))	 
	#
	return p

def create_mesh(param,region1,elemT):
	''' create mesh '''
	p = mdb.models['Model-1'].parts[param['name']]
	p.setElementType(regions=(region1,), elemTypes=elemT)
	p.seedPart(size=param['selt'], deviationFactor=0.01, minSizeFactor=0.1)
	p.generateMesh()

def create_assembly(param):
	''' create assembly '''
	a = mdb.models['Model-1'].rootAssembly;a.DatumCsysByDefault(CARTESIAN)
	p = mdb.models['Model-1'].parts[param['name']]; a.Instance(name=param['name'], part=p, dependent=ON)
	p1 = mdb.models['Model-1'].parts['cylinder'];   a.Instance(name='cylinder', part=p1, dependent=ON)
	session.viewports['Viewport: 1'].setValues(displayedObject=a)
	return a
	
def create_interaction(idim,param):
	''' create interaction '''
	mdb.models['Model-1'].ContactProperty('prop_cont_cyl')
	mdb.models['Model-1'].interactionProperties['prop_cont_cyl'].NormalBehavior(
	 pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
	a = mdb.models['Model-1'].rootAssembly;
	region1=a.instances['cylinder'].surfaces['cylinder'];region2=a.instances[param['name']].surfaces['top']
	if idim==1:
	  mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='cont_cylinder', 
	   createStepName='Initial', main=region1, secondary=region2, sliding=SMALL, 
	   enforcement=NODE_TO_SURFACE, thickness=OFF, interactionProperty='prop_cont_cyl', surfaceSmoothing=NONE, 
	   adjustMethod=NONE, smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
	else:
	  mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='cont_cylinder', createStepName='Initial', 
	   main=region1, secondary=region2, sliding=SMALL, thickness=ON, interactionProperty='prop_cont_cyl',
	   adjustMethod=NONE, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

def create_steps(idim,param, simu):
	''' create steps '''
	#loading
	mdb.models['Model-1'].ImplicitDynamicsStep(name='loading', previous='Initial', nlgeom=ON,
	  timePeriod=int(ceil(abs(simu['dmax']))*2.), initialInc=1e-3, minInc=1e-5, maxInc=0.5, maxNumInc=10000, 
	  application=QUASI_STATIC, nohaf=OFF, amplitude=RAMP, alpha=DEFAULT, initialConditions=OFF)
	#unloading	
	mdb.models['Model-1'].ImplicitDynamicsStep(name='unloading', previous='loading', nlgeom=ON, 
	  timePeriod=ceil(abs(simu['dmax'])/2.), initialInc=1e-2, minInc=1e-5, maxInc=1.0, maxNumInc=10000,    
	  application=QUASI_STATIC, nohaf=OFF, amplitude=RAMP, alpha=DEFAULT, initialConditions=OFF)
	#ouput
	a=mdb.models['Model-1'].rootAssembly; 
	#sample
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(timeInterval=1.0, region=MODEL,
	 variables=('S', 'E', 'PE', 'EE', 'U', 'SF', 'CSTRESS', 'CSTATUS'))
	if idim==1: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(sectionPoints=(1, 2, 3, 4, 5))
	#load
	region1=a.instances['cylinder'].sets['ref_load']
	out1=('U1','RF1') if simu['load']=='horizontal' else ('U2','RF2')
	mdb.models['Model-1'].FieldOutputRequest(name='F-Output-2', 
	 createStepName='loading', variables=('U', 'RF', ), timeInterval=1.0, 
	region=region1, sectionPoints=DEFAULT, rebar=EXCLUDE)
	mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=out1,
	 timeInterval=1.0, region=region1, sectionPoints=DEFAULT, rebar=EXCLUDE)
	 
def create_loads(idim,param,simu):
	''' create loads '''
	a = mdb.models['Model-1'].rootAssembly
	#fix
	region1 = a.instances[param['name']].sets['base']
	mdb.models['Model-1'].DisplacementBC(name='fixY', createStepName='Initial', 
	  region=region1, u1=SET, u2=SET, ur3=UNSET, amplitude=UNSET, 
	  distributionType=UNIFORM, fieldName='', localCsys=None)
	#symmetry in 3D
	if idim==3:
	  mdb.models['Model-1'].ZsymmBC(name='zsym', createStepName='Initial', 
	    region=a.instances[param['name']].sets['z0'], localCsys=None)
	  mdb.models['Model-1'].boundaryConditions['fixY'].setValues(u3=SET)
	#spring only for 1D
	if simu['case']==1:  
	  mdb.models['Model-1'].rootAssembly.engineeringFeatures.SpringDashpotToGround(
	    dashpotBehavior=OFF, dashpotCoefficient=0.0, dof=6, name='spring', 
	    orientation=None, region=region1, springBehavior=ON, springStiffness=simu['val'])
	else:
	  mdb.models['Model-1'].boundaryConditions['fixY'].setValues(ur3=SET)        
	#loading 		
	region1 = a.instances['cylinder'].sets['ref_load']
	if simu['load']=='horizontal':
	 if simu['case']==3:
	  u1in=abs(simu['dmax'])*cos(simu['val']); u2in=abs(simu['dmax'])*sin(simu['val']) 
	  mdb.models['Model-1'].DisplacementBC(name='movX', createStepName='loading', 
	    region=region1, u1=u1in, u2=u2in, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
	    distributionType=UNIFORM, fieldName='', localCsys=None)     
	 else:
	  mdb.models['Model-1'].DisplacementBC(name='movX', createStepName='loading', 
	    region=region1, u1=abs(simu['dmax']), u2=0., ur3=0., amplitude=UNSET, fixed=OFF, 
	    distributionType=UNIFORM, fieldName='', localCsys=None)
	 #       
	 if idim==3: mdb.models['Model-1'].boundaryConditions['movX'].setValues(u3=0., ur1=0., ur2=0.) 
	 mdb.models['Model-1'].boundaryConditions['movX'].setValuesInStep(stepName='unloading', u1=0.0, u2=0.0)
	else:
	 if simu['case']==3:
	  u1in=abs(simu['dmax'])*sin(simu['val']); u2in=abs(simu['dmax'])*cos(simu['val']) 
	  mdb.models['Model-1'].DisplacementBC(name='movY', createStepName='loading', 
	    region=region1, u1=u1in, u2=-u2in, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
	    distributionType=UNIFORM, fieldName='', localCsys=None)
	 else:        
	  mdb.models['Model-1'].DisplacementBC(name='movY', createStepName='loading', 
	    region=region1, u1=0., u2=-abs(simu['dmax']), ur3=0., amplitude=UNSET, fixed=OFF, 
	    distributionType=UNIFORM, fieldName='', localCsys=None)
	 #       
	 if idim==3: mdb.models['Model-1'].boundaryConditions['movY'].setValues(u3=0., ur1=0., ur2=0.) 
	 mdb.models['Model-1'].boundaryConditions['movY'].setValuesInStep(stepName='unloading', u1=0.0, u2=0.0)
	  
def create_job(idim,irun=1):
	''' create job '''
	jobname='portique_'+str(idim)
	#
	mdb.Job(name=jobname, model='Model-1', description='', type=ANALYSIS, 
	 atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
	 memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
	 explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF, 
	 modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
	 scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
	#
	if irun==1:
	  mdb.jobs[jobname].submit(consistencyChecking=OFF)	
	  mdb.jobs[jobname].waitForCompletion()
	  return jobname	  
	return []
	
def post(idim,jobname,param,simu):
	''' generate output '''
	o3 = session.openOdb(name=os.getcwd()+'/'+jobname+'.odb')
	#view
	out1=['U1','RF1'] if simu['load']=='horizontal' else ['U2','RF2']
	session.viewports['Viewport: 1'].setValues(displayedObject=o3)
	session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(mirrorAboutYzPlane=False)
	session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(mirrorAboutXyPlane=False)
	if idim==3:
	 session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(mirrorAboutXyPlane=True)
	session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF))
	session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=-1)
	session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, out1[0]))
	session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(deformationScaling=UNIFORM, visibleEdges=FEATURE)
	#get results
	p=mdb.models['Model-1'].parts[param['name']];str1='CYLINDER.REF_LOAD';coef=[1.0, 1.0, 1.0, 2.0]
	U2 = xyPlot.xyDataListFromField(odb=o3, outputPosition=NODAL, variable=(('U', NODAL, ((COMPONENT, out1[0]), )), ), nodeSets=(str1, ))
	RF2 = xyPlot.xyDataListFromField(odb=o3, outputPosition=NODAL, variable=(('RF', NODAL, ((COMPONENT, out1[1]), )), ), nodeSets=(str1, ))
	if idim==1:
	 RM3 = xyPlot.xyDataListFromField(odb=o3, outputPosition=NODAL, variable=(('SM', INTEGRATION_POINT), ), nodeSets=(param['name'].upper()+'.TRAV', ))
	#save to CSV
	fid1=open(os.getcwd()+'/01_res'+jobname+'.txt','w')
	#
	fid1.write('Resultats du portique \n'+'Modelisation '+str(idim)+'D ('+str(len(p.elements))+' elements) \n\n')
	if idim==1:
	 fid1.write('temps;   deplacement;      force;        moment;   \n')
	else:
	 fid1.write('temps;   deplacement;      force;        \n')
	#
	for i1 in xrange(len(U2[0].data)):
	  if idim==1:
	   str1 = '{:5.0f};   {:+8.4e};   {:+8.4e};   {:+8.4f}  \n'.format(U2[0].data[i1][0],U2[0].data[i1][1],coef[idim]*RF2[0].data[i1][1],RM3[0].data[i1][1])
	  else:
	   str1 = '{:5.0f};   {:+8.4e};   {:+8.4e};   \n'.format(U2[0].data[i1][0],U2[0].data[i1][1],coef[idim]*RF2[0].data[i1][1])
	  fid1.write(str1.replace('.',','))
 	#
	fid1.close()

def make_model(idim, param, mat, simu):	
 ''' make model '''
 simu = check_input(param,simu); 
 #Domain 
 create_material(mat);
 if idim==2:
  geo2D(param,mat,simu)  	
 elif idim==3:
  geo3D(param,mat,simu)	
 else:
  beam_geo(param,mat,simu);idim=1	
 #
 create_assembly(param)
 create_interaction(idim,param)
 create_steps(idim,param, simu)
 create_loads(idim,param,simu)
 #
 return idim, simu
 
 
#-----------------------------------------------------------------------------	
#%%  MAIN
#-----------------------------------------------------------------------------
idim, simu = make_model(idim, param, mat, simu)
jobname=create_job(idim,irun=1)
if type(jobname)==str: 
  post(idim,jobname,param,simu)










