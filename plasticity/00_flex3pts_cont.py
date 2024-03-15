# -*- coding: mbcs -*- 

''' 
 ARTS ET METIERS - PGE GIM2 - PLASTICITY
 
 SIMULATION 3 POINTS BENDING
      (with contact)
 
 Contributed by M. Guskov, N. Ranc and E. Monteiro 
 Copyright (c) 2018-2024 by ENSAM Paris, All Rights Reserved.	
'''

from abaqus import *
from abaqusConstants import *
from caeModules import *
import numpy as np

Mdb()

#---------------------------------------------------------------------------
#%% PARAMETERS (unit: MM-MPa)
#---------------------------------------------------------------------------
#choice of the model (1-beam; 2-solid2D; 3-solid3D)
idim=1
#name, dimensions of the sample [X,Y,Z], radius of cylinder, space between fix points, 
#size of elements, linear/quadratic interpolation (False/True), plane strain/stress in 2D ('strain'/'stress')
param=dict([('name','sample'),('dim',[300., 14., 14.]), ('rad',5.), ('base',250.),
  ('selt',2.),('quad',True), ('type2d','stress') ])
#name, Density, Young modulus, Poisson ratio, initial yield stress, linear isotropic hardening coefficient 
mat=dict([('name','steel'),('dens',7850.e-12),('E',2.1e5),('nu',0.3),('Re',300.),('E1',1.0e3)])
#imposed displacement
simu=dict([('dmax',30.)])




#---------------------------------------------------------------------------
#%% MODELING
#---------------------------------------------------------------------------
def check_dim(param):
	''' check input data '''
	if (param['base']-param['dim'][0])>=-1.e-3:param['base']=None
	return param
	
def sketch1D(param):
	''' 1D sketch '''
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',sheetSize=2.*max(param['dim']))
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=STANDALONE)
	if param['base']==None:
	 s1.Line(point1=(0.0, 0.0), point2=(param['dim'][0]/2., 0.0))
	else:
	 s1.Line(point1=(0.0, 0.0), point2=(param['base']/2., 0.0))
	 s1.Line(point1=(param['base']/2., 0.0), point2=(param['dim'][0]/2., 0.0))
	 
	return s1
	
def sketch2D(param):
	''' 2D sketch '''
	s1=sketch1D(param)
	s1.Line(point1=(param['dim'][0]/2., 0.0), point2=(param['dim'][0]/2., param['dim'][2]))
	s1.Line(point1=(param['dim'][0]/2., param['dim'][2]), point2=(0.0, param['dim'][2]))
	s1.Line(point1=(0.0, param['dim'][2]), point2=(0.0, 0.0))
	return s1

def sketch2D_cyl(xc,yc,rc):
	''' draw cylinder '''
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2.*abs(rc))
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints; s1.setPrimaryObject(option=STANDALONE)
	#s1.ArcByCenterEnds(center=(xc, yc+rc), point1=(xc-rc, yc+rc), point2=(xc+rc, yc+rc), direction=COUNTERCLOCKWISE)
	s1.ArcByCenterEnds(center=(xc, yc+rc), point1=(xc-rc*cos(pi/3.), yc+rc*(1.-sin(pi/3.))), point2=(xc, yc), direction=COUNTERCLOCKWISE)
	s1.ArcByCenterEnds(center=(xc, yc+rc), point1=(xc, yc), point2=(xc+rc*cos(pi/3.), yc+rc*(1.-sin(pi/3.))), direction=COUNTERCLOCKWISE)
	return s1

def cylinder2D(namec,xc,yc,rc):
	''' contact elements '''
	s1=sketch2D_cyl(xc,yc,rc)
	p1 = mdb.models['Model-1'].Part(name=namec, dimensionality=TWO_D_PLANAR, type=DISCRETE_RIGID_SURFACE)
	p1 = mdb.models['Model-1'].parts[namec]
	p1.BaseWire(sketch=s1)
	s1.unsetPrimaryObject()
	p1 = mdb.models['Model-1'].parts[namec]
	session.viewports['Viewport: 1'].setValues(displayedObject=p1)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#specific set and surface
	p1.Surface(side2Edges=p1.edges.getByBoundingBox(), name=namec)	
	p1.ReferencePoint(point=p1.InterestingPoint(edge=p1.edges[0], rule=CENTER))
	p1.Set(referencePoints=(p1.referencePoints.values()[0],), name='ref_'+namec)
	
	#mesh 
	elemType1 = mesh.ElemType(elemCode=R2D2, elemLibrary=STANDARD)
	p1.setElementType(regions=(p1.surfaces[namec].edges,), elemTypes=(elemType1, ))
	p1.seedPart(size=param['rad']/10., deviationFactor=0.01, minSizeFactor=0.01)
	p1.generateMesh()
	
	return p1

def cylinder3D(namec,xc,yc,rc):
	''' contact elements '''
	s1=sketch2D_cyl(xc,yc,rc)
	p1 = mdb.models['Model-1'].Part(name=namec, dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
	p1 = mdb.models['Model-1'].parts[namec]
	p1.BaseShellExtrude(sketch=s1, depth=param['dim'][1]/2.)
	s1.unsetPrimaryObject()
	p1 = mdb.models['Model-1'].parts[namec]
	session.viewports['Viewport: 1'].setValues(displayedObject=p1)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#specific set and surface
	p1.Surface(side1Faces=p1.faces.getByBoundingBox(), name=namec)	
	p1.ReferencePoint(point=p1.InterestingPoint(edge=p1.edges.getByBoundingBox(zMax=1.e-3)[0], rule=CENTER))
	p1.Set(referencePoints=(p1.referencePoints.values()[0],), name='ref_'+namec)
	
	#mesh 
	elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=STANDARD)
	elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=STANDARD)
	p1.setElementType(regions=(p1.surfaces[namec].faces,), elemTypes=(elemType1, elemType2 ))
	p1.seedPart(size=param['rad']/10., deviationFactor=0.01, minSizeFactor=0.01)
	p1.generateMesh()
	
	return p1
			
def beam_geo(param,mat):
	''' beam geometry'''
	#beam model	
	s1=sketch1D(param)
	p = mdb.models['Model-1'].Part(name=param['name'], dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts[param['name']]
	p.BaseWire(sketch=s1)
	s1.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts[param['name']]
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#specific set
	p = mdb.models['Model-1'].parts[param['name']];v=p.vertices
	v1 = v.getByBoundingSphere(center=(0.0, 0.0, 0.0),radius=1.e-3);p.Set(vertices=v1, name='middle');p.Set(vertices=v1, name='trav');
	xc=param['dim'][0]/2. if param['base']==None else param['base']/2.
	v2 = v.getByBoundingSphere(center=(xc, 0.0, 0.0),radius=1.e-3);p.Set(vertices=v2, name='base');
	
	#specific surfaces	
	e1=	p.edges.getByBoundingBox();p.Surface(name='top', side1Edges=(e1,));p.Surface(name='bot', side2Edges=(e1,))
	
	#beam section
	mdb.models['Model-1'].RectangularProfile(name='prof1', a=param['dim'][1], b=param['dim'][2])
	mdb.models['Model-1'].BeamSection(name='sec', integration=DURING_ANALYSIS, 
	   poissonRatio=mat['nu'], profile='prof1', material=mat['name'], temperatureVar=LINEAR, consistentMassMatrix=False)
	f = p.edges.getByBoundingBox();	region1 = regionToolset.Region(edges=f)
	p.SectionAssignment(region=region1, sectionName='sec', offset=0.0, 
          offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
	p.assignBeamSectionOrientation(region=region1, method=N1_COSINES, n1=(0.0, 0.0, -1.0))	
	
	#beam mesh
	r1 = p.edges.getByBoundingBox();p.Set(name='all', edges=r1)
	if param['quad']:
	 #quadratic beam
	 elemType1 = mesh.ElemType(elemCode=B22, elemLibrary=STANDARD)
	else:
	 #linear beam
	 elemType1 = mesh.ElemType(elemCode=B21, elemLibrary=STANDARD) 
	
	return p,r1,(elemType1,)

def geo2D(param,mat):
	''' 2D geometry'''	
	# 2D model	
	s1=sketch2D(param)
	p = mdb.models['Model-1'].Part(name=param['name'], dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts[param['name']]
	p.BaseShell(sketch=s1)
	s1.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts[param['name']]
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']
		
	#specific sets
	p = mdb.models['Model-1'].parts[param['name']];e=p.edges;v = p.vertices;
	e1 = e.getByBoundingBox(xMax=1.e-3);p.Set(edges=e1, name='middle');
	v1 = v.getByBoundingSphere(center=(0.0, param['dim'][2], 0.0),radius=1.e-3);p.Set(vertices=v1, name='trav');
	xc=param['dim'][0]/2. if param['base']==None else param['base']/2.
	v2 = v.getByBoundingSphere(center=(xc, 0.0, 0.0),radius=1.e-3);p.Set(vertices=v2, name='base');
		
	#specific surfaces	
	e2=	e.getByBoundingBox(yMin=param['dim'][2]*0.95);p.Surface(name='top', side1Edges=(e2,))
	e3=	e.getByBoundingBox(yMax=1.e-3);p.Surface(name='bot', side1Edges=(e3,))
	
	#2D section 
	mdb.models['Model-1'].HomogeneousSolidSection(name='sec', material=mat['name'], thickness=param['dim'][1])
	p = mdb.models['Model-1'].parts[param['name']]
	f = p.faces.getByBoundingBox();	region1 = regionToolset.Region(faces=f)
	p.SectionAssignment(region=region1, sectionName='sec', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
		
	#2D mesh
	r1 = p.faces.getByBoundingBox();p.Set(name='all', faces=r1)
	if param['quad']:	  
	 #quadratic
	 if param['type2d']=='strain':
	  #plane strain
	  elemType1 = mesh.ElemType(elemCode=CPE8, elemLibrary=STANDARD)
	  elemType2 = mesh.ElemType(elemCode=CPE6, elemLibrary=STANDARD)
	 else:	 
	  #plane stress
	  elemType1 = mesh.ElemType(elemCode=CPS8, elemLibrary=STANDARD)
	  elemType2 = mesh.ElemType(elemCode=CPS6, elemLibrary=STANDARD)
	else:
	 #linear 
	 if param['type2d']=='strain':
	  #plane strain
	  elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
	  elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
	 else:	 
	  #plane stress
	  elemType1 = mesh.ElemType(elemCode=CPS4, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
	  elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
		
	return p,r1,(elemType1, elemType2)

def geo3D(param,mat):
	''' 3D geometry'''
	#3D model	
	s1=sketch2D(param)
	p = mdb.models['Model-1'].Part(name=param['name'], dimensionality=THREE_D, type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts[param['name']]
	p.BaseSolidExtrude(sketch=s1, depth=param['dim'][1]/2.)
	s1.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts[param['name']]
	session.viewports['Viewport: 1'].setValues(displayedObject=p)
	del mdb.models['Model-1'].sketches['__profile__']
	
	#specific set
	e1 = p.edges.getByBoundingBox(xMax=1.0e-3, yMin=param['dim'][2]-1.e-3);	p.Set(edges=e1, name='trav');
	
	if param['base']!=None:
	  xc=param['base']/2.
	  f = p.faces.getByBoundingBox(yMax=param['dim'][2]/2.)
	  v1 = p.vertices.getByBoundingSphere(center=(xc, 0.0, 0.0),radius=1.e-3)
	  v2 = p.vertices.getByBoundingSphere(center=(xc, 0.0, param['dim'][1]/2.),radius=1.e-3)
	  p.PartitionFaceByShortestPath(point1=v1[0], point2=v2[0], faces=f)
	else:
	  xc=param['dim'][0]/2.
	e1 = p.edges.getByBoundingBox(xMin=xc*0.95, xMax=xc*1.05)
	p.Set(edges=e1, name='base');
	
	f = p.faces.getByBoundingBox(zMax=1.e-3);p.Set(faces=f, name='z0')
	f = p.faces.getByBoundingBox(xMax=1.e-3);p.Set(faces=f, name='middle');	
	
	#specific surfaces	
	f2=	p.faces.getByBoundingBox(yMin=param['dim'][2]*0.95);p.Surface(name='top', side12Faces=(f,f2))
	f3=	p.faces.getByBoundingBox(yMax=1.e-3);p.Surface(name='bot', side12Faces=(f3,))
	
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
 		 
	return p,r1,(elemType1, elemType2, elemType3)

def create_materials(mat):
 ''' create materials '''
 mdb.models['Model-1'].Material(name=mat['name'])
 mdb.models['Model-1'].materials[mat['name']].Density(table=((mat['dens'], ), ))
 mdb.models['Model-1'].materials[mat['name']].Elastic(table=((mat['E'], mat['nu']), ))
 mdb.models['Model-1'].materials[mat['name']].Plastic(hardening=ISOTROPIC, table=((mat['Re'], 0.0), (mat['Re']+10.0*mat['E1'],10.0)))
 
def create_assembly(param):
 ''' create assembly'''
 a = mdb.models['Model-1'].rootAssembly;a.DatumCsysByDefault(CARTESIAN)
 p = mdb.models['Model-1'].parts[param['name']]; a.Instance(name=param['name'], part=p, dependent=ON)
 p = mdb.models['Model-1'].parts['trav']; a.Instance(name='trav', part=p, dependent=ON)
 p = mdb.models['Model-1'].parts['base']; a.Instance(name='base', part=p, dependent=ON)
 session.viewports['Viewport: 1'].setValues(displayedObject=a)
 
def create_contact(idim, param):
 ''' create contact '''
 a = mdb.models['Model-1'].rootAssembly;
 mdb.models['Model-1'].ContactProperty('prop_cont_trav')
 mdb.models['Model-1'].interactionProperties['prop_cont_trav'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)
 region1=a.instances['trav'].surfaces['trav'];region2=a.instances[param['name']].surfaces['top']
 if idim==2:
  mdb.models['Model-1'].interactionProperties['prop_cont_trav'].GeometricProperties(contactArea=param['dim'][1], padThickness=None)
 #TRAV
 if idim==1:
  mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='cont_trav', 
    createStepName='Initial', main=region1, secondary=region2, sliding=SMALL, 
    enforcement=NODE_TO_SURFACE, thickness=OFF, interactionProperty='prop_cont_trav', surfaceSmoothing=NONE, 
    adjustMethod=NONE, smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
 else:
  mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='cont_trav', createStepName='Initial', 
    main=region1, secondary=region2, sliding=SMALL, thickness=ON, interactionProperty='prop_cont_trav',
    adjustMethod=NONE, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
 #BASE
 mdb.models['Model-1'].ContactProperty('prop_cont_base')
 mdb.models['Model-1'].interactionProperties['prop_cont_base'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)	
 if idim==2:
  mdb.models['Model-1'].interactionProperties['prop_cont_base'].GeometricProperties(contactArea=param['dim'][1], padThickness=None)
 region1=a.instances['base'].surfaces['base'];region2=a.instances[param['name']].surfaces['bot']
 if idim==1:
  mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='cont_base', 
    createStepName='Initial', main=region1, secondary=region2, sliding=SMALL, 
    enforcement=NODE_TO_SURFACE, thickness=OFF, interactionProperty='prop_cont_base', surfaceSmoothing=NONE, 
    adjustMethod=NONE, smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
 else:
  mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='cont_base', createStepName='Initial', 
    main=region1, secondary=region2, sliding=SMALL, thickness=ON, interactionProperty='prop_cont_base',
    adjustMethod=NONE, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

def create_steps(idim, param, simu):
 ''' create steps '''
 #step: load
 mdb.models['Model-1'].ImplicitDynamicsStep(name='loading', previous='Initial', nlgeom=ON,
    timePeriod=int(ceil(abs(simu['dmax']))*2.), initialInc=1e-6, minInc=1e-9, maxInc=0.5, maxNumInc=10000000, 
    application=QUASI_STATIC, nohaf=OFF, amplitude=RAMP, alpha=DEFAULT, initialConditions=OFF)
 #step: unload		
 mdb.models['Model-1'].ImplicitDynamicsStep(name='unloading', previous='loading', nlgeom=ON, 
    timePeriod=ceil(abs(simu['dmax'])*2.), initialInc=1e-2, minInc=1e-5, maxInc=0.5, maxNumInc=10000,  
    application=QUASI_STATIC, nohaf=OFF, amplitude=RAMP, alpha=DEFAULT, initialConditions=OFF)
 #outputs
 a=mdb.models['Model-1'].rootAssembly; 
 #sample
 mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(timeInterval=0.5, region=MODEL,
    variables=('S', 'E', 'PE', 'EE', 'U', 'SF', 'CSTRESS', 'CSTATUS'))
 if idim==1: mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(sectionPoints=(1, 2, 3, 4, 5))
 #load
 region1=a.instances['trav'].sets['ref_trav']
 mdb.models['Model-1'].FieldOutputRequest(name='F-Output-2', 
    createStepName='loading', variables=('U', 'RF', ), timeInterval=0.5, 
    region=region1, sectionPoints=DEFAULT, rebar=EXCLUDE)
 mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=('U2', 'RF2', ),
     timeInterval=0.5, region=region1, sectionPoints=DEFAULT, rebar=EXCLUDE)
	 
def create_loads(idim, param, simu):
 ''' create loads '''
 a = mdb.models['Model-1'].rootAssembly
 #SYM
 region1=a.SetByBoolean(name='middle', sets=(a.instances[param['name']].sets['middle'],a.instances['trav'].sets['ref_trav']))
 mdb.models['Model-1'].XsymmBC(name='xsym', createStepName='Initial', region=region1, localCsys=None)
 #BASE
 region1 = a.instances['base'].sets['ref_base']
 mdb.models['Model-1'].EncastreBC(name='fixall', createStepName='Initial', region=region1, localCsys=None)
 #SYM 3D	
 if idim==3:
  region1=a.SetByBoolean(name='z0', sets=(a.instances[param['name']].sets['z0'],a.instances['trav'].sets['ref_trav']))
  mdb.models['Model-1'].ZsymmBC(name='zsym', createStepName='Initial', region=region1, localCsys=None)
 #GRAVITY
 region1 = a.instances[param['name']].sets['all']
 mdb.models['Model-1'].Gravity(name='grav', createStepName='loading', region=region1, comp2=-9.81e3, distributionType=UNIFORM, field='')
 #LOADING
 region1 = a.instances['trav'].sets['ref_trav']
 mdb.models['Model-1'].DisplacementBC(name='movY', createStepName='loading', 
    region=region1, u1=0.0, u2=-abs(simu['dmax']), ur3=0.0, amplitude=UNSET, fixed=OFF, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
 #UNLOADING	
 mdb.models['Model-1'].boundaryConditions['movY'].setValuesInStep(stepName='unloading', u2=0.0)
 
def create_job(idim,irun=1):
 ''' create job '''
 jobname='flex3pts_cont_'+str(idim)
 mdb.Job(name=jobname, model='Model-1', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
    
 if irun==1:
	  mdb.jobs[jobname].submit(consistencyChecking=OFF)	
	  mdb.jobs[jobname].waitForCompletion()
	  return jobname	  
 return []
 
def post_treat(jobname, idim, param):
 ''' post treatment '''
 #load
 o3 = session.openOdb(name=os.getcwd()+'/'+jobname+'.odb')
 #path
 p = mdb.models['Model-1'].parts[param['name']]
 in1=np.argsort([x.coordinates[1] for x in p.sets['middle'].nodes]).tolist();id1=[x.label for x in p.sets['middle'].nodes]
 session.Path(name='path_middle', type=NODE_LIST, expression=((param['name'].upper(), tuple([id1[x] for x in in1])), ))
 #view 1
 session.viewports['Viewport: 1'].setValues(displayedObject=o3)
 session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(mirrorAboutXyPlane=False)
 session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(mirrorAboutYzPlane=True)
 if idim==3: 
  session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(mirrorAboutXyPlane=True)
 session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF))
 session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=-1)
 session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable( variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'))
 session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(deformationScaling=UNIFORM, uniformScaleFactor=1., visibleEdges=FEATURE)
 session.viewports['Viewport: 1'].maximize();origin1=session.viewports['Viewport: 1'].origin;
 height1=session.viewports['Viewport: 1'].height;width1=session.viewports['Viewport: 1'].width;
 session.viewports['Viewport: 1'].view.fitView()
 #CSV
 p=mdb.models['Model-1'].parts[param['name']];str1=param['name'].upper()+'.TRAV';coef=[2.0, 2.0, 2.0, 4.0]
 U2 = xyPlot.xyDataListFromField(odb=o3, outputPosition=NODAL, variable=(('U', NODAL, ((COMPONENT, 'U2'), )), ), nodeSets=(str1, ))
 RF2 = xyPlot.xyDataListFromField(odb=o3, outputPosition=NODAL, variable=(('RF', NODAL, ((COMPONENT, 'RF2'), )), ), nodeSets=('TRAV.REF_TRAV', ))
 if idim==1:
  RM3 = xyPlot.xyDataListFromField(odb=o3, outputPosition=NODAL, variable=(('SM', INTEGRATION_POINT), ), nodeSets=(str1, ))
 # 
 fid1=open(os.getcwd()+'/01_res'+jobname+'.txt','w')
 fid1.write('Resultats de flexion 3 points avec contact\n'+'Modelisation '+str(idim)+'D ('+str(len(p.elements))+' elements) \n\n')
 if idim==1:
  fid1.write('temps;   deplacement;      force;        moment;   \n')
 else:
  fid1.write('temps;   deplacement;      force;        \n')
 for i1 in xrange(len(U2[0].data)):
  if idim==1:
   str1 = '{:5.0f};   {:+8.4e};   {:+8.4e};   {:+8.4f}  \n'.format(U2[0].data[i1][0],U2[0].data[i1][1],coef[idim]*RF2[0].data[i1][1],RM3[0].data[i1][1])
  else:
   str1 = '{:5.0f};   {:+8.4e};   {:+8.4e};   \n'.format(U2[0].data[i1][0],U2[0].data[i1][1],coef[idim]*RF2[0].data[i1][1])
  fid1.write(str1.replace('.',','))
 
 fid1.close()


def make_model(idim, param, mat, simu):	
 ''' make model '''
 #MATERIALS
 create_materials(mat)
 #CAD
 param=check_dim(param)
 xc=param['dim'][0]/2. if param['base']==None else param['base']/2.
 if idim==2:
  p,r1,elemT=geo2D(param,mat) 
  p1=cylinder2D('trav',0.0,param['dim'][2],param['rad'])
  p2=cylinder2D('base',xc,0.0,-param['rad'])
 elif idim==3:
  p,r1,elemT=geo3D(param,mat)
  p1=cylinder3D('trav',0.0,param['dim'][2],param['rad'])
  p2=cylinder3D('base',xc,0.0,-param['rad']) 
 else:
  p,r1,elemT=beam_geo(param,mat);idim=1
  p1=cylinder2D('trav',0.0,0.0,param['rad'])
  p2=cylinder2D('base',xc,0.0,-param['rad']) 
 #MESH
 p = mdb.models['Model-1'].parts[param['name']]
 p.setElementType(regions=(r1,), elemTypes=elemT)
 p.seedPart(size=param['selt'], deviationFactor=0.1, minSizeFactor=0.1)
 p.generateMesh()
 #ASSEMBLY
 create_assembly(param)
 #CONTACT
 create_contact(idim, param)
 #STEPS
 create_steps(idim, param, simu)
 #LOADS
 create_loads(idim, param, simu)
 #out
 return idim, param



	


#---------------------------------------------------------------------------
#%% MAIN
#---------------------------------------------------------------------------
idim, param = make_model(idim, param, mat, simu)	
jobname = create_job(idim,irun=1)
if type(jobname)==str: 
 post_treat(jobname, idim, param)


