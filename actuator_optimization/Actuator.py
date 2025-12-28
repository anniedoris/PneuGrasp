from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import math

class Actuator:
    def __init__(self, definition):

        # Import definition of parameters
        self.definition = definition

        # Define specific geometric properties
        self.t = self.definition['geometry']['t']
        self.r_core = self.definition['geometry']['r_core']
        self.b_core = self.definition['geometry']['b_core']
        self.h_core = self.definition['geometry']['h_core']
        self.h_upper = self.definition['geometry']['h_upper']
        self.w_core = self.definition['geometry']['w_core']
        self.theta_core = self.definition['geometry']['theta_core']
        self.n = self.definition['geometry']['n']
        self.wa_core = self.definition['geometry']['wa_core']
        self.w_extrude = self.definition['geometry']['w_extrude']
        self.h_round = self.definition['geometry']['h_round']
        self.ho_round = self.definition['geometry']['ho_round']
        self.alpha = self.definition['geometry']['alpha']
        self.round_core = self.definition['geometry']['round_core']
        self.round_skin = self.definition['geometry']['round_skin']

        # Material parameters
        self.material_name = self.definition['material_properties']['name']
        self.mat_model_type = self.definition['material_properties']['model']
        self.param_1 = self.definition['material_properties']['P1']
        self.param_2 = self.definition['material_properties']['P2']
        self.density = self.definition['material_properties']['density']

        # Pressure load
        self.pressure_load = self.definition['pressure']

        # Mesh parameters
        self.mesh = self.definition['geometry']['mesh']

        # # Type of simulation
        # self.sim_type = self.definition['sim_type']

        # Define specific model properties
        self.job_name = 'M1'
        self.model = None
        self.assembly = None
        self.actuator_instance = None
        self.actuator_part = None
        self.core_faces = [] # Store the point-on location of core faces
        self.current_job = None
        self.mass = None

    # Clear the model space, if we don't, abaqus will quit when re-running script

    def load_params(self, def_dict):

        # Import definition of parameters
        self.definition = def_dict

        # Define specific geometric properties
        self.t = self.definition['geometry']['t']
        self.r_core = self.definition['geometry']['r_core']
        self.b_core = self.definition['geometry']['b_core']
        self.h_core = self.definition['geometry']['h_core']

        # Set h_upper to the same as the thickness for now
        self.h_upper = self.definition['geometry']['t']
        self.w_core = self.definition['geometry']['w_core']
        self.theta_core = self.definition['geometry']['theta_core']
        self.n = self.definition['geometry']['n']
        self.wa_core = self.definition['geometry']['wa_core']
        self.w_extrude = self.definition['geometry']['w_extrude']
        self.h_round = self.definition['geometry']['h_round']
        self.alpha = self.definition['geometry']['alpha']

        # Keep ho_round parallel with h_round for now
        self.ho_round = self.h_round + self.t + self.t * math.sin(math.radians(self.alpha))
        self.round_core = self.definition['geometry']['round_core']
        self.round_skin = self.definition['geometry']['round_skin']

        # Material parameters
        self.material_name = self.definition['material_properties']['name']
        self.mat_model_type = self.definition['material_properties']['model']
        self.param_1 = self.definition['material_properties']['P1']
        self.param_2 = self.definition['material_properties']['P2']
        self.density = self.definition['material_properties']['density']

        # Pressure load
        self.pressure_load = self.definition['pressure']

        # Mesh size
        self.mesh = self.definition['geometry']['mesh']

        # Define specific model properties
        self.model = None
        self.assembly = None
        self.actuator_instance = None
        self.actuator_part = None
        self.core_faces = []  # Store the point-on location of core faces
        self.current_job = None

    def clear_model_space(self):

        # Delete any pre-existing parts
        for key in mdb.models['Model-1'].parts.keys():
            del mdb.models['Model-1'].parts[key]

        try:
            # Delete any pre-existing instances
            for key in mdb.models['Model-1'].rootAssembly.features.keys():
                del mdb.models['Model-1'].rootAssembly.features[key]
        except:
            # Delete any pre-existing instances
            for key in mdb.models['Model-1'].rootAssembly.features.keys():
                del mdb.models['Model-1'].rootAssembly.features[key]

    # Function that generates the core geometry
    def generate_core(self):

        # Helper function that shifts a point by a given distance (used primarily in x direction)
        def shift(point, x_shift, y_shift):
            return(point[0] + x_shift, point[1] + y_shift)

        # Definition of the points of the core, see diagram!
        PC1 = (self.t, self.t)
        PC2 = shift(PC1, 0, self.b_core)
        PC3 = shift(PC2, 0, self.h_core)
        PC4 = shift(PC2, self.r_core*math.tan(self.theta_core), self.r_core)
        PC5 = shift(PC2, self.h_core * math.tan(self.theta_core), self.h_core)
        PC6 = shift(PC5, self.w_core, 0)
        PC7 = shift(PC6, (self.h_core - self.r_core) * math.tan(self.theta_core), -self.h_core + self.r_core)
        PC8 = shift(PC7, 2*self.wa_core, 0)
        PC9 = shift(PC6, self.h_core * math.tan(self.theta_core), -self.h_core)
        PC10 = shift(PC9, 0, -self.b_core)

        # Initialize the core part
        core = mdb.models['Model-1'].Part(name='Core', dimensionality=THREE_D,
            type=DEFORMABLE_BODY)
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
            sheetSize=200.0)

        # Distance between two repeated points
        step = self.w_core + 2*self.wa_core + 2*(self.h_core - self.r_core)*math.tan(self.theta_core)

        # Generate the geometry
        for i in range(self.n):
            if i==0:
                s.Line(point1=PC1, point2=PC2)
                s.Line(point1=PC2, point2=PC3)
                s.Line(point1=PC3, point2=PC6)
            else:
                s.Line(point1=shift(PC4, step * i, 0), point2=shift(PC5, step * i, 0))
                s.Line(point1=shift(PC5, step * i, 0), point2=shift(PC6, step * i, 0))
            s.Line(point1=shift(PC6, step * i, 0), point2=shift(PC7, step * i, 0))

            if i!=(self.n-1):
                s.ArcByStartEndTangent(point1=shift(PC7, step * i, 0), point2=shift(PC8, step * i, 0),
                                       vector=(math.sin(self.theta_core), -math.cos(self.theta_core)))
            else:
                s.Line(point1=shift(PC7, step * i, 0),
                       point2=shift(PC9, step * i, 0))
                s.Line(point1=shift(PC9, step*i, 0), point2=shift(PC10, step*i, 0))
                s.Line(point1=PC1,
                       point2=shift(PC10, step*i, 0))

        # Save the sketch for reference later
        mdb.models['Model-1'].ConstrainedSketch(name='Core Side Sketch', objectToCopy=s)

        # Extrude the sketch
        core.BaseSolidExtrude(sketch=s, depth=self.w_extrude)

        # Generate the rounded part of the core
        rounded_core = mdb.models['Model-1'].Part(name='Rounded Core', dimensionality=THREE_D,
            type=DEFORMABLE_BODY)
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
            sheetSize=200.0)

        # Definition of points for the extrusion face
        PC11 = (0, 0)
        PC15 = shift(PC11, self.w_extrude, 0)
        PC12 = shift(PC11, self.h_round*math.tan(self.alpha), self.h_round)
        PC13 = shift(PC11, self.w_extrude / 2, self.b_core + self.h_core)
        PC14 = shift(PC15, -self.h_round*math.tan(self.alpha), self.h_round)

        # Creation of lines for the extrusion face
        total_length = step*(self.n-1) + 2*self.h_core*math.tan(self.theta_core) + self.w_core
        s.Line(point1=PC11, point2=PC15)
        s.Line(point1=PC11, point2=PC12)
        s.Arc3Points(point1=PC12,
                     point2=PC14,
                     point3=PC13)
        s.Line(point1=PC14, point2=PC15)

        # Exterior box lines
        s.Line(point1=(-self.t, -self.t), point2=(-self.t, self.b_core + self.h_core + self.t))
        s.Line(point1=(self.w_extrude + self.t, self.b_core + self.h_core + self.t),
               point2=(-self.t, self.b_core + self.h_core + self.t))
        s.Line(point1=(self.w_extrude + self.t, self.b_core + self.h_core + self.t),
               point2=(self.w_extrude + self.t, -self.t))
        s.Line(point1=(-self.t, -self.t), point2=(self.w_extrude + self.t, -self.t))

        # Save sketch for reference later
        mdb.models['Model-1'].ConstrainedSketch(name='Rounded Core Sketch', objectToCopy=s)
        rounded_core.BaseSolidExtrude(sketch=s, depth=total_length)

        # Create the rounded core, cut the core from the rounded core profile
        a = mdb.models['Model-1'].rootAssembly
        a.Instance(name='Core-A', part=core, dependent=OFF)
        a.Instance(name='Rounded Core-A', part=rounded_core, dependent=OFF)
        start_point = (self.w_extrude/2, 0, 0)
        end_point = (self.t, self.t, self.w_extrude/2)
        trans_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1], end_point[2] - start_point[2])
        a.translate(instanceList=('Rounded Core-A', ), vector=trans_vector)
        a.rotate(instanceList=('Rounded Core-A', ), axisPoint=(self.t, self.t, self.w_extrude/2),
                axisDirection=(0.0, 1.0, 0.0), angle=90.0)
        a.InstanceFromBooleanCut(name='Final Core',
                instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Core-A'],
                cuttingInstances=(a.instances['Rounded Core-A'], ),
                originalInstances=SUPPRESS)

        return

    # Function generates the skin of the actuator
    def generate_skin(self):

        def shift(point, x_shift, y_shift):
            return(point[0] + x_shift, point[1] + y_shift)

        # Initiate the skin part
        skin = mdb.models['Model-1'].Part(name='Skin', dimensionality=THREE_D,
                                          type=DEFORMABLE_BODY)

        # Initialize sketch
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)


        # Define a step for the skin
        step = self.t + self.w_core + 2 * (self.h_core - self.r_core) * math.tan(self.theta_core) + self.wa_core + (
                    self.wa_core - self.t)

        # Define all the points of the skin
        PS1 = (0, 0)
        PS2 = shift(PS1, 0, self.t + self.b_core)
        PS3 = shift(PS2, 0, self.h_core + self.h_upper)
        PS4 = shift(PS1, self.t - self.t * math.cos(self.theta_core),
                    self.t + self.b_core + self.t * math.sin(self.theta_core))
        PS5 = shift(PS4, self.r_core * math.tan(self.theta_core), self.r_core)
        PS6 = shift(PS5, (self.h_upper + self.h_core - self.r_core - self.t * math.sin(self.theta_core)) * math.tan(
            self.theta_core), (self.h_upper + self.h_core - self.r_core - self.t * math.sin(self.theta_core)))
        PS7 = (self.t + self.h_core * math.tan(self.theta_core) + self.w_core + (self.h_core - self.r_core) * math.tan(
            self.theta_core) + self.t * math.cos(
            self.theta_core) - (
                           self.h_upper + (self.h_core - self.r_core) - self.t * math.sin(self.theta_core)) * math.tan(
            self.theta_core),
               self.t + self.b_core + self.h_core + self.h_upper)
        PS8 = (self.t + self.h_core * math.tan(self.theta_core) + self.w_core + (self.h_core - self.r_core) * math.tan(
            self.theta_core) + self.t * math.cos(
            self.theta_core), self.t + self.b_core + self.r_core + self.t * math.sin(self.theta_core))
        PS9 = shift(PS8, self.r_core*math.tan(self.theta_core), -self.r_core)
        PS10 = shift(PS9, self.t - self.t * math.cos(self.theta_core), -self.t * math.sin(self.theta_core))
        PS11 = shift(PS10, 0, -self.b_core -self.t)

        # Generate the geometry
        for i in range(self.n):
            if i == 0:
                s.Line(point1=PS1, point2=PS2)
                s.Line(point1=PS2, point2=PS3)
                s.Line(point1=PS3, point2=PS7)
            else:
                s.Line(point1=shift(PS5, step * i, 0), point2=shift(PS6, step * i, 0))
                s.Line(point1=shift(PS6, step * i, 0), point2=shift(PS7, step * i, 0))
            s.Line(point1=shift(PS7, step * i, 0), point2=shift(PS8, step * i, 0))

            if i != (self.n - 1):
                s.ArcByStartEndTangent(point1=shift(PS8, step * i, 0), point2=shift(PS5, step * (i+1), 0),
                                       vector=(math.sin(self.theta_core), -math.cos(self.theta_core)))
            else:
                s.Line(point1=shift(PS8, step * i, 0),
                       point2=shift(PS9, step * i, 0))
                s.Line(point1=shift(PS9, step * i, 0),
                       point2=shift(PS10, step * i, 0))
                s.Line(point1=shift(PS10, step * i, 0), point2=shift(PS11, step * i, 0))
                s.Line(point1=PS1,
                       point2=shift(PS11, step * i, 0))

        # Save sketch for reference later
        mdb.models['Model-1'].ConstrainedSketch(name='Skin Sketch', objectToCopy=s)
        w_extrude_mold = self.w_extrude + 2 * (
                    self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha))

        # Extrude the base
        skin.BaseSolidExtrude(sketch=s, depth=w_extrude_mold)

        # Create the rounded mold part
        rounded_skin = mdb.models['Model-1'].Part(name='Rounded Skin', dimensionality=THREE_D,
                                                  type=DEFORMABLE_BODY)
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                    sheetSize=200.0)

        # Back to the first definition of step
        step = self.w_core + 2 * self.wa_core + 2 * (self.h_core - self.r_core) * math.tan(self.theta_core)
        total_length_mold = step * (self.n - 1) + 2 * self.h_core * math.tan(self.theta_core) + self.w_core + 2 * self.t

        # Define the points associated with the rounded profile
        PS12 = (-self.t * math.cos(self.alpha) - (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha), -self.t)
        PS13 = shift(PS12, self.ho_round * math.tan(self.alpha), self.ho_round)
        PS14 = shift(PS12, w_extrude_mold / 2, self.b_core + self.h_core + self.h_upper + self.t)
        PS16 = shift(PS12, w_extrude_mold, 0)
        PS15 = shift(PS16, - (self.ho_round * math.tan(self.alpha)), self.ho_round)

        # Define points of the rounded part
        s.Line(point1=PS12, point2=PS13)
        s.Arc3Points(point1=PS13, point2=PS15, point3=PS14)
        s.Line(point1=PS15, point2=PS16)
        s.Line(point1=PS12, point2=PS16)

        # Define points of the core
        A1 = shift(PS12, -self.t, -self.t)
        A2 = shift(A1, 0, 3 * self.t + self.b_core + self.h_core + self.h_upper)
        A3 = shift(A2, 2 * self.t + w_extrude_mold, 0)
        A4 = shift(A3, 0, -(3 * self.t + self.b_core + self.h_core + self.h_upper))

        # Create box around cut out
        s.Line(point1=A1, point2=A2)
        s.Line(point1=A2, point2=A3)
        s.Line(point1=A3, point2=A4)
        s.Line(point1=A4, point2=A1)

        # Save the sketch
        mdb.models['Model-1'].ConstrainedSketch(name='Rounded Skin', objectToCopy=s)

        # Extrude the rounded skin
        rounded_skin.BaseSolidExtrude(sketch=s, depth=total_length_mold)

        # Performing the rounding operation
        a = mdb.models['Model-1'].rootAssembly
        a.Instance(name='Mold-A', part=skin, dependent=OFF)
        a.Instance(name='Rounded Mold-A', part=rounded_skin, dependent=OFF)
        end_point = (0, 0, w_extrude_mold / 2.0)
        start_point = (w_extrude_mold / 2.0 + PS12[0], -self.t, 0)
        trans_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1], end_point[2] - start_point[2])
        a.translate(instanceList=('Rounded Mold-A',), vector=trans_vector)
        a.rotate(instanceList=('Rounded Mold-A',), axisPoint=(0, 0, w_extrude_mold / 2.0),
                 axisDirection=(0.0, 1.0, 0), angle=90.0)
        a.InstanceFromBooleanCut(name='Final Skin',
                                 instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Mold-A'],
                                 cuttingInstances=(a.instances['Rounded Mold-A'],),
                                 originalInstances=SUPPRESS)

    # Performs final cutting operation of the core from the skeleton
    def cut_core_skeleton(self):
        a = mdb.models['Model-1'].rootAssembly
        w_extrude_mold = self.w_extrude + 2 * (
                self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha))
        a.translate(instanceList=('Final Core-1',), vector=(0, 0, (w_extrude_mold - self.w_extrude) / 2.0))
        print("Translation Value:")
        print((w_extrude_mold - self.w_extrude) / 2.0)

        # Perform fillet operations here, right before subtraction operation, so that translation still possible
        # Fillet for the skin
        if self.round_skin != 0:
            final_skin = mdb.models['Model-1'].parts['Final Skin']
            edges_final_skin = final_skin.edges
            final_skin.Round(radius=self.round_skin, edgeList=tuple(edges_final_skin))

        # Fillet for the core
        if self.round_core != 0:
            final_core = mdb.models['Model-1'].parts['Final Core']
            edges_final_core = final_core.edges
            final_core.Round(radius=self.round_core, edgeList=tuple(edges_final_core))

        # Partition the core in half
        center_datum = mdb.models['Model-1'].parts['Final Core'].DatumPlaneByPrincipalPlane(principalPlane=XYPLANE,
                                                                     offset=self.w_extrude / 2.0)
        desired_datum = mdb.models['Model-1'].parts['Final Core'].datums[center_datum.id]  # Can access attributes of a feature this way
        all_actuator_cells = mdb.models['Model-1'].parts['Final Core'].cells
        mdb.models['Model-1'].parts['Final Core'].PartitionCellByDatumPlane(datumPlane=desired_datum, cells=all_actuator_cells)

        # Store locations of the core faces so we can make a pressure surface later
        core_instance = a.instances['Final Core-1']

        a.InstanceFromBooleanCut(name='Actuator',
                                 instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Final Skin-1'],
                                 cuttingInstances=(a.instances['Final Core-1'],),
                                 originalInstances=SUPPRESS)
        self.assembly = mdb.models['Model-1'].rootAssembly
        self.model = mdb.models['Model-1']
        self.actuator_instance = self.assembly.instances['Actuator-1']
        self.actuator_part = self.model.parts['Actuator']

        return

    def test_object(self):
        s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                     sheetSize=200.0)
        g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
        s1.setPrimaryObject(option=STANDALONE)
        s1.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(30.0, 0.0))
        p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D,
                                       type=DEFORMABLE_BODY)
        p = mdb.models['Model-1'].parts['Part-1']
        p.BaseSolidExtrude(sketch=s1, depth=20.0)
        s1.unsetPrimaryObject()
        return

    def grasp_object_setup(self):
        object_height = 40.0
        object_radius = 38.1
        translate_down = 11.47

        # Generate the object to grasp
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',
                                                    sheetSize=200.0)
        s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(object_radius, 0.0))
        p = mdb.models['Model-1'].Part(name='object', dimensionality=THREE_D,
                                       type=DISCRETE_RIGID_SURFACE)
        p.BaseSolidExtrude(sketch=s, depth=object_height)
        rp = p.ReferencePoint(point=(0.0, 0.0, 0.0))  # Create a reference point for fixing later

        # Make the grasping object be a shell object rather than a solid object
        p = mdb.models['Model-1'].parts['object']
        c1 = p.cells
        p.RemoveCells(cellList=c1[0:1])


        # Put the grasping object in the assembly
        a = mdb.models['Model-1'].rootAssembly
        a.Instance(name='object-1', part=p, dependent=OFF)
        w_extrude_mold = self.w_extrude + 2 * (
                self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha))
        x_motion = 16.38
        y_motion = -object_radius - translate_down
        z_motion = (w_extrude_mold/2.0) - (object_height/2.0)
        a.translate(instanceList=('object-1',), vector=(x_motion, 0.0, 0.0)) # Move in the x direction
        a.translate(instanceList=('object-1',), vector=(0.0, y_motion, 0.0))  # Move in the y direction
        a.translate(instanceList=('object-1',), vector=(0.0, 0.0, z_motion))  # Move in the z direction

        # Mesh the object
        partInstances = (a.instances['object-1'],)
        a.seedPartInstance(regions=partInstances, size=self.mesh, deviationFactor=0.1,
                           minSizeFactor=0.1)
        a.generateMesh(regions=partInstances)

        # Add the contact interactions between the part and the bottom of the actuator
        # Surface for the object part of the contact
        all_object_faces = a.instances['object-1'].faces
        side1Faces1 = all_object_faces.getSequenceFromMask(mask=('[#1 ]',), )
        region1 = a.Surface(side1Faces=side1Faces1, name='M_Surf-Object')

        # Surfaces for the actuator part of the contact
        all_actuator_faces = a.instances['Actuator-1'].faces
        xmin = -1000
        ymin = -0.001
        zmin = -1000
        xmax = 1000
        ymax = 0.001
        zmax = 1000
        side1Faces1 = all_actuator_faces.getByBoundingBox(xMin=xmin, yMin=ymin, zMin=zmin, xMax=xmax, yMax=ymax, zMax=zmax)
        region2 = a.Surface(side1Faces=side1Faces1, name='S_Surf-Actuator')

        # Define the interaction property
        mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='Int-3',
                                                         createStepName='Initial', master=region1, slave=region2,
                                                         sliding=FINITE, thickness=ON, interactionProperty='IntProp-1',
                                                         adjustMethod=NONE, initialClearance=OMIT, datumAxis=None,
                                                         clearanceRegion=None)

        # Fix the rigid object so that it doesn't move during simulation
        r1 = a.instances['object-1'].referencePoints
        rp_key = r1.keys()[0]
        choose_ref = (r1[rp_key],)
        region = a.Set(referencePoints=choose_ref, name='ObjectRP')
        mdb.models['Model-1'].EncastreBC(name='Fixed Object', createStepName='Initial',
                                         region=region, localCsys=None)

        # Request force output from the simulation
        mdb.models['Model-1'].HistoryOutputRequest(name='H-Output-Force',
                                                   createStepName='Pressure-Step',
                                                   variables=('RF1', 'RF2', 'RF3', 'RM1',
                                                              'RM2', 'RM3'), region=region, sectionPoints=DEFAULT,
                                                   rebar=EXCLUDE)
        return

    # Create the material model, the material section, and assign section to actuator
    def generate_material(self):
        self.model.Material(name=self.material_name)
        self.material_model = self.model.materials[self.material_name]

        # Define the material parameters
        self.material_model.Density(table=((self.density,),))
        if self.mat_model_type == "Neo-Hookean":
            self.material_model.Hyperelastic(
                materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE,
                volumetricResponse=VOLUMETRIC_DATA, table=((self.param_1, self.param_2),))

        # Create a section for the solid actuator
        mdb.models['Model-1'].HomogeneousSolidSection(name='Actuator-Section',
                                                      material=self.material_name, thickness=None)
        all_actuator_cells = self.actuator_part.cells
        all_actuator_set = self.actuator_part.Set(cells=all_actuator_cells, name='All Actuator Cells')
        self.actuator_part.SectionAssignment(region=all_actuator_set, sectionName='Actuator-Section', offset=0.0,
                            offsetType=MIDDLE_SURFACE, offsetField='',
                            thicknessAssignment=FROM_SECTION)

    # Create a partition down the center of the actuator and make instance independent
    def partition_down_center(self):

        # For partitioning the cell
        w_extrude_mold = self.w_extrude + 2 * (
                self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha))
        center_datum = self.actuator_part.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=w_extrude_mold/2.0)
        desired_datum = self.actuator_part.datums[center_datum.id] # Can access attributes of a feature this way
        all_actuator_cells = self.actuator_part.cells
        self.actuator_part.PartitionCellByDatumPlane(datumPlane=desired_datum, cells=all_actuator_cells)

        # Make the instance independent
        self.assembly.makeIndependent(instances=(self.actuator_instance,))
        return

    # Create the step for the pressure application
    def generate_steps(self):

        self.model.StaticStep(name='Pressure-Step', previous='Initial',
                              maxNumInc=10000, initialInc=0.1, maxInc=0.1, nlgeom=ON, minInc=1 * 10 ** (-5))

        self.model.steps['Pressure-Step'].Restart(frequency=1, numberIntervals=0, overlay=ON, timeMarks=OFF)

        # Fixed incrementation
        # mdb.models['Model-1'].StaticStep(name='Pressure-Step-Fixed',
        #                                  previous='Pressure-Step', maxNumInc=10000, timeIncrementationMethod=FIXED,
        #                                  initialInc=0.001, noStop=OFF)


    def apply_pressure(self):

        # Find the pressure surface
        actuator_instance_faces = self.actuator_instance.faces
        step = self.w_core + 2 * self.wa_core + 2 * (self.h_core - self.r_core) * math.tan(self.theta_core)
        total_length = step * (self.n - 1) + 2 * self.h_core * math.tan(self.theta_core) + self.w_core
        A = self.t/10

        xmin = self.t - A
        xmax = self.t + total_length + 2 * A
        ymin = self.t - A
        ymax = ymin + self.b_core + self.h_core + 2 * A
        zmin = self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha) - A
        zmax = zmin + self.w_extrude + 2 * A

        all_faces = actuator_instance_faces.getByBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)
        
        # Get rid of some pesky surfaces that linger after above operation and shouldn't be part of the pressure surface
        def shift(point, x_shift, y_shift):
            return(point[0] + x_shift, point[1] + y_shift)
        
        PS1 = (0, 0)
        PS4 = shift(PS1, self.t - self.t * math.cos(self.theta_core),
                    self.t + self.b_core + self.t * math.sin(self.theta_core))
        PS5 = shift(PS4, self.r_core * math.tan(self.theta_core), self.r_core)

        x = self.t*(math.cos(self.theta_core))
        d = 2*(self.wa_core - x)
        R = (d/2.0)/(math.cos(self.theta_core))
        H = R*math.sin(self.theta_core)
        chord_partial = R - H
        print("Chord Partial")
        print(chord_partial)
        xmin2 = -A
        xmax2 = total_length + A
        ymin2 = PS5[1] - chord_partial - 1.0
        ymax2 = PS5[1] + A
        zmin2 = self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha) - A
        zmax2 = zmin + self.w_extrude + 2 * A
        
        all_faces2 = actuator_instance_faces.getByBoundingBox(xmin2, ymin2, zmin2, xmax2, ymax2, zmax2)
            
        #Create pressure surfaces, perform boolean operation to get the correct pressure surface
        self.assembly.Surface(side1Faces=all_faces, name='Pressure Surface All')
        self.assembly.Surface(side1Faces=all_faces2, name='Pressure Surface Remove')
        a = mdb.models['Model-1'].rootAssembly
        a.SurfaceByBoolean(name = 'Pressure Surface', surfaces = [a.surfaces["Pressure Surface All"], a.surfaces["Pressure Surface Remove"]], operation = DIFFERENCE)

        # Set up fluid cavity
        # Set the model initial conditions necessary for fluid cavity
        mdb.models['Model-1'].setValues(absoluteZero=0, universalGas=8.314)

        # Create fluid cavity properties
        mdb.models['Model-1'].FluidCavityProperty(name='Fluid Cavity Property',
                                                  definition=PNEUMATIC, molecularWeight=0.02)

        # Create the reference point for the fluid cavity
        a = mdb.models['Model-1'].rootAssembly
        fluid_cavity_rp_id = a.ReferencePoint(point=(0.0, 0.0, 0.0)).id

        # Create the fluid cavity reference point set
        a = mdb.models['Model-1'].rootAssembly
        r1 = a.referencePoints
        fluid_cavity_rp_for_set = (r1[fluid_cavity_rp_id],)
        a.Set(referencePoints=fluid_cavity_rp_for_set, name='Fluid Cavity RP')

        # Create the fluid cavity
        a = mdb.models['Model-1'].rootAssembly
        region1 = a.sets['Fluid Cavity RP']
        a = mdb.models['Model-1'].rootAssembly
        region2 = a.surfaces['Pressure Surface']
        mdb.models['Model-1'].FluidCavity(name='Fluid Cavity',
                                          createStepName='Initial', cavityPoint=region1, cavitySurface=region2,
                                          interactionProperty='Fluid Cavity Property')

        # Define the pressure
        mdb.models['Model-1'].FluidCavityPressureBC(name='FluidCavityPressure',
                                                    createStepName='Pressure-Step', magnitude=self.pressure_load, fixed=OFF,
                                                    fluidCavity='Fluid Cavity', amplitude=UNSET)

        # Set initial temperature for the fluid cavity
        a = mdb.models['Model-1'].rootAssembly
        region = a.sets['Fluid Cavity RP']
        mdb.models['Model-1'].Temperature(name='Predefined Field-1',
                                          createStepName='Initial', region=region, distributionType=UNIFORM,
                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0,))

       # Request volume and pressure outputs
        regionDef = mdb.models['Model-1'].rootAssembly.sets['Fluid Cavity RP']
        mdb.models['Model-1'].HistoryOutputRequest(name='H-Output-2',
                                                   createStepName='Pressure-Step', variables=('PCAV', 'CVOL'),
                                                   region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)

    def apply_BCs(self):
        actuator_instance_faces = self.actuator_instance.faces
        w_extrude_mold = self.w_extrude + 2 * (
                self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(self.alpha))

        A = self.t / 10
        xmin = -A
        xmax = A
        ymin = -A
        ymax = (ymin + self.b_core + self.r_core + self.h_core + self.h_upper + 2 * A)
        zmin = -A
        zmax = (zmin + w_extrude_mold + 2 * A)

        print(w_extrude_mold)

        print("Fixed Faces")
        print(xmin, xmax)
        print(ymin, ymax)
        print(zmin, zmax)

        # found_faces = actuator_instance_faces.getByBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)
        found_faces = actuator_instance_faces.getByBoundingBox(-0.0001, -10000, -10000, 0.0001, 10000, 10000)
        print(found_faces)

        fixed_set = self.assembly.Set(faces=found_faces, name='Fixed Faces')
        self.model.EncastreBC(name='Fixed', createStepName='Pressure-Step',
                                         region=fixed_set, localCsys=None)

    # Create contact properties
    def contact_interactions(self):
        self.model.ContactProperty('IntProp-1')
        self.model.interactionProperties['IntProp-1'].TangentialBehavior(
            formulation=FRICTIONLESS)
        all_faces = self.assembly.instances['Actuator-1'].faces
        region = self.assembly.Surface(side1Faces=all_faces, name='All_Faces')
        self.model.SelfContactStd(name='Int-1',
                                             createStepName='Pressure-Step', surface=region,
                                             interactionProperty='IntProp-1', thickness=ON)

    # Mesh the part
    def generate_mesh(self):
        import mesh

        # Specify element type
        all_cells = self.assembly.instances['Actuator-1'].cells
        self.assembly.setMeshControls(regions=all_cells, elemShape=TET, technique=FREE)
        elemType1 = mesh.ElemType(elemCode=C3D20R, elemLibrary=STANDARD)
        elemType2 = mesh.ElemType(elemCode=C3D15, elemLibrary=STANDARD)
        elemType3 = mesh.ElemType(elemCode=C3D10H, elemLibrary=STANDARD)

        cell_set = self.assembly.Set(cells=all_cells, name='All Cells')

        self.assembly.setElementType(regions=cell_set, elemTypes=(elemType1, elemType2,
                                                           elemType3))

        # Seeding
        partInstances = (self.assembly.instances['Actuator-1'],)
        self.assembly.seedPartInstance(regions=partInstances, size=self.mesh, deviationFactor=0.1,
                           minSizeFactor=0.1)
        self.assembly.generateMesh(regions=partInstances)

        return

    def generate_inspection_set(self):

        # Create tracking nodes
        all_nodes = self.actuator_instance.nodes
        step = self.w_core + 2 * self.wa_core + 2 * (self.h_core - self.r_core) * math.tan(self.theta_core)
        total_length = step * (self.n - 1) + 2 * self.h_core * math.tan(self.theta_core) + self.w_core
        buffer = 0.01
        xmin = -buffer
        ymin = -buffer + self.round_skin + self.round_skin*math.tan(self.alpha)
        zmin = self.w_extrude + 2 * (
                    self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(
                self.alpha)) - buffer - (self.round_skin + self.round_skin*math.tan(self.alpha))*math.tan(self.alpha)
        xmax = total_length + 2*self.t + buffer
        ymax = buffer + self.round_skin + self.round_skin*math.tan(self.alpha)
        zmax = self.w_extrude + 2 * (
                    self.t * math.cos(self.alpha) + (self.t + self.t * math.sin(self.alpha)) * math.tan(
                self.alpha)) + buffer - (self.round_skin + self.round_skin*math.tan(self.alpha))*math.tan(self.alpha)
        tracking_nodes = all_nodes.getByBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)
        self.assembly.Set(nodes=tracking_nodes, name='Tracking')

        # Create nodes for stress analysis
        all_elements = self.actuator_instance.elements
        step = self.w_core + 2 * self.wa_core + 2 * (self.h_core - self.r_core) * math.tan(self.theta_core)
        total_length = step * (self.n - 1) + 2 * self.h_core * math.tan(self.theta_core) + self.w_core
        buffer = 0.001

        def shift(point, x_shift, y_shift):
            return(point[0] + x_shift, point[1] + y_shift)
        PS1 = (0, 0)
        PS2 = shift(PS1, 0, self.t + self.b_core)
        PS3 = shift(PS2, 0, self.h_core + self.h_upper)
        PS4 = shift(PS1, self.t - self.t * math.cos(self.theta_core),
                    self.t + self.b_core + self.t * math.sin(self.theta_core))
        PS5 = shift(PS4, self.r_core * math.tan(self.theta_core), self.r_core)
        PS6 = shift(PS5, (self.h_upper + self.h_core - self.r_core - self.t * math.sin(self.theta_core)) * math.tan(
            self.theta_core), (self.h_upper + self.h_core - self.r_core - self.t * math.sin(self.theta_core)))
        PS7 = (self.t + self.h_core * math.tan(self.theta_core) + self.w_core + (self.h_core - self.r_core) * math.tan(
            self.theta_core) + self.t * math.cos(
            self.theta_core) - (
                       self.h_upper + (self.h_core - self.r_core) - self.t * math.sin(self.theta_core)) * math.tan(
            self.theta_core),
               self.t + self.b_core + self.h_core + self.h_upper)
        PS8 = (self.t + self.h_core * math.tan(self.theta_core) + self.w_core + (self.h_core - self.r_core) * math.tan(
            self.theta_core) + self.t * math.cos(
            self.theta_core), self.t + self.b_core + self.r_core + self.t * math.sin(self.theta_core))

        buffer = 0.01
        xmin = PS8[0] - buffer
        ymin = self.t + self.b_core - buffer
        zmin = 0.0
        print("Step")
        print(step)
        print(PS5)
        xmax = (shift(PS5, step, 0))[0] + buffer
        ymax = self.t + self.b_core + self.r_core + self.t
        zmax = self.w_extrude + 3*self.t
        stress_nodes = all_elements.getByBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)
        self.assembly.Set(nodes=stress_nodes, name='StressNodes')

        return

    def fluid_cavity(self):

        # Create reference point for fluid cavity
        step = self.w_core + 2 * self.wa_core + 2 * (self.h_core - self.r_core) * math.tan(self.theta_core)
        total_length = step * (self.n - 1) + 2 * self.h_core * math.tan(self.theta_core) + self.w_core
        ref_point = self.assembly.ReferencePoint(point=(total_length/2.0, (self.t + self.b_core)/2.0, self.w_extrude/2.0))
        assembly = mdb.models['Model-1'].rootAssembly
        assembly.Set(name='RP', referencePoints=(assembly.referencePoints[assembly.referencePoints.keys()[-1]], ))
        self.assembly = assembly

        # # Generate the actual cavity
        # model = mdb.models['Model-1']
        # model.FluidCavityProperty(name='FluidCavityProperty', definition=PNEUMATIC, molecularWeight=0.029)
        # model.FluidCavity(name='FluidCavity', createStepName='Initial', cavityPoint=assembly.sets['RP'],
        #                   cavitySurface=assembly.surfaces['Pressure Surface'],
        #                   interactionProperty='FluidCavityProperty')
        #
        # # Request volume output
        # model.HistoryOutputRequest(name='VolumeOutput', createStepName='Pressure-Step', variables=('PCAV', 'CVOL'),
        #                            region=assembly.sets['RP'], sectionPoints=DEFAULT, rebar=EXCLUDE)
        #
        # # Set default values needed
        # model.setValues(absoluteZero=0, universalGas=8.314)


    def write_job(self, model_name):
        self.current_job = mdb.Job(name=model_name, model='Model-1', description='', type=ANALYSIS,
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1,
                numGPUs=0)
        mdb.jobs[model_name].writeInput(consistencyChecking=OFF)
        return self.current_job

    def get_mass(self):
        assembly1 = mdb.models['Model-1'].rootAssembly
        weight = assembly1.getMassProperties()
        self.mass = weight['mass'] * 1000.0
        mdb.saveAs('testing6.cae')
        return

