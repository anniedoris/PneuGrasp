from odbAccess import *
import os
import ast
import numpy as np


start_model = 379
end_model = 384
model_list = np.arange(start_model, end_model + 1, 1)
final_pressure = 100.0
model_list = ['M' + str(i) for i in model_list]

for model_name in model_list:
    try:
        odb_name = model_name + '.odb'
        odb = openOdb(path=odb_name)
        file = True
    except:
        file = False
    if file:

        # for frame in odb.steps['Pressure-Step'].frames:
        #     print(frame)
        #     with open(model_name + '_' + str(frame.frameId) + '.txt', 'w') as f:
        #         field = frame.fieldOutputs['LE'].getSubset(position=ELEMENT_NODAL)
        #         print(field.values)
                # field = field.getScalarField(invariant=MAX_PRINCIPAL)
                # max_strain = max([(g.data, g.elementLabel, g.integrationPoint) for g in field.values])[0]
                # print(max_strain)
                #         f.write(str(max_strain))
                #         f.write('\n')
                # # max_strain = max([(g.data, g.elementLabel, g.integrationPoint) for g in field.values])[0]

                # # f.write(str(max_strain))
                # # f.write('\n')



        with open(model_name + '.txt', 'w') as f:

            # # Extract pressure values to write to the file
            f.write("DATA\n")
            f.write('Pressure')
            f.write('\n')
            pressure_data = odb.steps['Pressure-Step'].historyRegions['Node ASSEMBLY.1'].historyOutputs['PCAV'].data
            for inc in pressure_data:
                f.write(str(inc[1]))
                f.write('\n')

            # Extract volume values to write to the file
            f.write("DATA\n")
            f.write('Volume')
            f.write('\n')
            pressure_data = odb.steps['Pressure-Step'].historyRegions['Node ASSEMBLY.1'].historyOutputs['CVOL'].data
            for inc in pressure_data:
                f.write(str(inc[1]))
                f.write('\n')

            # # Write pressure values to the file
            # f.write("DATA\n")
            # f.write('Pressure')
            # f.write('\n')
            # for frame in odb.steps['Pressure-Step'].frames:
            #     pressure_step = float(frame.description.split('=')[1])
            #     f.write(str(pressure_step * final_pressure))
            #     f.write('\n')
            #

            # Write mean strain values to the file
            f.write("DATA\n")
            f.write('Average Strain')
            f.write('\n')
            for frame in odb.steps['Pressure-Step'].frames:
                field = frame.fieldOutputs['LE'].getScalarField(invariant=MAX_PRINCIPAL)
                avg_result = np.mean([g.data for g in field.values])
                f.write(str(avg_result))
                f.write('\n')

            # Write max strain values to the file
            f.write("DATA\n")
            f.write('Max Strain')
            f.write('\n')
            for frame in odb.steps['Pressure-Step'].frames:
                field = frame.fieldOutputs['LE'].getScalarField(invariant=MAX_PRINCIPAL)
                max_strain = max([(g.data, g.elementLabel, g.integrationPoint) for g in field.values])[0]
                f.write(str(max_strain))
                f.write('\n')

            # Write coordinates of the side to the file
            # Isolate the set of interest
            set_of_interest = 'TRACKING'
            desired_set = odb.rootAssembly.nodeSets[set_of_interest]

            # Get the initial coordinates of the desired set
            initial_coords = []

            for node in odb.rootAssembly.nodeSets[set_of_interest].nodes[0]:
                initial_coords.append({'label': node.label, 'x_coord': node.coordinates[0], 'y_coord': node.coordinates[1],
                                       'z_coord': node.coordinates[2]})
            for initial_node in initial_coords:
                f.write("DATA\n")
                f.write("Node " + str(initial_node['label']))
                f.write("\n")
                for frame in odb.steps['Pressure-Step'].frames:
                    displacements = frame.fieldOutputs['U'].getSubset(region=desired_set)
                    for node in displacements.values:
                        if initial_node['label'] == node.nodeLabel:
                            f.write(str(node.data[0] + initial_node['x_coord']))
                            f.write(',')
                            f.write(str(node.data[1] + initial_node['y_coord']))
                            f.write('\n')















            # # Extract the volume of the cavity
            # for frame in odb.steps['Pressure-Step'].frames:
            #     print(frame)
                # set_of_interest = 'FLUID CAVITY RP'
                # desired_set = odb.rootAssembly.nodeSets[set_of_interest]

                # print(frame)
                # print(desired_set)

            # Get the mass of the actuator
            # assembly = odb.rootAssembly
            # print(assembly)
            # weight = assembly.getMassProperties()
            # print(weight)

            # print(type(field))
            # maxp = max([(g.data, g.elementLabel, g.integrationPoint) for g in field.values])
            # avg_result = np.mean([g.data for g in field.values])
            # print(maxp)
            # print(avg_result)
            # f.write("Average Strain")
            # f.write(str("\n"))
            # f.write(str(avg_result))
            # f.write(str("\n"))
            # f.write("Max Strain")
            # f.write(str("\n"))
            # f.write(str(maxp[0]))
            # f.write(',')
            # f.write(str(maxp[1]))
            # f.write(str("\n"))
            # f.write("Element Max Strain")
            # f.write(str("\n"))

        # lastFrame = odb.steps['Pressure-Step'].frames[-1]
        # stress = lastFrame.fieldOutputs['S']
        # max_von_mises = -10
        # element = None
        # integration_point = None
        # for node in stress.values:
        #     print(node.elementLabel)
        #     print(node.mises)
        #     if node.mises > max_von_mises:
        #         max_von_mises = node.mises
        #         element = node.elementLabel
        #         node = node.integrationPoint
        # print("Max Von Mises")
        # print(max_von_mises)
        # print(element)
        # print(node)

        # # Isolate the set of interest
        # desired_set = odb.rootAssembly.nodeSets[set_of_interest]
        #
        # # Get the initial coordinates of the desired set
        # initial_coords = []
        #
        # for node in odb.rootAssembly.nodeSets[set_of_interest].nodes[0]:
        #     initial_coords.append({'label': node.label, 'x_coord': node.coordinates[0], 'y_coord': node.coordinates[1],
        #                            'z_coord': node.coordinates[2]})
        #
        # for frame in odb.steps['Pressure-Step'].frames:
        #     pressure_step = float(frame.description.split('=')[1])
        #     f.write('Pressure')
        #     f.write('\n')
        #     f.write(str(pressure_step))
        #     f.write('\n')
        #     displacements = frame.fieldOutputs['U'].getSubset(region=desired_set)
        #     for node in displacements.values:
        #         for initial_node in initial_coords:
        #             if initial_node['label'] == node.nodeLabel:
        #                 f.write(str(node.data[0] + initial_node['x_coord']))
        #                 f.write(',')
        #                 f.write(str(node.data[1] + initial_node['y_coord']))
        #                 f.write(',')
        #                 f.write(str(node.nodeLabel))
        #                 f.write('\n')


     # for initial_node in initial_coords:
    #
    #                     if initial_node['label'] == node.nodeLabel:
    #                         f.write(str((final_pressure/num_inc)*frame_num))
    #                         f.write(',')
    #                         f.write(str(node.data[0] + initial_node['x_coord']))
    #                         f.write(',')
    #                         f.write(str(node.data[1] + initial_node['y_coord']))
    #                         f.write(',')
    #                         f.write(str(node.data[2] + initial_node['z_coord']))
    #                         f.write('\n')

    # # Get the initial coordinates of the set of interest
    # initial_coords = []
    #
    # for node in odb.rootAssembly.nodeSets[set_of_interest].nodes[0]:
    #     initial_coords.append({'label': node.label, 'x_coord': node.coordinates[0], 'y_coord': node.coordinates[1],
    #                            'z_coord': node.coordinates[2]})
    #
    # frame_num = 0
    #
    # desired_set = odb.rootAssembly.nodeSets[set_of_interest]
    #
    # with open('TipSweep' + file_name + '.txt', 'w') as f:
    #     f.write('Pressure, X Position, Y Position, Z Position')
    #     f.write(str('\n'))
    #     for frame in odb.steps['Pressure-Step'].frames:
    #         displacements = frame.fieldOutputs['U'].getSubset(region=desired_set)
    #
    #         for node in displacements.values:
    #
    #             if node.nodeLabel==tip_node_label:
    #
    #                 for initial_node in initial_coords:
    #
    #                     if initial_node['label'] == node.nodeLabel:
    #                         f.write(str((final_pressure/num_inc)*frame_num))
    #                         f.write(',')
    #                         f.write(str(node.data[0] + initial_node['x_coord']))
    #                         f.write(',')
    #                         f.write(str(node.data[1] + initial_node['y_coord']))
    #                         f.write(',')
    #                         f.write(str(node.data[2] + initial_node['z_coord']))
    #                         f.write('\n')
    #
    #         frame_num = frame_num + 1

    # class Extractor:
    #     def __init__(self, Actuator):
    #`
#         self.Actuator = Actuator
#         self.actuator_dict_list = []
#         self.key_numbers = []
#
#         # Initialize a list file that contains name parameter pairs if it doesn't already exist
#         print("Current Working Directory: ", os.getcwd())
#         if "key_file.txt" in os.listdir(os.getcwd()):
#             pass
#         else:
#             with open("key_file.txt", 'w') as f_w:
#                 f_w.close()
#
#     # Check to see if a key already exists for the model, otherwise assign it a key value
#     def find_key(self):
#
#         # Load all previously run actuators
#         with open("key_file.txt", 'r') as f_r:
#             lines = f_r.readlines()
#             for line in lines:
#                 curr_dict = ast.literal_eval(line)
#                 self.actuator_dict_list.append(curr_dict)
#
#         # Find all new actuators
#         print("Run Actuators")
#         print(self.actuator_dict_list)
#
#         # Check to see if the actuator has already been run
#         if len(self.actuator_dict_list) != 0:
#             print("Previously run actuators")
#             for run_actuator in self.actuator_dict_list:
#
#                 # Check for equivalence of actuator dictionaries
#                 def dict_equivalence(d1, d2):
#                     if (d1['pressure'] == d2['pressure']) and (d1['material_properties'] == d2['material_properties']):
#                         for key in d1['geometry']:
#                             if round(d1['geometry'][key], 10) != round(d2['geometry'][key], 10):
#                                 return False
#                         return True
#                     else:
#                         return False
#
#                 equivalence_result = dict_equivalence(run_actuator['actuator'], self.Actuator.definition)
#                 if equivalence_result:
#                     return (True, run_actuator['key'])
#
#         # Otherwise generate a new key and write actuator to the file
#         if len(self.actuator_dict_list) == 0:
#             next_key_num = 1
#         else:
#             next_key_num = self.actuator_dict_list[-1]['key'] + 1
#
#         with open("key_file.txt", 'a') as f_a:
#             keyed_dict = {'actuator': self.Actuator.definition, 'key': next_key_num}
#             f_a.write(str(keyed_dict))
#
#         return (False, next_key_num)

        # with open("key_file.txt", 'w') as f_w:
        #     num = 1
        #     keyed_dict = {'key': num, 'actuator': self.Actuator.definition}
        #     f_w.write(str(keyed_dict))


  # with open("key_file.txt", 'r') as f_r:
        #     even_odd_counter = 1
        #     for line in f_r:
        #         if even_odd_counter % 2 == 0:
        #             print("line")
        #             print(line)
        #             dict_one = json.loads(line)
        #             print(dict_one)
        #             self.actuator_dict_list.append(json.loads(line))
        #         else:
        #             self.key_numbers.append(int(line))
        #         even_odd_counter += 1
        #
        # print("Actuator Dict List")
        # print(self.actuator_dict_list)
        #
        # print("Key Numbers")
        # print(self.key_numbers)

# file_name = "1,70T.odb"
# set_of_interest = 'TRACKING'
# tip_node_label = 358
# final_pressure = 100.0
# num_inc = 100.0
#
# odb = openOdb(path=file_name)
#
# # Get the initial coordinates of the set of interest
# initial_coords = []
#
# for node in odb.rootAssembly.nodeSets[set_of_interest].nodes[0]:
#     initial_coords.append({'label': node.label, 'x_coord': node.coordinates[0], 'y_coord': node.coordinates[1],
#                            'z_coord': node.coordinates[2]})
#
# frame_num = 0
#
# desired_set = odb.rootAssembly.nodeSets[set_of_interest]
#
# with open('TipSweep' + file_name + '.txt', 'w') as f:
#     f.write('Pressure, X Position, Y Position, Z Position')
#     f.write(str('\n'))
#     for frame in odb.steps['Pressure-Step'].frames:
#         displacements = frame.fieldOutputs['U'].getSubset(region=desired_set)
#
#         for node in displacements.values:
#
#             if node.nodeLabel==tip_node_label:
#
#                 for initial_node in initial_coords:
#
#                     if initial_node['label'] == node.nodeLabel:
#                         f.write(str((final_pressure/num_inc)*frame_num))
#                         f.write(',')
#                         f.write(str(node.data[0] + initial_node['x_coord']))
#                         f.write(',')
#                         f.write(str(node.data[1] + initial_node['y_coord']))
#                         f.write(',')
#                         f.write(str(node.data[2] + initial_node['z_coord']))
#                         f.write('\n')
#
#         frame_num = frame_num + 1







        #
        # # First and last frames
        # initialFrame = odb.steps['Pressure-Step'].frames[0]
        # finalFrame = odb.steps['Pressure-Step'].frames[-1]
        #
        # # Set we are interested in
        # desired_set = odb.rootAssembly.nodeSets[set_of_interest]
        #
        # # Displacemnts
        # initial_displacement = initialFrame.fieldOutputs['U'].getSubset(region=desired_set)
        # final_displacement = finalFrame.fieldOutputs['U'].getSubset(region=desired_set)
        #
        # with open('First' + file_name + '.txt', 'w') as f:
        #     f.write('Node Label, X Position, Y Position, Z Position')
        #     f.write(str('\n'))
        #
        #     for node in initial_displacement.values:
        #
        #         for initial_node in initial_coords:
        #
        #             if initial_node['label'] == node.nodeLabel:
        #
        #                 f.write(str(node.nodeLabel))
        #                 f.write(',')
        #                 f.write(str(node.data[0] + initial_node['x_coord']))
        #                 f.write(',')
        #                 f.write(str(node.data[1] + initial_node['y_coord']))
        #                 f.write(',')
        #                 f.write(str(node.data[2] + initial_node['z_coord']))
        #                 f.write('\n')
        #
        # with open('Final' + file_name + '.txt', 'w') as f:
        #     f.write('Node Label, X Position, Y Position, Z Position')
        #     f.write(str('\n'))
        #
        #     for node in final_displacement.values:
        #
        #         for initial_node in initial_coords:
        #
        #             if initial_node['label'] == node.nodeLabel:
        #                 f.write(str(node.nodeLabel))
        #                 f.write(',')
        #                 f.write(str(node.data[0] + initial_node['x_coord']))
        #                 f.write(',')
        #                 f.write(str(node.data[1] + initial_node['y_coord']))
        #                 f.write(',')
        #                 f.write(str(node.data[2] + initial_node['z_coord']))
        #                 f.write('\n')
        #

        #
        # print(node.label)
        # print(node.coordinates[0])
        # print(node.coordinates[1])
        # print(node.coordinates[2])
        # f.write(str(node.label))
        # f.write(',')
        # f.write(str(node.coordinates[0]))
        # f.write(',')
        # f.write(str(node.coordinates[1]))
        # f.write(',')
        # f.write(str(node.coordinates[2]))
        # f.write(str('\n'))



# firstFrame = odb.steps['GravStep'].frames[0]
#
# # print(odb.steps)
# # print(odb.steps['GravStep'])
# # print(odb.steps['GravStep'].frames)
#
# # Create a variable that refers to the displacement 'U'
# # in the last frame of the first step.
#
# displacement_i = firstFrame.fieldOutputs['U']
#
# # Create a variable that refers to the node set 'PUNCH'
# # located at the center of the hemispherical punch.
# # The set is  associated with the part instance 'PART-1-1'.
#
# # print(odb.rootAssembly.nodeSets['BOTTOM'])
# # print(odb.rootAssembly.instances['ACTUATOR-1'].nodeSets)
#
# bottom_center_line = odb.rootAssembly.nodeSets['BOTTOMCENTERLINE']
#
# # Create a variable that refers to the displacement of the node
# # set in the last frame of the first step.
#
# bottom_center_line_displacement_i = displacement_i.getSubset(region=bottom_center_line)
#
# # Finally, print some field output data from each node
# # in the node set (a single node in this example).
#
# with open('InitialPositions' + file_name + '.txt', 'w') as f:
#     f.write('Node Label, X Position, Y Position, Z Position')
#     f.write(str('\n'))
#     for v in bottom_center_line_displacement_i.values:
#         f.write(str(v.nodeLabel))
#         f.write(',')
#         f.write(str(v.data[0]))
#         f.write(',')
#         f.write(str(v.data[1]))
#         f.write(',')
#         f.write(str(v.data[2]))
#         f.write(str('\n'))
#         # print 'Position = ', v.position,'Type = ',v.type
#         # print 'Node label = ', v.nodeLabel
#         # print 'X displacement = ', v.data[0]
#         # print 'Y displacement = ', v.data[1]
#         # print 'Displacement magnitude =', v.magnitude
