from odbAccess import *
import os
import ast
import numpy as np

class Extractor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.done = False

    def get_data(self):
        model_list = [self.model_name]
        for model_name in model_list:
            try:
                odb_name = model_name + '.odb'
                odb = openOdb(path=odb_name)
                file = True
            except:
                file = False
            if file:

                with open(model_name + '.txt', 'w') as f:

                    # # Extract pressure values to write to the file
                    f.write("DATA\n")
                    f.write('Pressure')
                    f.write('\n')
                    pressure_data = odb.steps['Pressure-Step'].historyRegions['Node ASSEMBLY.1'].historyOutputs[
                        'PCAV'].data
                    for inc in pressure_data:
                        f.write(str(inc[1]))
                        f.write('\n')

                    # Extract volume values to write to the file
                    f.write("DATA\n")
                    f.write('Volume')
                    f.write('\n')
                    pressure_data = odb.steps['Pressure-Step'].historyRegions['Node ASSEMBLY.1'].historyOutputs[
                        'CVOL'].data
                    for inc in pressure_data:
                        f.write(str(inc[1]))
                        f.write('\n')

                    # # Write mean strain values to the file
                    # f.write("DATA\n")
                    # f.write('Average Strain')
                    # f.write('\n')
                    # for frame in odb.steps['Pressure-Step'].frames:
                    #     field = frame.fieldOutputs['LE'].getScalarField(invariant=MAX_PRINCIPAL)
                    #     avg_result = np.mean([g.data for g in field.values])
                    #     f.write(str(avg_result))
                    #     f.write('\n')

                    # Write max strain values to the file
                    f.write("DATA\n")
                    f.write('Max Strain')
                    f.write('\n')
                    for frame in odb.steps['Pressure-Step'].frames:
                        field = frame.fieldOutputs['LE'].getScalarField(invariant=MAX_PRINCIPAL)
                        max_strain = max([(g.data, g.elementLabel, g.integrationPoint) for g in field.values])[0]
                        f.write(str(max_strain))
                        f.write('\n')

                    # Write the reaction force values to the file
                    all_keys = odb.steps['Pressure-Step'].historyRegions.keys()  # Pressure-Step
                    print(all_keys)
                    for i in all_keys:
                        if 'OBJECT' in i:
                            desired_key = i
                    print(desired_key)
                    # print(odb.steps['Pressure-Step'].historyRegions)
                    # print(odb.steps['Pressure-Step'].historyRegions)
                    reaction1 = odb.steps['Pressure-Step'].historyRegions[desired_key].historyOutputs['RF1'].data
                    reaction2 = odb.steps['Pressure-Step'].historyRegions[desired_key].historyOutputs['RF2'].data

                    def extract_force(tuples):
                        just_force = []
                        for tuple in tuples:
                            just_force.append(tuple[1])
                        return just_force

                    R1 = extract_force(reaction1)
                    R2 = extract_force(reaction2)
                    Rtot = []

                    for i, f1 in enumerate(R1):
                        f2 = R2[i]
                        Rtot.append(math.sqrt(f1 ** 2 + f2 ** 2))
                    print(R1)
                    print(R2)
                    print(Rtot)

                    f.write("DATA\n")
                    f.write('R1\n')
                    for f1 in R1:
                        f.write(str(f1))
                        f.write('\n')

                    f.write("DATA\n")
                    f.write('R2\n')
                    for f2 in R2:
                        f.write(str(f2))
                        f.write('\n')

                    f.write("DATA\n")
                    f.write('RTOT\n')
                    for ftot in Rtot:
                        f.write(str(ftot))
                        f.write('\n')

                    # Write coordinates of the side to the file
                    # Isolate the set of interest
                    set_of_interest = 'TRACKING'
                    desired_set = odb.rootAssembly.nodeSets[set_of_interest]

                    # Get the initial coordinates of the desired set
                    initial_coords = []

                    for node in odb.rootAssembly.nodeSets[set_of_interest].nodes[0]:
                        initial_coords.append(
                            {'label': node.label, 'x_coord': node.coordinates[0], 'y_coord': node.coordinates[1],
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
        print("Finished Extracting Data")