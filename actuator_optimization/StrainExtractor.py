from odbAccess import *
import os
import ast
import numpy as np

def update_model_name():
    return

def extract_results():
    return

model_list = np.arange(1, 15, 1)
model_list = ['M' + str(i) for i in model_list]


for file_name in model_list:
    with open(file_name + 'strains' + '.txt', 'w') as f:
        file_name = file_name + '.odb'
        file = True
        try:
            odb = openOdb(path=file_name)
            print("Opening file")
            print(file_name)
        except:
            file = False

        if file:

            lastFrame = odb.steps['Pressure-Step'].frames[-1]
            field = lastFrame.fieldOutputs['LE'].getScalarField(invariant = MAX_PRINCIPAL)
            print(type(field))
            maxp = max([(g.data, g.elementLabel, g.integrationPoint) for g in field.values])
            avg_result = np.mean([g.data for g in field.values])
            print(maxp)
            print(avg_result)
            f.write("Average Strain")
            f.write(str("\n"))
            f.write(str(avg_result))
            f.write(str("\n"))
            f.write("Max Strain")
            f.write(str("\n"))
            f.write(str(maxp[0]))
            f.write(',')
            f.write(str(maxp[1]))
            f.write(str("\n"))
            f.write("Element Max Strain")
            f.write(str("\n"))


