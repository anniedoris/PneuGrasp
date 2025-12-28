from CMA import *
import subprocess
import math
from ProcessedModels import *
from ExperimentManager import *
import os
import nevergrad

#### EDIT HERE ####

experiment_report_file_name = 'CMA-thickness-test.txt'
instrum = ng.p.Instrumentation(
            ng.p.Array(shape=(2,), lower=(1.6, 2.9999999), upper=(3.0, 3.0))
        )
budget = 1
num_workers = 1
#####

optimizer = ng.optimizers.registry['CMA'](parametrization=instrum, budget=budget, num_workers=num_workers)
ExperimentManager = ExperimentManager(optimizer, experiment_report_file_name)

# Initialize the files that haven't been initialized yet
ExperimentManager.initialize_key_file()
ExperimentManager.initialize_lumped_results()
ExperimentManager.optimize()



# # Define the parameters and the parameter bounds
# instrum = ng.p.Instrumentation(
#             ng.p.Array(shape=(2,), lower=(1.6, 2.9999999), upper=(3.0, 3.0))
#         )
#
# # Define the number of workers and the budget
# CMA = CMA(budget=1, num_workers=1, instrum=instrum)
#
#
# with open(experiment_name, 'w') as overview:
#
#     for i in range(CMA.optimizer.budget):
#
#         x = CMA.get_trial_parameters()
#         trial_params = x.value[0][0]
#         print("Trial Parameters")
#         print(trial_params)
#
#         overview.write('Experiment: ' + str(i))
#
#         overview.write("\n")
#
#         overview.write(str(trial_params[0]))
#         overview.write("\n")
#         overview.write(str(trial_params[1]))
#         overview.write("\n")
#
#         fname = 'actuator_optimization/actuator_defintion.py'
#         parameters = [{'parameter': 't', 'value': trial_params[0]},
#                       {'parameter': 'r_core', 'value': 3},
#                       {'parameter': 'b_core', 'value': 3.63},
#                       {'parameter': 'h_core', 'value': 12.37},
#                       {'parameter': 'h_upper', 'value': 1.70},
#                       {'parameter': 'w_core', 'value': 1.658},
#                       {'parameter': 'theta_core', 'value': math.radians(7.125)},
#                       {'parameter': 'n', 'value': 1},
#                       {'parameter': 'wa_core', 'value': 5},
#                       {'parameter': 'w_extrude', 'value': 15.63},
#                       {'parameter': 'h_round', 'value': 13.65},
#                       {'parameter': 'ho_round', 'value': 13.65 + 1.70 + 1.70 * math.sin(math.radians(5))},
#                       {'parameter': 'alpha', 'value': math.radians(5)},
#                       {'parameter': 'round_core', 'value': 0.5},
#                       {'parameter': 'round_skin', 'value': 1.0},
#                       {'parameter': 'mesh', 'value': 1.6}]
#
#         with open(fname, 'w') as f:
#             f.write('import math')
#             f.write('\n')
#             f.write('geometry_dict = {')
#             f.write('\n')
#             for i, param in enumerate(parameters):
#                 if i != len(parameters) - 1:
#                     f.write('\t\'' + str(param['parameter']) + '\': ' + str(param['value']) + ',')
#                     f.write('\n')
#                 else:
#                     f.write('\t\'' + str(param['parameter']) + '\': ' + str(param['value']) + '}')
#             f.write('\n')
#             f.write('material_dict = {\'name\': \'Smooth Sil 950\', \'model\': \'Neo-Hookean\', \'P1\': 0.34, \'P2\': 0, \'density\': 1.24*(10**-9)}')
#             f.write('\n')
#             f.write('actuator_definition = {\'geometry\': geometry_dict, \'material_properties\': material_dict, \'pressure\': 0.1}')
#
#         print("Terminal")
#         subprocess.run(['abaqus', 'cae', '-noGUI', 'C:/Users/and008/Documents/Models V2/actuator_optimization/main_abaqus.py'], shell=True)
#
#         with open('model_name.txt', 'r') as fr:
#             lines = fr.readlines()
#             for line in lines:
#                 model_name = line
#
#         print("Model Name")
#         print(model_name)
#         overview.write(str(model_name))
#         overview.write("\n")
#
#         df_model = ProcessedModel(model_name)
#         df_model.load_data()
#         score = df_model.data['Max Strain'].iloc[-1]
#         print("Score")
#         print(score)
#
#         overview.write(str(score))
#         overview.write("\n")
#
#         CMA.get_loss(x, score)
#
#     CMA.report_optimal()
#     print(CMA.recommendation)