import os
import sys
import time
# from subprocess import check_output
# from Actuator import *
# from ExperimentManager import *
# from actuator_defintion import *
# from Sweep import *
# from GridSearch import *
# from Extractor import *
import math
from abaqus import *
from abaqusConstants import *
import sys

# Force updating of abaqus files, if we import normally, abaqus doesn't remember
execfile('Actuator.py')
execfile('Extractor.py')
execfile('actuator_defintion.py')

Actuator = Actuator(actuator_definition)
Actuator.load_params(actuator_definition)
Actuator.clear_model_space()
Actuator.generate_core()
Actuator.generate_skin()
Actuator.cut_core_skeleton()
Actuator.generate_material()
Actuator.partition_down_center()
Actuator.generate_steps()
Actuator.apply_pressure()
Actuator.apply_BCs()
Actuator.contact_interactions()
Actuator.generate_mesh()
Actuator.generate_inspection_set()
Actuator.grasp_object_setup()
Actuator.get_mass()
current_job = Actuator.write_job('model')
current_job.submit()
current_job.waitForCompletion()

Extractor = Extractor('model')
Extractor.get_data()


# ExperimentManager = ExperimentManager("sweep", Sweep, actuator_definition)
#
# # Generate a key file if one doesn't already exist
# ExperimentManager.initialize_key_file()
#
# # Generate a lumped results file if one doesn't already exist
# ExperimentManager.initialize_lumped_results()
#
# ExperimentManager.

# # Start going through the experiment manager
# while ExperimentManager.remaining_experiments:
#     ExperimentManager.get_params() # Load parameters for the experiment
#     ExperimentManager.check_run_status() # See if the model has been run before
#
#     # Write the model name to the list if it has been run before
#     if ExperimentManager.run_before:
#         ExperimentManager.experiment_models.append(ExperimentManager.model_name)
#         print >> sys.__stdout__, 'Model #' + str(ExperimentManager.model_name)
#         print >> sys.__stdout__, '\tModel run before:' + str(ExperimentManager.run_before)
#         print >> sys.__stdout__, '\tModel parameters:' + str(ExperimentManager.Actuator.definition['geometry'])
#
#     else:
#
#         try:
#             ExperimentManager.set_up_model()
#             if testing:
#                 print("Testing")
#                 print >> sys.__stdout__, 'Model parameters:' + str(
#                     ExperimentManager.Actuator.definition['geometry'])
#
#             else:
#                 ExperimentManager.write_to_keyfile()
#                 ExperimentManager.experiment_models.append(ExperimentManager.model_name)
#                 print >> sys.__stdout__, 'Model #' + str(ExperimentManager.model_name)
#                 print >> sys.__stdout__, '\tModel run before:' + str(ExperimentManager.run_before)
#                 print >> sys.__stdout__, '\tModel parameters:' + str(
#                     ExperimentManager.Actuator.definition['geometry'])
#                 print >> sys.__stdout__, '\tGeometry generated'
#
#         except Exception as e:
#             print >> sys.__stdout__, 'Issue'
#             print >> sys.__stdout__, e
#             # if e == 'name is empty':
#
#             ExperimentManager.geometry_generated = False
#             print >> sys.__stdout__, 'Geometry Generation Failed'
#             print >> sys.__stdout__, '\tModel parameters:' + str(ExperimentManager.Actuator.definition['geometry'])
#
#
#         # Only run the experiment if the geometry generates and we are not in testing mode
#         if ExperimentManager.geometry_generated:
#             if not testing:
#                 print >> sys.__stdout__, '\tComputing'
#                 start_time = time.time()
#                 ExperimentManager.run_model() # Run computation
#                 end_time = time.time()
#                 print >> sys.__stdout__, end_time
#                 total_time = end_time - start_time
#                 print >> sys.__stdout__, '\tComputation Time: ' + str(end_time)
#                 print >> sys.__stdout__, '\tModel Complete'
#
#                 # Store the run time for the model
#                 ExperimentManager.run_time = total_time
#
#                 # Write results to the lumped parameter file
#                 ExperimentManager.lumped_results_update()
#
#                 ExperimentManager.write_name()
#
#     # Extract results
#     # print >> sys.__stdout__, '\tStripped ' + ExperimentManager.model_name.strip('M')
#     # print >> sys.__stdout__, '\tExtracting Data...'
#     # Extractor = Extractor(int(ExperimentManager.model_name.strip('M')))
#     # print >> sys.__stdout__, '\tData Extracted'
#     # metric = Extractor.get_data()
#     # print >> sys.__stdout__, '\tScoring Metric: ' + str(metric)
#
#     # Move to the next experiment
#     ExperimentManager.experiment_number += 1
#
# # Report all the relevant models
# print >> sys.__stdout__, 'Models for the Experiment:'
# print >> sys.__stdout__, ExperimentManager.experiment_models


#
# # Pulling user-defined parameters of the actuators
# definition_dict = actuator_definition
#
# # Set up actuator and extractor
# Actuator = Actuator(definition_dict)
# Extractor = Extractor(Actuator)
#
# # Find the key associated with the extractor
# run, key = Extractor.find_key()
#
# # Set up the actuator model
# Actuator.clear_model_space()
# Actuator.generate_core()
# Actuator.generate_skin()
# Actuator.cut_core_skeleton()
# Actuator.generate_material()
# Actuator.partition_down_center()
# Actuator.generate_steps()
# Actuator.apply_pressure()
# Actuator.apply_BCs()
# Actuator.contact_interactions()
# Actuator.generate_mesh()
# Actuator.generate_inspection_set()