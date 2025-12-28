import os
import sys
import ast
import math
import shutil
import subprocess
from concurrent import futures
import nevergrad as ng

class ExperimentManager:
    def __init__(self, optimizer, experiment_report_file_name):
        self.optimizer = optimizer
        self.experiment_report_file = experiment_report_file_name

        # self.optimization_type = optimization_type
        # self.Optimizer = Optimizer
        # self.actuator_definition = actuator_definition
        # self.experiment_number = 1
        # self.remaining_experiments = True
        # self.Actuator = Actuator(actuator_definition)
        # self.model_name = None
        # self.actuator_dict_list = [] # Chronicles actuators that have already been run
        # self.current_job = None
        # self.mass = None
        # self.run_time = None
        # self.run_before = None
        # self.experiment_models = [] # Keeps track of all the model names that have been run in experiemnt
        # self.geometry_generated = True
        # self.next_key_num = None

    def initialize_key_file(self):
        if "key_file.txt" in os.listdir(os.getcwd()):
            pass
        else:
            with open("key_file.txt", 'w') as f_w:
                f_w.close()

    def initialize_lumped_results(self):
        if "lumped_results.txt" in os.listdir(os.getcwd()):
            pass
        else:
            with open("lumped_results.txt", 'w') as f_w:
                f_w.close()

    def optimize(self):
        with open(self.experiment_report_file, 'w') as experiment_summary:

            def loss_function(x):
                return sum((x - 0.5) ** 2)

            # filepath = 'log-file.txt'
            #
            # logger = ng.callbacks.ParametersLogger(filepath)
            # self.optimizer.register_callback("tell", logger)

            with futures.ProcessPoolExecutor(max_workers=self.optimizer.num_workers) as executor:
                recommendation = self.optimizer.minimize(loss_function, executor=executor, batch_mode = False)

            print(recommendation)

            # # Indicator if the geometry can be physically generated or not
            # not_physical = True
            #
            # while not_physical:
            #
            #     # Get the trial parameters
            #     x = self.optimizer.ask()
            #     trial_params = x.value[0][0]
            #
            #     # TODO - Check that the model is physically realizable
            #     not_physical = False
            #
            # # Get a unique name for the set of parameters to be run
            # # if geometry_generate:
            # actuator_dict_list = []
            # with open("key_file.txt", 'r') as f_r:
            #     lines = f_r.readlines()
            #     for line in lines:
            #         curr_dict = ast.literal_eval(line)
            #         actuator_dict_list.append(curr_dict)
            #     try:
            #         next_key_num = actuator_dict_list[-1]['key'] + 1
            #     except:
            #         next_key_num = 1
            #
            # # Write the model name to the experiment summary
            # model_name = 'C' + str(next_key_num)
            # print("Model name: " + str(model_name))
            # experiment_summary.write(model_name)
            # experiment_summary.write("\n")
            #
            # # Write the trial parameters
            # experiment_summary.write("\tTrial Param:" + str(trial_params))
            # print("\tTrial Parameters:" + str(trial_params))
            #
            # # Create a folder with that model name in the directory
            # os.mkdir(model_name)
            #
            # # Populate the folder with a file that has the relevant geometry in it
            # fname = model_name + '/actuator_defintion.py'
            # parameters = [{'parameter': 't', 'value': trial_params[0]},
            #               {'parameter': 'r_core', 'value': 3},
            #               {'parameter': 'b_core', 'value': 3.63},
            #               {'parameter': 'h_core', 'value': 12.37},
            #               {'parameter': 'h_upper', 'value': 1.70},
            #               {'parameter': 'w_core', 'value': 1.658},
            #               {'parameter': 'theta_core', 'value': math.radians(7.125)},
            #               {'parameter': 'n', 'value': 1},
            #               {'parameter': 'wa_core', 'value': 5},
            #               {'parameter': 'w_extrude', 'value': 15.63},
            #               {'parameter': 'h_round', 'value': 13.65},
            #               {'parameter': 'ho_round', 'value': 13.65 + 1.70 + 1.70 * math.sin(math.radians(5))},
            #               {'parameter': 'alpha', 'value': math.radians(5)},
            #               {'parameter': 'round_core', 'value': 0.5},
            #               {'parameter': 'round_skin', 'value': 1.0},
            #               {'parameter': 'mesh', 'value': 1.6}]
            #
            # # Write the geometry file to the folder
            # with open(fname, 'w') as f:
            #     f.write('import math')
            #     f.write('\n')
            #     f.write('geometry_dict = {')
            #     f.write('\n')
            #     for i, param in enumerate(parameters):
            #         if i != len(parameters) - 1:
            #             f.write('\t\'' + str(param['parameter']) + '\': ' + str(param['value']) + ',')
            #             f.write('\n')
            #         else:
            #             f.write('\t\'' + str(param['parameter']) + '\': ' + str(param['value']) + '}')
            #     f.write('\n')
            #     f.write('material_dict = {\'name\': \'Smooth Sil 950\', \'model\': \'Neo-Hookean\', \'P1\': 0.34, \'P2\': 0, \'density\': 1.24*(10**-9)}')
            #     f.write('\n')
            #     f.write('actuator_definition = {\'geometry\': geometry_dict, \'material_properties\': material_dict, \'pressure\': 0.1}')
            #
            # # Copy over the other relevant files to the geometry folder
            # shutil.copy('actuator_optimization/main_abaqus.py', 'C1/main_abaqus.py')
            # shutil.copy('actuator_optimization/Actuator.py', 'C1/Actuator.py')
            # shutil.copy('actuator_optimization/Extractor.py', 'C1/Extractor.py')
            #
            # print("Current Directory")
            # print(os.getcwd())
            # os.chdir(model_name)
            # print(os.getcwd())
            # print("New Dir")
            #
            # subprocess.run(
            #     ['abaqus', 'cae', '-noGUI', 'C:/Users/and008/Documents/Models CMA/' + model_name + '/main_abaqus.py'],
            #     shell=True)






    # Loads the parameters for the experiment depending on the optimizer type
    def get_params(self):
        if self.optimization_type == "sweep":

            # Update the actuator definition parameters
            self.actuator_definition['geometry'][self.Optimizer.sweep_parameter] = self.Optimizer.current_value

            # Indicate that this is the final experiment
            if round(self.Optimizer.current_value, 5) == round(self.Optimizer.max_value, 5):
                self.remaining_experiments = False

            # Augment for the subsequent experiment
            self.Optimizer.current_value = self.Optimizer.current_value + self.Optimizer.step

        if self.optimization_type == 'gridsearch':

            # Get the latest set of parameters
            self.Optimizer.generate_params()

            for parameter in self.Optimizer.current_parameters:
                self.actuator_definition['geometry'][parameter['parameter']] = parameter['value']

            if self.Optimizer.last_experiment:
                self.remaining_experiments = False

        if self.optimization_type == 'single':
            self.remaining_experiments = False

    def write_name(self):
        with open("model_name.txt", 'w') as f:
            f.write(ExperimentManager.model_name)

    # See if the model has been run already, generate model name
    def check_run_status(self):

        # Find the model name based on key file
        with open("key_file.txt", 'r') as f_r:
            lines = f_r.readlines()
            for line in lines:
                curr_dict = ast.literal_eval(line)
                self.actuator_dict_list.append(curr_dict)

        # Check to see if the actuator has already been run
        if len(self.actuator_dict_list) != 0:

            self.next_key_num = False
            for run_actuator in self.actuator_dict_list:

                # Check for equivalence of actuator dictionaries
                def dict_equivalence(d1, d2):
                    if (d1['pressure'] == d2['pressure']) and (
                            d1['material_properties'] == d2['material_properties']):
                        for key in d1['geometry']:
                            if round(d1['geometry'][key], 10) != round(d2['geometry'][key], 10):
                                return False
                        return True
                    else:
                        return False

                equivalence_result = dict_equivalence(run_actuator['actuator'], self.Actuator.definition)
                if equivalence_result:
                    run_before = True
                    print("\tModel Run Before")
                    self.next_key_num = run_actuator['key']
                    break

            if not self.next_key_num:
                self.next_key_num = self.actuator_dict_list[-1]['key'] + 1
                run_before = False

        # Otherwise generate a new set of parameters
        else:
            self.next_key_num = 1
            run_before = False
        self.run_before = run_before

        if self.run_before:
            self.model_name = 'M' + str(self.next_key_num)

    def write_to_keyfile(self):

        # Write to the key file
        if not self.run_before:
            with open("key_file.txt", 'a') as f_a:
                print("\tNew Model")
                keyed_dict = {'actuator': self.actuator_definition, 'key': self.next_key_num}
                f_a.write(str(keyed_dict))
                f_a.write(str('\n'))

        self.model_name = 'M' + str(self.next_key_num)

    def set_up_model(self):
        # if self.optimization_type == "sweep":
        #
        #     # Update the actuator definition parameters
        #     self.actuator_definition['geometry'][self.Optimizer.sweep_parameter] = self.Optimizer.current_value
        #
        #     # Find the model name based on key file
        #     with open("key_file.txt", 'r') as f_r:
        #         lines = f_r.readlines()
        #         for line in lines:
        #             curr_dict = ast.literal_eval(line)
        #             self.actuator_dict_list.append(curr_dict)
        #
        #     # Check to see if the actuator has already been run
        #     if len(self.actuator_dict_list) != 0:
        #
        #         next_key_num = False
        #         for run_actuator in self.actuator_dict_list:
        #
        #             # Check for equivalence of actuator dictionaries
        #             def dict_equivalence(d1, d2):
        #                 if (d1['pressure'] == d2['pressure']) and (
        #                         d1['material_properties'] == d2['material_properties']):
        #                     for key in d1['geometry']:
        #                         if round(d1['geometry'][key], 10) != round(d2['geometry'][key], 10):
        #                             return False
        #                     return True
        #                 else:
        #                     return False
        #
        #             equivalence_result = dict_equivalence(run_actuator['actuator'], self.Actuator.definition)
        #             if equivalence_result:
        #                 run_before = True
        #                 print("\tModel Run Before")
        #                 next_key_num = run_actuator['key']
        #                 break
        #
        #         if not next_key_num:
        #             next_key_num = self.actuator_dict_list[-1]['key'] + 1
        #             run_before = False
        #
        #     # Otherwise generate a new set of parameters
        #     else:
        #         next_key_num = 1
        #         run_before = False
        #     self.run_before = run_before
        #
        #    # Write to the key file
        #     if not run_before:
        #         with open("key_file.txt", 'a') as f_a:
        #             print("\tNew Model")
        #             keyed_dict = {'actuator': self.actuator_definition, 'key': next_key_num}
        #             f_a.write(str(keyed_dict))
        #             f_a.write(str('\n'))
        #
        #     self.model_name = 'M' + str(next_key_num)
        #     print("\tModel Name:" + str(self.model_name))
        #
        #     # Indicate that this is the final experiment
        #     if round(self.Optimizer.current_value, 5) == round(self.Optimizer.max_value, 5):
        #         self.remaining_experiments = False
        #
        #     # Augment for the next experiment
        #     self.Optimizer.current_value = self.Optimizer.current_value + self.Optimizer.step

            # Set up the actual model
            self.Actuator.load_params(self.actuator_definition)
            self.Actuator.clear_model_space()
            self.Actuator.generate_core()
            self.Actuator.generate_skin()
            self.Actuator.cut_core_skeleton()
            self.Actuator.generate_material()
            self.Actuator.partition_down_center()
            self.Actuator.generate_steps()
            self.Actuator.apply_pressure()
            self.Actuator.apply_BCs()
            self.Actuator.contact_interactions()
            self.Actuator.generate_mesh()
            self.Actuator.generate_inspection_set()
            self.Actuator.get_mass()

    def run_model(self):
        self.current_job = self.Actuator.write_job(self.model_name)
        self.current_job.submit()
        print("\tComputing...")
        print(self.current_job)
        self.current_job.waitForCompletion()

    def lumped_results_update(self):
        if not self.run_before:
            with open("lumped_results.txt", 'a') as f_a:
                keyed_dict = {'key': self.model_name, 'time': self.run_time, 'mass': self.Actuator.mass}
                f_a.write(str(keyed_dict))
                f_a.write(str('\n'))

    #
    # def run_model(self):
    #     return
    #
    # def extract_results(self):
    #     return

