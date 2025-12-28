import numpy as np
import itertools

class GridSearch:
    def __init__(self, range_parameters):
        self.range_parameters = range_parameters
        self.current_parameters = [] # Define the current set of parameters
        self.last_experiment = False
        self.master_parameters = []

        master_list = []
        for range_dict in self.range_parameters:
            self.master_parameters.append(range_dict['parameter'])
            master_list.append(list(np.linspace(range_dict['lb'], range_dict['ub'], range_dict['num'])))

        self.master_list = master_list
        self.all_combos = list(itertools.product(*self.master_list))
        self.index = 0

    def generate_params(self):
        self.current_parameters = []
        values = self.all_combos[self.index]

        for i, param_name in enumerate(self.master_parameters):
            self.current_parameters.append({'parameter': param_name, 'value': values[i]})

        if self.index == len(self.all_combos) - 1:
            self.last_experiment = True

        self.index += 1