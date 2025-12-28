import pandas as pd
import itertools
from sklearn import linear_model
import math
# pd.set_option('max_columns', None)

class ProcessedModel:
    def __init__(self, model_name):
        self.name = model_name
        self.text_file = model_name + '.txt'
        self.data = pd.DataFrame()
        self.tip_node = None
        self.actuator = None
        self.num_elements = None
        self.time = None
        self.nodal_strain = None
        self.nodal_strain_avg_after = None
        self.mass = None

        # Parameters for evaluating models at the same curvature
        self.curvature_pressure = None
        self.curvature_radius = None
        self.curvature_bending_angle = None
        self.curvature_average_strain = None
        self.curvature_max_strain = None
        self.curvature_volume = None

        # Energy metrics
        self.energy_same_pressure = None
        self.energy_same_curvature = None


    def load_data(self):
        with open(self.text_file, 'r') as f:
            def data_split(x):
                if x.startswith('DATA'):
                    data_split.count += 1
                return data_split.count
            data_split.count = 0

            # Break the data whenever there is a "DATA" signal
            for key, grp in itertools.groupby(f, data_split):
                list_version = list(grp)
                data_name = list_version[1].strip('\n')
                data = list_version[2:]
                if "Node" in data_name:
                    data = [i.strip('\n') for i in data]
                    x_data = []
                    y_data = []
                    for item in data:
                        x_coor, y_coor = item.split(',')
                        x_data.append(float(x_coor))
                        y_data.append(float(y_coor))
                    self.data[data_name + ' X'] = x_data
                    self.data[data_name + ' Y'] = y_data
                else:
                    data = [float(i.strip('\n')) for i in data]
                    self.data[data_name] = data

    def compute_grasping_force(self):
        u1 = self.data['R1']
        u2 = self.data['R2']
        u1_prime = u1 * math.cos(math.radians(55)) + u2 * math.sin(math.radians(55))
        u2_prime = -u1 * math.sin(math.radians(55)) + u2 * math.cos(math.radians(55))
        self.data['U1_prime'] = u1_prime
        self.data['grasping_force'] = u2_prime

        # Get absolute value to make forces positive
        self.data['grasping_force'] = abs(self.data['grasping_force'])

        # Remove time steps where the force is zero
        self.data = self.data[self.data['grasping_force'] > 0.0]
        return

    def strain_force_loss(self):
        if self.data.empty:
            return_value = float('inf')
            strain_val = float('inf')
            grasping_val = float('inf')
        else:
            self.data['loss'] = 0.5*self.data['Max Strain'] - 0.5*self.data['grasping_force']
            # print(self.data)
            # id_min = self.data.idxmin()['loss']
            # print("ID MIN")
            # print(id_min)
            return_value = self.data['loss'].min()
            # print("loss value right here")
            # print(return_value)

            # print("resetting indices")
            self.data.reset_index(inplace=True)
            # print("new idmin")
            # print(self.data.idxmin())
            idmin_loss = self.data.idxmin()["loss"]
            # print(idmin_loss)

            # print("max strain val")
            # print(self.data)
            strain_val = self.data.iloc[idmin_loss]["Max Strain"]
            # print(strain_val)

            # print("grasping force val")
            grasping_val = self.data.iloc[idmin_loss]["grasping_force"]
            # print(grasping_val)
            # print(self.data['grasping_force'])
            # print("loss column")
            # print(self.data["loss"])
            # print("")
            # strain_value = self.data['Max Strain'].iloc[id_min]
            # force_value = self.data['grasping_force'].iloc[id_min]
        # print("DATA FRAME")
        # print(self.data)
        print("RETURNING LOSS OF: " + str(return_value) + ' ' + str(strain_val) + ' ' + str(grasping_val))
        return return_value, strain_val, grasping_val


    def get_curvature_data(self):
        working_df = self.data
        cols = self.data.columns
        all_columns = list(cols)
        node_columns = []
        other_columns = []
        for col in all_columns:
            if 'Node' in col:
                node_columns.append(col)
            else:
                other_columns.append(col)

        working_df = working_df.drop(columns=other_columns)

        radii = []
        for i, current_row in working_df.iterrows():
            x_values = []
            y_values = []

            if i == 0:
                R = None
            else:
                for key, value in current_row.iteritems():
                    if "X" in key:
                        x_values.append(value)
                    else:
                        y_values.append(value)
                regression_df = pd.DataFrame()
                regression_df['X'] = x_values
                regression_df['Y'] = y_values
                regression_df['X^2 + Y^2'] = regression_df['X'] * regression_df['X'] + regression_df['Y'] * \
                                             regression_df['Y']
                X = regression_df[['X', 'Y']]
                Y = regression_df['X^2 + Y^2']
                regr = linear_model.LinearRegression()
                regr.fit(X, Y)
                A = regr.coef_[0]
                B = regr.coef_[1]
                C = regr.intercept_
                k = A / 2.0
                m = B / 2.0
                R = 0.5 * math.sqrt(4 * C + A * A + B * B)
                # circle = plt.Circle((k, m), R, color='r')
                # fig, ax = plt.subplots()
                # ax.add_patch(circle)
                # # plt.show()
                # ax.scatter(regression_df['X'], regression_df['Y'])
                # plt.xlabel('[mm]')
                # plt.ylabel('[mm]')
                # plt.show()
            radii.append(R)
        self.data['Radius'] = radii
        self.data['Curvature'] = 1 / self.data['Radius']

    def max_strain_at_curvature(self, curvature_value):
        if self.data['Curvature'].iloc[-1] > curvature_value:

            for i, value in self.data['Curvature'].iteritems():
                if i != (len(self.data['Curvature']) - 1):
                    if self.data['Curvature'].iloc[i] < curvature_value < self.data['Curvature'].iloc[i + 1]:
                        lower_value = self.data['Curvature'].iloc[i]
                        lower_value_index = i
                        upper_value = self.data['Curvature'].iloc[i + 1]
                        upper_value_index = i + 1

            # Find pressures for a certain curvature
            curvature_pressure = ((self.data['Pressure'].iloc[upper_value_index] - self.data['Pressure'].iloc[lower_value_index])/(upper_value - lower_value))*(curvature_value - lower_value) + self.data['Pressure'].iloc[lower_value_index]
            self.curvature_pressure = curvature_pressure

            def interpolate(lower_index, upper_index, string_objective, desired_pressure):
                interpolated_value = ((self.data[string_objective].iloc[upper_index] - self.data[string_objective].iloc[
                    lower_index]) / (self.data['Pressure'].iloc[upper_index] - self.data['Pressure'].iloc[
                    lower_index])) * (desired_pressure - self.data['Pressure'].iloc[lower_index]) + \
                                     self.data[string_objective].iloc[lower_index]
                return interpolated_value

            max_strain_curvature = interpolate(lower_value_index, upper_value_index, 'Max Strain', curvature_pressure)
            self.curvature_max_strain = max_strain_curvature
        else:
            self.curvature_max_strain = float('inf')