import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import cm
import matplotlib
import numpy as np
import ast
import math
from ProcessedModels import *
from sklearn import linear_model
import oapackage


start_model = 348
end_model = 362
sweep_variable = 't'
curvature_value = 0.007
#
# start_model = 133
# end_model = 147

model_list = np.arange(start_model, end_model + 1, 1)
model_list = ['M' + str(i) for i in model_list]
all_models = []

# model_list.remove('M63')
# model_list.remove('M180')
# model_list.remove('M163')
# model_list.remove('M173')
# model_list.remove('M174')
# model_list.remove('M175')
# model_list.remove('M176')
# model_list.remove('M177')
# model_list.remove('M188')

# model_list = model_list + ['M22']
# model_list = ['M22']

# model_list = model_list + ['M22']
# model_list = ['M22', 'gravity_ydir']

# Initialize the post-processed models, load them into dataframes
for model in model_list:
    print("Model:")
    print(model)
    current_model = ProcessedModel(model)
    current_model.load_data()

    # Load geometric parameters #####
    with open('C:/Users/and008/Documents/Models V2/' + "key_file.txt", 'r') as f_r:
        lines = f_r.readlines()
        for line in lines:
            curr_dict = ast.literal_eval(line)
            if int(curr_dict['key']) == int(model.strip('M')):
                current_model.actuator = curr_dict['actuator']['geometry']

    # # Load geometric parameters
    # with open('C:/Users/and008/Documents/Models V2/' + "lumped_results.txt", 'r') as f_r:
    #     lines = f_r.readlines()
    #     for line in lines:
    #         curr_dict = ast.literal_eval(line)
    #         if curr_dict['key'] == model:
    #             current_model.num_elements = curr_dict['elements']
    #             current_model.time = curr_dict['time']

    # Load mass of actuators ######
    with open('C:/Users/and008/Documents/Models V2/' + "lumped_results.txt", 'r') as f_r:
        lines = f_r.readlines()
        for line in lines:
            curr_dict = ast.literal_eval(line)
            if curr_dict['key'] == model:
                current_model.mass = curr_dict['mass']

    # Identify the tip node
    zero_pressure_data = current_model.data.iloc[0]
    max_x = 0.0
    tip_node = 0
    for i, value in zero_pressure_data.iteritems():
        if "Node" in i:
            if "X" in i:
                if value > max_x:
                    max_x = value
                    tip_node = i
                    tip_node = tip_node.split(' ')[1]
                    current_model.tip_node = tip_node

    # Find the bending angle
    cols = current_model.data.columns
    tip_cols = []
    for col in cols:
        if current_model.tip_node in col:
            tip_cols.append(col)
    tip_data = current_model.data[tip_cols]

    def bending_angle(x):
        raw_angle = math.degrees(math.atan(abs(x[0]) / abs(x[1])))
        if x[0] < 0:
            raw_angle = raw_angle + 90.0
        else:
            raw_angle = 90.0 - raw_angle
        raw_angle = raw_angle*2.0
        return raw_angle

    bending_angles = tip_data.apply(bending_angle, axis=1)
    current_model.data['Bending Angle'] = bending_angles

    # Find the radius of curvature
    last_pressurized_step = current_model.data.iloc[-1]
    working_df = current_model.data
    cols = current_model.data.columns
    all_columns = list(cols)
    node_columns = []
    other_columns = []
    for col in all_columns:
        if 'Node' in col:
            node_columns.append(col)
        else:
            other_columns.append(col)

    working_df = working_df.drop(columns = other_columns)

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
            regression_df['X^2 + Y^2'] = regression_df['X']*regression_df['X'] + regression_df['Y']*regression_df['Y']
            X = regression_df[['X','Y']]
            Y = regression_df['X^2 + Y^2']
            regr = linear_model.LinearRegression()
            regr.fit(X, Y)
            A = regr.coef_[0]
            B = regr.coef_[1]
            C = regr.intercept_
            k = A/2.0
            m = B/2.0
            R = 0.5 * math.sqrt(4*C + A*A + B*B)
            circle = plt.Circle((k, m), R, color='r')
            # fig, ax = plt.subplots()
            # ax.add_patch(circle)
            # # plt.show()
            # ax.scatter(regression_df['X'], regression_df['Y'])
            # plt.show()
        radii.append(R)
    current_model.data['Radius'] = radii
    current_model.data['Curvature'] = 1/current_model.data['Radius']
    all_models.append(current_model)

# colors = ['red', 'blue']
# labels = ['No Gravity', 'Gravity']
# markers = ['o', '^']
# i = 0
# for model in all_models:
#     x_values = []
#     y_values = []
#     count = 0
#     try:
#         model.data.drop(columns=['Pressure', 'Volume', 'Average Strain', 'Max Strain', 'Bending Angle', 'Radius', 'Curvature'], inplace=True)
#     except:
#         model.data.drop(
#             columns=['Pressure', 'Average Strain', 'Max Strain', 'Bending Angle', 'Radius', 'Curvature'],
#             inplace=True)
#     for item in model.data.iloc[-1]:
#         if count % 2 == 0:
#             x_values.append(item)
#         else:
#             y_values.append(item)
#         count = count + 1
#
#     plt.scatter(x_values, y_values, color=colors[i], label=labels[i], marker=markers[i])
#     i= i + 1
# plt.axis('equal')
# plt.legend()
# plt.xlabel('Position (mm)')
# plt.ylabel('Position (mm)')
# plt.show()

# colors = ['red', 'blue']
# labels = ['No Gravity', 'Gravity']
# markers = ['o', '^']
# i = 0
#
# fig, axs = plt.subplots(1,2)
# for model in all_models:
#     print(model.data)
#     axs[0].plot(model.data['Pressure']*1000, model.data['Bending Angle'], color=colors[i], label=labels[i], marker=markers[i])
#     i = i + 1
#
# percent_difference_list = []
# for i, row in all_models[0].data.iterrows():
#     no_grav = row['Bending Angle']
#     grav = all_models[1].data.iloc[i]['Bending Angle']
#     percent_difference = (abs(no_grav-grav)/(grav))*100.0
#     print(percent_difference)
#     percent_difference_list.append(percent_difference)
#
# axs[0].legend()
# axs[0].set_xlabel('Pressure (kPa)')
# axs[0].set_ylabel('Bending Angle (degrees)')
# axs[1].set_xlabel('Pressure (kPa)')
# axs[1].set_ylabel('Percent Difference (%)')
# axs[1].plot(all_models[0].data['Pressure'].iloc[1:]*1000.0, percent_difference_list[1:], color='black', marker='o')
# plt.show()


# Graphing goes after here

# for model in all_models:
#     # plt.plot(model.data['Pressure'], model.data['Bending Angle'])
#     plt.plot(model.data['Pressure'], model.data['Curvature'], '-o')


# Function to graph the visualization suite for all models over all pressures
def visualization_suite(model_list, sweep_variable):

    # Get min and max parameters
    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    sweep_values = []
    for model in model_list:
        sweep_values.append(model.actuator[sweep_variable])
    max_value = max(sweep_values)
    min_value = min(sweep_values)

    # Graphing
    for model in model_list:
        cmap = plt.cm.viridis
        normalize = plt.Normalize(vmin=min_value, vmax=max_value)
        colors = np.ones(len(model.data))
        colors = colors*model.actuator[sweep_variable]

        # cb = plt.scatter(tip_node_df['X Position'], tip_node_df['Y Position'], c=colors, s=15, cmap=cmap,
        #             norm=normalize, label=label)


        model.data['Pressure'] = model.data['Pressure'] * 1000
        ax[0,0].scatter(model.data['Pressure'], model.data['Average Strain'],  c=[cmap(normalize(colors[0]))], s=15)
        ax[0,0].plot(model.data['Pressure'], model.data['Average Strain'],  c=[cmap(normalize(colors[0]))][0])
        ax[0,0].set_xlabel('Pressure (kPa)')
        ax[0,0].set_ylabel('Average Strain')
        ax[0, 1].scatter(model.data['Pressure'], model.data['Max Strain'], c=[cmap(normalize(colors[0]))], s=15)
        ax[0, 1].plot(model.data['Pressure'], model.data['Max Strain'], c=[cmap(normalize(colors[0]))][0])
        ax[0, 1].set_xlabel('Pressure (kPa)')
        ax[0, 1].set_ylabel('Max Strain')
        ax[0, 2].scatter(model.data['Pressure'], model.data['Volume'] - model.data['Volume'].iloc[0], c=[cmap(normalize(colors[0]))], s=15)

        ax[0, 2].plot(model.data['Pressure'], model.data['Volume'] - model.data['Volume'].iloc[0], c=[cmap(normalize(colors[0]))][0])
        ax[1, 0].scatter(model.data['Pressure'], model.data['Bending Angle'], c=[cmap(normalize(colors[0]))], s=15)
        ax[1, 0].plot(model.data['Pressure'], model.data['Bending Angle'], c=[cmap(normalize(colors[0]))][0])
        ax[1, 0].set_ylabel('Bending Angle (Degrees)')
        ax[1, 0].set_xlabel('Pressure (kPa)')
        ax[1, 1].scatter(model.data['Pressure'], model.data['Radius'], c=[cmap(normalize(colors[0]))], s=15)
        ax[1, 1].plot(model.data['Pressure'], model.data['Radius'], c=[cmap(normalize(colors[0]))][0])
        ax[1, 1].set_ylabel('Radius of Curvature (mm)')
        ax[1, 1].set_xlabel('Pressure (kPa)')
        ax[1, 2].scatter(model.data['Pressure'], model.data['Curvature'], c=[cmap(normalize(colors[0]))], s=15)
        ax[1, 2].plot(model.data['Pressure'], model.data['Curvature'], c=[cmap(normalize(colors[0]))][0])
        ax[1, 2].set_ylabel('Curvature (1/mm)')
        ax[1, 2].set_xlabel('Pressure (kPa)')
        ax[0, 2].set_ylabel('Change in Volume (mm^3)')
        ax[0, 2].set_xlabel('Pressure (kPa)')
        ax[1, 2].set_xlabel('Pressure (kPa)')
        # ax[1,1].plot(model.data['Pressure'], model.data['Radius'])
        # ax[1, 1].set_xlabel('Pressure')
        # ax[1, 1].set_ylabel('Bending Radius')
        # ax[1,2].plot(model.data['Pressure'], model.data['Curvature'])
        # ax[1, 2].set_xlabel('Pressure')
        # ax[1, 2].set_ylabel('Curvature')
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[:, 2])
    cbar.ax.set_title(sweep_variable + " (mm)", fontsize=8)
    plt.show()

    for model in model_list:
        plt.plot(model.data['Volume'] - model.data['Volume'].iloc[0], model.data['Pressure'],
                      c=[cmap(normalize(colors[0]))][0])


def visualization_same_pressure(model_list, sweep_variable):

    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    # sweep_values = []
    # for model in model_list:
    #     sweep_values.append(model.actuator[sweep_variable])
    # max_value = max(sweep_values)
    # min_value = min(sweep_values)

    # Graphing
    sweep_variables = []
    average_strains = []
    max_strains = []
    change_volumes = []
    bending_angles = []
    radii_list = []
    curvature_list = []

    strain_df = pd.read_csv("C:/Users/and008/Documents/Models V2/" + "Thickness_Strain.csv")

    for model in model_list:
        # cmap = plt.cm.viridis
        # normalize = plt.Normalize(vmin=min_value, vmax=max_value)
        # colors = np.ones(len(model.data))
        # colors = colors*model.actuator[sweep_variable]
        # print(colors)

        # cb = plt.scatter(tip_node_df['X Position'], tip_node_df['Y Position'], c=colors, s=15, cmap=cmap,
        #             norm=normalize, label=label)


        ax[0,0].scatter(model.actuator[sweep_variable], model.data['Average Strain'].iloc[-1], color='blue')
        sweep_variables.append(model.actuator[sweep_variable])
        average_strains.append(model.data['Average Strain'].iloc[-1])
        ax[0,0].set_xlabel(sweep_variable + " (mm)")
        ax[0,0].set_ylabel('Average Strain')
        ax[0,1].scatter(model.actuator[sweep_variable], model.data['Max Strain'].iloc[-1], color='red')
    #     ax[0, 1].plot(model.data['Pressure'], model.data['Max Strain'], c=[cmap(normalize(colors[0]))][0])
        ax[0, 1].set_xlabel(sweep_variable + " (mm)")
        ax[0, 1].set_ylabel('Max Strain')
        max_strains.append(model.data['Max Strain'].iloc[-1])
        ax[0,2].scatter(model.actuator[sweep_variable], model.data['Volume'].iloc[-1] - model.data['Volume'].iloc[0], color='black')
        ax[0, 2].set_xlabel(sweep_variable + " (mm)")
        ax[0, 2].set_ylabel('Change in Volume (mm^3)')
        change_volumes.append(model.data['Volume'].iloc[-1] - model.data['Volume'].iloc[0])
        ax[1, 0].scatter(model.actuator[sweep_variable], model.data['Bending Angle'].iloc[-1], color='green')
    #     ax[1, 0].plot(model.data['Pressure'], model.data['Bending Angle'], c=[cmap(normalize(colors[0]))][0])
        bending_angles.append(model.data['Bending Angle'].iloc[-1])
        ax[1, 0].set_ylabel('Bending Angle (Degrees)')
        ax[1, 0].set_xlabel(sweep_variable + " (mm)")
        ax[1, 1].scatter(model.actuator[sweep_variable], model.data['Radius'].iloc[-1], color='purple')
    #     ax[1, 1].plot(model.data['Pressure'], model.data['Radius'], c=[cmap(normalize(colors[0]))][0])
        radii_list.append(model.data['Radius'].iloc[-1])
        ax[1, 1].set_ylabel('Radius of Curvature (mm)')
        ax[1, 1].set_xlabel(sweep_variable + " (mm)")
        ax[1, 2].scatter(model.actuator[sweep_variable], model.data['Curvature'].iloc[-1], color='orange')
    #     ax[1, 2].plot(model.data['Pressure'], model.data['Curvature'], c=[cmap(normalize(colors[0]))][0])
        curvature_list.append(model.data['Curvature'].iloc[-1])
        ax[1, 2].set_ylabel('Curvature (1/mm)')
        ax[1, 2].set_xlabel(sweep_variable + " (mm)")
    #     ax[0, 2].set_ylabel('Mass (kg)')
    #     ax[1, 2].set_xlabel('Pressure (kPa)')
    #     # ax[1,1].plot(model.data['Pressure'], model.data['Radius'])
    #     # ax[1, 1].set_xlabel('Pressure')
    #     # ax[1, 1].set_ylabel('Bending Radius')
    #     # ax[1,2].plot(model.data['Pressure'], model.data['Curvature'])
    #     # ax[1, 2].set_xlabel('Pressure')
    #     # ax[1, 2].set_ylabel('Curvature')
    # sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax[:, 2])
    # cbar.ax.set_title("Thickness (mm)", fontsize=8)
    lines_df = pd.DataFrame(
        {'Sweep Variable': sweep_variables, 'Average Strains': average_strains, 'Max Strains': max_strains,
         'Volume': change_volumes, 'Bending Angle': bending_angles, 'Radius': radii_list, 'Curvature': curvature_list})
    lines_df.sort_values('Sweep Variable', inplace=True)
    # ax[0, 1].plot(strain_df['Thickness'], strain_df['Strain, 100 kpa'])
    ax[0, 0].plot(lines_df['Sweep Variable'], lines_df['Average Strains'], color='blue')
    ax[0, 1].plot(lines_df['Sweep Variable'], lines_df['Max Strains'], color='red')
    ax[0, 2].plot(lines_df['Sweep Variable'], lines_df['Volume'], color='black')
    ax[1, 0].plot(lines_df['Sweep Variable'], lines_df['Bending Angle'], color='green')
    ax[1, 1].plot(lines_df['Sweep Variable'], lines_df['Radius'], color='purple')
    ax[1, 2].plot(lines_df['Sweep Variable'], lines_df['Curvature'], color='orange')
    plt.show()

def pareto_optimal_set(model_list):
    sweep_variable = 'n'
    sweep_values = []

    for model in model_list:
        sweep_values.append(model.actuator[sweep_variable])
    max_value = max(sweep_values)
    min_value = min(sweep_values)

    cmap = plt.cm.Spectral
    normalize = plt.Normalize(vmin=min_value, vmax=max_value)
    # colors = np.ones(len(model.data))
    # colors = colors * model.actuator[sweep_variable]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # y_values = []
    # for model in model_list:
    #     if model.name == 'M22':
    #         color = 'red'
    #     ax[0].scatter(model.curvature_pressure, model.curvature_max_strain, color=cmap(normalize(model.actuator[sweep_variable])))
        # y_values.append(model.curvature_max_strain)

    nums = np.arange(147, 163, 1)
    print("Model Numbers Here")
    print(nums)
    nums = ['M' + str(i) for i in nums]
    sweep_models = ['M22'] + nums

    sweep_ind = []
    pareto = oapackage.ParetoDoubleLong()
    for i, model in enumerate(model_list):
        w = oapackage.doubleVector((model.curvature_pressure, 1.0/model.curvature_max_strain))
        pareto.addvalue(w, i)
        if model.name in sweep_models:
            sweep_ind.append(i)

    pareto.show(verbose=1)

    lst = pareto.allindices()  # the indices of the Pareto optimal designs


    sweep_df = pd.DataFrame(columns=['x_values', 'y_values'])
    for i, model in enumerate(model_list):
        model.curvature_pressure = model.curvature_pressure*1000
        if i in lst:
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='s', color=cmap(normalize(model.actuator[sweep_variable])))
        else:
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
                          color=cmap(normalize(model.actuator[sweep_variable])))
        if model.name == 'M22':
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='o',
                          color=cmap(normalize(model.actuator[sweep_variable])))
        if model.name in sweep_models:
            sweep_df = sweep_df.append({'x_values': model.curvature_pressure, 'y_values': model.curvature_max_strain}, ignore_index=True)
            sweep_ind.append(i)
    sweep_df.sort_values('x_values', inplace=True)

    for i, model in enumerate(model_list):
        if i in sweep_ind:
            if i in lst:
                ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='s',
                              color=cmap(normalize(model.actuator[sweep_variable])))
            else:
                ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
                              color=cmap(normalize(model.actuator[sweep_variable])))

    for i, model in enumerate(model_list):
        if model.name == 'M22':
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='o',
                          color=cmap(normalize(model.actuator[sweep_variable])))

    special_set = np.arange(189, 202, 1).tolist()
    special_set = ['M' + str(i) for i in special_set]
    new_model_list = model_list + special_set
    for i, model in enumerate(new_model_list):
        if model.name in special_set:
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='o',
                          color='red')

    # ax.plot(sweep_df['x_values'], sweep_df['y_values'], color='black', linestyle='solid', marker='.' )

    print("Pareto Indicies")
    print(lst)

    ax.set_xlabel('Pressure [kPa]', fontsize=14)
    ax.set_ylabel('Max Strain [-]', fontsize=14)
    ax.invert_yaxis()
    sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_title(sweep_variable, fontsize=14, pad=20)
    cbar.ax.tick_params(labelsize=14)
    # plt.ylim(max(y_values), min(y_values))

    plt.savefig('pareto_' + sweep_variable)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def pareto_special(model_list):

    # colors = np.ones(len(model.data))
    # colors = colors * model.actuator[sweep_variable]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # y_values = []
    # for model in model_list:
    #     if model.name == 'M22':
    #         color = 'red'
    #     ax[0].scatter(model.curvature_pressure, model.curvature_max_strain, color=cmap(normalize(model.actuator[sweep_variable])))
        # y_values.append(model.curvature_max_strain)

    first_models = []
    first_model_names = np.arange(33, 190, 1).tolist()
    first_model_names = ['M' + str(i) for i in first_model_names]
    second_models = []
    second_model_names = np.arange(190, 248, 1).tolist()
    second_model_names = ['M' + str(i) for i in second_model_names]
    third_models = []
    third_model_names = np.arange(249, 287, 1).tolist()
    third_model_names = ['M' + str(i) for i in third_model_names]

    print("First Model Names")
    print(first_model_names)

    print("Second Model Names")
    print(second_model_names)

    for model in model_list:
        print(model.name)
        model.curvature_pressure = model.curvature_pressure * 1000
        if model.name in first_model_names:
            first_models.append(model)
        if model.name in second_model_names:
            second_models.append(model)
        if model.name in third_model_names:
            third_models.append(model)

    print("First Models")
    print(first_models)

    print("Second Models")
    print(second_models)
    #
    # pareto = oapackage.ParetoDoubleLong()
    # for i, model in enumerate(first_models):
    #     w = oapackage.doubleVector((model.curvature_pressure, 1.0/model.curvature_max_strain))
    #     pareto.addvalue(w, i)
    #
    # pareto.show(verbose=1)
    #
    # lst = pareto.allindices()  # the indices of the Pareto optimal designs
    #
    #
    # for i, model in enumerate(first_models):
    #     model.curvature_pressure = model.curvature_pressure*1000
    #     if i in lst:
    #         ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='s', color='blue')
    #     else:
    #         ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
    #                       color='blue')
    #
    #
    # for i, model in enumerate(model_list):
    #     if model.name == 'M22':
    #         ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='o',
    #                       color='blue')
    #
    # # special_set = np.arange(189, 202, 1).tolist()
    # # special_set = ['M' + str(i) for i in special_set]
    # # new_model_list = model_list + special_set + ['M22']
    #
    # pareto = oapackage.ParetoDoubleLong()
    #
    # print("Checking Model List")
    # print(model_list)
    # for i, model in enumerate(model_list):
    #     w = oapackage.doubleVector((model.curvature_pressure, 1.0 / model.curvature_max_strain))
    #     pareto.addvalue(w, i)
    #
    # pareto.show(verbose=1)
    #
    # lst = pareto.allindices()  # the indices of the Pareto optimal designs
    # print("Pareto Indices")
    # print(lst)
    #
    # pareto_models = []
    #
    # for i, model in enumerate(model_list):
    #     if i in lst:
    #         print("Match")
    #         print(i)
    #         pareto_models.append(i)
    #
    # for i, model in enumerate(second_models):
    #     if model.name in pareto_models:
    #         ax.scatter(model.curvature_pressure, model.curvature_max_strain, edgecolors='black', marker='s', color='red')
    #     else:
    #         ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
    #                       color='red')

    for model in model_list:
        if model.name in first_model_names:
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
                       color='blue')
        elif model.name in second_model_names:
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
                       color='red')
        elif model.name in third_model_names:
            ax.scatter(model.curvature_pressure, model.curvature_max_strain, marker='o',
                       color='green')

    # ax.plot(sweep_df['x_values'], sweep_df['y_values'], color='black', linestyle='solid', marker='.' )

    ax.set_xlabel('Pressure [kPa]', fontsize=14)
    ax.set_ylabel('Max Strain [-]', fontsize=14)
    ax.invert_yaxis()
    # sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.ax.set_title(sweep_variable, fontsize=14, pad=20)
    # cbar.ax.tick_params(labelsize=14)
    # plt.ylim(max(y_values), min(y_values))

    # plt.savefig('pareto_' + sweep_variable)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def visualization_same_curvature(model_list, sweep_variable, curvature_value):

    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    # sweep_values = []
    # for model in model_list:
    #     sweep_values.append(model.actuator[sweep_variable])
    # max_value = max(sweep_values)
    # min_value = min(sweep_values)

    # Graphing
    sweep_variables = []
    average_strains = []
    max_strains = []
    change_volumes = []
    bending_angles = []
    radii_list = []
    curvature_list = []

    # strain_df = pd.read_csv("C:/Users/and008/Documents/Models V2/" + "Thickness_Strain.csv")

    other_strains_list = []

    val = 0
    for model in model_list:
        # cmap = plt.cm.viridis
        # normalize = plt.Normalize(vmin=min_value, vmax=max_value)
        # colors = np.ones(len(model.data))
        # colors = colors*model.actuator[sweep_variable]
        # print(colors)

        # cb = plt.scatter(tip_node_df['X Position'], tip_node_df['Y Position'], c=colors, s=15, cmap=cmap,
        #             norm=normalize, label=label)
        for i, value in model.data['Curvature'].iteritems():
            if i != (len(model.data['Curvature']) - 1):
                if model.data['Curvature'].iloc[i] < curvature_value < model.data['Curvature'].iloc[i + 1]:
                    lower_value = model.data['Curvature'].iloc[i]
                    lower_value_index = i
                    upper_value = model.data['Curvature'].iloc[i + 1]
                    upper_value_index = i + 1

        # Find pressures for a certain curvature
        curvature_pressure = ((model.data['Pressure'].iloc[upper_value_index] - model.data['Pressure'].iloc[lower_value_index])/(upper_value - lower_value))*(curvature_value - lower_value) + model.data['Pressure'].iloc[lower_value_index]
        model.curvature_pressure = curvature_pressure

        # Find the radius of curvature corresponding to a certain curvature
        curvature_radius = ((model.data['Radius'].iloc[upper_value_index] - model.data['Radius'].iloc[lower_value_index])/(model.data['Pressure'].iloc[upper_value_index] - model.data['Pressure'].iloc[lower_value_index]))*(model.curvature_pressure - model.data['Pressure'].iloc[lower_value_index]) + model.data['Radius'].iloc[lower_value_index]
        model.curvature_radius = curvature_radius

        # Find the volume corresponding to a certain curvature
        curvature_volume = ((model.data['Volume'].iloc[upper_value_index] - model.data['Volume'].iloc[
            lower_value_index]) / (model.data['Pressure'].iloc[upper_value_index] - model.data['Pressure'].iloc[
            lower_value_index])) * (model.curvature_pressure - model.data['Pressure'].iloc[lower_value_index]) + \
                           model.data['Volume'].iloc[lower_value_index]
        model.curvature_volume = curvature_volume

        # Find the bending angle corresponding to a certain curvature
        curvature_bending_angle = ((model.data['Bending Angle'].iloc[upper_value_index] - model.data['Bending Angle'].iloc[
            lower_value_index]) / (model.data['Pressure'].iloc[upper_value_index] - model.data['Pressure'].iloc[
            lower_value_index])) * (model.curvature_pressure - model.data['Pressure'].iloc[lower_value_index]) + \
                           model.data['Bending Angle'].iloc[lower_value_index]
        model.curvature_bending_angle = curvature_bending_angle

        # Find the average strain corresponding to a certain curvature
        curvature_average_strain = ((model.data['Average Strain'].iloc[upper_value_index] -
                                    model.data['Average Strain'].iloc[
                                        lower_value_index]) / (model.data['Pressure'].iloc[upper_value_index] -
                                                               model.data['Pressure'].iloc[
                                                                   lower_value_index])) * (
                                              model.curvature_pressure - model.data['Pressure'].iloc[
                                          lower_value_index]) + \
                                  model.data['Average Strain'].iloc[lower_value_index]
        model.curvature_average_strain = curvature_average_strain



        # Find the max strain corresponding to a certain curvature
        curvature_max_strain = ((model.data['Max Strain'].iloc[upper_value_index] -
                                     model.data['Max Strain'].iloc[
                                         lower_value_index]) / (model.data['Pressure'].iloc[upper_value_index] -
                                                                model.data['Pressure'].iloc[
                                                                    lower_value_index])) * (
                                           model.curvature_pressure - model.data['Pressure'].iloc[
                                       lower_value_index]) + \
                                   model.data['Max Strain'].iloc[lower_value_index]
        model.curvature_max_strain = curvature_max_strain

        # calc = ((strain_df['Max Time'].iloc[val] -
        #                              strain_df['Min Time'].iloc[
        #                                  val]) / (model.data['Pressure'].iloc[upper_value_index] -
        #                                                         model.data['Pressure'].iloc[
        #                                                             lower_value_index])) * (
        #                                    model.curvature_pressure - model.data['Pressure'].iloc[
        #                                lower_value_index]) + \
        #                            strain_df['Max Time'].iloc[val]
        #
        # other_strains_list.append(calc)
        #
        # val = val + 1

        print(model.name)
        print(model.data)
        print(lower_value_index, upper_value_index)

        # print(model.data['Curvature'])
        ax[0,0].scatter(model.actuator[sweep_variable], model.curvature_average_strain, color='blue')
        ax[0,0].set_xlabel(sweep_variable + " (mm)")
        ax[0,0].set_ylabel('Average Strain')
        ax[0,1].scatter(model.actuator[sweep_variable], model.curvature_max_strain, color='red')
    # #     ax[0, 1].plot(model.data['Pressure'], model.data['Max Strain'], c=[cmap(normalize(colors[0]))][0])
        ax[0, 1].set_xlabel(sweep_variable + " (mm)")
        ax[0, 1].set_ylabel('Max Strain')
    # #     # # ax[0,2].plot(model.data['Pressure'], model.data['Mass'])
        ax[1, 0].scatter(model.actuator[sweep_variable], model.curvature_bending_angle, color='green')
    # #     ax[1, 0].plot(model.data['Pressure'], model.data['Bending Angle'], c=[cmap(normalize(colors[0]))][0])
        ax[1, 0].set_ylabel('Bending Angle (Degrees)')
        ax[1, 0].set_xlabel(sweep_variable + " (mm)")
        ax[0, 2].scatter(model.actuator[sweep_variable], model.curvature_volume - model.data['Volume'].iloc[0], color='black')
        # #     ax[1, 0].plot(model.data['Pressure'], model.data['Bending Angle'], c=[cmap(normalize(colors[0]))][0])
        ax[0, 2].set_ylabel('Change in Volume (mm^3)')
        ax[0, 2].set_xlabel(sweep_variable + " (mm)")
    #     ax[1, 1].scatter(model.actuator[sweep_variable], model.data['Radius'].iloc[-1], color='purple')
    # #     ax[1, 1].plot(model.data['Pressure'], model.data['Radius'], c=[cmap(normalize(colors[0]))][0])
    #     ax[1, 1].set_ylabel('Radius of Curvature (mm)')
    #     ax[1, 1].set_xlabel(sweep_variable + " (mm)")
    #     ax[1, 2].scatter(model.actuator[sweep_variable], model.data['Curvature'].iloc[-1], color='orange')
    # #     ax[1, 2].plot(model.data['Pressure'], model.data['Curvature'], c=[cmap(normalize(colors[0]))][0])
    #     ax[1, 2].set_ylabel('Curvature (1/mm)')
    #     ax[1, 2].set_xlabel(sweep_variable + " (mm)")
    #     ax[0, 2].set_ylabel('Mass (kg)')
    #     ax[1, 2].set_xlabel('Pressure (kPa)')
        ax[1, 1].scatter(model.actuator[sweep_variable], model.curvature_radius, color = 'purple')
        ax[1, 1].set_xlabel(sweep_variable + " (mm)")
        ax[1, 1].set_ylabel('Bending Radius (mm)')
        ax[1, 2].scatter(model.actuator[sweep_variable], model.curvature_pressure, color='orange')
        ax[1, 2].set_xlabel(sweep_variable + " (mm)")
        ax[1, 2].set_ylabel('Pressure (kPa)')

        sweep_variables.append(model.actuator[sweep_variable])
        average_strains.append(model.curvature_average_strain)
        max_strains.append(model.curvature_max_strain)
        change_volumes.append(model.curvature_volume - model.data['Volume'].iloc[0])
        bending_angles.append(model.curvature_bending_angle)
        radii_list.append(model.curvature_radius)
        curvature_list.append(model.curvature_pressure)

    # sm = plt.cm.ScalarMappable(norm=normalize, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax[:, 2])
    # cbar.ax.set_title("Thickness (mm)", fontsize=8)
    lines_df = pd.DataFrame(
        {'Sweep Variable': sweep_variables, 'Average Strains': average_strains, 'Max Strains': max_strains,
         'Volume': change_volumes, 'Bending Angle': bending_angles, 'Radius': radii_list, 'Curvature': curvature_list})
    lines_df.sort_values('Sweep Variable', inplace=True)
    ax[0, 0].plot(lines_df['Sweep Variable'], lines_df['Average Strains'], color='blue')
    ax[0, 1].plot(lines_df['Sweep Variable'], lines_df['Max Strains'], color='red')
    # ax[0, 1].plot(strain_df['Thickness'], other_strains_list, color='red')
    ax[0, 2].plot(lines_df['Sweep Variable'], lines_df['Volume'], color='black')
    ax[1, 0].plot(lines_df['Sweep Variable'], lines_df['Bending Angle'], color='green')
    ax[1, 1].plot(lines_df['Sweep Variable'], lines_df['Radius'], color='purple')
    ax[1, 2].plot(lines_df['Sweep Variable'], lines_df['Curvature'], color='orange')

    plt.show()

def mass_graph(model_list, sweep_value):
    for model in model_list:
        plt.scatter(model.actuator[sweep_variable], model.mass, color='red')
        plt.xlabel(sweep_value + " (mm)")
        plt.ylabel("Mass (kg)")

    plt.show()

def energy(model_list, sweep_value, curvature_value):

    fig, ax = plt.subplots(2, constrained_layout=True)
    sweep_values = []
    for model in model_list:
        sweep_values.append(model.actuator[sweep_variable])
    max_value = max(sweep_values)
    min_value = min(sweep_values)

    for model in model_list:
        model.data['Delta Volume'] = model.data['Volume'] - model.data['Volume'].iloc[0]
        model.energy_same_pressure = np.trapz(model.data['Pressure'], x = model.data['Delta Volume'])/(1000*1000)

        for i, value in model.data['Curvature'].iteritems():
            if i != (len(model.data['Curvature']) - 1):
                if model.data['Curvature'].iloc[i] < curvature_value < model.data['Curvature'].iloc[i + 1]:
                    lower_value = model.data['Curvature'].iloc[i]
                    lower_value_index = i
                    upper_value = model.data['Curvature'].iloc[i + 1]
                    upper_value_index = i + 1
        curvature_pressure = ((model.data['Pressure'].iloc[upper_value_index] - model.data['Pressure'].iloc[
            lower_value_index]) / (upper_value - lower_value)) * (curvature_value - lower_value) + \
                             model.data['Pressure'].iloc[lower_value_index]
        model.curvature_pressure = curvature_pressure

        delta_curvature_volume = ((model.data['Delta Volume'].iloc[upper_value_index] - model.data['Delta Volume'].iloc[
            lower_value_index]) / (model.data['Pressure'].iloc[upper_value_index] - model.data['Pressure'].iloc[
            lower_value_index])) * (model.curvature_pressure - model.data['Pressure'].iloc[lower_value_index]) + \
                           model.data['Delta Volume'].iloc[lower_value_index]

        energy_by_curvature = model.data.copy()

        drop_inds = np.arange(lower_value_index + 1, len(energy_by_curvature))
        energy_by_curvature.drop(index=drop_inds.tolist(), inplace=True)
        energy_by_curvature = energy_by_curvature.append({'Delta Volume': delta_curvature_volume, 'Pressure': curvature_pressure}, ignore_index=True)

        model.energy_same_curvature = np.trapz(energy_by_curvature['Pressure'], x=energy_by_curvature['Delta Volume'])/(1000*1000)

        print("Pressure")
        print(energy_by_curvature['Pressure'])
        ax[0].scatter(model.actuator[sweep_variable], model.energy_same_pressure, color='black')
        ax[1].scatter(model.actuator[sweep_variable], model.energy_same_curvature, color='maroon')
        ax[1].set_xlabel(str(sweep_variable) + " (mm)")
        ax[1].set_ylabel("Energy (J)")
        ax[0].set_ylabel("Energy (J)")
        ax[0].set_title("Energy Required for Pressurization to " + str(round(model.data['Pressure'].iloc[-1], 2)) + " kPa")
        ax[1].set_title(
            "Energy Required for Curvature of " + str(curvature_value) + " 1/mm")
        # ax[0, 0].plot(model.data['Pressure'], model.data['Average Strain'], c=[cmap(normalize(colors[0]))][0])
    plt.show()

def mesh_convergence(model_list):

    for model in model_list:
        unique_nodal_strains = pd.read_csv("C:/Users/and008/Documents/Models V2/" + "UniqueNodalStrainsMeshRefinement.csv")
        for i, row in unique_nodal_strains.iterrows():
            if row['Model'] == model.name:
                model.nodal_strain = row['Strain']
                model.nodal_strain_avg_after = row['Strain Avg After']


    # fig, ax = plt.subplots()
    # for model in model_list:
    #     ax.scatter(model.num_elements, model.data['Bending Angle'].iloc[-1], color='blue')
    #     ax.set_xlabel('Number Elements')
    #     ax.set_ylabel('Bending Angle (degrees)')
    #     ax.set_xscale('log')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # for model in model_list:
    #     ax.scatter(model.num_elements, model.data['Bending Angle'].iloc[-1], color='blue')
    #     ax.set_xlabel('Number Elements')
    #     ax.set_ylabel('Bending Angle (degrees)')
    #     # ax.set_xscale('log')
    # plt.show()


    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.2))

    by_element_size = pd.DataFrame(columns=["Number of Elements", "Bending Angle", "Volume", "Average Strain", "Max Strain", "Time", "Mesh Size", "Nodal Strain", "Strain Avg After"])
    for model in model_list:
        by_element_size = by_element_size.append({"Number of Elements": model.num_elements, "Bending Angle": model.data['Bending Angle'].iloc[-1], "Strain Avg After": model.nodal_strain_avg_after, "Volume": model.data['Volume'].iloc[-1], "Nodal Strain":model.nodal_strain, "Average Strain": model.data['Average Strain'].iloc[-1], "Max Strain": model.data['Max Strain'].iloc[-1], "Time": model.time, "Mesh Size": model.actuator["mesh"]}, ignore_index=True)
    by_element_size.sort_values("Number of Elements", inplace=True)

    p1, = ax.plot(by_element_size['Number of Elements'], by_element_size['Max Strain'], color='g', marker='.', linestyle='-', label="Integration Point Max Strain")
    p2, = twin1.plot(by_element_size['Number of Elements'], by_element_size['Nodal Strain'], color='orange', marker='.', linestyle='-', label="Nodal Max Strain")
    p3, = twin2.plot(by_element_size['Number of Elements'], by_element_size['Strain Avg After'], color='black', marker='.', linestyle='-', label="Nodal Strain Avgerage After")

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0.6, 1.1)
    # twin1.set_ylim(0.6, 1.1)
    # twin2.set_ylim(0.6, 1.1)

    ax.set_xlabel("Number of Elements")
    ax.set_ylabel("Integration Point Max Strain")
    twin1.set_ylabel("Nodal Max Strain")
    twin2.set_ylabel("Nodal Max Strain Average After")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2 ,p3])

    plt.show()
    # plot(by_element_size['Number of Elements'], by_element_size['Max Strain'], color='g', marker='.', linestyle='-',
    #      label="Integration Point Max Strain")
    # p2, = twin1.plot(by_element_size['Number of Elements'], by_element_size['Nodal Strain'], color='orange', marker='.',
    #                  linestyle='-', label="Nodal Max Strain")
    # p3, = twin2.plot(by_element_size['Number of Elements'], by_element_size['Strain Avg After'], color='black',
    #                  marker='.', linestyle='-', label="Nodal Strain Avgerage After")

    plt.plot(by_element_size['Number of Elements'], by_element_size['Max Strain'], color='g', marker='.', linestyle='-',
         label="Integration Point")
    plt.plot(by_element_size['Number of Elements'], by_element_size['Nodal Strain'], color='orange', marker='.',
                     linestyle='-', label="Nodal Strain Average First")
    plt.plot(by_element_size['Number of Elements'], by_element_size['Strain Avg After'], color='black',
                     marker='.', linestyle='-', label="Nodal Strain Average After")
    plt.xlabel('Number of Elements')
    plt.ylabel('Max Strain')
    plt.legend()
    plt.show()
    #
    # plt.plot(by_element_size['Number of Elements'], by_element_size['Time']/60.0, marker='.', linestyle='-')
    # plt.xlabel("Number of Elements")
    # plt.ylabel("Time (min)")
    # plt.show()
    #
    # plt.plot(by_element_size["Mesh Size"], by_element_size['Number of Elements'], marker='.', linestyle='-')
    # plt.ylabel("Number of Elements")
    # plt.xlabel("Global Seed Size (mm)")
    # plt.show()


    #
    # for model in model_list:
    #     plt.scatter(model.num_elements, model.time/60.0, color='blue')
    #     plt.xlabel('Number Elements')
    #     plt.ylabel('Time (min)')
    # plt.show()
    #
    # for model in model_list:
    #     plt.scatter(model.num_elements, model.data['Max Strain'].iloc[-1])
    #     plt.xlabel('Number Elements')
    #     plt.ylabel('Max Strain')
    # plt.show()

#
visualization_suite(all_models, sweep_variable)
# # # # # #
visualization_same_pressure(all_models, sweep_variable)
# # #
# visualization_same_curvature(all_models, sweep_variable, curvature_value)
#
# pareto_special(all_models)
# pareto_optimal_set(all_models)

# mass_graph(all_models, sweep_variable)
# energy(all_models, sweep_variable, curvature_value)




# mesh_convergence(all_models)


#
# def update_model_name():
#     return
#
# def graph_strains(model_list, sweep_variable, min_value, max_value):
#
#     for model in model_list:
#         path_name = 'C:/Users/and008/Documents/Models/' + model_name + 'strains.txt'
#         with open(path_name, 'r') as fr:
#             for line
#
# def graph_tip_reachable_space(model_list, sweep_variable, min_value, max_value):
#     for model_name in model_list:
#         print("model name")
#         print(model_name)
#         model_name = 'M' + str(model_name)
#         path_name = 'C:/Users/and008/Documents/Models/' + model_name + '.txt'
#
#         data = pd.DataFrame(columns=['X Position', 'Y Position'])
#
#         try:
#             with open(path_name, 'r') as fr:
#                 next_line = False
#
#                 for line in fr:
#
#                     # Extract pressure
#                     if next_line:
#                         current_pressure = float(line)
#                         next_line = False
#
#                     if 'Pressure' in line:
#                         next_line = True
#
#                     if ',' in line:
#                         coords = line.split(',')
#                         x_coord = float(coords[0])
#                         y_coord = float(coords[1].strip('\n'))
#                         node = int(coords[2])
#                         data = data.append({'X Position': coords[0], 'Y Position': coords[1], 'Node': node, 'Pressure':current_pressure}, ignore_index=True)
#
#             pressure_grouped = data.groupby(by=['Pressure'])
#
#             for key, item in pressure_grouped:
#                 if key == 0.0:
#                     tip_node = None
#                     tip_node_x = -10
#                     for i, row in item.iterrows():
#                         if float(row['X Position']) > tip_node_x:
#                             tip_node_x = float(row['X Position'])
#                             tip_node = int(row['Node'])
#
#             node_grouped = data.groupby(by=['Node'])
#
#             for key, item in node_grouped:
#                 if key == tip_node:
#                     tip_node_df = item
#
#             print(tip_node_df.dtypes)
#             tip_node_df['X Position'] = tip_node_df['X Position'].astype(float)
#             tip_node_df['Y Position'] = tip_node_df['Y Position'].astype(float)
#             print(tip_node_df.dtypes)
#
#             # Find the model name based on key file
#             with open("key_file.txt", 'r') as f_r:
#                 lines = f_r.readlines()
#                 for line in lines:
#                     curr_dict = ast.literal_eval(line)
#                     print("Dict Line")
#                     print(line)
#                     if curr_dict['key'] == int(model_name.strip('M')):
#                         label = curr_dict['actuator']['geometry'][sweep_variable]
#                         geom_value = float(curr_dict['actuator']['geometry'][sweep_variable])
#                         print("geom value")
#                         print(geom_value)
#                         geom_value = math.degrees(geom_value)
#
#                         print("label")
#                         label = sweep_variable + '=' + str(geom_value)
#                         print(label)
#
#
#             cmap = plt.cm.viridis
#             normalize = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
#
#             colors = np.ones(len(tip_node_df))
#             colors = colors*geom_value
#             cb = plt.scatter(tip_node_df['X Position'], tip_node_df['Y Position'], c=colors, s=15, cmap=cmap,
#                         norm=normalize, label=label)
#
#             plt.axis('equal')
#             plt.xlabel('X Position (mm)')
#             plt.ylabel('Y Position (mm)')
#             # plt.legend()
#             plt.title('Tip Reachable Space - Pressure at 100 kPa')
#
#         except:
#             pass
#
#     cbar = plt.colorbar(cb)
#     cbar.set_label('Thickness (mm)')
#
#     plt.show()
#
# def graph_tip_endpoint(model_list, sweep_variable, min_value, max_value):
#     for model_name in model_list:
#         print("model name")
#         print(model_name)
#         model_name = 'M' + str(model_name)
#         path_name = 'C:/Users/and008/Documents/Models/' + model_name + '.txt'
#
#         data = pd.DataFrame(columns=['X Position', 'Y Position'])
#         try:
#             with open(path_name, 'r') as fr:
#                 next_line = False
#
#                 for line in fr:
#
#                     # Extract pressure
#                     if next_line:
#                         current_pressure = float(line)
#                         next_line = False
#
#                     if 'Pressure' in line:
#                         next_line = True
#
#                     if ',' in line:
#                         coords = line.split(',')
#                         x_coord = float(coords[0])
#                         y_coord = float(coords[1].strip('\n'))
#                         node = int(coords[2])
#                         data = data.append({'X Position': coords[0], 'Y Position': coords[1], 'Node': node, 'Pressure':current_pressure}, ignore_index=True)
#
#             pressure_grouped = data.groupby(by=['Pressure'])
#
#             for key, item in pressure_grouped:
#                 if key == 0.0:
#                     tip_node = None
#                     tip_node_x = -10
#                     for i, row in item.iterrows():
#                         if float(row['X Position']) > tip_node_x:
#                             tip_node_x = float(row['X Position'])
#                             tip_node = int(row['Node'])
#
#             node_grouped = data.groupby(by=['Node'])
#
#             for key, item in node_grouped:
#                 if key == tip_node:
#                     tip_node_df = item
#
#             print(tip_node_df.dtypes)
#             tip_node_df['X Position'] = tip_node_df['X Position'].astype(float)
#             tip_node_df['Y Position'] = tip_node_df['Y Position'].astype(float)
#             print(tip_node_df.dtypes)
#
#             # Find the model name based on key file
#             with open("key_file.txt", 'r') as f_r:
#                 lines = f_r.readlines()
#                 for line in lines:
#                     curr_dict = ast.literal_eval(line)
#                     print("Dict Line")
#                     print(line)
#                     if curr_dict['key'] == int(model_name.strip('M')):
#                         geom_value = float(curr_dict['actuator']['geometry'][sweep_variable])
#                         print("geom value")
#                         print(geom_value)
#
#                         print("label")
#                         label = sweep_variable + '=' + str(geom_value)
#                         print(label)
#
#             tip_node_df.reset_index(inplace=True)
#             idmax = tip_node_df['Pressure'].argmax()
#             final_tip_position = tip_node_df.iloc[idmax]
#             print(final_tip_position)
#
#             # cmap = matplotlib.cm.get_cmap('viridis')
#             # lowpres = cmap(norm(min_value))
#             # highpres = cmap(norm(max_value))
#
#             cmap = plt.cm.viridis
#             normalize = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
#
#             print("Geom Value")
#             print(geom_value)
#             geom_value = math.degrees(geom_value)
#             cb = plt.scatter(final_tip_position['X Position'], final_tip_position['Y Position'], c=geom_value, s=15, cmap=cmap,
#                         norm=normalize,
#                         label=label)
#             plt.axis('equal')
#             plt.xlabel('X Position (mm)')
#             plt.ylabel('Y Position (mm)')
#             plt.ylim(-120, 25)
#             plt.xlim(-35, 110)
#             # plt.legend()
#             plt.title('Final Tip Position - Pressure at 100 kPa')
#         except:
#             pass
#
#     cbar = plt.colorbar(cb)
#     cbar.set_label('Theta (degrees)')
#     plt.show()
#
# def graph_bending_angle(model_list, sweep_variable, min_value, max_value):
#     for model_name in model_list:
#         print("model name")
#         print(model_name)
#         model_name = 'M' + str(model_name)
#         path_name = 'C:/Users/and008/Documents/Models/' + model_name + '.txt'
#
#         data = pd.DataFrame(columns=['X Position', 'Y Position'])
#         try:
#             with open(path_name, 'r') as fr:
#                 next_line = False
#
#                 for line in fr:
#
#                     # Extract pressure
#                     if next_line:
#                         current_pressure = float(line)
#                         next_line = False
#
#                     if 'Pressure' in line:
#                         next_line = True
#
#                     if ',' in line:
#                         coords = line.split(',')
#                         x_coord = float(coords[0])
#                         y_coord = float(coords[1].strip('\n'))
#                         node = int(coords[2])
#                         data = data.append({'X Position': coords[0], 'Y Position': coords[1], 'Node': node, 'Pressure':current_pressure}, ignore_index=True)
#
#             pressure_grouped = data.groupby(by=['Pressure'])
#
#             for key, item in pressure_grouped:
#                 if key == 0.0:
#                     tip_node = None
#                     tip_node_x = -10
#                     for i, row in item.iterrows():
#                         if float(row['X Position']) > tip_node_x:
#                             tip_node_x = float(row['X Position'])
#                             tip_node = int(row['Node'])
#
#             node_grouped = data.groupby(by=['Node'])
#
#             for key, item in node_grouped:
#                 if key == tip_node:
#                     tip_node_df = item
#
#             print(tip_node_df.dtypes)
#             tip_node_df['X Position'] = tip_node_df['X Position'].astype(float)
#             tip_node_df['Y Position'] = tip_node_df['Y Position'].astype(float)
#             print(tip_node_df.dtypes)
#
#             # Find the model name based on key file
#             with open("key_file.txt", 'r') as f_r:
#                 lines = f_r.readlines()
#                 for line in lines:
#                     curr_dict = ast.literal_eval(line)
#                     print("Dict Line")
#                     print(line)
#                     if curr_dict['key'] == int(model_name.strip('M')):
#                         geom_value = float(curr_dict['actuator']['geometry'][sweep_variable])
#                         print("geom value")
#                         print(geom_value)
#
#                         print("label")
#                         label = sweep_variable + '=' + str(geom_value)
#                         print(label)
#
#             tip_node_df.reset_index(inplace=True)
#             idmax = tip_node_df['Pressure'].argmax()
#             final_tip_position = tip_node_df.iloc[idmax]
#             print(final_tip_position)
#
#             # cmap = matplotlib.cm.get_cmap('viridis')
#             # lowpres = cmap(norm(min_value))
#             # highpres = cmap(norm(max_value))
#
#             cmap = plt.cm.viridis
#             normalize = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
#
#             print("Geom Value")
#             print(geom_value)
#
#             y_value = final_tip_position['Y Position']
#             x_value = final_tip_position['X Position']
#             # if x_value > 0:
#             #     bending_angle = math.degrees(math.atan(abs(x_value)/abs(y_value)))
#             # else:
#             #     bending_angle = math.degrees(math.atan(abs(x_value)/abs(y_value)) + 90.0
#             bending_angle = math.degrees(math.atan2(final_tip_position['Y Position'], final_tip_position['X Position']))
#             # bending_angle = abs(bending_angle)
#             # if x_value < 0:
#             #     bending_angle = 360.0  abs(bending_angle)
#             print("Y Value")
#             print(y_value)
#             print("X Value")
#             print(x_value)
#             print("Bending Angle")
#             raw_angle = math.degrees(math.atan(abs(x_value)/abs(y_value)))
#             if x_value < 0:
#                 raw_angle = raw_angle + 90.0
#             else:
#                 raw_angle = 90.0 - raw_angle
#             raw_angle = raw_angle*2.0
#             print(raw_angle)
#             print("Geom value")
#             print(geom_value)
#             plt.scatter(math.degrees(geom_value), raw_angle, c='blue', s=15)
#             # cb = plt.scatter(final_tip_position['X Position'], final_tip_position['Y Position'], c=geom_value, s=15, cmap=cmap,
#             #             norm=normalize,
#             #             label=label)
#             plt.xlabel('Theta (degrees)')
#             plt.ylabel('Bending Angle')
#             # plt.ylim(-120, 25)
#             # plt.xlim(-35, 110)
#             # # plt.legend()
#             plt.title('Inflation to 100 kPa')
#         except:
#             pass
#
#     # cbar = plt.colorbar(cb)
#     # cbar.set_label('Theta (degrees)')
#     plt.show()

# graph_tip_reachable_space(model_list, sweep_variable, 10, 20)
# graph_tip_endpoint(model_list, sweep_variable, 10, 20)

# graph_max_stress(model_list, sweep_variable, 1.6, 3.0)
# graph_bending_angle(model_list, sweep_variable, 1.6, 3.0)

# graph_strains(model_list)

# plt.show()
#print(data)

# plt.plot(two_seven_five[' X Position'], two_seven_five[' Y Position'], label = '2.75 T', color='blue')
# plt.plot(two_two_five[' X Position'], two_two_five[' Y Position'], label = '2.25 T', color='orange')
# plt.plot(two_zero_zero[' X Position'], two_zero_zero[' Y Position'], label = '2.00 T', color='purple')
# plt.plot(one_eight_zero[' X Position'], one_eight_zero[' Y Position'], label = '1,80 T', color='red')
# plt.plot(one_seven_zero[' X Position'], one_seven_zero[' Y Position'], label = '1,70 T', color='green')
#
#
# print(one_eight_zero.iloc[-1])
# plt.plot(two_seven_five.iloc[-1][' X Position'], two_seven_five.iloc[-1][' Y Position'], 'o', color='blue')
# plt.plot(two_two_five.iloc[-1][' X Position'], two_two_five.iloc[-1][' Y Position'], 'o', color='orange')
# plt.plot(two_zero_zero.iloc[-1][' X Position'], two_zero_zero.iloc[-1][' Y Position'], 'o', color='purple')
# plt.plot(one_eight_zero.iloc[-1][' X Position'], one_eight_zero.iloc[-1][' Y Position'], 'o', color='red')
# plt.plot(one_seven_zero.iloc[-1][' X Position'], one_seven_zero.iloc[-1][' Y Position'], 'o', color='green')
#
# plt.axis('equal')
# plt.xlabel('X Position (mm)')
# plt.ylabel('Y Position (mm)')
# plt.legend()
# plt.show()

# initial_coords = pd.read_csv(path_name + '/' + 'FirstM1S,170kpa.txt')
# final_coords = pd.read_csv(path_name + '/' + 'FinalM1S,170kpa.txt')
# initial_coords_ng = pd.read_csv(path_name + '/' + 'FirstM1SNG,170kpa.txt')
# final_coords_ng = pd.read_csv(path_name + '/' + 'FinalM1SNG,170kpa.txt')
# tip_sweep = pd.read_csv(path_name + '/' + 'TipSweepM1S,170kpa.txt')

# fig, ax = plt.subplots(1)
#
# cmap = matplotlib.cm.get_cmap('Spectral')
# lowpres = cmap(0.0)
# highpres = cmap(170.0)
#
# # # Pressure
#
# # #normalize item number values to colormap
# # norm = matplotlib.colors.Normalize(vmin=0, vmax=170)
# #
# # #colormap possible values = viridis, jet, spectral
# # lowpres = cmap(norm(0.0),bytes=True)
# # highpres = cmap(norm(170.0),bytes=True)
#
# # cmap = cm.get_cmap('Spectral')
# # # for i, row in tip_sweep.iterrows():
# #     print(row[' X Position'])
# colorbar = plt.scatter(tip_sweep[' X Position'], tip_sweep[' Y Position'], c=tip_sweep['Pressure'], vmin=0, vmax=170.0, cmap='viridis')
#
# norm = matplotlib.colors.Normalize(vmin=0, vmax=170)
#
# cmap = matplotlib.cm.get_cmap('viridis')
# lowpres = cmap(norm(0.0))
# highpres = cmap(norm(170.0))
#
#
# #
# plt.scatter(initial_coords[' X Position'], initial_coords[' Y Position'], color=lowpres)
# plt.scatter(final_coords[' X Position'], final_coords[' Y Position'], color=highpres)
#
# cbar = plt.colorbar(colorbar)
# cbar.set_label('Pressure (kPa)')
# plt.axis('equal')
# plt.xlabel('X Position (mm)')
# plt.ylabel('Y Position (mm)')
# plt.show()


# plt.plot(initial_coords_ng[' X Position'], initial_coords_ng[' Y Position'], 'o', color='red', label='0 kPa NG')
# plt.plot(final_coords_ng[' X Position'], final_coords_ng[' Y Position'], 's', color='red', label='170 kPa NG')
# plt.plot(initial_coords[' X Position'], initial_coords[' Y Position'], 'o', color='blue', label='0 kPa')
# plt.plot(final_coords[' X Position'], final_coords[' Y Position'], 's', color='blue', label='170 kPa')


# plt.legend()



# files_in_directory = os.listdir(path_name)
#
# initial_coords = pd.read_csv(path_name + '/' + 'M1S,0kpa.txt')
# initial_coords = initial_coords.sort_values(by=['Node Label'])
# print(initial_coords)
# plt.plot(initial_coords[' X Position'], initial_coords[' Y Position'], 'o', label= '0 kPa')
#
# pressure_list = []
#
# for file in files_in_directory:
#     if 'txt' in file:
#
#         pressure = float(file.split(',')[1].split('.')[0].split('k')[0])
#
#         actuator_name = file.split(',')[0]
#         print(actuator_name)
#
#         pressure_list.append(pressure)
#
# pressure_list.sort()
# print(pressure_list)
#
# for pressure in pressure_list:
#
#     print(pressure)
#     if pressure != 0.0:
#         data = pd.read_csv(path_name + '/' + actuator_name + ',' + str(int(pressure)) + 'kpa.txt')
#         data = data.sort_values(by=['Node Label'])
#         data[' X Position'] = data[' X Position'] + initial_coords[' X Position']
#         data[' Y Position'] = data[' Y Position'] + initial_coords[' Y Position']
#         plt.plot(data[' X Position'], data[' Y Position'], 'o', label=str(pressure) + ' kPa')
# #
# # plt.plot(initial_coords[' X Position'], initial_coords[' Y Position'], 'o', label= '0 kPa')
# # # plt.ylim([-100, 10])
# plt.axis('equal')
# plt.xlabel('X Position (mm)')
# plt.ylabel('Y Position (mm)')
# plt.legend()
# plt.show()