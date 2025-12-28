import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import ast
from ProcessedModels2 import *

df_files = pd.DataFrame(columns = ["grasping_strength", "strain", "folder_name", "time"])

### The part for loading stuff from individual text files
for item in os.listdir('C:/Users/and008/Documents/cma_baseline'):
    if "txt" not in item:
        if "csv" not in item:
            # time = os.path.getctime(item)
            # time = os.stat(item).st_birthtime
            time = os.path.getmtime('C:/Users/and008/Documents/cma_baseline/' + item)
            # for sub_item in os.listdir(item):
            try:
                ProcessedModel1 = ProcessedModel('C:/Users/and008/Documents/cma_baseline/' + item + '/' + 'model')
                ProcessedModel1.load_data()
                print("Trying to extract data")
                ProcessedModel1.compute_grasping_force()
                return_value, strain_value, force_value = ProcessedModel1.strain_force_loss()
            except:
                print("This geometry was not feasible")
                # this catches the case where the geometry doesn't generate
                return_value = float('inf')
                strain_value = float('inf')
                force_value = float('inf')

            # Append all of the values
            df_files = df_files.append({'grasping_strength': force_value, 'strain': strain_value, 'folder_name': item, 'time': time},
                                   ignore_index=True)
print("Final Folder Df")
print(df_files)
df_files.sort_values(by=["time"], inplace=True, ascending=False)
print("Data Types")
print(df_files.dtypes)
print(df_files)
df_files.to_csv("baseline_file.csv")
# #######################

# df = pd.DataFrame(columns=['loss', 'generation', 't', 'b_core', 'r_core', 'w_extrude', 'wa_core', 'alpha'])
# with open('Test1.txt', 'r') as f:
#     lines = f.readlines()
#
# for line in lines:
#     line = ast.literal_eval(line)
#     # print(line['t'])
#     t = line['t']
#     b_core = line['b_core']
#     r_core = line['r_core']
#     w_extrude = line['w_extrude']
#     alpha = line['alpha']
#     wa_core = line['wa_core']
#
#     df = df.append({'loss': line['#loss'], 'generation': line['#generation'], 't':t, 'b_core': b_core, 'r_core':r_core, 'w_extrude':w_extrude, 'alpha':alpha, 'wa_core':wa_core}, ignore_index=True)
#
# print(df)
# df = df.iloc[::-1]
# print(df)
# df.to_csv('loss_file.csv')








# Dataframe should have loss, strain, grasping force, folder name, id number, and value of each parameter


# df = pd.DataFrame(columns = ["loss", "strength", "strain"])
# for item in os.listdir('D:/CMA Thru 2-7'):
#     if (item != "GraphTest.txt") and (item != "Test1.txt") and (item != "actuator_optimization"):
#         print(item)
#         try:
#             # Load the exported data into a dataframe
#             print("Starting model processing")
#             ProcessedModel1 = ProcessedModel('D:/CMA Thru 2-7/' + item + '/model')
#             ProcessedModel1.load_data()
#             print("Trying to extract data")
#             ProcessedModel1.compute_grasping_force()
#             return_value, strain_value, force_value = ProcessedModel1.strain_force_loss()
#         except:
#             print("Geometry not feasible")
#             # this catches the case where the geometry doesn't generate
#             return_value = float('inf')
#             strain_value = float('inf')
#             force_value = float('inf')
#         df = df.append({'loss': return_value, 'strength': force_value, 'strain': strain_value},
#                        ignore_index=True)
#         # path_parent = os.path.dirname(os.getcwd())
#         # os.chdir(path_parent)
#
# print("FINAL DF")
# print(df)
#
# groups = df.groupby(by='generation')
#
# groups_list = []
# for group in groups:
#     groups_list.append(group)
#
# print("Groups List")
# print(groups_list)
# groups_list.remove(groups_list[-1])
#
# fig, axs = plt.subplots(3,3, figsize=(11,8))
#
# i = 0
# for axs in axs.reshape(-1):
#     axs.plot(df['t'], df['loss'], '.')
#     # current_df = groups_list[i][1]
#     # gen_name = groups_list[i][0]
#     print("Here")
#     print(i)
#     print(groups_list[i][1])
#     axs.set_title("Generation "+ str(i + 1))
#     axs.plot(groups_list[i][1]['t'], groups_list[i][1]['loss'], '.', color = 'red')
#     i += 1
#     axs.set_ylabel('Strain [-]')
#     axs.set_xlabel('Thickness [mm]')
# plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.7, wspace=0.32, left=0.1, right=0.95)
# plt.savefig('FirstCMA.png')
# plt.show()