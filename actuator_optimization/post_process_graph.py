import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

pd.set_option('display.max_rows', None)
df = pd.read_csv('loss_file.csv')
df_grouped = df.groupby(by='generation')

def get_cmap(n, name='seismic'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
N = 23
cmap = get_cmap(N)

# # All generations on one graph
# for generation, little_df in df_grouped:
#     print(generation)
#     print(little_df)
#     plt.plot(little_df['grasping_strength'], little_df['strain'], '.', color=cmap(generation - 1))
#     plt.xlabel('Grasping Force [N]')
#     plt.ylabel('Strain [-]')
# plt.show()

#### MULTI GENERATION PLOT ####
# colormap = plt.cm.bwr #or any other colormap
# without_infinity = df[df['loss'] < 200]
# normalize = matplotlib.colors.Normalize(vmin=-0.5, vmax=without_infinity['loss'].max())
# row_num = 6
# col_num = 4
# fig, axs = plt.subplots(row_num, col_num, sharex=True, sharey=True)
# tuples = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3),(5, 0), (5, 1), (5, 2), (5, 3)]
# for generation, little_df in df_grouped:
#     if generation < 23:
#         little_df_to_plot = little_df[little_df['loss'] < 200]
#         for i, point in little_df_to_plot.iterrows():
#             axs[tuples[generation-1][0], tuples[generation-1][1]].plot(point['grasping_strength'], point['strain'], '.', c= plt.cm.plasma(normalize(point['loss'])))
#         axs[tuples[generation-1][0], tuples[generation-1][1]].set_xlim(0, 7)
#         axs[tuples[generation - 1][0], tuples[generation - 1][1]].set_ylim(0, 1.5)
#         # axs[tuples[generation - 1][0], tuples[generation - 1][1]].invert_xaxis()
#         axs[tuples[generation - 1][0], tuples[generation - 1][1]].set_title('Generation ' + str(generation))
# fig.text(0.5, 0.04, 'Grasping Force (GF) [N]', ha='center')
# fig.text(0.04, 0.5, 'Max Strain ($\epsilon_m$) [-]', va='center', rotation='vertical')
# fig.set_size_inches(7.5, 7.5)
# fig.tight_layout(rect=[0.05,0.05,0.95,0.95])
# fig.delaxes(axs[5,3])
# fig.delaxes(axs[5,2])
# cmap = matplotlib.cm.ScalarMappable(norm=normalize, cmap=matplotlib.cm.plasma)
# cmap.set_array([])
# fig.colorbar(cmap, ax=axs[5, 2:4], orientation='horizontal', pad=-10.0, label='Cost C = 0.5*$\epsilon_m$ - 0.5*GF')
# axs[5,2].spines['top'].set_visible(False)
# axs[5,2].spines['right'].set_visible(False)
# axs[5,2].spines['bottom'].set_visible(False)
# axs[5,2].spines['left'].set_visible(False)
# axs[5,3].spines['top'].set_visible(False)
# axs[5,3].spines['right'].set_visible(False)
# axs[5,3].spines['bottom'].set_visible(False)
# axs[5,3].spines['left'].set_visible(False)
# plt.savefig('multiobjective_cma_generations.png')
# plt.show()
############################

#### OVERALL COST FUNCTION PLOT ####
# # Best fit line
# df_no_inf = df[df['loss'] < 200]
# print(df_no_inf)
# regr = linear_model.LinearRegression()
# regr.fit(df_no_inf[['Unnamed: 0']], df_no_inf['loss'])
# prediction_results = regr.predict(df_no_inf[['Unnamed: 0']])
# baseline_cost_value = 0.2240224981508615
#
# plt.plot(df_no_inf['Unnamed: 0'], df_no_inf['loss'], '.', label='Candidate')
# plt.axhline(y = baseline_cost_value, xmin=0.045, xmax=0.96, c='green', label='Baseline Design')
# plt.plot(df_no_inf['Unnamed: 0'], prediction_results, color='red', label='Best Fit Line')
# plt.xlabel('Actuator Candidate Number')
# plt.ylabel('Cost')
# plt.legend()
# plt.savefig('cost_multi_cma.png')
# plt.show()
####################################

#### OVERALL COST FUNCTION PLOT ####
# df_no_inf = df[df['loss'] < 200]
# print(df_no_inf)
# regr = linear_model.LinearRegression()
# regr.fit(df_no_inf[['Unnamed: 0']], df_no_inf['strain'])
#
# prediction_results = regr.predict(df_no_inf[['Unnamed: 0']])
# plt.plot(df_no_inf['Unnamed: 0'], df_no_inf['strain'], '.', label='Candidate')
# baseline_strain_value = 1.189437032
# baseline_grasping_force_value = 0.741392035
#
# plt.axhline(y = baseline_strain_value, xmin=0.045, xmax=0.96, c='green', label='Baseline Design')
# plt.plot(df_no_inf['Unnamed: 0'], prediction_results, color='red', label='Best Fit Line')
# plt.xlabel('Actuator Candidate Number')
# plt.ylabel('Max Strain [-]')
# plt.legend()
# plt.savefig('strain_multi_cma.png')
# plt.show()
######################################

#### OVERALL COST FUNCTION PLOT ####
# df_no_inf = df[df['loss'] < 200]
# print(df_no_inf)
# regr = linear_model.LinearRegression()
# regr.fit(df_no_inf[['Unnamed: 0']], df_no_inf['grasping_strength'])
#
# prediction_results = regr.predict(df_no_inf[['Unnamed: 0']])
# plt.plot(df_no_inf['Unnamed: 0'], df_no_inf['grasping_strength'], '.', label='Candidate')
# # baseline_strain_value = 1.189437032
# baseline_grasping_force_value = 0.741392035
#
# plt.axhline(y = baseline_grasping_force_value, xmin=0.045, xmax=0.96, c='green', label='Baseline Design')
# plt.plot(df_no_inf['Unnamed: 0'], prediction_results, color='red', label='Best Fit Line')
# plt.xlabel('Actuator Candidate Number')
# plt.gca().invert_yaxis()
# plt.ylabel('Grasping Force [N]')
# plt.legend()
# plt.savefig('grasping_strength_multi_cma.png')
# plt.show()
######################################

##### SPIDER PLOT FOR PARAMETERS ########
# fig, axs = plt.subplots(6, 4, subplot_kw=dict(projection='polar'), figsize = (20, 12))
# results_df = df[df['loss'] < 200]
# min_value = results_df['loss'].min()
# max_value = results_df['loss'].max()
# cmap = plt.cm.plasma
# normalize = plt.Normalize(vmin=-0.5, vmax=max_value)
# label_loc = np.linspace(start=0, stop=2.0* np.pi, num=7)
# print("Labels")
# print(label_loc)
# parameters = ['T', 'BC', 'RC', 'WA', 'WE', 'Alpha', 'T']
#
# tuples = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3),(5, 0), (5, 1), (5, 2), (5, 3)]
# for generation, little_df in df_grouped:
#     if generation < 23:
#         little_df_to_plot = little_df[little_df['loss'] < 200]
#         for i, point in little_df_to_plot.iterrows():
#             candidate = [point['t'], point['b_core'], point['r_core'], point['wa_core'], point['w_extrude'], point['alpha'], point['t']]
#             axs[tuples[generation-1][0], tuples[generation-1][1]].plot(label_loc, candidate, color=cmap(normalize(point['loss'])))
#             axs[tuples[generation-1][0], tuples[generation-1][1]].set_yticklabels([])
#             axs[tuples[generation-1][0], tuples[generation-1][1]].set_xticklabels([])
#             axs[tuples[generation-1][0], tuples[generation-1][1]].set_thetagrids(np.degrees(label_loc), labels=parameters, fontsize=8)
#             axs[tuples[generation-1][0], tuples[generation-1][1]].set_ylim([0.0, 1.0])
#             axs[tuples[generation - 1][0], tuples[generation - 1][1]].set_title('Generation ' + str(generation))
#             # axs[tuples[generation - 1][0], tuples[generation - 1][1]].set_yticklabels(['0.5', '1.0'])
# # for spine in axs[5,2].spines:
# #     axs[5,2].spines[spine].set_visible(False)
# # # axs[5,2].spines['bottom'].set_visible(False)
# # # axs[5,2].spines['left'].set_visible(False)
# # # axs[5,3].spines['top'].set_visible(False)
# # # axs[5,3].spines['right'].set_visible(False)
# # # axs[5,3].spines['bottom'].set_visible(False)
# # # axs[5,3].spines['left'].set_visible(False)
# fig.set_size_inches(7.5, 10.5)
# fig.delaxes(axs[5,3])
# fig.delaxes(axs[5,2])
#
# # RdYlBu_r
# cmap = matplotlib.cm.ScalarMappable(norm=normalize, cmap=matplotlib.cm.plasma)
# cmap.set_array([])
# fig.colorbar(cmap, ax=axs[5, 2:4], orientation='horizontal', pad=-10.0, label='Cost')
# fig.tight_layout()
# plt.savefig('spider_plots.png')
# plt.show()
##########################################################

#### PLOTS OF INDIVIDUAL PARAMETERS ####
# df_no_inf = df[df['loss'] < 200]
# print(df_no_inf)
# regr = linear_model.LinearRegression()
# regr.fit(df_no_inf[['Unnamed: 0']], df_no_inf['alpha'])
#
# prediction_results = regr.predict(df_no_inf[['Unnamed: 0']])
# plt.plot(df_no_inf['Unnamed: 0'], df_no_inf['alpha'], '.', label='Candidate')
# # baseline_strain_value = 1.189437032
# baseline_grasping_force_value = 0.741392035
#
# plt.axhline(y = baseline_grasping_force_value, xmin=0.045, xmax=0.96, c='green', label='Baseline Design')
# plt.plot(df_no_inf['Unnamed: 0'], prediction_results, color='red', label='Best Fit Line')
# plt.xlabel('Actuator Candidate Number')
# # plt.gca().invert_yaxis()
# plt.ylabel('Alpha Normalized')
# plt.legend()
# plt.savefig('alpha_normalized.png')
# plt.show()
######################################

##### SINGLE GENERATION SPIDER PLOTS ########
# target_generation = 22
# fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection='polar'), figsize = (20, 12))
# results_df = df[df['loss'] < 200]
# min_value = results_df['loss'].min()
# max_value = results_df['loss'].max()
# cmap = plt.cm.plasma
# normalize = plt.Normalize(vmin=-0.5, vmax=max_value)
# label_loc = np.linspace(start=0, stop=2.0* np.pi, num=7)
# print("Labels")
# print(label_loc)
# parameters = ['T', 'BC', 'RC', 'WA', 'WE', 'Alpha', 'T']
#
# tuples = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3),(5, 0), (5, 1), (5, 2), (5, 3)]
# for generation, little_df in df_grouped:
#     if generation < 23:
#         little_df_to_plot = little_df[little_df['loss'] < 200]
#         for i, point in little_df_to_plot.iterrows():
#             if generation == target_generation:
#                 candidate = [point['t'], point['b_core'], point['r_core'], point['wa_core'], point['w_extrude'], point['alpha'], point['t']]
#                 axs.plot(label_loc, candidate, color=cmap(normalize(point['loss'])))
#                 axs.set_yticklabels([])
#                 axs.set_xticklabels([])
#                 axs.set_thetagrids(np.degrees(label_loc), labels=parameters, fontsize=8)
#                 axs.set_ylim([0.0, 1.0])
#                 axs.set_title('Generation ' + str(generation))
#             # axs[tuples[generation - 1][0], tuples[generation - 1][1]].set_yticklabels(['0.5', '1.0'])
# # for spine in axs[5,2].spines:
# #     axs[5,2].spines[spine].set_visible(False)
# # # axs[5,2].spines['bottom'].set_visible(False)
# # # axs[5,2].spines['left'].set_visible(False)
# # # axs[5,3].spines['top'].set_visible(False)
# # # axs[5,3].spines['right'].set_visible(False)
# # # axs[5,3].spines['bottom'].set_visible(False)
# # # axs[5,3].spines['left'].set_visible(False)
# fig.set_size_inches(7.5, 10.5)
# # fig.delaxes(axs[5,3])
# # fig.delaxes(axs[5,2])
#
# # RdYlBu_r
# cmap = matplotlib.cm.ScalarMappable(norm=normalize, cmap=matplotlib.cm.plasma)
# cmap.set_array([])
# fig.colorbar(cmap, ax=axs, orientation='horizontal', pad=-10.0, label='Cost')
# fig.tight_layout()
# plt.savefig('cmap.png')
# plt.show()
##########################################################

#### INDIVIDUAL DUAL OBJECTIVE PLOTS ####
target_generation = 1
colormap = plt.cm.bwr #or any other colormap
without_infinity = df[df['loss'] < 200]
normalize = matplotlib.colors.Normalize(vmin=-0.5, vmax=without_infinity['loss'].max())
row_num = 4
col_num = 1
fig, axs = plt.subplots(row_num, col_num, sharex=True, sharey=True)
tuples = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3),(5, 0), (5, 1), (5, 2), (5, 3)]

for generation, little_df in df_grouped:
    if generation < 23:
        if generation == 1:
            little_df_to_plot = little_df[little_df['loss'] < 200]
            for i, point in little_df_to_plot.iterrows():
                axs[0].plot(point['grasping_strength'], point['strain'], '.', markersize=20, c= plt.cm.plasma(normalize(point['loss'])))
            axs[0].set_xlim(-0.5, 7)
            axs[0].set_ylim(0, 1.6)
            axs[0].set_title('Cost C = 0.5*$\epsilon_m$ - 0.5*GF')
            # axs[tuples[generation - 1][0], tuples[generation - 1][1]].invert_xaxis()
            # axs[0].set_title('Generation ' + str(generation))

        if generation == 7:
            little_df_to_plot = little_df[little_df['loss'] < 200]
            for i, point in little_df_to_plot.iterrows():
                axs[1].plot(point['grasping_strength'], point['strain'], '.', markersize=20, c= plt.cm.plasma(normalize(point['loss'])))
            axs[1].set_xlim(-0.5, 7)
            axs[1].set_ylim(0, 1.6)
            axs[1].set_title(' ')
            # axs[tuples[generation - 1][0], tuples[generation - 1][1]].invert_xaxis()
            # axs[1].set_title('Generation ' + str(generation))

        if generation == 14:
            little_df_to_plot = little_df[little_df['loss'] < 200]
            for i, point in little_df_to_plot.iterrows():
                axs[2].plot(point['grasping_strength'], point['strain'], '.', markersize=20, c= plt.cm.plasma(normalize(point['loss'])))
            axs[2].set_xlim(-0.5, 7)
            axs[2].set_ylim(0, 1.6)
            axs[2].set_title(' ')
            # axs[tuples[generation - 1][0], tuples[generation - 1][1]].invert_xaxis()
            # axs[2].set_title('Generation ' + str(generation))

        if generation == 21:
            little_df_to_plot = little_df[little_df['loss'] < 200]
            for i, point in little_df_to_plot.iterrows():
                axs[3].plot(point['grasping_strength'], point['strain'], '.', markersize=20, c= plt.cm.plasma(normalize(point['loss'])))
            axs[3].set_xlim(-0.5, 7)
            axs[3].set_ylim(0, 1.6)
            # axs[tuples[generation - 1][0], tuples[generation - 1][1]].invert_xaxis()
            axs[3].set_title(' ')
            # axs[3].set_title('Generation ' + str(generation))

fig.text(0.5, 0.04, 'Grasping Force (GF) [N]', ha='center')
fig.text(0.04, 0.5, 'Max Strain ($\epsilon_m$) [-]', va='center', rotation='vertical')
fig.set_size_inches(3.25, 9)
fig.tight_layout(rect=[0.05,0.05,0.95,0.95])
axs[0].tick_params(axis='y', labelsize=14)
axs[0].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].tick_params(axis='x', labelsize=14)
axs[2].tick_params(axis='y', labelsize=14)
axs[2].tick_params(axis='x', labelsize=14)
axs[3].tick_params(axis='y', labelsize=14)
axs[3].tick_params(axis='x', labelsize=14)
# fig.delaxes(axs[5,3])
# fig.delaxes(axs[5,2])
cmap = matplotlib.cm.ScalarMappable(norm=normalize, cmap=matplotlib.cm.plasma)
cmap.set_array([])
fig.colorbar(cmap, ax=axs, orientation='horizontal', pad=-10.0, label='Cost C = 0.5*$\epsilon_m$ - 0.5*GF')
# axs[5,2].spines['top'].set_visible(False)
# axs[5,2].spines['right'].set_visible(False)
# axs[5,2].spines['bottom'].set_visible(False)
# axs[5,2].spines['left'].set_visible(False)
# axs[5,3].spines['top'].set_visible(False)
# axs[5,3].spines['right'].set_visible(False)
# axs[5,3].spines['bottom'].set_visible(False)
# axs[5,3].spines['left'].set_visible(False)
plt.savefig('cost_text.png')
plt.show()
############################

##### FOUR GENERATION SPIDER PLOTS ########
# target_generation = 22
# fig, axs = plt.subplots(4, 1, subplot_kw=dict(projection='polar'), figsize = (20, 12))
# results_df = df[df['loss'] < 200]
# min_value = results_df['loss'].min()
# max_value = results_df['loss'].max()
# cmap = plt.cm.plasma
# normalize = plt.Normalize(vmin=-0.5, vmax=max_value)
# label_loc = np.linspace(start=0, stop=2.0* np.pi, num=7)
# print("Labels")
# print(label_loc)
# parameters = ['T', 'BC', 'RC', 'WA', 'WE', 'Alpha', 'T']
# arial_font = {'fontname': 'Arial'}
#
# tuples = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3),(5, 0), (5, 1), (5, 2), (5, 3)]
# for generation, little_df in df_grouped:
#     if generation < 23:
#         little_df_to_plot = little_df[little_df['loss'] < 200]
#         for i, point in little_df_to_plot.iterrows():
#             if generation == 1:
#                 candidate = [point['t'], point['b_core'], point['r_core'], point['wa_core'], point['w_extrude'], point['alpha'], point['t']]
#                 axs[0].plot(label_loc, candidate, color=cmap(normalize(point['loss'])))
#                 axs[0].set_yticklabels([])
#                 axs[0].set_xticklabels([])
#                 axs[0].set_thetagrids(np.degrees(label_loc), labels=parameters, fontsize=14, fontname="Arial")
#                 axs[0].set_ylim([0.0, 1.0])
#                 axs[0].set_title(' ')
#                 # axs[0].set_title('Generation ' + str(generation))
#             if generation == 7:
#                 candidate = [point['t'], point['b_core'], point['r_core'], point['wa_core'], point['w_extrude'], point['alpha'], point['t']]
#                 axs[1].plot(label_loc, candidate, color=cmap(normalize(point['loss'])))
#                 axs[1].set_yticklabels([])
#                 axs[1].set_xticklabels([])
#                 axs[1].set_thetagrids(np.degrees(label_loc), labels=parameters, fontsize=14, fontname="Arial")
#                 axs[1].set_ylim([0.0, 1.0])
#                 axs[1].set_title(' ')
#                 # axs[1].set_title('Generation ' + str(generation))
#             if generation == 14:
#                 candidate = [point['t'], point['b_core'], point['r_core'], point['wa_core'], point['w_extrude'], point['alpha'], point['t']]
#                 axs[2].plot(label_loc, candidate, color=cmap(normalize(point['loss'])))
#                 axs[2].set_yticklabels([])
#                 axs[2].set_xticklabels([])
#                 axs[2].set_thetagrids(np.degrees(label_loc), labels=parameters, fontsize=14, fontname="Arial")
#                 axs[2].set_ylim([0.0, 1.0])
#                 axs[2].set_title(' ')
#                 # axs[2].set_title('Generation ' + str(generation))
#             if generation == 21:
#                 candidate = [point['t'], point['b_core'], point['r_core'], point['wa_core'], point['w_extrude'], point['alpha'], point['t']]
#                 axs[3].plot(label_loc, candidate, color=cmap(normalize(point['loss'])))
#                 axs[3].set_yticklabels([])
#                 axs[3].set_xticklabels([])
#                 axs[3].set_thetagrids(np.degrees(label_loc), labels=parameters, fontsize=14, fontname="Arial")
#                 axs[3].set_ylim([0.0, 1.0])
#                 axs[3].set_title(' ')
#                 # axs[3].set_title('Generation ' + str(generation))
#             # axs[tuples[generation - 1][0], tuples[generation - 1][1]].set_yticklabels(['0.5', '1.0'])
# # for spine in axs[5,2].spines:
# #     axs[5,2].spines[spine].set_visible(False)
# # # axs[5,2].spines['bottom'].set_visible(False)
# # # axs[5,2].spines['left'].set_visible(False)
# # # axs[5,3].spines['top'].set_visible(False)
# # # axs[5,3].spines['right'].set_visible(False)
# # # axs[5,3].spines['bottom'].set_visible(False)
# # # axs[5,3].spines['left'].set_visible(False)
# fig.set_size_inches(3.25, 9)
# # fig.delaxes(axs[5,3])
# # fig.delaxes(axs[5,2])
#
# # RdYlBu_r
# cmap = matplotlib.cm.ScalarMappable(norm=normalize, cmap=matplotlib.cm.plasma)
# cmap.set_array([])
# # fig.colorbar(cmap, ax=axs, orientation='horizontal', pad=-10.0, label='Cost')
# fig.tight_layout(rect=[0.05,0.05,0.95,0.95])
# plt.savefig('four_generations_spider_plots.png')
# plt.show()
##########################################################