import matplotlib.pyplot as plt
import pandas as pd
import ast

df = pd.DataFrame(columns=['loss', 'sim_num', 'generation', 't', 'w_extrude', 'wa_core', 'alpha', 'b_core', 'r_core'])
with open('GraphTest.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    # print(line)
    line = ast.literal_eval(line)
    # print(line['t'])
    # thickness = line['t']*(3.0-1.6) + 1.6
    X = [line['t'], line['r_core'], line['b_core'], line['wa_core'], line['w_extrude'], line['alpha']]
    simID = hash(tuple(X))
    df = df.append(
        {'loss': line['#loss'], 'generation': line['#generation'], 'sim_num': simID, 't': line['t'],
         'w_extrude': line['w_extrude'], 'wa_core': line['wa_core'], 'alpha': line['alpha'], 'b_core': line['b_core'],
         'r_core': line['r_core']}, ignore_index=True)

print(df)

df2 = df.iloc[:-7, :]
print(df2)
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