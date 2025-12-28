import matplotlib.pyplot as plt
import pandas as pd
import ast

df = pd.DataFrame(columns=['t', 'loss', 'generation'])
with open('N3ThicknessOptimization.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    print(line)
    line = ast.literal_eval(line)
    print(line['t'])
    thickness = line['t']*(3.0-1.6) + 1.6
    df = df.append({'t': thickness, 'loss': line['#loss'], 'generation': line['#generation']}, ignore_index=True)
print(df)

groups = df.groupby(by='generation')

groups_list = []
for group in groups:
    groups_list.append(group)

print("Groups List")
print(groups_list)
groups_list.remove(groups_list[-1])

fig, axs = plt.subplots(3,3, figsize=(11,8))

i = 0
for axs in axs.reshape(-1):
    axs.plot(df['t'], df['loss'], '.')
    # current_df = groups_list[i][1]
    # gen_name = groups_list[i][0]
    print("Here")
    print(i)
    print(groups_list[i][1])
    axs.set_title("Generation "+ str(i + 1))
    axs.plot(groups_list[i][1]['t'], groups_list[i][1]['loss'], '.', color = 'red')
    i += 1
    axs.set_ylabel('Strain [-]')
    axs.set_xlabel('Thickness [mm]')
plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.7, wspace=0.32, left=0.1, right=0.95)
plt.savefig('FirstCMA.png')
plt.show()