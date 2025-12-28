import matplotlib.pyplot as plt
import pandas as pd
import os

# path_name = '/Users/annie/Documents/Thesis/M1/4.9 Mesh'
path_name = '/Users/annie/Documents/Thesis/M1/Experiments/Working Space'

# files_in_directory = os.listdir(path_name)

initial_coords = pd.read_csv(path_name + '/' + 'M1S,0kpa.txt')
initial_coords = initial_coords.sort_values(by=['Node Label'])
print(initial_coords)
tip = initial_coords[initial_coords['Node Label'] == 1]
tipX = tip[' X Position'].iloc[0]
tipY = tip[' Y Position'].iloc[0]
#
# plt.plot(initial_coords[' X Position'], initial_coords[' Y Position'], 'o', label= '0 kPa')

data = pd.read_csv(path_name + '/' + 'M1S,170kpa.txt')
for i, row in data.iterrows():
    tip_x_disp = row[' X Position']
    tip_y_disp = row[' Y Position']
    tip_x_position = tip_x_disp + tipX
    tip_y_position = tip_y_disp + tipY
    plt.plot([tip_x_position], [tip_y_position], 'o', color='red')
    # print(tip_x_initial)
    # print(tip_y_initial)

# data0 = pd.read_csv(path_name + '/' + 'M1S,170kpafull.txt')
# print(data0)
# data0 = data0.sort_values(by=['Node Label'])
# print(data0.columns)
# data0[' X Position'] = -(data0[' X Position'] - initial_coords[' X Position'])
# data0[' Y Position'] = data0[' Y Position'] - initial_coords[' Y Position']
# plt.plot(data0[' X Position'], data0[' Y Position'], 'o')


plt.axis('equal')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.show()



# # data = data.sort_values(by=['Node Label'])
# data[' X Position'] = data[' X Position'] + initial_coords[' X Position']
# data[' Y Position'] = data[' Y Position'] + initial_coords[' Y Position']
#
# for row in data.iterrows():
#     print(row)




# plt.plot(data[' X Position'], data[' Y Position'], 'o', label=file)

# pressure_list = []

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

# for file in files_in_directory:
#     data = pd.read_csv(path_name + '/' + file)
#     print(data)
#     data = data.sort_values(by=['Node Label'])
#     data[' X Position'] = data[' X Position'] + initial_coords[' X Position']
#     data[' Y Position'] = data[' Y Position'] + initial_coords[' Y Position']
#     plt.plot(data[' X Position'], data[' Y Position'], 'o', label=file)
# #
# # plt.plot(initial_coords[' X Position'], initial_coords[' Y Position'], 'o', label= '0 kPa')
# # # plt.ylim([-100, 10])
# plt.axis('equal')
# plt.xlabel('X Position (mm)')
# plt.ylabel('Y Position (mm)')
# plt.legend()
# plt.show()