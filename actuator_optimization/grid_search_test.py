from GridSearch import *

GridSearch1 = GridSearch([
    {'parameter': 'n', 'lb': 5, 'ub': 7, 'num': 3},
    {'parameter': 'w_extrude', 'lb': 2, 'ub': 5, 'num': 4}])

print(GridSearch1.master_list)
print(GridSearch1.master_parameters)
print(GridSearch1.all_combos)

for i in range(12):
    GridSearch1.generate_params()
    print(GridSearch1.current_parameters)
    print(GridSearch1.last_experiment)