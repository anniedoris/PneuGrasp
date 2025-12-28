import os
import nevergrad as ng
from concurrent import futures
from ModifiedParametersLogger import *
import math
import shutil
import subprocess
from ProcessedModels import *
import ast
import pandas as pd

experiment_name = 'RestartTest3'
folder_name = 'cma_opt/restart3'
# past_experiments = 'LongRun9_1_23.txt' #For inoculating old experiments, set as None if there aren't any
past_experiments = None

# Define the parameters
instrum = ng.p.Instrumentation(
            t = ng.p.Scalar(init=0.9016326716541065, lower=0.0, upper=1.0),
            r_core = ng.p.Scalar(init=0.5729380313717183, lower=0.0, upper=1.0),
            b_core = ng.p.Scalar(init=0.5316614744861082, lower=0.0, upper=1.0),
            wa_core = ng.p.Scalar(init=0.3398175247762982, lower=0.0, upper=1.0),
            w_extrude = ng.p.Scalar(init=0.8179732452475367, lower=0.0, upper=1.0),
            alpha = ng.p.Scalar(init=0.516958502478302, lower=0.0, upper=1.0)
        )
budget = 10000
num_workers = 20

# Unnormalize the parameters
def unnormalize(X, min_values, max_values):
    Xnew = []
    for i, value in enumerate(X):
        xunorm = value * (max_values[i] - min_values[i]) + min_values[i]
        Xnew.append(xunorm)
    return Xnew

# Specify geometric constraints
def constraint4(params):
    X = [params[1]['t'], params[1]['r_core'], params[1]['b_core'], params[1]['wa_core']]
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5], [3.5, 10.0, 6.0, 9.0])
    print("Constraint4")
    print(X)
    print(unnormX)
    if unnormX[3] >= unnormX[0]:
        return False

def constraint5(params):
    X = [params[1]['t'], params[1]['r_core'], params[1]['b_core'], params[1]['wa_core']]
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5], [3.5, 10.0, 6.0, 9.0])
    print("Constraint5")

    if 0.5 < unnormX[2]:
        return True
    else:
        return False

def constraint7(params):
    X = [params[1]['t'], params[1]['r_core'], params[1]['b_core'], params[1]['wa_core']]
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5], [3.5, 10.0, 6.0, 9.0])
    print("Constraint7")

    if unnormX[0] < unnormX[2] + unnormX[1]:
        return True
    else:
        return False

def constraint8(params):
    X = [params[1]['t'], params[1]['r_core'], params[1]['b_core'], params[1]['wa_core']]
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5], [3.5, 10.0, 6.0, 9.0])
    print("Constraint8")

    if unnormX[1] < 12.37 - (0.5-0.5*math.sin(math.radians(7.125))):
        return True
    else:
        return False

def constraint9(params):
    X = [params[1]['t'], params[1]['r_core'], params[1]['b_core'], params[1]['wa_core']]
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5], [3.5, 10.0, 6.0, 9.0])
    print("Constraint9")

    if 13.65 < 12.37 + unnormX[2]:
        return True
    else:
        return False

def constraintabaqus(params):
    X = [params[1]['t'], params[1]['r_core'], params[1]['b_core'], params[1]['wa_core']]
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5], [3.5, 10.0, 6.0, 9.0])
    print("Constraint9")

    if 13.65 < 12.37 + unnormX[2]:
        return True
    else:
        return False

# instrum.register_cheap_constraint(constraint4)
# instrum.register_cheap_constraint(constraint5)
# instrum.register_cheap_constraint(constraint7)
# instrum.register_cheap_constraint(constraint8)
# instrum.register_cheap_constraint(constraint9)

#num_workers is 6

# Define the loss function
# def loss_function(t, rand):
#     X = [t, w_extrude]
#     simID = hash(tuple(X))
#     print("SimID:" + str(simID))
#     return (t - 0.5) ** 2

# Define the loss function, where abaqus operations are called

def loss(t, r_core, b_core, wa_core, w_extrude, alpha):

    # Get the unique ID number
    X = [t, r_core, b_core, wa_core, w_extrude, alpha]
    simID = hash(tuple(X))

    # Create the directory for this simulation
    os.mkdir(str(simID))

    # print("Normalized")
    # print(X)
    # print("Unnormalized")
    unnormX = unnormalize(X, [1.5, 0.0, 0.0, 1.5, 10.0, 1.0], [3.5, 10.0, 6.0, 9.0, 25.0, 15.0])
    # print(unnormX)

    # Generate the set of parameters we are looking for
    parameters = [{'parameter': 't', 'value': unnormX[0]},
                  {'parameter': 'r_core', 'value': unnormX[1]},
                  {'parameter': 'b_core', 'value': unnormX[2]},
                  {'parameter': 'h_core', 'value': 12.37},
                  {'parameter': 'h_upper', 'value': 1.70},
                  {'parameter': 'w_core', 'value': 1.658},
                  {'parameter': 'theta_core', 'value': math.radians(7.125)},
                  {'parameter': 'n', 'value': 10},
                  {'parameter': 'wa_core', 'value': unnormX[3]},
                  {'parameter': 'w_extrude', 'value': unnormX[4]},
                  {'parameter': 'h_round', 'value': 13.65},
                  {'parameter': 'ho_round', 'value': 13.65 + 1.70 + 1.70 * math.sin(math.radians(5))},
                  {'parameter': 'alpha', 'value': math.radians(unnormX[5])},
                  {'parameter': 'round_core', 'value': 0.5},
                  {'parameter': 'round_skin', 'value': 1.0},
                  {'parameter': 'mesh', 'value': 1.6}]

    # Write the geometry file to the folder
    fname = str(simID) + '/actuator_defintion.py'
    with open(fname, 'w') as f:
        f.write('import math')
        f.write('\n')
        f.write('geometry_dict = {')
        f.write('\n')
        for i, param in enumerate(parameters):
            if i != len(parameters) - 1:
                f.write('\t\'' + str(param['parameter']) + '\': ' + str(param['value']) + ',')
                f.write('\n')
            else:
                f.write('\t\'' + str(param['parameter']) + '\': ' + str(param['value']) + '}')
        f.write('\n')
        f.write('material_dict = {\'name\': \'Smooth Sil 950\', \'model\': \'Neo-Hookean\', \'P1\': 0.34, \'P2\': 0, \'density\': 1.24*(10**-9)}')
        f.write('\n')
        f.write('actuator_definition = {\'geometry\': geometry_dict, \'material_properties\': material_dict, \'pressure\': 0.20}')

    print("Current Directory")
    print(os.getcwd())

    # Copy over the other relevant files
    shutil.copy('actuator_optimization/main_abaqus.py', str(simID) + '/main_abaqus.py')
    shutil.copy('actuator_optimization/Actuator.py', str(simID) + '/Actuator.py')
    shutil.copy('actuator_optimization/Extractor.py', str(simID) + '/Extractor.py')

    # Switch into the desired directory
    os.chdir(str(simID))

    print("Inside Directory")
    print(os.getcwd())

    # Call the script that runs all abaqus commands
    FNULL = open(os.devnull, 'w')
    subprocess.run(
        ['abaqus', 'cae', '-noGUI', 'C:/Users/mechb/Annie/' + folder_name + '/' + str(simID) + '/main_abaqus.py'],
        shell=True)
    FNULL.close()

    try:
        # Load the exported data into a dataframe
        print("Starting model processing")
        ProcessedModel1 = ProcessedModel('model')
        ProcessedModel1.load_data()
        print("Trying to extract data")
        ProcessedModel1.compute_grasping_force()
        return_value = ProcessedModel1.strain_force_loss()

    except:
        print("Geometry not feasible")
        # this catches the case where the geometry doesn't generate
        return_value = float('inf')
        
    # Delete the files that we don't need
    for item in os.listdir(os.getcwd()):
        if (item.endswith(".odb")) or (item.endswith(".txt")) or (item.endswith(".sta")) or (item.endswith(".cae")):
            pass
        else:
            os.remove(item)

    # Move back out one directory to the main Models/CMA directory
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    # # If the model doesn't reach the target pressure, assign it a very high loss value
    # if round(ProcessedModel1.data['Pressure'].iloc[-1], 2) != 0.15:
    #     return_value = float('inf')

    # For now, return strain as the metric of interest
    return return_value

if __name__=='__main__':

    # Initialize the optimizer
    optimizer = ng.optimizers.CMA(parametrization=instrum, budget=budget, num_workers=num_workers)
    experiment_name_logger_file = experiment_name + '.txt'

    # Read in existing information if there is some
    past_experiment_df = pd.DataFrame(columns=['loss', 't', 'r_core', 'b_core', 'wa_core', 'w_extrude', 'alpha'])
    if past_experiments != None:
        with open(past_experiments, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = ast.literal_eval(line)
            loss_value = line['#loss']
            t_value = line['t']
            r_core_value = line['r_core']
            b_core_value = line['b_core']
            wa_core_value = line['wa_core']
            w_extrude_value = line['w_extrude']
            alpha_value = line['alpha']
            row_dict = {'loss': loss_value,'t': t_value, 'r_core': r_core_value, 'b_core': b_core_value, 'wa_core': wa_core_value, 'w_extrude': w_extrude_value, 'alpha': alpha_value}
            past_experiment_df = pd.concat([past_experiment_df, pd.DataFrame([row_dict])], ignore_index=True)
    
        print("Previous experiments")
        print(past_experiment_df)
        print(past_experiment_df.dtypes)

        # Feed the information into the optimizer
        for i, row in past_experiment_df.iterrows():
            print("Feeding in previous design " + str(i))
            candidate = optimizer.parametrization.spawn_child(new_value=(tuple(), {"t": row['t'], "r_core": row['r_core'], "b_core": row['b_core'], "wa_core": row['wa_core'], "w_extrude": row['w_extrude'], "alpha": row['alpha']}))
            optimizer.tell(candidate, row['loss'])

    # Remove the logger file if there is already one there
    if experiment_name_logger_file in os.listdir():
        os.remove(experiment_name_logger_file)

    # Tell the optimizer to write to the logger_file
    logger = ModifiedParametersLogger(experiment_name_logger_file)
    optimizer.register_callback("tell", logger)

    # Run the parallelization
    with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(loss, executor=executor, batch_mode=False)

    print(recommendation.value)