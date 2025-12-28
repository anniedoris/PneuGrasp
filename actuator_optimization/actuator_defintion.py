import math
geometry_dict = {
	't': 2.244467595209397,
	'r_core': 3,
	'b_core': 3.63,
	'h_core': 12.37,
	'h_upper': 1.7,
	'w_core': 1.658,
	'theta_core': 0.12435470920459599,
	'n': 10,
	'wa_core': 5,
	'w_extrude': 15.63,
	'h_round': 13.65,
	'ho_round': 15.498164762671019,
	'alpha': 0.08726646259971647,
	'round_core': 0.5,
	'round_skin': 1.0,
	'mesh': 1.6}
material_dict = {'name': 'Smooth Sil 950', 'model': 'Neo-Hookean', 'P1': 0.34, 'P2': 0, 'density': 1.24*(10**-9)}
actuator_definition = {'geometry': geometry_dict, 'material_properties': material_dict, 'pressure': 0.150}