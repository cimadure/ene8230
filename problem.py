from model import ModelShaving
# from docplex.mp.advmodel import AdvModel
import numpy as np
import pandas as pd
import sys

# import cplex

print('Load')

print('Load Pb and other data')
filename = "ConsommationUniversite.pickle"
df = pd.read_pickle(filename)

year = df['year'].unique()
print(year)

an = df[df['year'] == year[0]]
Pb = an['Power Clipped [kW]'].to_numpy(copy=True)

Pb = Pb[:11]

print(Pb.shape)
# print(df.info)
# [69504 rows x 14 columns]>

print('Set problem parameters')

params = {
    # Variables	Descriptions	Unités
    'SOCmin': 20.0,  # État de charge minimum des véhicules électriques	kWh
    'SOCmax': 80.0,  # État de charge maximum des véhicules électriques	kWh
    'Pch_min': 5.0,  # Puissance de charge minimum 	kW
    'Pch_max_n': np.array([[7.5], [50.0]]),  # Puissance de charge maximum en utilisant les bornes de type n	kW
    'Pdis_min': 5.0,  # Puissance de décharge minimum	kW
    'Pdis_max_n': np.array([[5.0], [40.0]]),  # Puissance de décharge maximum en utilisant les bornes de type n	kW
    'beta_ch': 0.90,  # Efficacité de charge	%
    'beta_dis': 0.90,  # Efficacité de décharge	%
    'NEVs': 1000,  # Nombre de véhicules électriques	Qté
    'Rut': np.array([[0.25], [0.25], [0.25], [0.25]]),  # Ratio d’utilisateurs avec le profil i	%

    'delta_t': 15 / 60,  # 15mins
    'MAX_OPTIM': 1e8,  # Facteur majorant du programme d'optimisation

    'Pb': Pb,  # Puissance appelée par le bâtiment appelée à l’instant t	kW

    # Si(t)  #1, Si le profil d’utilisateur est raccordé à une borne à l’instant t

}

mdl = ModelShaving('V2B', params=params)
mdl.problem_variables()
mdl.problem_constraints()

# mdl.add_kpi(mdl.max(mdl.v), "Max Pch(t)");

print(mdl.print_information())

solus = mdl.solve()
print("Obj", mdl.solution.get_objective_value())
print("--------------------------------------------")
print(mdl.print_solution(print_zeros=False))
print(solus)
#print(mdl.describe_objectives())
print('end')
