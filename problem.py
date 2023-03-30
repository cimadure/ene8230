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
print(df.shape)
year = df['year'].unique()
print(year)

an = df[df['year'] == year[0]]
Pb = an['Power Clipped [kW]'].to_numpy(copy=True)

Pb = Pb[:40]

print(Pb.shape)

# print(df.info)
# [69504 rows x 14 columns]>

l = Pb.shape[0]
print("maximum de Pb(t):{m} avec T={t}".format(m=max(Pb), t=l))
#Si = 10 * np.random.random_sample((l, 4)) - 5
#Si = np.random.randint(0, 2, size=(l, 1))
Si = np.ones((l, 4))
#print(Si)

print('Set problem parameters')
n_ev = 1000.0
# Prius 4.4kWh, Leaf 24kWh
soc_max = 0.8*24*1000
soc_min = 0.2*4.4*1000
print("{mi} <= SOC <= {ma}".format(mi=soc_min, ma=soc_max))

params = {
    # Variables	Descriptions	Unités
    'SOCmin': soc_min,  # État de charge minimum des véhicules électriques	kWh
    'SOCmax': soc_max,  # État de charge maximum des véhicules électriques	kWh
    'Pch_min': 5.0,  # Puissance de charge minimum 	kW
    'Pdis_min': 5.0,  # Puissance de décharge minimum	kW
    'Pch_max_n': [7.5, 50.0],  # Puissance de charge maximum en utilisant les bornes de type n	kW
    'Pdis_max_n': [5.0, 40.0],  # Puissance de décharge maximum en utilisant les bornes de type n	kW
    #'Pch_max_n': [50.0],  # Puissance de charge maximum en utilisant les bornes de type n	kW
    #'Pdis_max_n': [40.0],  # Puissance de décharge maximum en utilisant les bornes de type n	kW

    'beta_ch': 0.93,  # Efficacité de charge	%
    'beta_dis': 0.90,  # Efficacité de décharge	%
    'NEVs': n_ev,  # Nombre de véhicules électriques	Qté
    'Rut': [0.21, 0.22, 0.23, 0.34],  # Ratio d’utilisateurs avec le profil i	%
    #'Rut': [1.0],  # Ratio d’utilisateurs avec le profil i	%

    'delta_t': 15 / 60,  # 15mins
    'MAX_OPTIM': 1e8,  # Facteur majorant du programme d'optimisation

    'Pb': Pb,  # Puissance appelée par le bâtiment appelée à l’instant t	kW

    'Si': Si,#1, Si le profil d’utilisateur est raccordé à une borne à l’instant t

}


def main(args):

    mdl = ModelShaving('V2B', params=params)
    mdl.problem_variables()
    #print(mdl.Pch_tot__t)
    #print(mdl.Pdis_tot__t)
    mdl.problem_constraints()

    mdl.minimize( mdl.problem_cout_depassement() )


    # mdl.add_kpi(mdl.max(mdl.v), "Max Pch(t)");
    print('\n')
    print(mdl.print_information())
    solus = mdl.solve()
    if solus:
        print("Obj", mdl.solution.get_objective_value())
        print(vars(mdl))
        print(mdl.print_solution(print_zeros=False))
        print(solus)
        print(mdl.describe_objectives())
    print("--------------------------------------------")

    print('end')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

