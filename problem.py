from model import ModelShaving
# from docplex.mp.advmodel import AdvModel
import numpy as np
import pandas as pd
import pickle
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

Pb = Pb[30:40] # 40
idx = an.index[30:40]

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

bn = n_ev
params = {
    # Variables	Descriptions	Unités
    'SOCmin': soc_min,  # État de charge minimum des véhicules électriques	kWh
    'SOCmax': soc_max,  # État de charge maximum des véhicules électriques	kWh
    'Pch_min': 0*(4.4*(15/60))*bn,  # Puissance de charge minimum 	kW
    'Pdis_min': 0*(4.4*(15/60))*bn,  # Puissance de décharge minimum	kW
    'Pch_max_n': [bn*7.5, bn*50.0],  # Puissance de charge maximum en utilisant les bornes de type n	kW
    'Pdis_max_n': [bn*5.0, bn*40.0],  # Puissance de décharge maximum en utilisant les bornes de type n	kW
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
    # more: https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.solution.html#docplex.mp.solution.SolveSolution
    mdl = ModelShaving('V2B', params=params)
    mdl.problem_variables()
    #print(mdl.Pch_tot__t)
    #print(mdl.Pdis_tot__t)
    mdl.problem_constraints()

    mdl.minimize(mdl.problem_cout_depassement())

    mdl.add_kpi(mdl.min(mdl.problem_cout_depassement()), "Min Cout dépassement")

    print('\n')
    print(mdl.print_information())
    solus = mdl.solve()
    assert solus, "!!! Solve of the model fails"
    print("********************************************")
    print("Obj", mdl.solution.get_objective_value())
    print(mdl.print_solution(print_zeros=False))
    print("--------------------------------------------")
    print(mdl.report())
    print("--------------------------------------------")
    print('end')
    with open('solution.pickle', mode='wb') as f:
        pickle.dump(solus, f)

    #all = solus.as_df(name_key='name', value_key='value')
    #print(all)

    dk = pd.DataFrame({'Pb batiment': mdl.params['Pb'],
                       'Pr reseau': solus.get_value_list(mdl.Pr__t),
                       'P charge totale': solus.get_value_list(mdl.Pch_tot__t),
                       'P décharge totale': solus.get_value_list(mdl.Pdis_tot__t),
                       'Si group 1': mdl.params['Si'][:, 0],
                       'Si group 2': mdl.params['Si'][:, 1],
                       'Si group 3': mdl.params['Si'][:, 2],
                       'Si group 4': mdl.params['Si'][:, 3]}
                      ).set_index(idx)

    print(dk)

    return 0

    # #numrows = mdl.linear_constraints.get_num()
    # #numcols = mdl.variables.get_num()
    # print("============================================")
    # print()
    # #print("Solution status = ", mdl.solution.get_status())
    # print("Solution value  = ", mdl.solution.get_objective_value())
    # slack = mdl.solution.get_linear_slacks()
    # pi = mdl.solution.get_dual_values()
    # x = mdl.solution.get_values()
    # dj = mdl.solution.get_reduced_costs()
    # for i in   mdl.iter_constraints():
    #     print("Row %d:  Slack = %10f  Pi = %10f" % (i, slack[i], pi[i]))
    # #for j in range(numcols):
    # #    print("Column %d:  Value = %10f Reduced cost = %10f" %
    # #          (j, x[j], dj[j]))


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

