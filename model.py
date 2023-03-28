import numpy as np
# from docplex.mp.advmodel import AdvModel
from docplex.mp.model import Model


def func(x):
    return x + 1


"""
-graph, tableau: Profile des utilisateurs  
-graph: profile avant vs P avec V2B 
-tableau: # bornes par type 
Par année: 
-tableau: valeurs des coûts 
"""

"""
		[self.add_range(lb=self.params['SOCmin'], expr='SOC__{n}{i}'.format(n=n,i=i), ub=self.params['SOCmax']) for n in self.ens['N'] for i in self.ens['I']]
"""


# import cplex._internal._procedural as proc
# print(proc.__file__)


# https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html
# https://notebook.community/IBMDecisionOptimization/docplex-examples/examples/mp/jupyter/tutorials/Beyond_Linear_Programming
class ModelShaving(Model):

    def __init__(self, *args, **kwargs):
        self.params = kwargs.pop('params', None)
        if self.params:
            self.ens = {"utilsateur": np.shape(self.params['Rut'])[0],  # i
                        "borne": np.shape(self.params['Pch_max_n'])[0],  # n
                        "instant": np.shape(self.params['Pb'])[0],  # t
                        "mois": 12}  # m
            self.ens['I'] = range(1, self.ens['utilsateur'] + 1)
            self.ens['N'] = range(1, self.ens['borne'] + 1)
            self.ens['T'] = range(0, self.ens['instant'])
            self.ens['M'] = range(1, self.ens['mois'] + 1)
        # print(self.ens)

        super(ModelShaving, self).__init__(*args, **kwargs)

    def problem_variables(self):
        """
        SOCi(t)	État de charge de la flotte des véhicules électriques à l’instant t	kWh
        Pch,ni(t)	Puissance de charge à l’instant (t)	kW
        Pdis,ni(t)	Puissance de décharge à l’instant (t)	kW
        delta_ch,i(t)	1, Si le profil d’utilisateur i est en charge
        0, Sinon	Bin
        delta_dis,i(t)	1, Si le profil d’utilisateur i est en décharge
        0, Sinon	Bin
        Rborne,ni	Ratio de borne de type n utilisé par le profil d’utilisateur i	%
        """
        self.SOC__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'], lb=0,
                                                   ub=self.params['MAX_OPTIM'], name='SOC_')  # , key_format=None)

        self.Pch__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'], lb=0,
                                                   ub=self.params['MAX_OPTIM'], name='Pch_')  # , key_format=None)
        self.Pdis__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'], lb=0,
                                                    ub=self.params['MAX_OPTIM'], name='Pdis_')  # , key_format=None)

        self.delta_ch__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_ch_')
        self.delta_dis__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_dis_')

        self.Rborne__n_i = self.binary_var_matrix(self.ens['N'], self.ens['I'], name='Rborne_')

    def problem_constraint_prevent_simultaneous_charge_and_discharge(self):
        [self.add_constraint(self.delta_ch__i_t[i, t] + self.delta_dis__i_t[i, t] <= 1.0)
         for i in self.ens['I'] for t in self.ens['T']]

    def problem_constraint_SOC_range(self):
        [self.add_range(lb=self.params['SOCmin'], expr=self.SOC__n_i_t[n, i, t], ub=self.params['SOCmax'])
         for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]

    def problem_constraint_Pch_range(self):
        [self.add_range(lb=self.params['Pch_min'], expr=self.Pch__n_i_t[n, i, t], ub=self.params['Pch_max_n'][n-1])
         for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]
        #[self.add_constraint(self.params['Pch_min'] <= self.Pch__n_i_t[n, i, t])
        # for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]

#    , ub = self.params['Pch_max_n'][n]

    def problem_constraint_Pdis_range(self):
        [self.add_range(lb=self.params['Pch_min'], expr=self.Pdis__n_i_t[n, i, t], ub=self.params['Pdis_max_n'][n-1])
         for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]

    def problem_power_aggration__t(s_i, delta_i, r__ut_i, p__n_i, r__n_i):
        pass

    def problem_constraint_Pch_total__t(self):
        pass

    def problem_constraint_Pdis_total__t(self):
        pass

    def problem_constraint_Pch__n_i_t(self):
        pass

    def problem_constraint_Pdis__n_i_t(self):
        pass

    def problem_constraint_SOC__n_i_t(self):
        pass

    def problem_constraints(self):
        self.problem_constraint_prevent_simultaneous_charge_and_discharge()
        self.problem_constraint_SOC_range()
        self.problem_constraint_Pch_range()
        self.problem_constraint_Pdis_range()

# [self.add_range(lb=self.params['SOCmin'], expr='SOC__{n}{i}'.format(n=n,i=i), ub=self.params['SOCmax']) for n in self.ens['N'] for i in self.ens['I']]
# [self.add_range(lb=self.params['SOCmin'], expr=self.SOC__n_i_t[n,i], ub=self.params['SOCmax']) for n in self.ens['N'] for i in self.ens['I']];
# [self.le_constraint(self.delta_ch__i_t[i,t] + self.delta_dis__i_t[i,t], 1.0, name='ctr_ch_dis') for i in self.ens['I'] for t in self.ens['T']]

# %%
