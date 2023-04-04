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

    @staticmethod
    def var_dict_as_df(solution, var_dict, index, columns, index_first=False, prefix=None):
        key_column_names = list()

        if index_first:
            key_column_names = list(index)
            key_column_names.append(columns)
        else:
            key_column_names = list(columns)
            key_column_names.append(index)

        # print(key_column_names)
        vals = solution.get_value_df(var_dict, value_column_name='value', key_column_names=key_column_names)
        piv = vals.pivot(index=index, columns=columns, values='value')
        return piv.add_prefix(prefix) if prefix else piv

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
                                                   ub=self.params['MAX_OPTIM'],
                                                   name='Pch__n_i_t_')  # , key_format=None)
        self.Pdis__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'], lb=0,
                                                    ub=self.params['MAX_OPTIM'],
                                                    name='Pdis__n_i_t_')  # , key_format=None)

        self.con__n_i_t = self.binary_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'],
                                               name='con__n_i_t')
        self.discon__n_i_t = self.binary_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'],
                                                  name='discon__n_i_t')

        self.delta_ch__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_ch_')
        self.delta_dis__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_dis_')

        # TODO: WARNING: CHANGES (maybe use constains for ratio !), current value alway 0
        # self.Rborne__n_i = self.continuous_var_matrix(self.ens['N'], self.ens['I'], name='Rborne_')
        self.Rborne__n_i = 0.5 * np.ones(shape=(2, 4))
        self.Rborne__n_i[0, 0] = .75
        self.Rborne__n_i[1, 0] = .25

        print(self.Rborne__n_i)

        self.Pch__i_t = self.continuous_var_matrix(keys1=self.ens['I'], keys2=self.ens['T'], lb=0,
                                                   ub=self.params['MAX_OPTIM'], name='Pch__i_t_')

        self.Pdis__i_t = self.continuous_var_matrix(keys1=self.ens['I'], keys2=self.ens['T'], lb=0,
                                                    ub=self.params['MAX_OPTIM'], name='Pdis__i_t_')

        self.Pch_demand_i_t = self.continuous_var_matrix(keys1=self.ens['I'], keys2=self.ens['T'], lb=0,
                                                         ub=self.params['MAX_OPTIM'], name='Pch_demand_i_t_')

        self.Pdis_demand_i_t = self.continuous_var_matrix(keys1=self.ens['I'], keys2=self.ens['T'], lb=0,
                                                          ub=self.params['MAX_OPTIM'], name='Pdis_demand_i_t_')

        self.Pch_tot__t = self.continuous_var_list(keys=self.ens['T'], lb=0,
                                                   ub=self.params['MAX_OPTIM'], name='Pch_tot_')

        self.Pdis_tot__t = self.continuous_var_list(keys=self.ens['T'], lb=0,
                                                    ub=self.params['MAX_OPTIM'], name='Pdis_tot_')

        self.Pr__t = self.continuous_var_list(keys=self.ens['T'], lb=0, ub=self.params['MAX_OPTIM'], name='Pr_')

        self.Pr__i_t = self.continuous_var_matrix(keys1=self.ens['I'], keys2=self.ens['T'], lb=0,
                                                  ub=self.params['MAX_OPTIM'], name='Pr__i_t_')
        self.Pr__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'], lb=0,
                                                  ub=self.params['MAX_OPTIM'], name='Pr__n_i_t_')  # , key_format=None)

    def problem_constraint_prevent_simultaneous_charge_and_discharge_i_t(self):
        #return [self.add_constraint(self.delta_ch__i_t[i, t] + self.delta_dis__i_t[i, t] <= 1.0)
        #        for i in self.ens['I'] for t in self.ens['T']]

        y = self.params['Si']
        # s = self.con__n_i_t
        # z = self.discon__n_i_t
        #
        # # BEWARE !
        # # Contrainte pour initialisation
        for i in self.ens['I']:
            self.add_constraint(self.delta_ch__i_t[i, 0] == y[i - 1, 0])
        #
        [self.add_constraint(y[t, i - 1] - (self.delta_ch__i_t[i, t] + self.delta_dis__i_t[i, t]) == 0)
         for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]

    def problem_constraint_prevent_simultaneous_power_charge_and_discharge(self):
        return [self.add_constraint(self.Pch__n_i_t[n, i, t] * self.Pdis__n_i_t[n, i, t] == 0.0)
                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]


    def problem_constraint_SOC_range(self):
        return [self.add_range(lb=self.params['SOCmin'], expr=self.SOC__n_i_t[n, i, t], ub=self.params['SOCmax'])
                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]


    def problem_constraint_Pch_range(self):
        return [
            self.add_range(lb=self.params['Pch_min'], expr=self.Pch__n_i_t[n, i, t], ub=self.params['Pch_max_n'][n - 1])
            for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]
        # [self.add_constraint(self.params['Pch_min'] <= self.Pch__n_i_t[n, i, t])
        # for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]

        #    , ub = self.params['Pch_max_n'][n]


    def problem_constraint_Pdis_range(self):
        return [self.add_range(lb=self.params['Pch_min'], expr=self.Pdis__n_i_t[n, i, t],
                               ub=self.params['Pdis_max_n'][n - 1])
                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]


    def problem_constraint_Pch__n_i_t(self):
        # TODO: WARNING: self.delta_ch__i_t[i, t] , self.delta_dis__i_t[i, t] removed
        return [self.add_constraint(self.Pch__n_i_t[n, i, t]
                                    ==
                                    # self.params['NEVs'] * self.params['Rut'][i - 1] * self.Rborne__n_i[n, i]
                                    self.params['NEVs'] * self.params['Rut'][i - 1] * self.Rborne__n_i[n - 1, i - 1]
                                    #* self.params['Si'][t, i - 1]
                                     * self.delta_ch__i_t[i, 1]
                                    * self.params['Pb'][t]
                                    )
                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']
                ]


    def problem_constraint_Pdis__n_i_t(self):
        # TODO: WARNING: self.delta_ch__i_t[i, t] , self.delta_dis__i_t[i, t] removed
        return [self.add_constraint(self.Pdis__n_i_t[n, i, t]
                                    ==
                                    # self.params['NEVs'] * self.params['Rut'][i - 1]  * self.Rborne__n_i[n, i]
                                    self.params['NEVs'] * self.params['Rut'][i - 1] * self.Rborne__n_i[n - 1, i - 1]
                                    #* self.params['Si'][t, i - 1]
                                    * self.delta_dis__i_t[i, t]
                                    * self.params['Pb'][t]
                                    ) for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']
                ]


    def problem_constraint_SOC__n_i_t(self):
        # TODO: WARNING: self.delta_ch__i_t[i, t] , self.delta_dis__i_t[i, t] removed
        # Mod avec Pr__n_i_t
        return [self.add_constraint(self.SOC__n_i_t[n, i, t + 1] ==
                                    self.SOC__n_i_t[n, i, t]
                                    + self.params['beta_ch'] * self.Pch__n_i_t[n, i, t] * self.params['delta_t']
                                    - self.params['beta_dis'] * self.Pdis__n_i_t[n, i, t] * self.params['delta_t']
                                    # + self.Pr__n_i_t[n, i, t]
                                    ) for n in self.ens['N'] for i in self.ens['I'] for t in
                range(0, self.ens['instant'] - 1)]


    def problem_constraint_Pch__i_t(self):
        return [self.add_constraint(self.Pch__i_t[i, t] == self.sum(self.Pch__n_i_t[n, i, t] for n in self.ens['N']))
                for i in self.ens['I'] for t in self.ens['T']
                ]


    def problem_constraint_Pdis__i_t(self):
        return [self.add_constraint(self.Pdis__i_t[i, t] == self.sum(self.Pdis__n_i_t[n, i, t] for n in self.ens['N']))
                for i in self.ens['I'] for t in self.ens['T']
                ]


    def problem_power_aggration__t(self, s_i, delta_i=None, r__ut_i=None, p__n_i=None, r__n_i=None):
        return [self.params['NEVs'] * (
            self.sum(s_i[t, i - 1] * delta_i[i, t] * r__ut_i[i - 1]

                     * self.sum(p__n_i[n, i, t] * r__n_i[n, i] for n in self.ens['N'])
                     for i in self.ens['I'])
            for t in self.ens['T'])]


    def problem_constraint_Pch_total__t(self):
        return [self.add_constraint(self.Pch_tot__t[t] == self.sum(self.Pch__i_t[i, t] for i in self.ens['I']))
                for t in self.ens['T']
                ]


    def problem_constraint_Pdis_total__t(self):
        return [self.add_constraint(self.Pdis_tot__t[t] == self.sum(self.Pdis__i_t[i, t] for i in self.ens['I']))
                for t in self.ens['T']
                ]


    def problem_constraint_Pr__t(self):
        return [self.add_constraint(self.Pr__t[t] == self.params['Pb'][t] + self.Pch_tot__t[t] - self.Pdis_tot__t[t])
                for t in self.ens['T']
                ]

        # WITH delta :  - problem type is: MIQCP
        # Error: Model has non-convex quadratic constraint, index=0


    def problem_constraint_Pr__n_i_t(self):
        return [self.add_constraint(self.Pr__n_i_t[n, i, t]
                                    ==
                                    self.params['Pb'][t]
                                    + self.Pch__n_i_t[n, i, t]  # * self.delta_ch__i_t[i, t]
                                    - self.Pdis__n_i_t[n, i, t]
                                    # * (1 - self.delta_ch__i_t[i, t])#* self.delta_dis__i_t[i, t]
                                    )

                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']
                ]


        #    def problem_constraint_chrge_discharge__t(self):
        #        [self.add_constraint(0 ==  self.Pch_tot__t[t] * self.Pdis_tot__t[t] )
        #                            for t in self.ens['T']
        #          ]


    def problem_cout_depassement(self):
        c = 1.0  # next: cout = [summer, winter]
        return self.params['delta_t'] * sum(c * self.Pr__t[t] for t in self.ens['T'])


    def problem_constraint_unit_commitment(self):
        # TODO
        PU = self.params['SOCmin'] / self.params['delta_t']
        PD = self.params['SOCmax'] / self.params['delta_t']
        C = [1.0, 1.0, 1.0, 1.0]
        W = 1
        x = self.SOC__n_i_t
        y = self.params['Si']
        s = self.con__n_i_t
        z = self.discon__n_i_t

        # BEWARE !
        # Contrainte pour initialisation
        #3
        # for i in self.ens['I']:
         #    self.add_constraint(self.delta_ch__i_t[i, 0] == y[i - 1, 0])


        for n in self.ens['N']:
            for i in self.ens['I']:
                t = 0
                self.add_constraint(s[n, i, t] >= y[t, i - 1])

                for t in range(1, self.ens['instant']):
                    self.add_constraint(s[n, i, t] >= (y[t, i - 1] - y[t - 1, i - 1]))
                    self.add_constraint(z[n, i, t] >= (y[t - 1, i - 1] - y[t, i - 1]))

                for (ta, td) in zip(self.params['arrivee'][i - 1], self.params['depart'][i - 1]):
                    self.add_constraint(x[n, i, ta] == self.params['SOCmin'])
                    self.add_constraint(x[n, i, td] == self.params['SOCmin'])

                # self.add_constraints([self.sum(s[n, i, t - min((t + 1, W)) + 1:t + 1]) <= y[t - 1, i - 1] for t in self.ens['T']])

                # for t in ...
                # self.add_constraint(x[n, i, t] >= s[n, i, t]*self.params['SOCmin'])
                ##self.add_constraint(x[n, i, t] >= (1-z[n, i, t])*self.params['SOCmax'])
                # self.add_constraint(x[n, i, t] >= z[n, i, t]*0.9*self.params['SOCmax'])
                #    pass

                #         #self.add_constraints([x[i, j] <= y[i, j] * C[i, 0] for j in range(J)])
                #         #self.add_constraint(self.delta_ch__i_t[i, 0] == self.params['Si'][i, 0])
                #         #self.add_constraints([s[i, j] >= y[i, j] - y[i, j - 1] for j in range(1, J)])
                for t in range(1, self.ens['instant']):
                    self.add_constraint(x[n, i, t] - x[n, i, t - 1] <= PU)
                    self.add_constraint(x[n, i, t - 1] - x[n, i, t] <= PD)


    def problem_constraints(self):
        self.problem_constraint_prevent_simultaneous_charge_and_discharge_i_t()
        #self.problem_constraint_prevent_simultaneous_power_charge_and_discharge()

        self.problem_constraint_SOC_range()
        self.problem_constraint_Pch_range()
        self.problem_constraint_Pdis_range()

        # self.problem_constraint_SOC__n_i_t()

        # self.problem_constraint_Pch__n_i_t()
        # self.problem_constraint_Pdis__n_i_t()

        self.problem_constraint_Pch__i_t()
        self.problem_constraint_Pdis__i_t()

        self.problem_constraint_Pch_total__t()
        self.problem_constraint_Pdis_total__t()

        self.ctr_Pr__t = self.problem_constraint_Pr__t()

        self.problem_constraint_Pr__n_i_t()
        self.problem_constraint_unit_commitment()
