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
                        "mois": np.shape(self.params['t_min__m'])[0]}  # m
            self.ens['I'] = range(1, self.ens['utilsateur'] + 1)
            self.ens['N'] = range(1, self.ens['borne'] + 1)
            self.ens['T'] = range(0, self.ens['instant'])
            # self.ens['M'] = range(1, self.ens['mois'] + 1)
            self.ens['M'] = range(0, self.ens['mois'])
        print(self.ens)
        self.cout_energie = None
        self.cout_puissance = None
        self.cout_infrastucture = None

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

        self.delta_ch__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_ch_')
        self.delta_dis__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_dis_')

        # TODO: WARNING: CHANGES (maybe use constains for ratio !), current value alway 0
        # self.Rborne__n_i = self.continuous_var_matrix(self.ens['N'], self.ens['I'], name='Rborne_')
        self.Nborne__n_i = self.integer_var_matrix(self.ens['N'], self.ens['I'], name='Nborne__n_i_')


#        self.Rborne__n_i = np.array([[504,	8,	152,	304],[168,	8,	152,	304]])
#        self.Rborne__n_i = np.array([[32,	1,	10,	19],[11,	1,	10,	19]])

        self.Rborne__n_i = np.array([[34,	1,	16,	31], [34,	1,	16,	31]]) # 50% 50%
        #self.Rborne__n_i = np.array([[0,	0,	0,	0], [68, 2, 31, 61]]) #100%  0%
        #self.Rborne__n_i = np.array([[68, 2, 31, 61], [0,	0,	0,	0]]) #100%  0%

        #print(self.Rborne__n_i)

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

        self.Pr_t_max__m = self.continuous_var(lb=0, ub=self.params['MAX_OPTIM'], name='Pr_t_max__m')

        self.Pr_max__m = self.continuous_var_list(keys=self.ens['M'], lb=0, ub=self.params['MAX_OPTIM'], name='Pr_max__m')
        self.Pess_max__m = self.continuous_var_list(keys=self.ens['M'], lb=0, ub=self.params['MAX_OPTIM'], name='Pess_max__m')
        self.E__m = self.continuous_var_list(keys=self.ens['M'], lb=0, ub=self.params['MAX_OPTIM'], name='E__m')
        self.cout_infra_m = self.continuous_var_list(keys=self.ens['M'], lb=0, ub=self.params['MAX_OPTIM'], name='cout_infra_m')


        self.delta_soc_direction = self.binary_var_list(keys=self.ens['T'], name='delta_soc_dir_')


        ### NEW Modeling with ESS
        self.Pess__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'],
                                                    lb=-self.params['MAX_OPTIM'],
                                                    ub=self.params['MAX_OPTIM'],
                                                    name='Pess__n_i_t_')  # , key_format=None)


        self.Pess__t = self.continuous_var_list(keys=self.ens['T'],
                                                   lb=-self.params['MAX_OPTIM'],
                                                   ub=self.params['MAX_OPTIM'], name='Pess__t_')

        self.Pabs__n_i_t = self.continuous_var_cube(keys1=self.ens['N'], keys2=self.ens['I'], keys3=self.ens['T'], lb=0,
                                                   ub=self.params['MAX_OPTIM'],
                                                   name='Pabs__n_i_t_')
    def problem_constraint_prevent_simultaneous_charge_and_discharge_i_t(self):
        # CECI EST UNE NOUVELLE VERSION QUI IMPLIQUE QUE Si[n] soit activé
        # Si Si alors soit ch ou dech
        # return [self.add_constraint(self.delta_ch__i_t[i, t] + self.delta_dis__i_t[i, t] <= 1.0)
        #        for i in self.ens['I'] for t in self.ens['T']]
        y = self.params['Si']
        [self.add_constraint(y[t, i - 1] - (self.delta_ch__i_t[i, t] + self.delta_dis__i_t[i, t]) <= 1)
         for i in self.ens['I'] for t in self.ens['T']]
        [self.add_constraint(y[t, i - 1] - (self.delta_ch__i_t[i, t] + self.delta_dis__i_t[i, t]) >= 0)
         for i in self.ens['I'] for t in self.ens['T']]

    # validé
    def problem_constraint_SOC_range(self):
        return [self.add_range(lb=self.params['SOCmin'],
                               expr=self.SOC__n_i_t[n, i, t],
                               ub=self.params['SOCmax'],
                               )
                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]


    def problem_constraint_SOC__n_i_t(self):
        # TODO: WARNING: self.delta_ch__i_t[i, t] , self.delta_dis__i_t[i, t] removed
        # Mod avec Pr__n_i_t
        return [self.add_constraint(self.SOC__n_i_t[n, i, t + 1] ==
                                    self.SOC__n_i_t[n, i, t]
                                    +
                                    self.params['Si'][t, i - 1] *
                                    #1.0 *
                                    (

                                      self.params['beta_ch'] * self.Pch__n_i_t[n, i, t] * self.params['delta_t']
                                       #* self.delta_ch__i_t[i, t   ]
                                      #* self.delta_soc_direction[t]
                                    - self.params['beta_dis'] * self.Pdis__n_i_t[n, i, t] * self.params['delta_t']
                                       # * self.delta_dis__i_t[i, t]
                                    # * (1-self.delta_ch__i_t[i, t])#self.delta_dis__i_t[i, t]
                                    # + self.Pr__n_i_t[n, i, t]
                                    )
                                    ) for n in self.ens['N'] for i in self.ens['I'] for t in
                range(0, self.ens['instant'] - 1)]

    def problem_constraint_SOC__v3_n_i_t(self):
        for i in self.ens['I']:
            for c in range(0, len(self.params['arrivee'])-1):
                #print(c)
                for t in range(self.params['arrivee'][i - 1][c]+1, self.params['depart'][i - 1][c]+1):
                    #print(t)
                    [self.add_constraint(self.SOC__n_i_t[n, i, t] ==
                                            self.SOC__n_i_t[n, i, t-1]
                                            +
                                            (
                                              self.params['beta_ch'] * self.Pess__n_i_t[n, i, t] * self.params['delta_t']
                                            )
                                            ) for n in self.ens['N']]

                    # CONTRAINTES DE RANGE A L'INTERIEUR DE CETTE BOUCLE
                    [self.add_range(lb=self.params['SOCmin'],
                                    expr=self.SOC__n_i_t[n, i, t],
                                    ub=self.params['SOCmax'],
                                    #rng_name='SOC_range_in_SOC_V3_{n}{i}{t}'%(n,i,t)
                                    ) for n in self.ens['N']]
                    #else:
                    #    [self.add_constraint(self.SOC__n_i_t[n, i, t] == 0) for n in self.ens['N']]
                        #pass


    def problem_power_aggration__t(self, t, s_i=None, delta_i_t=None, r__ut_i=None, p__n_i_t=None, r__n_i=None):
        # WARNING: NO delta_i_t
       # return #self.params['NEVs'] * \
       #     (
       return  self.sum(
                    s_i[t, i - 1] *
                    # delta_i_t[i, t] *
                     #* r__ut_i[i - 1]
                     self.sum(p__n_i_t[n, i, t] * r__n_i[n - 1, i - 1] for n in self.ens['N'])
                     for i in self.ens['I'])

    def problem_power_aggration_Si_with_delta__t(self, t, s_i=None, delta_i_t=None, r__ut_i=None, p__n_i_t=None, r__n_i=None):
       return  self.sum(
                    s_i[t, i - 1] *
                    delta_i_t[i, t] *
                     self.sum(p__n_i_t[n, i, t] * r__n_i[n - 1, i - 1] for n in self.ens['N'])
                     for i in self.ens['I'])

    def problem_power_aggration_only_delta__t(self, t, s_i=None, delta_i_t=None, r__ut_i=None, p__n_i_t=None, r__n_i=None):
       return  self.sum(
                    delta_i_t[i, t] *
                     self.sum(p__n_i_t[n, i, t] * r__n_i[n - 1, i - 1] for n in self.ens['N'])
                     for i in self.ens['I'])

    def problem_power_aggration_only_Si__t(self, t, s_i=None, delta_i_t=None, r__ut_i=None, p__n_i_t=None, r__n_i=None):
       return  self.sum(
                    s_i[t, i - 1] *
                     self.sum(p__n_i_t[n, i, t] * r__n_i[n - 1, i - 1] for n in self.ens['N'])
                     for i in self.ens['I'])

    def problem_power_aggration_only_sum__t(self, t, s_i=None, delta_i_t=None, r__ut_i=None, p__n_i_t=None, r__n_i=None):
       return  self.sum(
                     self.sum(p__n_i_t[n, i, t] * r__n_i[n - 1, i - 1] for n in self.ens['N'])
                     for i in self.ens['I'])



    def problem_constraint_Pr__t(self):
        return [self.add_constraint(self.Pr__t[t] == self.params['Pb'][t] + self.Pch_tot__t[t] - self.Pdis_tot__t[t])
                #        return [self.add_constraint(self.Pr__t[t] == self.params['Pb'][t] + self.Pch_tot__t[t]*self.delta_ch__t[t] - self.Pdis_tot__t[t]*self.delta_dis__t[t] )
                for t in self.ens['T']
                ]
        # WITH delta :  - problem type is: MIQCP

        # Error: Model has non-convex quadratic constraint, index=0
    #5.1
    def problem_constraint_Pr__t_v3(self):
        return [self.add_constraint(self.Pr__t[t] == self.params['Pb'][t] + self.Pess__t[t])
                            for t in self.ens['T']
                ]

    def problem_constraint_Pess__t_v3(self):
        [self.add_constraint(self.Pess__t[t] == self.problem_power_aggration_only_sum__t(t=t,
        s_i = self.params['Si'],
        #delta_i_t = self.delta_ch__i_t,
        p__n_i_t = self.Pess__n_i_t,
        r__n_i = self.Rborne__n_i)) for t in self.ens['T']]


    def problem_constraint_Pess_range_v3(self):
        return [self.add_range(lb=-self.params['Pdis_max_n'][n - 1],
                               expr=self.Pess__n_i_t[n, i, t],
                               ub=self.params['Pch_max_n'][n - 1]
                 )
                for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]


    def problem_cout_energie(self):
        # Min∑_(t=1)^H▒(C_E^ *P_r^  (t)*∆t)
        # next: cout = [summer, winter]
        return self.params['delta_t'] * sum(self.params['C__E'] * self.Pr__t[t] for t in self.ens['T'])

    def problem_cout_puissance(self):
        # ∑_(m=1)^12▒(C_P^ *(P_m^max+)
#        return self.sum(self.params['C__P'] * self.Pr_t_max__m for m in self.ens['M'])
        return self.sum(self.params['C__P'] * self.Pr_max__m[m] for m in self.ens['M'])

    def problem_cout_infrastructure(self):
        # ∑_(i=1) ^ I▒(∑_(n=1) ^ 2▒(C_(b, n) * N_EVs * R_(ut, i) * R_(borne, ni)) )
        return sum(
            sum(self.params['C__b_n'][n - 1, i - 1] * # self.params['NEVs'] * self.params['Rut'][i - 1] *
                self.Rborne__n_i[n - 1, i - 1] for n in self.ens['N'])
            for i in self.ens['I']) *  sum(self.params['delta_t'] for t in self.ens['T'])


    def problem_constraint_SOC__n_i_t_arrivee(self):
        x = self.SOC__n_i_t
        for n in self.ens['N']:
            for i in self.ens['I']:
                [self.add_constraint(x[n, i,t] == self.params['SOCmin']*1.2,
                                     ctname='ctr_SOC_n_i_t_arrivee_{n}_{i}_{t}'.format(n=i, i=i,t=t))
                 for t in self.params['arrivee'][i - 1]]

    def problem_constraint_SOC__n_i_t_depart(self):
        x = self.SOC__n_i_t
        for n in self.ens['N']:
            for i in self.ens['I']:
                [self.add_constraint(x[n, i, t] >= self.params['SOCmax']) for t in self.params['depart'][i - 1]]


    def problem_constraint_Pr_t__max__m(self):
        for m in self.ens['M']:
            mini = self.params['t_min__m'][m - 0]
            maxi = self.params['t_max__m'][m - 0]

            maxi = maxi - mini + 1
            mini = 0
            print(m, mini, maxi)
            [self.add_constraint(self.Pr_t_max__m >= self.Pr__t[t]) for t in range(mini, maxi)]
#            [self.add_constraint(self.Pr_max__m[m] >= self.Pr__t[t]) for t in range(mini, maxi)]

    def problem_constraint_Pr_t__threshold__m(self):
        for m in self.ens['M']:
            mini = self.params['t_min__m'][m - 0]
            maxi = self.params['t_max__m'][m - 0]

            maxi = maxi - mini + 1
            mini = 0
            print(m, mini, maxi)
            [self.add_constraint(self.Pr_t_max__m >= self.params['P_souscrite']) for t in range(mini, maxi)]
#            [self.add_constraint(self.Pr_max__m[m] >= self.params['P_souscrite']) for t in range(mini, maxi)]

    def problem_constraints(self):

        self.cout_energie = self.problem_cout_energie()
        self.cout_puissance = self.problem_cout_puissance()
        self.cout_infrastructure = self.problem_cout_infrastructure()
        f_objective = self.cout_energie + self.cout_puissance + self.cout_infrastructure
        self.minimize(f_objective)
        # TODO: mettre les problem_cout_... en vecteurs/retirer la dernière somme pour avoir min et max
        self.add_kpi(self.cout_energie, "Cout Energie ($)")
        self.add_kpi(self.cout_puissance, "Cout Puissance ($)")
        self.add_kpi(self.cout_infrastructure, "Cout infrastructure ($)")
        self.add_kpi(self.max(self.params['Pb_max__m']), "max Puissance Pb_max[m] clipped (kW)")
        self.add_kpi(self.max(self.Pr__t), "max Puissance Preseau clipped (kW)")

        self.problem_constraint_SOC__v3_n_i_t()
        self.problem_constraint_SOC__n_i_t_arrivee()
        self.problem_constraint_SOC__n_i_t_depart()

        self.problem_constraint_Pr__t_v3()
        self.problem_constraint_Pr_t__max__m()
        self.problem_constraint_Pr_t__threshold__m()

        self.problem_constraint_Pess__t_v3()
        self.problem_constraint_Pess_range_v3()
        [self.add_constraint(self.Pess__n_i_t[n, i, t] * (1 - self.params['Si'][t - 0, i - 1]) == 0)
         for n in self.ens['N'] for i in self.ens['I'] for t in self.ens['T']]

        for m in self.ens['M']:
            mini = self.params['t_min__m'][m]
            maxi = self.params['t_max__m'][m]
            e = maxi - mini + 1
            s = 0

            [self.add_constraint(self.Pr_max__m[m] == self.max(self.Pr__t[t] for t in range(s, e) ))]

            [self.add_constraint(self.Pess_max__m[m] == self.max(self.Pess__t[t] for t in range(s, e) ))]

            [self.add_constraint(self.E__m[m] == self.params['delta_t'] * self.sum(self.Pr__t[t] for t in range(s, e) ))]

            [self.add_constraint(self.cout_infra_m[m] ==
                                 sum(
                                     sum(self.params['C__b_n'][n - 1, i - 1] *
                                         self.Rborne__n_i[n - 1, i - 1] for n in self.ens['N'])
                                        for i in self.ens['I']) * sum(self.params['delta_t'] for t in range(s, e)))]



    def problem_constraint_SOC__n_i_t_latch_on(self):
        W = 3
        y = self.params['Si']
        s = self.delta_ch__i_t
        for n in self.ens['N']:
            for i in self.ens['I']:
                self.add_constraints(
                    [self.sum(s[n, i, t - self.min(t + 1, W) + 1:t + 1]) <= y[t - 1, i - 1] for t in self.ens['T']])

    def problem_constraint_uc_soc_ramp_up_and_soc_ramp_down(self):
        PU = self.params['SOCmin'] / self.params['delta_t']
        PD = self.params['SOCmax'] / self.params['delta_t']
        x = self.SOC__n_i_t
        for n in self.ens['N']:
            for i in self.ens['I']:
                for t in range(1, self.ens['instant']):
                    self.add_constraint(x[n, i, t] - x[n, i, t - 1] <= PU)
                    self.add_constraint(x[n, i, t - 1] - x[n, i, t] <= PD)

