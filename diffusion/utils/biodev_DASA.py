# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 

import scipy

from pingouin import ancova
import pingouin as pg

class DASA_metric():
        
    def __init__(self):
        self.df_combined_real_fake = None
        
    def load_distribution(self, path):
        
        df_reals = pd.read_csv(path, sep="\t", header=None)
        
        self.df_combined_real_fake = pd.DataFrame()
        
        for index, row in df_reals.iterrows():
            for col_name in df_reals.columns:
                self.df_combined_real_fake = self.df_combined_real_fake.append(pd.DataFrame([[col_name, row[col_name], 'Real']],
                                                                    columns=['frame', 'prediction','real_fake']),
                                                                          ignore_index=True)
                
    def compute_DASA(self, fake_sequence):
        
        df_predictions = fake_sequence

        # df_combined_sample = pd.DataFrame()
        
        df_combined_sample = self.df_combined_real_fake.copy(deep=True)

        for idx, col_name in enumerate(df_predictions):
            df_combined_sample = df_combined_sample.append(pd.DataFrame([[idx, col_name, 'Fake']], 
                                                                 columns=['frame', 'prediction','real_fake']),
                                                                 ignore_index=True)        
        
        DASA = 0
        ancova_test = ancova(data=df_combined_sample, dv='prediction', covar='frame', between='real_fake')

        correlation = pg.corr(df_combined_sample.tail(15)["frame"], 
                              df_combined_sample.tail(15)["prediction"],
                              method="spearman")
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=df_combined_sample.tail(15)["frame"],
                                                                 y=df_combined_sample.tail(15)["prediction"])

        
        ''' 
        std_err = cuanto mejor más cerca de 0
        correlation["r"] = Cuanto mejor más cerca de 1
        ancova_test["p-unc"] = Cuanto mejor, más cerca de 1
        ancova_test["F"] = Cuanto mejor más cerca de 0
        '''
        
        
        # if(np.isnan(float(correlation["r"]))):
        #     DASA = ((1-std_err))*ancova_test["F"][0]
        # else:
        #     DASA = (1-float(correlation["r"])*(1-std_err))*ancova_test["F"][0]
        
#         print("CORRELATION:",float(correlation["r"]))
#         print("STDERROR:",std_err)
#         print("ANCOVA:",1 - ancova_test["p-unc"][0])
        
        if(np.isnan(float(correlation["r"]))):
            DASA = std_err + (1 - ancova_test["p-unc"][0])
        else:
            DASA = (1-float(correlation["r"])) + std_err + (1 - ancova_test["p-unc"][0])

        return DASA
