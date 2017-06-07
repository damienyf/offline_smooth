# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:31:21 2017

@author: 757743
"""
import pandas as pd

class Smooth:

    def __init__(self,dirmkt,**kwargs):
        self.key_mkt = dirmkt
        self.load_data = kwargs.get('load_data', False)
        self.key_id = kwargs.get('key_id', [])

        if not self.load_data:
            try:
                self.raw_all = pd.read_csv(r'data/{0}'.format(self.key_mkt))
            except Exception as e:
                print('failed to load market locally:', str(e))
        else:
            self.raw_all = self.load_from_database(self.key_mkt)

    def load_from_database(self, dirmkt):
        pass

    def set_key_id(self):
        self.key_id = self.raw_all.query("pool_cd = 'M'")['FCST_ID','LCL_FLW_IND'].drop_duplicates().values

    def get_key_mkt(self):
        return print(self.key_mkt)

    def deseasonalize(self):
        pass

    def smooth_initial(self):
        pass     

    def outlier_removal(self):
        pass
    
    def smooth_output(self):
        pass

    def resesonalize(self):
        pass

    

if __name__ == '__main__':
    testmkt = Smooth('mcodfw', load_data = False)
    testmkt.get_key_mkt()