# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:31:21 2017

@author: 757743
"""
import pandas as pd
import numpy as np

class Smooth:

    def __init__(self,dirmkt,**kwargs):
        self.key_mkt = dirmkt
        self.load_data = kwargs.get('load_data', False)
        self.key_id = kwargs.get('key_id', [])
        self.raw_all = None
        if not self.load_data:
            try:
                self.raw_all = pd.read_csv(r'data/{0}.csv'.format(self.key_mkt))
            except Exception as e:
                print('failed to load market locally:', str(e))
        else:
            self.raw_all = self.load_from_database(self.key_mkt)
        
        if not self.raw_all is None:
            self.set_key_id()
            self.deseasonalize()
            self.begin_smooth()
            
            
    def begin_smooth(self):
        col_list = ['fcst_id', 'flight_dep_date', 'lcl_flw_ind', 
                    'fcst_cls', 'fcst_prd', 'td', 'id']
                    
        for key in zip(self.key_id, np.arange(1,8)):
            self.raw_key = self.raw_all.query("fcst_id == {0} & lcl_flw_ind == '{1}' & fcst_prd == {2}".format(
                                              key[0][0], key[0][1], key[1] ))[col_list]
            
            raw_id
            new = pd.concat([raw_key, new_id], axis = 1)
#            seprate raw id and raw td and transpose cls into columns
            
    def load_from_database(self, dirmkt):
        pass

    def set_key_id(self):
        self.key_id = self.raw_all.query("pool_cd == 'M'")[['fcst_id','lcl_flw_ind']].drop_duplicates().values

    def get_key_mkt(self):
        return print(self.key_mkt)

    def get_key_id(self):
        return print(self.key_id)
        
    def deseasonalize(self):
        
#        change the data type to save some space
        for c, dtype in zip(self.raw_all.columns, self.raw_all.dtypes):
            if dtype == np.float64:
                self.raw_all[c] = self.raw_all[c].astype(np.float32)
            if dtype == np.int64:
                self.raw_all[c] = self.raw_all[c].astype(np.int32)
                
                
        self.raw_all = self.raw_all.query("pool_cd == 'M'")
        self.raw_all.dropna(axis = 0, subset =['tt', 'it', 'td', 'id', 'seas_index'], inplace = True )
        self.raw_all.eval('td = td /seas_index', inplace =True)
        self.raw_all.eval('id = id /seas_index', inplace =True)
        self.raw_all.sort_values()
        
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
    testmkt.get_key_id()