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
        
        self.alpha = 0.1
        col_list = ['fcst_id', 'flight_dep_date', 'lcl_flw_ind', 
                    'fcst_cls', 'fcst_prd', 'td', 'id']
                    
        for key in zip(self.key_id, np.arange(1,8), np.arange(1,8)):
            self.raw_key = self.raw_all.query("fcst_id == {0} & lcl_flw_ind == '{1}' & fcst_prd == {2} & dow == {3}".format(
                                              key[0][0], key[0][1], key[1] , key[2]))[col_list]
            
            
            self.raw_key = self.raw_key.pivot(index='flight_dep_date', columns='fcst_cls', values='td')
            self.smooth_initial()
            
            self.outlier_removal()
            
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
        self.raw_all.sort_values([ 'fcst_id', 'lcl_flw_ind', 'fcst_prd', 'fcst_cls','flight_dep_date'], inplace = True)
        
    def smooth_initial(self):
        
        if self.raw_key.shape[0] >= 10:
            
            s_key = self.raw_key.copy()
            for column in s_key:
                s_key[column][0] = s_key[column][:10].mean()
                s_key.rename(columns = {column : str(column) + 's' }, inplace = True)
            s_key[1:] = 0

            self.xs_key = pd.concat([self.raw_key, s_key], axis=1)
#            create smoothed value. pands ewm can do it
            for r in range(1, self.xs_key.shape[0]):
                self.xs_key.iloc[r, 10:] =  self.xs_key.iloc[r, 0:10].values * self.alpha + self.xs_key.iloc[r -1 , 10:] * (1 - self.alpha)
            

            
    def outlier_removal(self):

        self.d_key = self.xs_key.iloc[:,0:10] - self.xs_key.iloc[:,10:20].values
        for column in self.d_key:
            self.d_key[column][0] = self.d_key[column][:10].mean()
            self.d_key.rename(columns = {column : str(column) + 'd' } , inplace = True)
            
#        order and set weight
        for i in range(0, self.d_key.shape[0]):
            weight = (1- self.alpha) ** (self.d_key.shape[0] - i)
            self.d_key['w'][i] = weight
#        dev 1
        d1_key = self.d_key.copy()
        d1_key = self.d_key.iloc[:,0:10].multiply(self.d_key['w'], axis = 'index')
#        dev2
        d2_key = self.d_key.copy()
        d2_key.iloc[:, 0:10] = d2_key.iloc[:, 0:10] - d1_key.sum()/self.d_key['w'].sum()
        d2_key.iloc[:, 0:10] = d2_key.iloc[:, 0:10] ** 2
        d2_key.iloc[:, 0:10] = d2_key.iloc[:, 0:10].multiply(d2_key['w'], axis = 'index')
#        weighted s.e.
        wtd_se = np.sqrt(d2_key.iloc[:, 0:10].sum())/d2_key['w'].sum()
#        reset index to 1s so dataframe can broadcast 
        wtd_se = wtd_se.reset_index()
        wtd_se['fcst_cls'] = wtd_se['fcst_cls'].apply( lambda x: x[:-1] + 's')
        wtd_se = wtd_se.set_index('fcst_cls').squeeze()

        
#       confidence invertal
        cu_key = self.xs_key.iloc[:, 10:] +  wtd_se * 1.5
        cl_key = self.xs_key.iloc[:, 10:] -  wtd_se * 1.5
        cl_key[cl_key<0] = 0

#        as matrix creates view
        x_matrix = self.raw_key.as_matrix()
        cu_matrix = cu_key.as_matrix()
        cl_matrix = cl_key.as_matrix()
        
        self.flag_matrix =  (x_matrix < cl_matrix) | (x_matrix > cu_matrix)
        x_matrix[self.flag_matrix] = None
        
#            create x matrix for actual values
#        self.x_matrix = self.raw_key.as_matrix()
#            create s matrix for smoothed values
#        self.s_matrix = np.zeros((self.x_matrix.shape[0], self.x_matrix.shape[1] + 1))
#            add order/index to the s matrix
#        self.s_matrix[:,0] = np.arange(self.x_matrix.shape[0])
            
    def smooth_output(self):
        pass

    def resesonalize(self):
        pass

    

if __name__ == '__main__':
    testmkt = Smooth('mcodfw', load_data = False)
    testmkt.get_key_mkt()
    testmkt.get_key_id()