# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:31:21 2017

@author: 757743
"""
import pandas as pd
import numpy as np
import itertools
import gc

class Smooth:

    def __init__(self,dirmkt,**kwargs):
        self.key_mkt = dirmkt
        self.load_data = kwargs.get('load_data', False)
        self.key_id = kwargs.get('key_id', [])
        self.raw_all = None
        if not self.load_data:
            try:
                print(self.key_mkt, 'loading data from csv...')
                self.raw_all = pd.read_csv(r'data/{0}.csv'.format(self.key_mkt))
            except Exception as e:
                print('failed to load market locally:', str(e))
        else:
            self.raw_all = self.load_from_database(self.key_mkt)
        
    def begin_smooth(self):

        if not self.raw_all is None:
            self.deseasonalize()
            self.set_key_id()
            self.smooth()
            
    def smooth(self):
     
        self.alpha = 0.1
        col_list = ['fcst_id', 'flight_dep_date', 'lcl_flw_ind', 
                    'fcst_cls', 'fcst_prd', 'td', 'id']
        self.td_output = None
        self.id_output = None
        self.iter_list = itertools.product(self.key_id, np.arange(1,8), np.arange(1,8))
        iter_length = len(self.key_id) * 7 * 7
        for cnt, self.key in enumerate(self.iter_list):
            self.raw_key = self.raw_all.query("fcst_id == {0} & lcl_flw_ind == '{1}' & fcst_prd == {2} & dow == {3}".format(
                                              self.key[0][0], self.key[0][1], self.key[1] , self.key[2]))[col_list]

            if self.raw_key.shape[0] > 10:                               
                print('processing key:', self.key, 'current: {0} of {1}'.format(cnt, iter_length))          
                self.raw_key_t = self.raw_key.pivot(index='flight_dep_date', columns='fcst_cls', values='td')
                self.smooth_initial()
                self.outlier_removal()
                self.smooth_output()
                
                self.td_key = self.base_key
                
                if not isinstance(self.td_output,pd.DataFrame):
                    self.td_output = self.td_key
                else:
                    self.td_output = self.td_output.append(self.td_key)
                    
                del self.raw_key_t, self.base_key
                gc.collect()
                   
                self.raw_key_t = self.raw_key.pivot(index='flight_dep_date', columns='fcst_cls', values='id')
                self.smooth_initial()
                self.outlier_removal()
                self.smooth_output()
                
                self.id_key = self.base_key       
                
                if not isinstance(self.id_output,pd.DataFrame):
                    self.id_output = self.id_key
                else:
                    self.id_output = self.id_output.append(self.id_key)
            else:
                print('key data missing:', self.key, 'current: {0} of {1}'.format(cnt, iter_length))
#            seprate raw id and raw td and transpose cls into columns
         
         
    def load_from_database(self, dirmkt):
        pass

    def set_key_id(self):
        
        print('Collecting fcst keys..')
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
                
        print('start sorting and deseasonalization...')
        self.raw_all = self.raw_all.query("pool_cd == 'M'")
        self.raw_all.dropna(axis = 0, subset =['tt', 'it', 'td', 'id', 'seas_index'], inplace = True )
        self.raw_all.eval('td = td /seas_index', inplace =True)
        self.raw_all.eval('id = id /seas_index', inplace =True)
        self.raw_all['flight_dep_date'] = pd.to_datetime(self.raw_all['flight_dep_date'])
        self.raw_all.sort_values([ 'fcst_id', 'lcl_flw_ind', 'fcst_prd', 'dow','fcst_cls','flight_dep_date'], inplace = True)
        
    def smooth_initial(self):
        
        s_key = self.raw_key_t.copy()        
        if s_key.shape[0] >= 10:
            for column in s_key:
                s_key[column][0] = s_key[column][:10].mean()
                s_key.rename(columns = {column : str(column) + 's' }, inplace = True)
            s_key[1:] = 0

        else:
            for column in s_key:
                s_key[column][0] = s_key[column].mean()
                s_key.rename(columns = {column : str(column) + 's' }, inplace = True)
            s_key[1:] = 0
        
        self.xs_key = pd.concat([self.raw_key_t, s_key], axis=1)
#            create smoothed value. pandas ewm can do it
        for r in range(1, self.xs_key.shape[0]):
            self.xs_key.iloc[r, 10:] =  self.xs_key.iloc[r, 0:10].values * self.alpha + self.xs_key.iloc[r -1 , 10:] * (1 - self.alpha)
            

            
    def outlier_removal(self):

        self.d_key = self.xs_key.iloc[:,0:10] - self.xs_key.iloc[:,10:20].values
        for column in self.d_key:
            self.d_key[column][0] = self.d_key[column][:10].mean()
            self.d_key.rename(columns = {column : str(column) + 'd' } , inplace = True)
        
        self.d_key['w'] = 0
#        order and set weight
        for i in range(0, self.d_key.shape[0]):
            weight = (1- self.alpha) ** (self.d_key.shape[0] - i)

#            self.d_key['w'][i] = weight
            self.d_key.iloc[i, 10] = weight

#        dev 1
        d1_key = self.d_key.copy()
        d1_key = self.d_key.iloc[:,0:10].multiply(self.d_key['w'], axis = 'index')
#        dev2
        d2_key = self.d_key.copy()
        d2_key.iloc[:, 0:10] = d2_key.iloc[:, 0:10] - d1_key.sum()/self.d_key['w'].sum()
        d2_key.iloc[:, 0:10] = d2_key.iloc[:, 0:10] ** 2
        d2_key.iloc[:, 0:10] = d2_key.iloc[:, 0:10].multiply(d2_key['w'], axis = 'index')
#        weighted s.e.
#        wtd_se = np.sqrt(d2_key.iloc[:, 0:10].sum())/d2_key['w'].sum()
        wtd_se = d2_key.iloc[:, 0:10].copy()
        for r in range (0, d2_key.shape[0]):
            rp = r+ 1
            wtd_se.iloc[r, 0:10] = np.sqrt(d2_key.iloc[0:rp, 0:10].sum())/d2_key['w'][0:rp].sum()
#        reset index to 1s so dataframe can broadcast 
#        wtd_se = wtd_se.reset_index()
#        wtd_se['fcst_cls'] = wtd_se['fcst_cls'].apply( lambda x: x[:-1] + 's')
#        wtd_se = wtd_se.set_index('fcst_cls').squeeze()
        for column in wtd_se:
            wtd_se.rename(columns = {column : column[:-1] + 's' }, inplace = True)
        
#       confidence invertal
        cu_key = self.xs_key.iloc[:, 10:] +  wtd_se * 1.5
        cl_key = self.xs_key.iloc[:, 10:] -  wtd_se * 1.5
        cl_key[cl_key<0] = 0

#        as matrix creates view
        x_matrix = self.raw_key_t.as_matrix()
        cu_matrix = cu_key.as_matrix()
        cl_matrix = cl_key.as_matrix()
        
        self.flag_matrix =  (x_matrix < cl_matrix) | (x_matrix > cu_matrix)
        x_matrix[self.flag_matrix] = None
        

    def smooth_output(self):
        self.base_key = []
        for i in range(0, 10):
#            s_cls =  self.raw_key_t.iloc[:, [i]]
#            convert multiple columns into signle column dataframe to apply final smoothing
#            we can no longer apply method on all 10 clasees due to removed outliers
            s_cls =  self.raw_key_t.iloc[:, [i]].copy()
            s_cls.dropna(axis = 0, inplace = True)
            if s_cls.shape[0] > 0:
#            rename the column name to be x
                col = i + 1
                s_cls.rename(columns= {col: 'x'}, inplace =True)
                s_cls['x'].astype(np.float32)
                s_cls['s'] = s_cls['x']
    #            get initial value
                if s_cls.shape[0] > 10:
                    s_cls.iloc[0, 1] = s_cls['x'][:10].mean()
                else:
                    s_cls.iloc[0, 1] = s_cls['x'].mean()
    #            use buildin ewm method
                s_cls['s_std'] = s_cls['s'].ewm(alpha=self.alpha).std(bias=False).shift(1)
                s_cls['s'] = s_cls['s'].ewm(alpha=self.alpha).mean().shift(1)
                s_cls['s_date'] = s_cls.index.values
                s_cls['s_date'] = s_cls['s_date'].shift(1)
                
                s_cls.drop('x', axis = 1, inplace =True)
    #            a flag for outlier

#            merge with original data
                orig_cls = self.xs_key.iloc[:, [i]].copy()
                orig_cls.rename(columns= {col: 'x'}, inplace =True)
                s_cls['flag'] = False
                s_cls = orig_cls.merge(s_cls , how ='left', right_index =True, left_index =True)
    
    #            a forward fill, so initial values will not be filled
                s_cls.iloc[:,:-1] = s_cls.iloc[:,:-1].fillna(method = 'ffill')
                s_cls['fcst_cls'] = col
                s_cls['flag'].fillna(value = True, inplace = True)
            
            if i == 0:
                self.base_key = s_cls
                
            elif isinstance(self.base_key, pd.DataFrame):
                self.base_key = self.base_key.append(s_cls)
                
            elif isinstance(s_cls, pd.DataFrame):
                self.base_key = s_cls
                
#        add all the key values back
        if isinstance(self.base_key, pd.DataFrame):
            self.base_key['fcst_id'] = self.key[0][0]
            self.base_key['lcl_flw_ind'] = self.key[0][1]
            self.base_key['fcst_prd'] = self.key[1]
            self.base_key['dow'] = self.key[2]
            self.base_key['pool_cd'] = 'M'

    def get_full_output(self, **kwargs):
        """
        output the one-period-ahead fcst to indexed by actual demand 
        reseasonalized forecast
        """
        save = kwargs.get('save', None)
        
        self.td_output.rename( columns = {'x':'td', 's':'td_fcst', 's_std':'td_fcst_std'
                                         , 's_date':'eff_from_date', 'flag':'td_flag'}, inplace =True)
        self.id_output.rename( columns = {'x':'id', 's':'id_fcst', 's_std':'id_fcst_std'
                                         , 's_date':'eff_from_date', 'flag':'td_flag'}, inplace =True)
        
        base_col = ['flight_dep_date', 'dow', 'fcst_id', 'lcl_flw_ind', 'fcst_prd','fcst_cls',
                     'traffic_ct', 'td', 'id', 'mwp', 'seas_index', 'tt', 'it'
                     ,'tt_perc', 'it_perc'
                     ]
        merge_txt = ['flight_dep_date', 'dow', 'fcst_id', 'lcl_flw_ind', 'fcst_prd','fcst_cls']
        full_output = self.raw_all[base_col].merge(pd.merge(self.td_output.reset_index(), self.id_output.reset_index()
                                                    ,how='inner', on = merge_txt ), how = 'left', on = merge_txt)
                     
        if not save:
            return full_output
        else:
            full_output.to_csv(r'output\{}'.format(self.key_mkt))
    
    def get_smooth_output(self):
        """
        output the deseas fcst by effective date range
        """
        pass
    
if __name__ == '__main__':
    testmkt = Smooth('mcodfw', load_data = False)
#    testmkt.begin_smooth()


#import time                                                
#
#def timeme(method):
#    def wrapper(*args, **kw):
#        startTime = int(round(time.time() * 1000))
#        result = method(*args, **kw)
#        endTime = int(round(time.time() * 1000))
#
#        print(endTime - startTime,'ms')
#        return result
#
#    return wrapper