#NOTE: Everything Till Feature Addition is same as in 'Output_code_1.py'
#THIS CODE performs cleaning and feature addition in train_data and outputs train_data from Jan,2015 to Aug,2015. 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import gc
from sklearn import preprocessing

prod_cols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

train_data = pd.read_csv("../input/santander-pr/train.csv")
test_data=pd.read_csv("../input/santander-pr/test.csv")
 
gc.collect() # Collects garbage memory. 



# <-------------------------------Data Cleaning Starts------------------------------->

#Drop ['nomprov','ult_fec_cli_1t','tipodom'] features because of following reason:
#nomprov has one to one mapping with codprov 
#'ult_fec_cli_1t' and 'conyuemp' has majority values as NULL
#tipodom has all entries '1.0'
def drop_columns(data):
    data.drop(['nomprov','ult_fec_cli_1t','tipodom'],axis='columns',inplace=True)
drop_columns(train_data)
drop_columns(test_data)

#Convert indresi feature into 0,1 values rather than String
def transform_indresi(x):
    if(x=='S'):
        return 1
    else:
        return 0
train_data['indresi']=train_data['indresi'].transform(transform_indresi)
test_data['indresi']=test_data['indresi'].transform(transform_indresi)


test_data["renta"]   = pd.to_numeric(test_data["renta"], errors="coerce")
unique_prov = test_data[test_data.cod_prov.notnull()].cod_prov.unique()
grouped = test_data.groupby("cod_prov")["renta"].median()
 
def impute_renta(df): # fill null values of renta by the median of corresponding cod_prov.
    df["renta"]   = pd.to_numeric(df["renta"], errors="coerce")       
    for cod in unique_prov:
        df.loc[df['cod_prov']==cod,['renta']] = df.loc[df['cod_prov']==cod,['renta']].fillna({'renta':grouped[cod]}).values
    df.renta.fillna(df["renta"].median(), inplace=True)
    
def transform_renta(x): #applying 0-1 normalization.
    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value
    return (x-min_value)/range_value
impute_renta(train_data)
train_data['renta']=train_data['renta'].transform(transform_renta)
test_data['renta']=test_data['renta'].transform(transform_renta)
# print(train_data['renta'])

''' 
Fill null values in 'Conyuemp' with 'N'.
Conyuemp says if the customer is spouse of employee or not.
A spouse of employee would like to completely fill the details and there wouldn't be any null values.
therefore, it makes sense to fill 'N' in every null place.
'''
train_data['conyuemp']=train_data['conyuemp'].fillna('N') 
le = preprocessing.LabelEncoder()
le.fit(train_data.conyuemp)
train_data['conyuemp']=le.transform(train_data.conyuemp) 
 
test_data['conyuemp']=test_data['conyuemp'].fillna('N')
test_data['conyuemp']=le.transform(test_data.conyuemp) 


def get_cod_prov(data):
    data['cod_prov']=data['cod_prov'].fillna(-1.0)
    data['cod_prov']=data['cod_prov'].astype(int)
get_cod_prov(train_data)
get_cod_prov(test_data)

gc.collect()

#Segmento : Fill null values with 'Unknown' and do label encoding afterwards.
feature_name='segmento'
train_data[feature_name].fillna('Unknown',inplace=True)
le = preprocessing.LabelEncoder()
le.fit(train_data.segmento)
train_data[feature_name]=le.transform(train_data.segmento)
 
test_data['segmento'].fillna('Unknown',inplace=True)
test_data['segmento']=le.transform(test_data.segmento)
#train_data.drop([feature_name],axis=1,inplace=True)


# Canal_entrada: Got this dictionary from a notebook and we used it shamelessly without knowing what it is. And we still don't know.

canal_dict = {'KAI': 35,'KBG': 17,'KGU': 149,'KDE': 47,'KAJ': 41,'KCG': 59,
 'KHM': 12,'KAL': 74,'KFH': 140,'KCT': 112,'KBJ': 133,'KBL': 88,'KHQ': 157,'KFB': 146,
              'KFV': 48,'KFC': 4,
 'KCK': 52,'KAN': 110,'KES': 68,'KCB': 78,'KBS': 118,'KDP': 103,'KDD': 113,'KBX': 116,
              'KCM': 82,
 'KAE': 30,'KAB': 28,'KFG': 27,'KDA': 63,'KBV': 100,'KBD': 109,'KBW': 114,'KGN': 11,
 'KCP': 129,'KAK': 51,'KAR': 32,'KHK': 10,'KDS': 124,'KEY': 93,'KFU': 36,'KBY': 111,
 'KEK': 145,'KCX': 120,'KDQ': 80,'K00': 50,'KCC': 29,'KCN': 81,'KDZ': 99,'KDR': 56,
 'KBE': 119,'KFN': 42,'KEC': 66,'KDM': 130,'KBP': 121,'KAU': 142,'KDU': 79,
 'KCH': 84,'KHF': 19,'KCR': 153,'KBH': 90,'KEA': 89,'KEM': 155,'KGY': 44,'KBM': 135,
 'KEW': 98,'KDB': 117,'KHD': 2,'RED': 8,'KBN': 122,'KDY': 61,'KDI': 150,'KEU': 72,
 'KCA': 73,'KAH': 31,'KAO': 94,'KAZ': 7,'004': 83,'KEJ': 95,'KBQ': 62,'KEZ': 108,
 'KCI': 65,'KGW': 147,'KFJ': 33,'KCF': 105,'KFT': 92,'KED': 143,'KAT': 5,'KDL': 158,
 'KFA': 3,'KCO': 104,'KEO': 96,'KBZ': 67,'KHA': 22,'KDX': 69,'KDO': 60,'KAF': 23,'KAW': 76,
 'KAG': 26,'KAM': 107,'KEL': 125,'KEH': 15,'KAQ': 37,'KFD': 25,'KEQ': 138,'KEN': 137,
 'KFS': 38,'KBB': 131,'KCE': 86,'KAP': 46,'KAC': 57,'KBO': 64,'KHR': 161,'KFF': 45,
 'KEE': 152,'KHL': 0,'007': 71,'KDG': 126,'025': 159,'KGX': 24,'KEI': 97,'KBF': 102,
 'KEG': 136,'KFP': 40,'KDF': 127,'KCJ': 156,'KFR': 144,'KDW': 132,-1: 6,'KAD': 16,
 'KBU': 55,'KCU': 115,'KAA': 39,'KEF': 128,'KAY': 54,'KGC': 18,'KAV': 139,'KDN': 151,
 'KCV': 106,'KCL': 53,'013': 49,'KDV': 91,'KFE': 148,'KCQ': 154,'KDH': 14,'KHN': 21,
 'KDT': 58,'KBR': 101,'KEB': 123,'KAS': 70,'KCD': 85,'KFL': 34,'KCS': 77,'KHO': 13,
 'KEV': 87,'KHE': 1,'KHC': 9,'KFK': 20,'KDC': 75,'KFM': 141,'KHP': 160,'KHS': 162,
 'KFI': 134,'KGV': 43}
train_data['canal_entrada'].fillna(-1,inplace=True) #Filling with mode value
train_data['canal_entrada'] = train_data['canal_entrada'].map(lambda x: canal_dict[x]).astype(np.int16)
 
test_data['canal_entrada'].fillna(-1,inplace=True)
test_data['canal_entrada'] = test_data['canal_entrada'].map(lambda x: canal_dict[x]).astype(np.int16)


def get_age(data): #remove outliers, then fill age with mean value and then do 0-1 normalization.
    data["age"]   = pd.to_numeric(data["age"], errors="coerce")
    data.loc[data.age < 18,"age"]  = data.loc[(data.age >= 18) & (data.age <= 30),"age"].mean(skipna=True)
    data.loc[data.age > 100,"age"] = data.loc[(data.age >= 30) & (data.age <= 100),"age"].mean(skipna=True)
    data["age"].fillna(data["age"].mean(),inplace=True)
    data["age"] = data["age"].astype(int)    
    maxage = data['age'].max()
    minage = data['age'].min()
    rangeage = maxage-minage
    data['age'].apply(lambda x: round((x-minage)/maxage))
    
get_age(train_data)
get_age(test_data)

def get_indrel_1mes(data): #Fill with mode value
    data['indrel_1mes']=pd.to_numeric(data['indrel_1mes'],errors="coerce")
    fill_val=data['indrel_1mes'].mode()[0]    
    data['indrel_1mes'].fillna(fill_val,inplace=True)
get_indrel_1mes(train_data)
get_indrel_1mes(test_data)
#train_data=train_data.drop(['indrel_1mes'],axis='columns')

gc.collect()

#Label encoding on ind_actividad after filling null value with -1
feature_name='ind_actividad_cliente'
train_data[feature_name].fillna(-1.0,inplace=True)
le.fit(train_data['ind_actividad_cliente'])
train_data['ind_actividad_cliente']=le.transform(train_data['ind_actividad_cliente'])

test_data[feature_name].fillna(-1.0,inplace=True)
test_data['ind_actividad_cliente']=le.transform(test_data['ind_actividad_cliente'])
#train_data.drop([feature_name],axis=1,inplace=True)

gc.collect()

'''
Fill fecha_alta with the year value by taking only the first 4 characters of date.
If date is '2020-12-19', it will transform to 2020.
And fill null values with '0'
and then convert to numeric type from string.
''' 
def get_fecha_alta(data):
    def transform_fecha_alta(x):
        if(x in [np.nan,'',-1.0]):
            return x
        else:
            return x[0:4]
    data['fecha_alta']=data['fecha_alta'].transform(transform_fecha_alta)
    tmp_data=data['fecha_alta']
    tmp_data=tmp_data.dropna(how='any')
#     print(tmp_data.unique())
    tmp_data=tmp_data.astype(int)
    median_val=tmp_data.median()
    data['fecha_alta'].fillna(str(round(median_val,0)),inplace=True)
    data['fecha_alta']=data['fecha_alta'].astype(float)
get_fecha_alta(train_data)
get_fecha_alta(test_data)


# Label Encoding for ind_empleado
feature_name='ind_empleado'
fill_val = train_data[feature_name].mode()[0]
train_data[feature_name].fillna(fill_val,inplace=True)
le = preprocessing.LabelEncoder()
le.fit(train_data['ind_empleado'])
train_data['ind_empleado']=le.transform(train_data['ind_empleado'])
 
test_data[feature_name].fillna(fill_val,inplace=True)
test_data['ind_empleado']=le.transform(test_data['ind_empleado'])

gc.collect()

#Label Encoding for ind_nuevo also.
feature_name='ind_nuevo'
fill_val = train_data[feature_name].mode()[0]
train_data[feature_name].fillna(fill_val,inplace=True)
le.fit(train_data['ind_nuevo'])
train_data['ind_nuevo']=le.transform(train_data['ind_nuevo'])
 
test_data[feature_name].fillna(fill_val,inplace=True)
test_data['ind_nuevo']=le.transform(test_data['ind_nuevo'])
#train_data.drop([feature_name],axis=1,inplace=True)

# mp is mapping from country name to it's distance from Spain. 
#Use this mapping to fill value for pais_residencia and fill null values with the mapping of spain.
def get_pais_residencia(data):
    mp={'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27}
    data['pais_residencia'].fillna('ES',inplace=True)
    def transform_pais_residencia(x):
        return str(mp[x])
    data['pais_residencia']=data['pais_residencia'].transform(transform_pais_residencia)
    data['pais_residencia']=data['pais_residencia'].astype(int)
get_pais_residencia(train_data)
get_pais_residencia(test_data)

#fill null values with 'N' followed by label encoding.
def get_indfall(data):
    data['indfall'].fillna('N',inplace=True)
    def transform_indfall(x):
        if(x=='N'):
            return '0'
        else:
            return '1'
    data['indfall']=data['indfall'].transform(transform_indfall)
    data['indfall']=data['indfall'].astype(int)
get_indfall(train_data)
get_indfall(test_data)

#Sexo: Fill null values with unknown, then do label encoding.
feature_name='sexo'
train_data[feature_name].fillna('Unknown',inplace=True)
le.fit(train_data['sexo'])
train_data['sexo']=le.transform(train_data['sexo'])
 
test_data[feature_name].fillna('Unknown',inplace=True)
test_data['sexo']=le.transform(test_data['sexo'])
#train_data.drop([feature_name],axis=1,inplace=True)

#Fill null values with 'N' followed by label encoding.
def get_indext(data):
    data['indext'].fillna('N',inplace=True)
    def transform_indext(x):
        if(x=='N'):
            return '0'
        else:
            return '1'        
    data['indext']=data['indext'].transform(transform_indext)
    data['indext']=data['indext'].astype(float)
get_indext(train_data)
get_indext(test_data)


# Fill Null values in 'indrel' followed by Label encoding.
feature_name='indrel'
train_data[feature_name].fillna(1.0,inplace=True)
le.fit(train_data['indrel'])
train_data['indrel']=le.transform(train_data['indrel'])
 
test_data[feature_name].fillna(1.0,inplace=True)
test_data['indrel']=le.transform(test_data['indrel'])

#Fill null values in 'tiprel_1mes' with 'Unknown' followed by label encoding.
feature_name='tiprel_1mes'
train_data[feature_name].fillna('Unknown',inplace=True) #mode
le.fit(train_data['tiprel_1mes'])
train_data['tiprel_1mes']=le.transform(train_data['tiprel_1mes'])
 
test_data[feature_name].fillna('Unknown',inplace=True) #mode
test_data['tiprel_1mes']=le.transform(test_data['tiprel_1mes'])


#Antiguedad : Fill null values and 'NA' with 0 and then convert to integer type.
def get_antiguedad(data):
    def transform_antiguedad(x):
        x=str(x)
        x=x.strip()
        if(x=='NA'):
            return 0 
        elif(int(x)<=0):
            return '0'
        return x
    data['antiguedad']=data['antiguedad'].transform(transform_antiguedad)
    data['antiguedad']=(data['antiguedad'].astype(int))
get_antiguedad(train_data)
get_antiguedad(test_data)
# <-------------------------------Data Cleaning ends------------------------------->
train_data.dropna(how='any',inplace=True)
 
prod_cols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1'   ,
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1'       ,
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1'      ,
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1'      ,
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1'       ,
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1'      ,
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1'       ,
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
 
my_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1','ncodpers']

# <-------------------------------Feature Addition Starts--------------------------->

for col in prod_cols:
    train_data[col]=train_data[col].astype(int)

'''
Lag Features added: 
Lag features denotes the previous product value, here lag features are added for every product.
For example: x_lag1 value for a row will store the previous month value of that customer for product x.
similarly, 'x_lag'+str(i) will store the value of product x for that customer i months back. 
'''
# segregate data month wise into variables d1 to d8.


train_data=train_data[(train_data.fecha_dato=='2015-08-28') | (train_data.fecha_dato=='2015-07-28') | (train_data.fecha_dato=='2015-06-28') | (train_data.fecha_dato=='2015-05-28') | (train_data.fecha_dato=='2015-04-28') | (train_data.fecha_dato=='2015-03-28') | (train_data.fecha_dato=='2015-02-28') | (train_data.fecha_dato=='2015-01-28')]
d1 = train_data[train_data.fecha_dato=='2015-08-28']
d2 = train_data[train_data.fecha_dato=='2015-07-28']
d3 = train_data[train_data.fecha_dato=='2015-06-28']
d4 = train_data[train_data.fecha_dato=='2015-05-28']
d5 = train_data[train_data.fecha_dato=='2015-04-28']
d6 = train_data[train_data.fecha_dato=='2015-03-28']
d7 = train_data[train_data.fecha_dato=='2015-02-28']
d8 = train_data[train_data.fecha_dato=='2015-01-28']

# perform left join so as to add lag features using d1 to d8 as second variable
train_data = d1.merge(d2[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag1'))
train_data = train_data.merge(d3[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag2'))
train_data = train_data.merge(d4[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag3'))
train_data = train_data.merge(d5[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag4'))
train_data = train_data.merge(d6[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag5'))
train_data = train_data.merge(d7[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag6'))
train_data = train_data.merge(d8[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag7'))

#Similarly add lag features for test_data
test_data = test_data.merge(d1[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag1'))
test_data = test_data.merge(d2[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag2'))
test_data = test_data.merge(d3[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag3'))
test_data = test_data.merge(d4[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag4'))
test_data = test_data.merge(d5[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag5'))
test_data = test_data.merge(d6[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag6'))
test_data = test_data.merge(d7[my_cols],how='left',on='ncodpers',suffixes=(None,'_lag7'))

del d1,d2,d3,d4,d5,d6,d7,d8 #Not needed after this step. delete to prevent unnecassary memory blockage

#Fill any left null value with 0 just to ensure nothing is null after this step.
train_data=train_data.fillna(0)


'''
Toggle features added:
x_sum00 denotes how many times did value change from 0 to 0 from any month i to month i+1 for a particular customer.

toggle features describe traits of a person. If the value of x_sum01 is high then the probability of person toggling is high if previous value is 0.

'''

#Adding toggle features for train_data
for c in prod_cols:
    sum00=np.zeros(len(train_data))
    sum01=np.zeros(len(train_data))
    sum10=np.zeros(len(train_data))
    sum11=np.zeros(len(train_data))
    for lag_no in range(1,7): 
        x=train_data[c+'_lag'+str(lag_no)]
        y=train_data[c+'_lag'+str(lag_no+1)]
        x=x.fillna(0)
        y=y.fillna(0)
        x=x.astype(int)
        y=y.astype(int)
        sum00+=(x^y^1)&(x^1) 	# will yield 1 only if x=0 and y=0, else 0.
        sum11+=(x^y^1)&(x)		# will yield 1 only if x=1 and y=1, else 0.
        sum01+=(x^y)&(y)		# will yield 1 only if x=0 and y=1, else 0.
        sum10+=(x^y)&(x)		# will yield 1 only if x=1 and y=0, else 0.
    train_data[c+'_sum00']=pd.Series(sum00)
    train_data[c+'_sum01']=pd.Series(sum01)
    train_data[c+'_sum10']=pd.Series(sum10)
    train_data[c+'_sum11']=pd.Series(sum11)

#We decided to keep only 3 lag features since more than 3 lag features would be overkill.
#Deleting lag4 to lag7 for every product.
for c in prod_cols:
    for lag_no in range(4,8):
        train_data.drop([c+'_lag'+str(lag_no)],inplace=True,axis='columns')

#We don't perform any operations on test_data since this is already done in 'Output_code_1.py'

# <-------------------------------Feature Addition done--------------------------->

train_data.drop('fecha_dato',axis='columns',inplace=True)

#Output the train_data  and test_data to "train_data_first_half.csv" and "test_data.csv" respectively.
# This output .csv files in combination with 'train_data_first_half.csv' file will be used in model.py file 
train_data.to_csv('train_data_first_half.csv',index=False)

