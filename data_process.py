import pandas as pd 
import numpy as np 

df1 = pd.read_csv('base-flat.csv')
##补充检验指标
df_exam = pd.read_csv('base-检验.csv')
item_list = df_exam.labname.drop_duplicates()

df_gp1 = df_exam.groupby('labname')
def isnum(n):
    try:    
        if type(n) == float or type(n) == int or type(n)== complex:
            return(True)
        else:
            return(False)
    except:
        return (False)
for item in item_list:
    item_result = df_gp1.get_group(item)
    df_gp2 = item_result.groupby('patientunitstayid')
    item_collection = pd.DataFrame()
    for i, (admdr, datalist) in enumerate(df_gp2):
        subject = datalist.sort_values(by='labresultoffset')
        subject = subject[['patientunitstayid','labresult']]
        item_collection = pd.concat([item_collection,subject.head(1)])
    item_collection.rename(columns={'labresult':item},inplace=True)
    clean_item = item_collection[item].apply(isnum)
    if np.sum(clean_item !=0) ==0:
        continue
    clean_project = item_collection[clean_item].copy()
    df1 = pd.merge(df1,clean_project,on='patientunitstayid',how='left')
    print('##########'+item+'###complete!!!!')

df_breath = pd.read_csv('base--呼吸指标.csv')
item_breath = df_breath.respchartvaluelabel.drop_duplicates()
df_gp3 = df_breath.groupby('respchartvaluelabel')

for item in item_breath:
    item_result = df_gp3.get_group(item)
    df_gp4 = item_result.groupby('patientunitstayid')
    item_collection = pd.DataFrame()
    for i, (admdr, datalist) in enumerate(df_gp4):
        subject = datalist.sort_values(by='respchartoffset')
        subject = subject[['patientunitstayid','respchartvalue']]
        item_collection = pd.concat([item_collection,subject.head(1)])
    item_collection.rename(columns={'respchartvalue':item},inplace=True)
    clean_item = item_collection[item].apply(isnum)
    if np.sum(clean_item !=0) ==0:
        continue
    clean_project = item_collection[clean_item].copy()
    df1 = pd.merge(df1,clean_project,on='patientunitstayid',how='left')
    print('##########'+item+'###complete!!!!')

df1.to_csv('final-merge.csv',index=None)


main_data = pd.read_csv('final-merge.csv')
df1 = main_data[['patientunitstayid', 'gender', 'age', 'admissionheight',
       'admissionweight',  'intubated',
       'vent', 'dialysis', 'eyes', 'motor', 'verbal', 'meds', 'Hgb', 'calcium',
       'anion gap', 'PTT', '-eos', 'PT', 'WBC x 1000', 'total bilirubin',
       '-basos', 'Hct', 'glucose', 'PT - INR', 'RBC', 'albumin',
       'alkaline phos.', '-polys', 'BUN', 'RDW', 'platelets x 1000',
       'AST (SGOT)', 'chloride', 'sodium', 'MCHC', 'bicarbonate', 'MCH', 'MCV',
       '-monos', 'total protein', '-lymphs', 'potassium', 'ALT (SGPT)',
       'creatinine', 'paO2', 'bedside glucose', 'pH', 'Base Excess',
       'troponin - I', 'HCO3', 'FiO2_x', 'paCO2', 'lactate', 'phosphate',
       'magnesium', 'O2 Sat (%)', 'urinary specific gravity', 'MPV',
       'Total RR', 'FiO2_y', 'Vent Rate', 'Tidal Volume (set)', 'TV/kg IBW',
       'PEEP', 'LPM O2', 'Plateau Pressure', 'Peak Insp. Pressure',
       'Mean Airway Pressure', 'Exhaled MV', 'RR (patient)', 'SaO2']]
label_data = pd.read_csv('apache.csv')
label_data = label_data[['patientunitstayid','apachescore','actualhospitalmortality']]
df2 = pd.merge(df1,label_data,on='patientunitstayid',how='left')
df2.to_excel('raw_data.xlsx',index=None)

df1 = pd.read_excel('raw_data.xlsx')
def sexConvert(x):
    if 'Female' in x:
        return 0
    if 'Male' in x:
        return 1
def labelConvert(x):
    if x=='ALIVE':
        return 0
    if x=='EXPIRED':
        return 1
df1['BMI'] = 10000* df1['admissionweight'] / (df1['admissionheight'] * df1['admissionheight'])
df1['gender'] = df1['gender'].map(sexConvert)
df1['actualhospitalmortality'] = df1['actualhospitalmortality'].map(labelConvert)
df1.to_excel('process_data.xlsx',index=None)
