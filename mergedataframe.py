
import pandas as pd 
'''
GitHub does not accept large files. Therefore experiment A data is split into 5 DataFrames. In this code, they are merged and returned as a single ExperimentA file.
'''

df1,df2,df3,df4,df5 = pd.read_pickle("ExperimentApart1.pkl"),pd.read_pickle("ExperimentApart2.pkl"),pd.read_pickle("ExperimentApart3.pkl"),pd.read_pickle("ExperimentApart4.pkl"),pd.read_pickle("ExperimentApart5.pkl")
df = pd.concat([df1,df2,df3,df4,df5])
df.to_pickle("ExperimentA.pkl")