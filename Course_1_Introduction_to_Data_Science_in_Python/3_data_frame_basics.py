import pandas as pd
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()

## each row in DF is a series
df.iloc[0]
df.loc['Store 1']
type(df.iloc[0]) # => pandas.core.series.Series
type(df.loc['Store 1'])  # => pandas.core.frame.DataFrame


### Columns

df.loc['Store 1', 'Cost']   # type  pandas.core.series.Series
df.loc['Store 1']['Cost']   # type  pandas.core.series.Series
df.loc[:,['Name', 'Cost']]  # type dataframe
df[['Name', 'Cost']]
df['Name'][df['Cost']>3]


df.T.loc['Cost']
df['Cost']


df = df.set_index([df.index, 'Name'])
df.index.names = ['Location', 'Name']
df = df.append(pd.Series(data={'Cost': 3.00, 'Item Purchased': 'Kitty Food'}, name=('Store 2', 'Kevyn')))



### drop
df.drop('Store 1', inplace=True)
del df['Store 1']

# drop column
df.drop('Cost', axis=1, inplace=True)

###
### Quering a DataFrame
###
df = pd.read_csv('olympics.csv', index_col = 0, skiprows=1)
df.head()


## change columns names
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold' + col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver' + col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze' + col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#' + col[1:]}, inplace=True)

# condition, boolean mask
df['Gold']>0

only_gold = df.where(df['Gold'] > 0)
only_gold['Gold'].count()
df['Gold'].count()
only_gold = only_gold.dropna()
df[(df['Gold.1'] > 0) & (df['Gold'] == 0)]

###
### Indexing DataFrames
###
df['Country']=df.index

# You'll see that when we create a new index from an existing column
# it appears that a new first row has been added with empty values.
df = df.set_index('Gold')
df.head()

# reset_index. This promotes the index into a column and creates a default numbered index.
df.reset_index(inplace=True)

df = pd.read_csv("census.csv")
df['SUMLEV'].unique()
df[df['SUMLEV']==50]

columns_to_keep = ['STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015']
df = df[columns_to_keep]

df = df.set_index(['STNAME', 'CTYNAME'])

###
### missing values
###

df = pd.read_csv('log.csv')
df.set_index('time', inplace=True)
df=df.sort_index()


df['Gold'].idxmax()
df[df['Gold']==max(df['Gold'])].index.tolist()[0]


gold_only = df[(df['Gold.1']>0) & (df['Gold.1']>0)]
gold_only = pd.DataFrame(gold_only)
gold_only['diff']=(gold_only['Gold']-gold_only['Gold.1'])/gold_only['Gold.2']
gold_only['diff'].idxmax()

census_df = pd.read_csv('census.csv')
census_df[census_df['SUMLEV']==50].groupby('STNAME').count()['SUMLEV'].idxmax()

aggDF = census_df[census_df['SUMLEV']==50].groupby('STNAME', as_index=False)['CENSUS2010POP'].agg(lambda x: sum(sorted(x, reverse=True)[0:3]))
sumlevDf = census_df[census_df['SUMLEV']==50].copy()

sumlevDf['MAX_POP'] = sumlevDf[['POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']].max(axis=1)
sumlevDf['MIN_POP'] = sumlevDf[['POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']].min(axis=1)
sumlevDf['MAX_CHANGE'] = sumlevDf['MAX_POP'] - sumlevDf['MIN_POP']
sumlevDf.sort_values(by='MAX_CHANGE', ascending=False, inplace=True)

#Question 8
sumlevDf = census_df[census_df['SUMLEV']==50].copy()
sumlevDf = sumlevDf[((sumlevDf['REGION']==1) | (sumlevDf['REGION']==2)) & (sumlevDf['POPESTIMATE2015']>sumlevDf['POPESTIMATE2014'])]
sumlevDf = sumlevDf[sumlevDf['CTYNAME'].str.startswith('Washington')]
sumlevDf[['STNAME','CTYNAME']].sort_index(ascending=True)

df = census_df
df=df[df['SUMLEV']==50]
df.set_index(['STNAME','CTYNAME'], inplace=True)

import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})

df.apply(min_max, axis=1)

(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'avg': np.average, 'sum': np.sum}))


(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))


df_ages = pd.DataFrame({'age': np.random.randint(21, 51, 8)})
df_ages['age_bins'] = pd.cut(x=df_ages['age'], bins=[20, 29, 39, 49])
df_ages['age_by_decade'] = pd.cut(x=df_ages['age'], bins=[20, 29, 39, 49], labels=['20s', '30s', '40s'])


pd.to_datetime('4.7.12', dayfirst=True)