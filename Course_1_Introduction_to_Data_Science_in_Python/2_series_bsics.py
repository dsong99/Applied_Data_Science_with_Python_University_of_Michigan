import pandas as pd


## A series has an index and data column
animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)
'''
0    Tiger
1     Bear
2    Moose
'''

###
### Using a dictionary to create a series, keys will be assigned to series index
###
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s.index # =>Index(['Archery', 'Golf', 'Sumo', 'Taekwondo'], dtype='object')


## or pasing index when creating a series with a list
s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])


###
### Querying a Series
### iloc -- use index position
### loc  -- use index name
###

s.iloc[1]
s.loc['Golf']

# pass in an integer parameter, the operator will behave to query via the iloc attribute.
# If you pass in an object, it will query as if you wanted to use the label based loc attribute.

s[1]            # same as s.iloc[1]
s['Golf']       # same as s.loc['Golf']


## using integers as index, will be complicated, and Pandas cannot determine whether it query by index or by label

sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)
# s[0] will raise an error, s[99] => 'Bhutan' since it uses index label
# s.iloc[0]  => 'Bhutan', have to use iloc explicitly to get position data
# Note, if index are integers, use iloc to get position data



### iteration like a dictionary
summary = 0
for item in s:
    summary+=item

# or
import numpy as np
summary = np.sum(s)

for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()



## timestamp is the index
t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1
type(t1.index)  # =>  pandas.core.indexes.datetimes.DatetimeIndex

