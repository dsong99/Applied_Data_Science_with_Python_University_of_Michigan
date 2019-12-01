## standard python
from array import *
array1 = array('i', [1, 2, 3, 4, 5])  # first param is type, i means integer

import numpy as np

##
## Creation
##
mylist = [1,2,3]

x= np.array(mylist)

# array of arrays 2x3
m = np.array([[1,2,3], [4,5,6]])
m.shape

n = np.arange(0,30, 2)
#=>array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

#Both reshape and resize change the shape of the numpy array;
# the difference is that using resize will affect the original array
# while using reshape create a new reshaped instance of the array.

n = n.reshape(3,5)

o = np.linspace(0,4,9)
#=>array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. ])

o.resize(3,3) #change in place
#=>array([[0. , 0.5, 1. ],
#       [1.5, 2. , 2.5],
#       [3. , 3.5, 4. ]])


np.ones((3,2))
np.zeros((3,2))
np.eye(3)   # create a matrix with diagnal values as 1
np.diag([1,2,3]) # put 1,2,3 in diagnal

np.array([1,2,3]*3) # => array([1, 2, 3, 1, 2, 3, 1, 2, 3])
np.repeat([1,2,3], 3) # => array([1, 1, 1, 2, 2, 2, 3, 3, 3])

p = np.ones([2,3], int)
'''
array([[1, 1, 1],
       [1, 1, 1]])
'''
np.vstack([p, 2*p])

'''
=>
array([[1, 1, 1],
       [1, 1, 1],
       [2, 2, 2],
       [2, 2, 2]])
'''

np.hstack([p, 2*p])
'''
array([[1, 1, 1, 2, 2, 2],
       [1, 1, 1, 2, 2, 2]])
'''

##
## Operations
##

x = np.array([1,2,3])
y = np.array([4,5,6])

x+y # => array([5, 7, 9])
x*y # => array([ 4, 10, 18])
x.dot(y) # => 32 vector dot product

x= np.array([y,2*y])
'''
array([[ 4,  5,  6],
       [ 8, 10, 12]])
'''

x.T # tanspose
'''
array([[ 4,  8],
       [ 5, 10],
       [ 6, 12]])
'''

x.dtype # dtype('int32')
x = x.astype('f') # cast array to a different type
x.dtype # dtype('float32')

# min(), max(), mean(), sum(), std()
x.max() # 45

# index of max, min
x.argmax() # => 5
x.argmin() # => 0

##
## Indexing / Slicing
##
s = np.arange(13)**2
# => array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144],  dtype=int32)

s[0], s[4], s[0:3] # => (0, 16, array([0, 1, 4], dtype=int32))

s[-5::-2] # starts from -5, counting backward by 2
# =>  array([64, 36, 16,  4,  0], dtype=int32)

#
# two dimensional array
#
r = np.arange(36)
r.resize(6,6)
'''
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35]])
'''

r[2,2] # => 14
r[3,3:6] # => array([21, 22, 23])
r[:2, :-1] # first two rows exclude the last column
r[-1, ::2] # every 2nd in the last row

r> 30 # gives an array of True or False
r[r>30] # => array([31, 32, 33, 34, 35])
r[r>30] = 30 # assign elements > 30 to 30

### Note on using slicing, change the sub array also changes the original array.
### To avoid this use copy to create a new array
###

r_copy = r.copy()
r_copy[:] = 0

###
### Iterating over arrays
###

test = np.random.randint(0,10, (4,3)) # create a 4x3 matrix with values between 0-10
'''
array([[8, 2, 8],
       [6, 9, 0],
       [8, 5, 0],
       [4, 2, 5]])
'''
for row in test:
    print (row)

len(test) # => 4

for i, row in enumerate(test):
    print('row', i, 'is', row)

'''
row 0 is [8 2 8]
row 1 is [6 9 0]
row 2 is [8 5 0]
row 3 is [4 2 5]
'''

test2 = test * 2
for i, j in zip(test, test2):
    print(i,'+', j, '=', i+j)

'''
[8 2 8] + [16  4 16] = [24  6 24]
[6 9 0] + [12 18  0] = [18 27  0]
[8 5 0] + [16 10  0] = [24 15  0]
[4 2 5] + [ 8  4 10] = [12  6 15]
'''

td = np.arange(0,36)
td[::7] # =>array([ 0,  7, 14, 21, 28, 35])
td2 = td.reshape(6,6)
td2.diagonal() # => array([ 0,  7, 14, 21, 28, 35])
td2.reshape(36)[::7]  # =>array([ 0,  7, 14, 21, 28, 35])
td2[2:4,2:4]    # index starts from 0
'''
array([[14, 15],
       [20, 21]])
'''

### Missing values, below all return False
np.nan == None
np.nan == np.nan
np.NaN == np.NAN

#use np.isnan() to check missing values