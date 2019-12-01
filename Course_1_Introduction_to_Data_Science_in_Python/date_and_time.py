import datetime as dt
import time as tm
tm.time()

dtnow = dt.datetime.fromtimestamp(tm.time())

dt.datetime.now()
dtnow.year,dtnow.month, dtnow.day, dtnow.hour, dtnow.minute

delta = dt.timedelta(days=100)
dt.date.today() -  delta

map(function, iterable, ...)

list1 = [1,2,3,4,5]
list2 = [6,7,8,9,10]
list3 = ['a','b','c','d']
m=map(min, list1, list2)
type(m) = map
for k in m:
    print (k)
1
2
3
4
5
list3 = ['a','b','c','d']
m=map(lambda x:str(x).upper(), list3)
for v in m:
    print (v)

m1 = reduce((lambda x, y: x if x<y else y), [1, 2, 3, 4])
min(list1)


people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_and_name(person):
    title = person.split()[0]
    lastname = person.split()[-1]
    return '{} {}'.format(title, lastname)

list(map(split_title_and_name, people))