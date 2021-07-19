import copy


a = ['1', '2', '3']
b = a
c = a[:]
d = copy.copy(a)
e = copy.deepcopy(a)

a.remove('1')

print(a)
print(b)
print(c)
print(d)
print(e)