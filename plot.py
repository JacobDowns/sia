from pylab import *

xs = loadtxt('xs.txt')
hs = loadtxt('hs.txt')
qs = loadtxt('qs.txt')

print xs
print hs
print hs.max()

#plot(xs, hs, 'ro-')
plot(xs, qs, 'ro-')
show()
