import numpy as np
from time import time

P = np.random.randn(200, 64)
Q = np.random.randn(40981, 64)

PT = P.transpose()
QT = Q.transpose()

start = time()
A = np.matmul(PT, np.matmul(P, np.matmul(QT, Q)))
#A = PT.dot(P).dot(QT).dot(Q)
a = np.trace(A)
print('a', a)
print(time() - start)

start = time()
B = np.matmul(P, np.matmul(QT, np.matmul(Q, PT)))
#B = P.dot(QT).dot(Q).dot(PT)
b = np.trace(B)
print('b', b)
print(time() - start)

start = time()
C = np.matmul(QT, np.matmul(Q, np.matmul(PT, P)))
#C = QT.dot(Q).dot(PT).dot(P)
c = np.trace(C)
print('c', c)
print(time() - start)

start = time()
D = np.matmul(Q, np.matmul(PT, np.matmul(P, QT)))
#D = Q.dot(PT).dot(P).dot(QT)
d = np.trace(D)
print('d', d)
print(time() - start)
