import numpy as np

def f(x):
    return 2/(1 + np.exp(-x)) - 1

def df(x):
    return 0.5 * (1 - x) * (1 + x)

w1 = np.array([[0.1, -0.3, -0.2], [-0.1, 0.4, 0.2]])
w2 = np.array([0.2, -0.3])

def go_forward(inp):
    sum = np.dot(w1, inp)
    out = np.array([f(x) for x in sum])
    sum = np.dot(w2, out)
    y = f(sum)
    return y, out

def train(epoch):
    global w1, w2
    N = 100000
    lmd = 1
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]
        y, out = go_forward(x[0:3])
        e = y - x[-1]
        delta = e*df(y)

        w2[0] = w2[0] - lmd * delta * out[0]
        w2[1] = w2[1] - lmd * delta * out[1]

        delta2 = w2*delta*df(out)

        w1[0, :] = w1[0, :] - np.array(x[0:3]) * lmd * delta2[0]
        w1[1, :] = w1[1, :] - np.array(x[0:3]) * lmd * delta2[1]

epoch = ([1, 0, 1, 1],
         [1, 0, 0, 1],
         [0, 1, 0, -1],
         [0, 0, 1, -1],
         [1, 1, 1, 1])

train(epoch)

for x in epoch:
    y, out = go_forward(x[0:3])
    print(f'The resulting value: {y} Desired value: {x[-1]}')
