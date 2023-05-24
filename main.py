import matplotlib.pyplot as plt
import numpy as np




# The data is wriiten as length, width and color (0 = blue;  1 = red)
data = [[3,1.5,1],
        [2,1,0],
        [4,1.5,1],
        [3,1,0],
        [3.5,.5,1],
        [2,.5,0],
        [5.5,1,1],
        [1,1,0]]

def get_unknown_flower() -> list:
    l = float(input("Enter the length of the unknown flower: "))
    w = float(input("Enter the width of the unknown flower: "))
    return list(l, w)

def sigmoid(x: float):
    return (1 / (1 + np.exp(-x)))


# unknown_flower = get_unknown_flower()


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

plt.plot(x,x**2)