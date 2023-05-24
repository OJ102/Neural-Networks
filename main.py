import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore



# The flower data is wriiten as length, width and color (0 = blue;  1 = red)
flower_data = [[3,1.5,1],
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
    l = [l,w]
    return l

def sigmoid(x: float):
    return (1 / (1 + np.exp(-x)))

def d_sigmoid(x):
    return (sigmoid(x) * (1 - sigmoid(x)))


unknown_flower = get_unknown_flower()


# Get learning rate
learning_rate = float(input("Enter learning rate: "))

# Set random weights and bias
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Costs array
costs = list()

# Training loop
for i in range(50000):
    random_index=np.random.randint(len(flower_data))
    data_point=flower_data[random_index]

    z = data_point[0] * w1 + data_point[1] * w2 + b
    prediction = sigmoid(z)

    target = data_point[2]

    cost = np.square(prediction - target)
    
    dcost_dpred = 2 * (prediction - target)
    dpred_dz = d_sigmoid(z)

    dz_dw1 = data_point[0]
    dz_dw2 = data_point[1]
    dz_db=1

    dcost_dz = dcost_dpred * dpred_dz
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db



    if i % 300 == 0:
        cost_sum = 0
        for j in range(len(flower_data)):
            p=flower_data[j]
            z = p[0] * w1 + p[1] * w2 + b
            prediction = sigmoid(z)

            target = p[2]

            cost_sum += np.square(prediction - target)
        costs.append(cost_sum)

# Graphs the cost sum function for user to check if the cost sum is minimizing effectively
def graph_costs():
    plt.plot(costs)
    plt.title("Cost sum graph")
    plt.ylabel("Sum Cost function value")
    plt.xlabel("Training loop ->")
    plt.show()

# Prediction cross checker
def prediction_crosscheck():
    for i in range(len(flower_data)):
        p=flower_data[i]
        print(p)
        z = p[0] * w1 + p[1] * w2 + b
        prediction = sigmoid(z)
        print(prediction)

# Unknown flower finder
print(unknown_flower)
z = unknown_flower[0] * w1 + unknown_flower[1] * w2 + b
prediction = sigmoid(z)
if prediction > 0.5:
    print("The flower is", Fore.RED + "Red")
elif prediction < 0.5:
    print("The flower is",Fore.BLUE + "Blue")
else:
    print("Result inconclusive")

print(Fore.WHITE)
