import nevergrad as ng
import numpy as np

# instrum = ng.p.Instrumentation(
#     ng.p.Array(shape=(1,)).set_bounds(lower=-12, upper=0.45)
# )
#
# def square(x):
#     return sum((x - 0.5) ** 2)
#
# # function only has one parameter, continuous with shape 2
# optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100, num_workers=1)
#
# for _ in range(optimizer.budget):
#     x = optimizer.ask()
#     print("Trying value: " + str(x.value[0][0]))
#     loss = square(x.value[0][0])
#     optimizer.tell(x, loss)
#
#
# recommendation = optimizer.provide_recommendation()
# # recommendation = optimizer.minimize(square)  # best value
# print(recommendation.value)

## CMA-ES is here

# instrum = ng.p.Instrumentation(
#     ng.p.Array(shape=(2,)).set_bounds(lower=-1, upper=3.0)
# )

instrum = ng.p.Instrumentation(
    # ng.p.Array(shape=(2,), lower=(1.6, 2.9999999999), upper=(3.0, 3.0))
    ng.p.Scalar(lower=1.6, upper=3.0)
)

def square(x):
    print("Inside Square")
    print(x[0])
    print(x[1])
    return sum((x - 0.5) ** 2)

# function only has one parameter, continuous with shape 2
optimizer = ng.optimizers.registry['CMA'](parametrization=instrum, budget=3, num_workers=1)

for i in range(optimizer.budget):
    print("Here is i")
    print(i)
    if i == 0:
        print(instrum.value)
    x = optimizer.ask()
    print("Trying value: " + str(x.value[0][0]))
    loss = square(x.value[0][0])
    print("Result: " + str(loss))
    optimizer.tell(x, loss)

recommendation = optimizer.provide_recommendation()
# recommendation = optimizer.minimize(square)  # best value
print(recommendation.value)
