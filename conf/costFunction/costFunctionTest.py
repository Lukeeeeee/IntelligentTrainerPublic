import gym
import numpy as np
import json
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


def liner_cost_function(name):
    env = gym.make(name)
    data = []
    label = []
    state = env.reset()
    for i in range(80000):

        print(i)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action=action)
        sample = []
        for i in range(len(list(state))):
            sample.append(state[i])
        for i in range(len(list(next_state))):
            sample.append(next_state[i])
        for i in range(len(list(action))):
            sample.append(action[i])
        label.append(float(reward))
        data.append(sample)
    data = np.array(data)
    poly_feature = PolynomialFeatures(degree=2)
    data = poly_feature.fit_transform(X=data)

    pass
    model = linear_model.Lasso(alpha=0.06)
    model.fit(X=data, y=label)
    data = []
    label = []
    for i in range(1000):
        # print(i)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action=action)
        sample = []
        for i in range(len(list(state))):
            sample.append(state[i])
        for i in range(len(list(next_state))):
            sample.append(next_state[i])
        for i in range(len(list(action))):
            sample.append(action[i])
        label.append(float(reward))
        data.append(sample)
    a = model.predict(X=poly_feature.fit_transform(X=np.array(data)))

    par = model.coef_
    inter = model.intercept_
    # res = np.sum(np.multiply(data, par), axis=1) + inter

    print("Error %f:" % mean_absolute_error(a, label))
    # print("Error %f:" % mean_squared_error(res, label))
    # print("Error %f:" % mean_squared_error(res, a))
    print(par)
    print(inter)

    test_data = []

    # cost_fn = CostFunction(parameters_file=COST_FUNCTION_PATH + '/Swimmer-v1_cost_function_parameters.json')
    # cost_fn.intec = inter
    # cost_fn.coef = par
    #
    # for d in data:
    #     test_data.append(cost_fn(state=d[0:8], next_state=d[8:16], action=d[16:18]))
    #
    # print("Error %f:" % mean_absolute_error(test_data, label))
    # # print("Error %f:" % mean_squared_error(test_data, res))
    # print("Error %f:" % mean_absolute_error(test_data, a))

    # with open(name + '_cost_function_parameters.json', 'w') as f:
    #     json.dump(list(model.coef_) + [model.intercept_.item()], fp=f, indent=4)
    # pass
    # return list(par) + [inter.item()]


if __name__ == '__main__':
    a1 = liner_cost_function(name='Swimmer-v1')
