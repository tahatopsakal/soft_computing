import numpy as np
from matplotlib import pyplot as plt
import time

def calc_min(first_derivative,learning_rate,stop_val,point_to_start,max_iter,func_itself = None):
    """

    Calculates the minima of the given function using gradient descent or steepest descent

    :param first_derivative: First derivative of the desired function
    :param learning_rate: Learning rate
    :param stop_val: Minimum cost value to stop
    :param point_to_start: Initial Condition
    :param max_iter: Maximum Number of Iterations
    :param func_itself: Optional. Will be plotted if given.
    :return: Return the minima of the func_itself
    """

    start_time = time.time()
    if func_itself == None:
        print('Plotting is not desired.. Continuing the process..')
    else:
        try:
            x_axis = np.linspace(-2,1,10000)
            vals = func_itself(x_axis)
            plt.plot(x_axis,vals)
        except:
            raise Exception(f'{func_itself} is not a good function form to plot. '
                            f'Try using lambda function or check the statement')

    cur_point = point_to_start
    cur_lim_check = 0
    for _discard in range(max_iter):
        cur_point = cur_point - learning_rate*first_derivative(cur_point)
        if abs(cur_point - cur_lim_check) < stop_val:
            print(f'Cost value converges and process has terminated before {max_iter} iterations')
            return cur_point
        elif time.time() - start_time > 30:
            print(f'Process has terminated because the execution takes too long. Did you try changing iterion value {max_iter} with something else?')
            return cur_point
        cur_lim_check  = cur_point

    return cur_point


# print(calc_min(first_derivative= lambda x:2*x+3,learning_rate=0.4,point_to_start=-10,stop_val=10^-6,max_iter=10, func_itself= lambda x : x*x + 3*x))
print(calc_min(first_derivative= lambda x:4*x*x*x -2*x+1,learning_rate=0.01,point_to_start=5,stop_val=10^-6,max_iter=1000000, func_itself= lambda x : x*x*x*x - x*x + x + 6))
plt.show()
