from scipy.optimize import minimize, OptimizeResult
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import functions as f
import argparse

"""
Test functions are available in https://www.sfu.ca/~ssurjano/optimization.html

"""


def rastrigin(x):
    if len(x.shape) == 2:
        N, n = x.shape
        axis = 1
    else:
        n = len(x)
        axis = 0
    y = np.sum(x**2 -10*np.cos(2*math.pi*x), axis=axis, keepdims=True)
    return y + 10*n

def schwefel(x):
    if len(x.shape) == 2:
        N, n = x.shape
        axis = 1
    else:
        n = len(x)
        axis = 0
    y = 418.9829*n - np.sum(x*np.sin(np.sqrt(np.abs(x))), axis=axis, keepdims=True)
    return y


def griewank(x):
    if len(x.shape) == 2:
        N, n = x.shape
        axis = 1
    else:
        n = len(x)
        axis = 0
    y = np.sum((x**2)/4000, axis=axis, keepdims=True) - np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))), axis=axis, keepdims=True) + 1
    return y


def test_func(x):
    if len(x.shape) == 2:
        N, n = x.shape
        axis = 1
    else:
        n = len(x)
        axis = 0
    y = np.sum(x**2, axis=axis, keepdims=True)
    return y 

def hdmr_opt(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, maxiter=None,
                 disp=False, return_all=False, finite_diff_rel_step=None,
                 **unknown_options):

    def Pn(m, x):
        if m == 0:
            return np.ones_like(x)
        elif m == 1:
            return x
        else:
            return (2*m-1)*x*Pn(m-1, x)/m - (m-1)*Pn(m-2, x)/m
    
    def L(a,b,m,x):
        return np.sqrt((2*m+1)/(b-a))*Pn(m, 2*(x-b)/(b-a)+1)

    def calculate_alpha_coeff(xs):
        N, n = xs.shape
        alpha = np.zeros((m, n))
        y = fun(xs)
        f0 = np.mean(y)
        for r in range(m): # Iterate degree of the Legendre polynomial
            for i in range(n): # Iterate number of variable
                alpha[r, i] = np.mean((y-f0) * L(a, b, r+1, np.array(xs[:, [i]])))
        return alpha
    
    def evalute_hdmr(x, f0, alpha):
        N, n = x.shape
        m, _ = alpha.shape
        y = f0 * np.ones((N,1))
        for r in range(m):  
            for i in range(n):  
                y = y + alpha[r, i] * L(a, b, r+1, np.array(x[:, [i]])) # Meta-model
        return y

    def one_dim_evaluate_hdmr(x):
        m, _ = alpha.shape
        f = 0
        for r in range(m):
            f = f + alpha[r,i] * L(a, b, r+1, x) # One dimensional functions
        return f
    
    def one_dim_evaluate_hdmr_for_test(x, idx_var):
        m, _ = alpha.shape
        f = np.zeros((N,1))
        for r in range(m):
            f = f + alpha[r,idx_var] * L(a, b, r+1, x)
        return f

    def plot_results():
        f = np.zeros((N,n))
        y = fun(xs)
        for idx in range(n):
            f[:,[idx]] = np.mean(y) + one_dim_evaluate_hdmr_for_test(xs[:,[idx]], idx)  

        columns_xf = [f'x{id+1}' for id in range(n)] + [f'f{id+1}' for id in range(n)]
        xf = np.concatenate([xs,f],axis=1)
        df_xf = pd.DataFrame(xf, columns=columns_xf)

        columns_xy = [f'x{id+1}' for id in range(n)] + ["y"]
        xy = np.concatenate([xs,y],axis=1)
        df_xy = pd.DataFrame(xy, columns=columns_xy)

        yhat = evalute_hdmr(xs, np.mean(y), alpha)
        xyhat = np.concatenate([xs,yhat],axis=1)
        df_xyhat = pd.DataFrame(xyhat, columns=columns_xy)

        fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(8, n*4))
        for jj in range(1,n+1):
            axs[jj-1, 0].scatter(df_xy[f"x{jj}"], df_xy["y"], color='red' ,label="test")
            axs[jj-1, 0].set_title('Test function')
            axs[jj-1, 0].set_xlabel(f"x{jj}")
            axs[jj-1, 0].set_ylabel('y')

            axs[jj-1, 1].scatter(df_xyhat[f"x{jj}"], df_xyhat["y"], color='red', label="hdmr")
            axs[jj-1, 1].scatter(df_xf[f"x{jj}"], df_xf[f"f{jj}"], color='blue', label="1d")
            axs[jj-1, 1].set_title('FEOM applied to function')
            axs[jj-1, 1].set_xlabel(f"x{jj}")
            axs[jj-1, 1].set_ylabel('y')

            axs[jj-1, 0].legend()
            axs[jj-1, 1].legend()
        plt.subplots_adjust(hspace=0.6, wspace=0.3)
        plt.savefig(file_name + '.png')

    def calculate_distances(x0, arr):
        return np.sqrt(np.sum((x0 - np.array(arr)) ** 2, axis=1))

    def find_closest_points(x0, arr, k):
        distances = calculate_distances(x0, arr)
        indexes = np.argsort(distances)[:k]
        return arr[indexes]

    if is_adaptive:
        results = []
    
        xs = (b - a)*np.random.random((N,n)) + a # Generate sampling data
        print('XS: ', xs.shape)
        alpha = calculate_alpha_coeff(xs)
        print("Alpha: ", alpha)
        temp_status = []
        for i in range(n):
            status = minimize(one_dim_evaluate_hdmr, np.array(x0[i]), method='BFGS') 
            temp_status.append(status.x[0])
        result = OptimizeResult(x=temp_status, fun=fun(x0, *args), success=True, message=" ", nfev=1, njev=0, nhev=0)
        result.nfev = N
        results.append(result)

        old_x0 = np.array(x0)
        new_x0 = np.array(result.x)
        new_a = -np.inf
        new_b = np.inf
        old_a = [a for _ in range(n)]
        old_b = [b for _ in range(n)]

        while True:
            print(f"old x0 = {old_x0}")
            print(f"new x0 = {new_x0}")

            print("---------------------------------------------------------------------------------")
            # convergence criteria
            if np.sqrt(np.sum((old_x0 - new_x0) ** 2)) < epsilon:
                print("Convergence occured!")
                break

            closest_points = find_closest_points(x0=new_x0, arr=xs, k=k)
            new_a = np.min(closest_points, axis=0)
            new_b = np.max(closest_points, axis=0)
            
            for i in range(n):
                old_range = old_b[i] - old_a[i]
                if new_b[i] - new_a[i] < clip * old_range:
                    # if np.abs(new_a[i]) < 0.7 * np.abs(old_a[i]):
                    #     new_a[i] = 0.7 * old_a[i]
                    # if np.abs(new_b[i]) < 0.7 * np.abs(old_b[i]):
                    #     new_b[i] = 0.7 * old_b[i]
                    middle_point = (old_b[i] + old_a[i]) / 2
                    new_range = clip * old_range
                    new_a[i] = middle_point - (new_range / 2)
                    new_b[i] = middle_point + (new_range / 2)

            print("Old a: ", old_a)
            print("Old b: ", old_b)
            print("New a: ", new_a)
            print("New b: ", new_b)

            print("Creating new sample...", end=" ")
            xs = (new_b - new_a)*np.random.random((N,n)) + new_a 
            print("Done!")

            # HDMR
            alpha = calculate_alpha_coeff(xs)
            # print("Alpha: ", alpha)
            temp_status = []
            for i in range(n):
                status = minimize(one_dim_evaluate_hdmr, np.array(new_x0[i]), method='BFGS') 
                temp_status.append(status.x[0])
            result = OptimizeResult(x=temp_status, fun=fun(new_x0, *args), success=True, message=" ", nfev=1, njev=0, nhev=0)
            result.nfev = N
            results.append(result)

            old_x0 = np.array(new_x0)
            new_x0 = np.array(result.x)
            old_a = new_a
            old_b = new_b
            plot_results()
    else:

        xs = (b-a)*np.random.random((N,n))+a # Generate sampling data
        print('XS: ', xs.shape)
        alpha = calculate_alpha_coeff(xs)
        print("Alpha: ", alpha)
        plot_results()
        temp_status = []
        for i in range(n):
            status = minimize(one_dim_evaluate_hdmr, np.array(x0[i]), method='BFGS') 
            temp_status.append(status.x[0])
        result = OptimizeResult(x=temp_status, fun=fun(x0, *args), success=True, message=" ", nfev=1, njev=0, nhev=0)
        result.nfev = N

    return result


parser = argparse.ArgumentParser(
                    prog='HDMR',
                    description='Program applies the hdmr-opt method and plots the results.')
parser.add_argument('--numSamples', type=int, help='Number of samples to calculate alpha coefficients.', required=True)
parser.add_argument('--numVariables', type=int, help='Number of variable of the test function.', required=True)
parser.add_argument('--function', help='Test function name.', required=True)
parser.add_argument('--min', type=int, help='Lower range of the test function.', required=True)
parser.add_argument('--max', type=int, help='Upper range of the test function.', required=True)
parser.add_argument('--randomInit', action='store_true', help='Initializes x0 as random numbers in the range of xs. Default is initializing as 0.')
parser.add_argument('--legendreDegree', type=int, default=7, help='Number of legendre polynomial. Default is 7.')
parser.add_argument('--adaptive', action='store_true', help='Uses iterative method when set.')
parser.add_argument('--numClosestPoints', type=int, help='Number of closest points to x0. Default is 1000.', default=100)
parser.add_argument('--epsilon', type=float, help='Epsilon value for convergence. Default is 0.1.', default=0.1)
parser.add_argument('--clip', type=float, help='Clipping value for updating interval (a, b). Default is 0.9.', default=0.9)

global_args = parser.parse_args()

if __name__ == "__main__":

    # N = 100 # Number of samples to calculate alpha coefficients
    # n = 2 # Number of variable
    # m = 7 # Degree of the Legendre polynomial
    # a = -5 # Range of the function
    # b = 5 # Range of the function
    print('Args: ', global_args)

    N = global_args.numSamples # Number of samples to calculate alpha coefficients
    n = global_args.numVariables # Number of variable
    test_function = getattr(f, global_args.function)
    m = global_args.legendreDegree # Degree of the Legendre polynomial
    a = global_args.min # Range of the function
    b = global_args.max # Range of the function
    is_adaptive = global_args.adaptive
    if is_adaptive:
        k = global_args.numClosestPoints # Number of resampled points
        clip = global_args.clip
        epsilon = global_args.epsilon # Epsilon value for convergence
        if not (0 < clip <= 1):
            raise ValueError("Clipping value should be in the interval of (0, 1]")
        file_name = f"results/adaptive_{global_args.function}_a{a}_b{b}_N{N}_m{m}_k{k}_c{clip:.2f}" 
    else:
        file_name = f"results/{global_args.function}_a{a}_b{b}_N{N}_m{m}" 
    
    if not global_args.randomInit:
        if global_args.function.split('_')[1] == '2d':
            x0 = np.array([0.0, 0.0]) # Initial value of function for optimizing process
        elif global_args.function.split('_')[1] == '10d':
            x0 = np.zeros((10,))
    else:
        file_name += '_randomInit'
        if global_args.function.split('_')[1] == '2d':
            x0 = np.random.rand(2) * (b - a) + a # Initial value of function for optimizing process
        elif global_args.function.split('_')[1] == '10d':
            x0 = np.random.rand(10) * (b - a) + a

    # status_bfgs = minimize(test_function, x0, method="BFGS") # Applying direct optimization method to the function
    # print(f"BFGS status: {status_bfgs}")

    status_hdmr = minimize(test_function, x0, args=(), method=hdmr_opt) # Applying hdmr-opt method to the function
    print(f"hdmr_opt status: {status_hdmr}")

    with open(file_name + '.txt', 'w') as f:
        # f.write("BFGS Status\n" + str(status_bfgs) + "\n\n")
        f.write("HDMR Status\n" + str(status_hdmr))
    f.close()