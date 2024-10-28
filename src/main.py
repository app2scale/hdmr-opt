from scipy.optimize import minimize, OptimizeResult
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from src.basis_functions import Legendre, Cosine
try:
    import src.functions as functions
except:
    import functions    
import argparse

is_streamlit = False
is_interactive = False

def hdmr_opt(fun, x0, args=(), jac=None, callback=None,
                 gtol=1e-5, maxiter=None,
                 disp=False, return_all=False, finite_diff_rel_step=None,
                 **unknown_options):
    global plt1, plt2, plt3, a, b, xs, number_of_runs

    def calculate_alpha_coeff(xs):
        N, n = xs.shape
        alpha = np.zeros((m, n))
        y = fun(xs)
        f0 = np.mean(y)
        for r in range(m): # Iterate degree of the Legendre polynomial
            for i in range(n): # Iterate number of variable
                alpha[r, i] = (b - a) * np.mean((y-f0) * BasisFunction(a, b, r+1, np.array(xs[:, [i]])))
        return alpha
    
    def evalute_hdmr(x, f0, alpha):
        N, n = x.shape
        m, _ = alpha.shape
        y = f0 * np.ones((N,1))
        for r in range(m):  
            for i in range(n):  
                y = y + alpha[r, i] * BasisFunction(a, b, r+1, np.array(x[:, [i]])) # Meta-model
        return y

    def one_dim_evaluate_hdmr(x):
        m, _ = alpha.shape
        f = 0
        for r in range(m):
            f = f + alpha[r,i] * BasisFunction(a, b, r+1, x) # One dimensional functions
        return f
    
    def one_dim_evaluate_hdmr_for_test(x, idx_var):
        m, _ = alpha.shape
        f = np.zeros((N,1))
        for r in range(m):
            f = f + alpha[r,idx_var] * BasisFunction(a, b, r+1, x)
        return f

    def plot_results():
        global a, b
        f = np.zeros((N,n))
        y = fun(xs)
        if not is_adaptive:
            for idx in range(n):
                f[:,[idx]] = np.mean(y) + one_dim_evaluate_hdmr_for_test(xs[:,[idx]], idx)
        else:
            a = np.min(a)
            b = np.max(b)
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
        return fig
    
    def plot_alpha_squared(alpha):
        m, n = alpha.shape

        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(n):
            ax.plot(range(m), alpha[:, i] ** 2, label=f'X {i+1}', linewidth=2)

        ax.set_xlabel('Degree of Legendre Polynomial', fontsize=12)
        ax.set_ylabel('Square of Alpha Coefficients', fontsize=12)
        ax.set_title('Square of Alpha Coefficients', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()

        return fig
        
    def plot_with_function():
        global is_interactive
        if is_interactive == True:
            import plotly.graph_objects as go
            Y = fun(xs)
            yhat = evalute_hdmr(xs, np.mean(Y, axis=0), alpha)

            X1, X2 = zip(*xs)
            X1 = np.array(X1).flatten()
            X2 = np.array(X2).flatten()
            yhat = np.array(yhat).flatten()

            # Scatter plot
            scatter_trace = go.Scatter3d(x=X1, y=X2, z=yhat, mode='markers', marker=dict(color='blue', size=2), name='Data Points')

            x1_min, x1_max = np.array((np.min(X1), np.max(X1)))
            x2_min, x2_max = np.array((np.min(X2), np.max(X2)))

            X1 = np.linspace(x1_min, x1_max, int(N/10))
            X2 = np.linspace(x2_min, x2_max, int(N/10))

            X1, X2 = np.meshgrid(X1, X2)
            Y = fun(np.column_stack((X1.ravel(), X2.ravel()))).reshape(X1.shape)

            # Surface plot
            surface_trace = go.Surface(x=X1, y=X2, z=Y, colorscale='jet', opacity=0.6, name='Function Surface')

            # Create the figure and add the traces
            fig = go.Figure(data=[scatter_trace, surface_trace])

            # Set labels for each axis
            fig.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='y'), height=650, width=800, 
                              title='HDMR Scatter & Original Function Surface Plot')

            return fig
        
        
        Y = fun(xs)
        yhat = evalute_hdmr(xs, np.mean(Y, axis=0), alpha)

        X1, X2 = zip(*xs)
        fig = plt.figure(figsize=(8, n*4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("HDMR Scatter & Original Function Surface Plot")
        # Scatter plot
        
        # ax.scatter(X1, X2, Y, c='r', marker='o', alpha=0.6)
        ax.scatter(X1, X2, yhat, c='b', marker='o', alpha=0.6, s=2)
        x1_min, x1_max = np.array((np.min(X1), np.max(X1)))
        x2_min, x2_max = np.array((np.min(X2), np.max(X2)))

        print("x1_min: ", x1_min)

        x1 = np.linspace(x1_min, x1_max, N)
        x2 = np.linspace(x2_min, x2_max, N)

        x1, x2 = np.meshgrid(x1, x2)
        y = fun(np.column_stack((x1.ravel(), x2.ravel()))).reshape(x1.shape)

        ax.plot_surface(x1, x2, y, cmap='jet', alpha=0.6)

        # Limit the Y axis so scatter is easier to see.
        # ax.set_zlim(np.min(yhat), np.max(yhat))

        # Set labels for each axis
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('y')
        return fig


    def calculate_distances(x0_, arr):
        return np.sqrt(np.sum((x0_ - np.array(arr)) ** 2, axis=1))

    def find_closest_points(x0_, arr, k):
        distances = calculate_distances(x0_, arr)
        indexes = np.argsort(distances)[:k]
        return arr[indexes]

    if is_adaptive:
        iter_count = 1
        results = []
    
        xs = (b - a)*np.random.random((N,n)) + a # Generate sampling data
        # print('XS: ', xs.shape)
        alpha = calculate_alpha_coeff(xs)
        # print("Alpha: ", alpha)
        temp_status = []
        for i in range(n):
            status = minimize(one_dim_evaluate_hdmr, np.array(x0[i]), method='BFGS') 
            temp_status.append(status.x[0])
        result = OptimizeResult(x=temp_status, fun=fun(x0, *args), success=True, message=" ", nfev=1, njev=0, nhev=0)
        result.nfev = N
        results.append(result)

        if np.sqrt(np.sum((x0 - np.array(result.x)) ** 2)) < epsilon:
            print("Convergence occured at the first iteration!")
        else:
            old_x0 = x0
            old_a = [a for _ in range(n)]
            old_b = [b for _ in range(n)]

            new_x0 = np.array(result.x)
            new_a = -np.inf
            new_b = np.inf
        
            while True:
                iter_count += 1
                print(f"old x0 = {old_x0}")
                print(f"new x0 = {new_x0}")

                print("---------------------------------------------------------------------------------")
                closest_points = find_closest_points(x0_=new_x0, arr=xs, k=k)
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


                print("Creating new sample...", end=" ")
                new_xs = (new_b - new_a)*np.random.random((N,n)) + new_a 
                print("Done!")

                # HDMR
                alpha = calculate_alpha_coeff(new_xs)
                # print("Alpha: ", alpha)
                temp_status = []
                for i in range(n):
                    status = minimize(one_dim_evaluate_hdmr, np.array(new_x0[i]), method='BFGS') 
                    temp_status.append(status.x[0])
                result = OptimizeResult(x=temp_status, fun=fun(new_x0, *args), success=True, message=" ", nfev=1, njev=0, nhev=0)
                result.nfev = N

                if np.sqrt(np.sum((old_x0 - new_x0) ** 2)) < epsilon:
                    print("Convergence occured!")
                    break
                else:
                    results.append(result)
                
                old_x0 = np.array(new_x0)
                xs = new_xs
                new_x0 = np.array(result.x)
                old_a = new_a
                old_b = new_b
            a = old_a
            b = old_b
            x0 = old_x0
        if number_of_runs == 1:
            plt1 = plot_results()
            plt3 = plot_alpha_squared(alpha)
            if n == 2:
                plt2 = plot_with_function()
            else:
                plt2 = None
        result = results[-1]
        result["nfev"] = result["nfev"] * iter_count
        result['n_iterations'] = iter_count
        return result
    else:
        xs = (b-a)*np.random.random((N,n))+a # Generate sampling data
        # print('XS: ', xs.shape)
        alpha = calculate_alpha_coeff(xs)
        # print("Alpha: ", alpha)
        temp_status = []
        for i in range(n):
            status = minimize(one_dim_evaluate_hdmr, np.array(x0[i]), method='BFGS') 
            temp_status.append(status.x[0])
        result = OptimizeResult(x=temp_status, fun=fun(x0, *args), success=True, message=" ", nfev=1, njev=0, nhev=0)
        result.nfev = N
        if number_of_runs == 1:
            plt1 = plot_results()
            if n == 2:
                plt2 = plot_with_function()
            else:
                plt2 = None
        return result

def main_function(N_, n_, function_name_, basis_function_, m_, a_, b_, random_init_, x0_, is_adaptive_, k_=None, epsilon_=None, clip_=None, number_of_runs_=1):
    global N, n, function_name, basis_function, BasisFunction, m, a, b, x0, random_init, is_adaptive, k, epsilon, clip, is_streamlit, number_of_runs

    if basis_function_ not in ["Legendre", "Cosine"]:
        raise ValueError("basis_function should be Cosine or Legendre.")
    
    N = N_
    n = n_
    function_name = function_name_
    basis_function = basis_function_
    m = m_
    a = a_
    b = b_
    random_init = random_init_
    is_adaptive = is_adaptive_
    number_of_runs = number_of_runs_

    if basis_function == "Legendre":
        BasisFunction = Legendre
    elif basis_function == "Cosine":
        BasisFunction = Cosine
    print(f"is_adaptive: {is_adaptive}")
    if is_adaptive:
        k = k_
        epsilon = epsilon_
        clip = clip_
        if not (0 < clip <= 1):
            raise ValueError("Clipping value should be in the interval of (0, 1]")
        file_name = f"results/adaptive_{function_name}_a{a}_b{b}_N{N}_m{m}_k{k}_c{clip:.2f}" 
    else:
        file_name = f"results/{function_name}_a{a}_b{b}_N{N}_m{m}" 
    
    
    # status_bfgs = minimize(test_function, x0, method="BFGS") # Applying direct optimization method to the function
    # print(f"BFGS status: {status_bfgs}")
    status_hdmr = None

    start=time.time()

    
    status_hdmr = []
    for i in range(number_of_runs):
        a = a_
        b = b_
        if not random_init:
            if is_streamlit:
                x0 = np.fromstring(x0_, dtype=float,sep=',')
            elif x0_:
                # print(",,,,,,,,,,,,,,", x0_)
                x0 = np.array(x0_)
            else:
                if function_name.split('_')[1] == '2d':
                    x0 = np.zeros(2)
                elif function_name.split('_')[1] == '10d':
                    x0 = np.zeros(10)
        else:
            file_name += '_randomInit'
            if function_name.split('_')[1] == '2d':
                x0 = np.random.rand(2) * (b - a) + a # Initial value of function for optimizing process
            elif function_name.split('_')[1] == '10d':
                x0 = np.random.rand(10) * (b - a) + a
            
        result = minimize(getattr(functions, function_name), x0, args=(), method=hdmr_opt) # Applying hdmr-opt method to the function
        # print("result", result)
        status_hdmr.append(result)

    end=time.time()
    runtime = end-start
    return status_hdmr, runtime, plt1, plt2, plt3, file_name

number_of_runs = None

plt1 = plt2 = plt3 = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='HDMR',
                    description='Program applies the hdmr-opt method and plots the results.')
    parser.add_argument('--numSamples', type=int, help='Number of samples to calculate alpha coefficients.', required=True)
    parser.add_argument('--numVariables', type=int, help='Number of variable of the test function.', required=True)
    parser.add_argument('--function', help='Test function name.', required=True)
    parser.add_argument('--min', type=float, help='Lower range of the test function.', required=False, default=None)
    parser.add_argument('--max', type=float, help='Upper range of the test function.', required=False, default=None)
    parser.add_argument('--x0', nargs='+', type=float, help='Starting point x0.', required=False)
    parser.add_argument('--randomInit', action='store_true', help='Initializes x0 as random numbers in the range of xs. Default is initializing as 0.')
    parser.add_argument('--basisFunction', type=str, default="Cosine", help='Basis function that will be used in HDMR. Legendre or Cosine. Default is Cosine.')
    parser.add_argument('--legendreDegree', type=int, default=7, help='Number of legendre polynomial. Default is 7.')
    parser.add_argument('--adaptive', action='store_true', help='Uses iterative method when set.')
    parser.add_argument('--numClosestPoints', type=int, help='Number of closest points to x0. Default is 1000.', default=100)
    parser.add_argument('--epsilon', type=float, help='Epsilon value for convergence. Default is 0.1.', default=0.1)
    parser.add_argument('--clip', type=float, help='Clipping value for updating interval (a, b). Default is 0.9.', default=0.9)
    parser.add_argument('--numberOfRuns', type=int, help='Number of test runs to calculate average error.', default=1)

    global_args = parser.parse_args()

    print('Args: ', global_args)
    N_ = global_args.numSamples # Number of samples to calculate alpha coefficients
    n_ = global_args.numVariables # Number of variable
    function_name_ = global_args.function
    basis_function_ = global_args.basisFunction
    m_ = global_args.legendreDegree # Degree of the Legendre polynomial
    
    a_ = global_args.min # Range of the function
    b_ = global_args.max # Range of the function
        
    if not (a_ and b_):
        with open('src/function_ranges.json', 'r') as file:
            function_ranges = json.load(file)[function_name_]
        a_, b_ = function_ranges
    
    is_adaptive_ = global_args.adaptive
    random_init_ = global_args.randomInit
    k_ = global_args.numClosestPoints
    epsilon_ = global_args.epsilon
    clip_ = global_args.clip
    x0_ = global_args.x0
    number_of_runs_ = global_args.numberOfRuns

    # print("clip_: ", clip_)
    status_hdmr, runtime, _, _, _, file_name = main_function(N_, n_, function_name_, basis_function_, m_, a_, b_, 
        random_init_, x0_=x0_, is_adaptive_=is_adaptive_, k_=k_, epsilon_=epsilon_, clip_=clip_, number_of_runs_=number_of_runs_)
    print(f"{runtime} seconds")
    print(f"hdmr_opt status: {status_hdmr}")

    
    with open(file_name + '.txt', 'w') as f:
        # f.write("BFGS Status\n" + str(status_bfgs) + "\n\n")
        f.write("HDMR Status\n" + str(status_hdmr))
    f.close()

    if number_of_runs == 1:
        plt.show()
    else:
        global x0
        # Calculate average relative error ->
        function_range = np.abs(a_ - b_)
        print(f"Function interval: ({a_}, {b_}), range: {function_range} ")

        sum_of_distances = 0.0
        sum_of_relative_distances = 0.0

        with open('src/optimum_points.json', 'r') as file:
            all_optimum_points = json.load(file)

        optimum_points = all_optimum_points[function_name]
        optimum_points = np.array(optimum_points).reshape(-1, 2)
        # print("Optimum points:", optimum_points)
        
        for i in range(number_of_runs):
            x_i = status_hdmr[i].x
            print(f"Run {i+1}   x = {x_i}")

            # Calculate the distance and the relative distance (divided by range of the function for relative distance)
            distances = np.linalg.norm(optimum_points - x_i, axis=1)
            # print("Distances:", distances)
            distance = np.min(np.min(distances))
            relative_distance = distance / function_range

            # *** We might have to consider outliars here, sometimes very big numbers occur (rarely)

            sum_of_distances += distance
            sum_of_relative_distances += relative_distance
        
        print(f"\nSum of distances from optimum points\n{optimum_points}\nis: {sum_of_distances}")
        print(f"Average error: {sum_of_distances / number_of_runs}\n")

        print(f"Sum of relative distances from optimum points\n{optimum_points}\nis: {sum_of_relative_distances}")
        print(f"Average relative error: {sum_of_relative_distances / number_of_runs}")