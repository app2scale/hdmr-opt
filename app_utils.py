# app_utils.py

# Default function intervals and initial points for benchmark functions
default_function_intervals = {
    "testfunc": (-5.0, 5.0),
    "camel3": (-5.0, 5.0),
    "camel16": (-5.0, 5.0),
    "treccani": (-5.0, 5.0),
    "goldstein": (-2.0, 2.0),
    "branin": (-5.0, 15.0),
    "rosenbrock": (-2.048, 2.048),
    "ackley": (-30.0, 30.0),
    "griewank": (-600.0, 600.0),
    "rastrigin": (-5.12, 5.12)
}

default_function_x0 = {
    "testfunc": '2.0, 2.0',
    "camel3": '2.0, 2.0',
    "camel16": '2.0, 2.0',
    "treccani": '2.0, 2.0',
    "goldstein": '2.0, 2.0',
    "branin": '2.0, 2.0',
    "rosenbrock": '2.0, 2.0',
    "ackley": '2.0, 2.0',
    "griewank": '2.0, 2.0',
    "rastrigin": '2.0, 2.0'
}

def get_function_interval(function_name):
    """
    Retrieves the predefined input interval for the specified benchmark function.

    Args:
        function_name (str): The name of the function (e.g., 'testfunc', 'rastrigin').

    Returns:
        tuple: A tuple containing the minimum and maximum values for the function's domain.

    Raises:
        KeyError: If the function_name does not exist in the predefined intervals.
    """
    try:
        interval = default_function_intervals[function_name]
    except KeyError:
        raise KeyError(f"Invalid function name: '{function_name}'. Please provide a valid function name.")
    
    return interval

def get_function_x0(function_name):
    """
    Retrieves the predefined initial guess for optimization for the specified benchmark function.

    Args:
        function_name (str): The name of the function (e.g., 'testfunc', 'rastrigin').

    Returns:
        str: A string representing the initial guess for the function in the format 'x1, x2, ...'.

    Raises:
        KeyError: If the function_name does not exist in the predefined initial guesses.
    """
    try:
        x0 = default_function_x0[function_name]
    except KeyError:
        raise KeyError(f"Invalid function name: '{function_name}'. Please provide a valid function name.")
    
    return x0

def get_dims(function_name):
    """
    Extracts the number of dimensions from the function name.

    Assumes the function name follows the convention 'functionname_dimd', 
    where dim is the number of dimensions (e.g., 'rosenbrock_2d' or 'griewank_10d').

    Args:
        function_name (str): The name of the function including dimensionality (e.g., 'rosenbrock_2d').

    Returns:
        int: The number of dimensions of the function.

    Raises:
        ValueError: If the function name does not follow the expected format.
    """
    try:
        return int(function_name.split('_')[1][:-1])
    except (IndexError, ValueError):
        raise ValueError(f"Function name '{function_name}' does not follow the expected format 'functionname_Xd'.")