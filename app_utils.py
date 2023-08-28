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
    try:
        interval = default_function_intervals[function_name]    
    except:
        raise KeyError("You have entered non existing function name.")
    
    return interval

def get_function_x0(function_name):
    try:
        x0 = default_function_x0[function_name]    
    except:
        raise KeyError("You have entered non existing function name.")
    
    return x0

def get_dims(function_name):
    return int(function_name.split('_')[1][:-1])