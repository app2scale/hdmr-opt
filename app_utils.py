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

def get_function_interval(function_name):
    try:
        interval = default_function_intervals[function_name]    
    except:
        raise KeyError("You have entered non existing funtion name.")
    
    return interval

def get_dims(function_name):
    return int(function_name.split('_')[1][:-1])