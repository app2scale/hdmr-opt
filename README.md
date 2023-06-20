# hdmr-opt
## General Information
This repository contains the codes to calculate global minimum points of the given functions. In this code, we use two method to optimize and compare. In hdmr-opt method, We get the one dimensional form of the given functions using HDMR. In other method, we directly apply BFGS method to the function.

## Python Files

### Module: `functions.py`

Contains the functions that are used as optimization problems. You can think this file as a library.

Also you can run `functions.py` to see the functional forms.

You can run functions.py as: `python src/functions.py <funtion_name>`
- It plots the given function.

Example script: `python src/functions.py camel3_2d`

### Module: `main.py`

`main.py` computes the global minimum points of given function with parameters. The outputs are status reports and HDMR plots.

Prints status reports and also writes them into one .txt file in `results` folder.
- The status report for BFGS method.
- The status report for HDMR method.

Also saves the HDMR plots as one .png file in `results` folder.

The output files are 
- `<file_name_for_given_parameters>.txt`
- `<file_name_for_given_parameters>.png`

#### Running `main.py`
You can enter the script `python src/main.py --help`

```
usage: HDMR [-h] [-m M] [--random_init] num_of_samples num_of_variable function_name min max

Program applies the hdmr-opt method and plots the results.

positional arguments:
  num_of_samples   Number of samples to calculate alpha coefficients.
  num_of_variable  Number of variable of the test function.
  function_name    Test function name.
  min              Lower range of the test function.
  max              Upper range of the test function.

options:
  -h, --help       show this help message and exit
  -m M             Number of legendre polynomial. Default is 7.
  --random_init    Initializes x0 as random numbers in the range of xs. Default is initializing as 0.
```

Script format (x_0 initializes as zeros):
- `python src/main.py <num_of_samples> <num_of_variable> <function_name> <min> <max> -m <legendre_polynomial_degree>`


Another script (x_0 initializes as random numbers between the range of xs):
- `python src/main.py <num_of_samples> <num_of_variable> <function_name> <min> <max> -m <legendre_polynomial_degree> --random_init`

An example: `python src/main.py 500 2 ackley_2d -30 30 -m 11 --random_init`