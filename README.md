---
title: Hdmr Opt
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
---

# HDMR Optimization

🚀 **HDMR Optimization** is a repository that provides code to compute the global minimum points of specified functions using High Dimensional Model Representation (HDMR). It employs two optimization techniques: **HDMR-based optimization** and the traditional **BFGS** (Broyden–Fletcher–Goldfarb–Shanno) method. The HDMR-based approach extracts the one-dimensional form of a given function, whereas the BFGS method directly optimizes the target function.

**Note**: The code is compatible with **Python 3.9**. It is recommended to set up a dedicated Conda or virtual environment to avoid compatibility issues due to library versions.

## Python Files

### Module: `functions.py`

Contains the functions that are used as optimization problems. You can think this file as a library.

Also you can run `functions.py` to see the functional forms.

You can run functions.py as: `python src/functions.py <funtion_name>`
- It plots the given function.

Example script: `python src/functions.py camel3_2d`

### Module: `functions_forecast.py`

Contains optimize_helper function which uses the xgboost library to do regression on transactions.csv

optimize_helper is used to optimize the predicted number of transactions on future dates and return Mean Absolute Percentage Error. You only need to specify learning rate and the subsample parameters.

### Module: `main.py`

`main.py` computes the global minimum points of given function with parameters. The outputs are status reports and HDMR plots. You can also use adaptive hdmr by setting adaptive parameter.

Prints status reports and also writes them into one .txt file in `results` folder.
- The status report for BFGS method.
- The status report for HDMR method.

Also saves the HDMR plots as one .png file in `results` folder.

The output files are 
- `<file_name_for_given_parameters>.txt`
- `<file_name_for_given_parameters>.png`

#### Running `main.py`
You can enter the script `python src/main.py --help`:

```
usage: HDMR [-h] --numSamples NUMSAMPLES --numVariables NUMVARIABLES --function FUNCTION --min MIN --max MAX [--x0 X0 [X0 ...]] [--randomInit] [--basisFunction BASISFUNCTION]
            [--legendreDegree LEGENDREDEGREE] [--adaptive] [--numClosestPoints NUMCLOSESTPOINTS] [--epsilon EPSILON] [--clip CLIP] [--numberOfRuns NUMBEROFRUNS]

Program applies the hdmr-opt method and plots the results.

options:
  -h, --help            show this help message and exit
  --numSamples NUMSAMPLES
                        Number of samples to calculate alpha coefficients.
  --numVariables NUMVARIABLES
                        Number of variable of the test function.
  --function FUNCTION   Test function name.
  --min MIN             Lower range of the test function.
  --max MAX             Upper range of the test function.
  --x0 X0 [X0 ...]      Starting point x0.
  --randomInit          Initializes x0 as random numbers in the range of xs. Default is initializing as 0.
  --basisFunction BASISFUNCTION
                        Basis function that will be used in HDMR. Legendre or Cosine. Default is Cosine.
  --legendreDegree LEGENDREDEGREE
                        Number of legendre polynomial. Default is 7.
  --adaptive            Uses iterative method when set.
  --numClosestPoints NUMCLOSESTPOINTS
                        Number of closest points to x0. Default is 1000.
  --epsilon EPSILON     Epsilon value for convergence. Default is 0.1.
  --clip CLIP           Clipping value for updating interval (a, b). Default is 0.9.
  --numberOfRuns NUMBEROFRUNS
                        Number of test runs to calculate average error.
```


An example script (default version, not adaptive hdmr): `python src/main.py --numSamples 1000 --numVariables 2 --function camel16_2d --min -5 --max 5`

An example script (Runs adaptive hdmr): `python src/main.py --numSamples 1000 --numVariables 2 --function camel16_2d --min -5 --max 5 --adaptive --epsilon 0.2`

**UPDATE (11.09.23)**: You can now also use `--x0 2.5 1.5` this x0 variable is added to command line so you can define x0 from command line if you want.

If adaptive parameter is set, the output file will be additional parameters and will be starting with 'adaptive' key.

**UPDATE (30.11.23)**: You can now use `--numberOfRuns` parameter to test the code with many trials and calculate the average error and average normalized error. Provide an integer here to have your code running any number of times. There are still issues with `--adaptive` parameter. While using this parameter with `--adaptive` be careful. Some extreme numbers may lead very high errors.

# Web UI Update
After you install all dependencies inside of the `requirements.txt` you can run the following code to run web ui on your browser.

You should be inside of the main folder. (Should be seeing results, src folder etc.)
1. Open the terminal in this folder.
2. Run `streamlit run app.py`

This basically runs the `app.py` which is the streamlit app file. You can find the ui elements inside of this file.

As default it runs the app on the `http://localhost:8501/`




