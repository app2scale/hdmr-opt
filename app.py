import streamlit as st
import matplotlib.pyplot as plt
import src.main as main
import app_utils as utils

# Streamlit configuration
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="HDMR-Opt",
    page_icon=":tada:",
    layout="wide"
)

# Custom styles for sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 300px;
        max-width: 400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state to control the display of the header and introductory text
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True

# Display Header and Introduction
def display_header():
    """Displays the header and introductory description of the HDMR optimization application."""
    st.title("HDMR-Based Global Optimization Project")

    page_text = """
        Welcome to the HDMR-Based Global Optimization Application, a sophisticated framework designed for addressing global optimization challenges through advanced mathematical methodologies. This application employs High-Dimensional Model Representation (HDMR), a robust technique that facilitates the exploration of complex optimization landscapes by decomposing multivariate functions into lower-dimensional representations.

        HDMR is particularly effective in high-dimensional spaces, where conventional optimization methods often struggle due to the "curse of dimensionality." The method approximates a target function, which maps from n-dimensional space to real numbers, using polynomial basis functions. This enables the expression of the target function in a simplified form, where the function can be approximated by a combination of its mean value and a weighted sum of the basis functions.

        Specifically, the approximation can be represented as:
        
        f(x) ≈ f₀ + Σ(αⱼ Φⱼ(x)),
        
        where:
        - f₀ is the mean of the function,
        - αⱼ are the coefficients determined through regression,
        - Φⱼ(x) are the basis functions that capture the interaction effects among the variables.

        This approximation not only reduces computational complexity but also provides insights into the function's structure, facilitating a more efficient search for global minima.

        This application features two distinct optimization techniques for users to explore and compare:

        1. **HDMR-Based Optimization**: This method leverages the properties of HDMR to iteratively refine the approximation of the target function. By evaluating the function over a strategically chosen sample set and updating the coefficients, the optimization process adapts dynamically to the underlying structure of the function, enhancing the search for the global minimum.

        2. **BFGS Algorithm**: The BFGS (Broyden–Fletcher–Goldfarb–Shanno) algorithm is a widely-used quasi-Newton method that approximates the Hessian matrix of second derivatives. This algorithm iteratively refines an estimate of the minimum by applying a specific update rule that adjusts the current solution based on the gradient of the objective function and an approximation of the Hessian.

        The BFGS algorithm serves as an excellent benchmark against which the performance of the HDMR-based approach can be measured, particularly in terms of convergence speed and accuracy.

        The primary objective of this application is to compute the global minimum points for various benchmark functions and to conduct a comprehensive performance comparison between the HDMR-based optimization and the BFGS algorithm. Users will gain insights into the strengths and limitations of each method, aiding in the selection of optimal strategies for their specific optimization problems.

        For a comprehensive list of test functions utilized in this application, please visit the following link: [Benchmark Functions](https://www.sfu.ca/~ssurjano/optimization.html).
    """
    st.write(page_text)

# Sidebar User Inputs
def get_user_inputs():
    """Collects user inputs from the sidebar and returns them."""
    st.sidebar.header('User Inputs')

    # Interactive plot option
    interactive_plot = st.sidebar.checkbox("Interactive Plot")

    # Number of samples and function selection
    N = st.sidebar.slider("Number of samples:", 100, 10000, 1000, 100)
    function_name = st.sidebar.selectbox("Test function:", available_functions)
    basis_function = st.sidebar.selectbox("Basis Function:", basis_functions)
    legendre_degree = st.sidebar.slider("Number of Basis Functions:", 1, 20, 7, 1)

    # Function interval settings
    st.sidebar.write("Function Interval:")
    col1, col2 = st.sidebar.columns(2)
    interval = utils.get_function_interval(function_name=function_name.split('_')[0])
    with col1:
        min_val = st.number_input("Min:", value=interval[0], format="%.3f", step=1.0)
    with col2:
        max_val = st.number_input("Max:", value=interval[1], format="%.3f", step=1.0)

    # Initial value configuration
    random_init = st.sidebar.checkbox("Random Initialization")
    if not random_init:
        default_x0 = utils.get_function_x0(function_name=function_name.split('_')[0])
        x0 = st.sidebar.text_input("Initial Values (format: x1,x2,...,xn):", str(default_x0))
    else:
        x0 = None

    # Adaptive HDMR settings
    is_adaptive = st.sidebar.checkbox("Adaptive HDMR")

    # Initialize these values with None in case Adaptive HDMR is not selected
    num_closest_points = None
    epsilon = None
    clip = None

    if is_adaptive:
        num_closest_points = st.sidebar.number_input("Number of closest points:", 1, N, 100)
        epsilon = st.sidebar.number_input("Epsilon:", value=0.5, min_value=0.0, step=0.1)
        clip = st.sidebar.number_input("Clip:", 0.05, 1.0, 0.95, 0.05)

    return (interactive_plot, N, function_name, basis_function, legendre_degree, 
            min_val, max_val, random_init, x0, is_adaptive, num_closest_points, epsilon, clip)

# HDMR Calculation Logic
def calculate_hdmr(inputs):
    """Performs the HDMR optimization and returns the results."""
    (interactive_plot, N, function_name, basis_function, legendre_degree, 
     min_val, max_val, random_init, x0, is_adaptive, num_closest_points, epsilon, clip) = inputs

    n = utils.get_dims(function_name)
    main.is_interactive = interactive_plot
    main.is_streamlit = True

    if is_adaptive:
        return main.main_function(N, n, function_name, basis_function, legendre_degree, 
                                  min_val, max_val, random_init, x0, is_adaptive, 
                                  num_closest_points, epsilon, clip), interactive_plot
    else:
        return main.main_function(N, n, function_name, basis_function, legendre_degree,
                                  min_val, max_val, random_init, x0, is_adaptive), interactive_plot

# Display Results and Plots
def display_results(status_hdmr, runtime, n, plt1, plt2, plt3, interactive_plot):
    """Displays the results and associated plots."""
    st.subheader("Results")
    st.write(f"HDMR Optimization Success: {status_hdmr.success} - X: {status_hdmr.x}")
    st.write(f"Execution Time: {runtime:.5f} seconds")

    with st.expander("Click to view detailed results", expanded=False):
        st.write(status_hdmr)

    st.subheader("Plots")

    if n == 2:
        col3, col4 = st.columns([0.4, 0.6] if interactive_plot else 2)
        with col3:
            st.pyplot(plt1)
        with col4:
            if interactive_plot:
                st.plotly_chart(plt2)
            else:
                st.pyplot(plt2)
        with col3:
            st.pyplot(plt3)
    else:
        st.pyplot(plt1)

# Main Application
if __name__ == '__main__':
    available_functions = ["testfunc_2d", "rastrigin_2d", "camel3_2d", "camel16_2d", "treccani_2d", 
                           "goldstein_2d", "branin_2d", "rosenbrock_2d", "ackley_2d", 
                           "rosenbrock_10d", "griewank_10d", "rastrigin_10d"]
    basis_functions = ["Legendre", "Cosine"]

    # Display the header and introductory text when the application starts
    if st.session_state.show_intro:
        display_header()

    # Get user inputs
    inputs = get_user_inputs()

    if st.sidebar.button("Calculate HDMR"):
        # Hide the introductory text when the button is clicked
        st.session_state.show_intro = False

        # Calculate HDMR optimization
        (status_hdmr, runtime, plt1, plt2, plt3, file_name), interactive_plot = calculate_hdmr(inputs)
        
        # Display results and plots
        display_results(status_hdmr[0], runtime, utils.get_dims(inputs[2]), plt1, plt2, plt3, interactive_plot)

    # Button to reset the application to the introductory page
    if not st.session_state.show_intro and st.sidebar.button("Back to Introduction"):
        st.session_state.show_intro = True
