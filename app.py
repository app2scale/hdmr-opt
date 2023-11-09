import streamlit as st
import matplotlib.pyplot as plt
import src.main as main
import app_utils as utils

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="HDMR-Opt",
    page_icon=":tada:",
    layout="wide"
)

st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 300px;
           max-width: 400px;
       }
       """,
        unsafe_allow_html=True,
    )  

# Header
page_text = """
    This repository contains the codes to calculate global minimum points of the given functions.
    In this code, we use two method to optimize and compare. In hdmr-opt method,
    We get the one dimensional form of the given functions using HDMR. In other method,
    we directly apply BFGS method to the function.
"""
with st.container():
    # st.subheader("HDMR Optimization")
    st.title("HDMR Optimization")
    st.write(page_text)

# available_functions = ["testfunc_2d", "camel3_2d", "camel16_2d", "treccani_2d", "goldstein_2d", "branin_2d", 
#                        "rosenbrock_2d", "ackley_2d"]

available_functions = ["testfunc_2d", "rastrigin_2d", "camel3_2d", "camel16_2d", "treccani_2d", "goldstein_2d", "branin_2d", 
                       "rosenbrock_2d", "ackley_2d", "rosenbrock_10d", "griewank_10d", "rastrigin_10d"]

basis_functions = ["Legendre", "Cosine"]

st.sidebar.header('User Inputs')

interactive_plot = st.sidebar.checkbox("Interactive Plot")

N = st.sidebar.slider("Number of samples:", 100, 10000, 1000, 100)

# n = st.sidebar.slider("Number of variables: ", 1, 10, 2, 1)
function_name = st.sidebar.selectbox("Test function:", available_functions)
basis_funtion = st.sidebar.selectbox("Basis Function:", basis_functions)
legendreDegree = st.sidebar.slider("Number of Basis:", 1, 20, 7, 1)

st.sidebar.write("Function interval: ")
col1, col2 = st.sidebar.columns(2)
interval = utils.get_function_interval(function_name=function_name.split('_')[0])
with col1:
    st.number_input("Min: ", value=interval[0], format="%.3f", step=1.0)
with col2:
    st.number_input("Max: ", value=interval[1], format="%.3f", step=1.0)


random_init = st.sidebar.checkbox("Random Initialization")
if not random_init:
    default_x0 = utils.get_function_x0(function_name=function_name.split('_')[0])
    x0 = st.sidebar.text_input("required format: x1,x2,...,xn",str(default_x0))
else:
    x0 = None

is_adaptive = st.sidebar.checkbox("Adaptive HDMR")

if is_adaptive:
    num_closest_points = st.sidebar.number_input("Number of closest points:", 1, N, 100)
    epsilon = st.sidebar.number_input("Epsilon:", value=0.5, min_value=0.0, step=0.1)
    clip = st.sidebar.number_input("Clip:", 0.05, 1.0, 0.95, 0.05)

if st.sidebar.button("Calculate HDMR"):

    n = utils.get_dims(function_name)
    print("n is: ", n)

    main.is_interactive = interactive_plot
    main.is_streamlit = True
    if is_adaptive:
        status_hdmr, runtime, plt1, plt2, plt3, file_name = main.main_function(N, n, function_name, basis_funtion, legendreDegree, 
                                        interval[0], interval[1], random_init, x0, is_adaptive, num_closest_points, epsilon, clip)
    else:
        status_hdmr, runtime, plt1, plt2, plt3, file_name = main.main_function(N, n, function_name, basis_funtion, legendreDegree,
                                        interval[0], interval[1], random_init, x0, is_adaptive)
    st.subheader("Results")
    st.write(f"hdmr_opt status Success: {status_hdmr.success} - X: {status_hdmr.x}")
    st.write(f"Runtime: {runtime:.5f} seconds")

    with st.expander("Click to see the full result", expanded=False):
        st.write(status_hdmr)

    st.subheader("Plots")
    
    if n == 2:
        if interactive_plot:
            col3, col4 = st.columns(spec=[0.4, 0.6], gap='small')
        else:
            col3, col4 = st.columns(spec=2, gap='small')

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
        
    


