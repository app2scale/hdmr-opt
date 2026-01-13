"""
Streamlit Web Interface for HDMR Optimization (Refactored)

Robust Streamlit UI for HDMR optimization with:
- Strong typing / safe unpacking from main_function
- Safe x0 parsing (string -> numeric vector)
- Defensive figure rendering (matplotlib / plotly)
- Clean session-state management

Author: HDMR Optimization Research Group
Refactor: Senior-style stabilization
Version: 3.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.figure

import src.main as hdmr_engine  # uses main_function + parse_x0
import app_utils as utils

# Optional: plotly for interactive plot
try:
    from plotly.graph_objects import Figure as PlotlyFigure
except Exception:  # pragma: no cover
    PlotlyFigure = None  # type: ignore


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="HDMR Optimization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep your original CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 320px;
        max-width: 450px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3c72;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2a5298;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Session State
# =============================================================================

DEFAULTS = {
    "show_intro": True,
    "optimization_done": False,
    "payload": None,  # will store results, figs, runtime, filename
    "config": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# UI - Header / Intro
# =============================================================================

def display_header() -> None:
    st.markdown('<div class="main-header">🎯 HDMR-Based Global Optimization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive Tool for High-Dimensional Function Optimization</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <h3>🔬 HDMR Method</h3>
            <p>High Dimensional Model Representation for efficient optimization</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <h3>📊 12+ Functions</h3>
            <p>Standard benchmark test functions for validation</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h3>⚡ Adaptive Mode</h3>
            <p>Iterative refinement for faster convergence</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    
    with st.expander("📖 About HDMR Optimization", expanded=True):
        st.markdown("""
        ### What is HDMR?
        
        **High Dimensional Model Representation (HDMR)** is an advanced technique that decomposes 
        complex, high-dimensional functions into simpler, lower-dimensional components. This makes 
        optimization more efficient and interpretable.
        
        #### Mathematical Foundation
        
        HDMR approximates a function f(**x**) as:
        
        ```
        f(x₁, x₂, ..., xₙ) ≈ f₀ + Σᵢ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ, xⱼ) + ...
        ```
        
        Where:
        - **f₀**: Mean value of the function
        - **fᵢ(xᵢ)**: Effect of individual variable xᵢ
        - **fᵢⱼ(xᵢ, xⱼ)**: Interaction between variables xᵢ and xⱼ
        
        #### Key Advantages
        
        - ✅ **Efficient**: Requires fewer function evaluations than grid/random search
        - ✅ **Scalable**: Works well in high-dimensional spaces (10D+)
        - ✅ **Interpretable**: Provides insights into variable importance
        - ✅ **Adaptive**: Can iteratively refine search space for better convergence
        
        #### How to Use This Tool
        
        1. Select a **test function** from the sidebar
        2. Configure **HDMR parameters** (samples, basis functions, etc.)
        3. Optionally enable **Adaptive HDMR** for iterative refinement
        4. Click **"Run Optimization"** to start
        5. View **results and visualizations** below
        
        #### Benchmark Functions
        
        This tool includes 12+ standard benchmark functions from optimization literature:
        - **Rastrigin**: Highly multimodal with many local minima
        - **Rosenbrock**: Narrow curved valley (banana function)
        - **Ackley**: Nearly flat outer region with deep center
        - **Six-Hump Camel**: Multiple local and global minima
        - And more...
        
        For detailed information on each function, see the 
        [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/optimization.html).
        """)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def get_user_inputs() -> Dict[str, Any]:
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # 1) Function
    st.sidebar.subheader("1️⃣ Test Function")
    available_functions = [
        "testfunc_2d", "rastrigin_2d", "camel3_2d", "camel16_2d",
        "treccani_2d", "goldstein_2d", "branin_2d", "rosenbrock_2d",
        "ackley_2d", "rosenbrock_10d", "griewank_10d", "rastrigin_10d"
    ]
    function_name = st.sidebar.selectbox("Select Function:", available_functions)

    n = int(utils.get_dims(function_name))
    interval = utils.get_function_interval(function_name.split("_")[0])
    st.sidebar.info(f"**Dimension:** {n}D\n\n**Domain:** {interval}")

    st.sidebar.markdown("---")

    # 2) HDMR params
    st.sidebar.subheader("2️⃣ HDMR Parameters")
    N = st.sidebar.slider(
        "Number of Samples (N):",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of sample points for coefficient calculation"
    )
    basis_function = st.sidebar.selectbox(
        "Basis Function:",
        ["Legendre", "Cosine"],
        help="Legendre: smooth functions | Cosine: periodic functions"
    )
    m = st.sidebar.slider(
        "Number of Basis Functions (m):",
        min_value=1,
        max_value=20,
        value=7,
        help="Higher values capture more complex patterns"
    )

    st.sidebar.markdown("---")

    # 3) Search space
    st.sidebar.subheader("3️⃣ Search Space")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        min_val = st.number_input("Min:", value=float(interval[0]), format="%.6f")
    with c2:
        max_val = st.number_input("Max:", value=float(interval[1]), format="%.6f")

    if min_val >= max_val:
        st.sidebar.error("Invalid bounds: Min must be smaller than Max.")
        # Keep a safe fallback so the rest of the app doesn't crash.
        # You can also choose to disable the Run button instead.
        max_val = min_val + 1e-6

    st.sidebar.markdown("---")

    # 4) Initialization
    st.sidebar.subheader("4️⃣ Initialization")
    random_init = st.sidebar.checkbox(
        "Random Initialization",
        value=False,
        help="Start from a random point within bounds"
    )

    x0_raw: Optional[str] = None
    if not random_init:
        # Build a dimension-aware default x0 string
        default_x0 = utils.get_function_x0(function_name.split("_")[0])

        def _to_str_vec(v: Any) -> Optional[str]:
            if v is None:
                return None
            if isinstance(v, str):
                return v
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=float).reshape(-1)
                return ", ".join([f"{x:.6f}" for x in arr])
            # scalar
            try:
                return f"{float(v):.6f}"
            except Exception:
                return None

        default_str = _to_str_vec(default_x0)

        # If default_x0 doesn't match n, provide a clean n-dim default
        if default_str is None:
            default_str = "0.0"
        else:
            # quick parse count (best-effort, without importing parse_x0 here)
            tokens = [t for t in default_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",") if t.strip() != ""]
            if len(tokens) not in (1, n):
                # show a stable default for the selected dimension
                default_str = ", ".join(["0.0"] * n)

        help_txt = (
            f"Provide {n} values (comma-separated). "
            f"You may also provide a single value (e.g., '0') to broadcast to all {n} dimensions. "
            f"Example for {n}D: " + (", ".join(["0"] * min(n, 6)) + (" ..." if n > 6 else ""))
        )

        x0_raw = st.sidebar.text_input(
            "Initial Point (x₀):",
            value=default_str,
            help=help_txt
        )

    st.sidebar.markdown("---")

    # 5) Adaptive
    st.sidebar.subheader("5️⃣ Adaptive Mode (Optional)")
    is_adaptive = st.sidebar.checkbox(
        "Enable Adaptive HDMR",
        value=False,
        help="Iteratively refine search space for better convergence"
    )

    num_closest_points = None
    epsilon = None
    clip = None
    maxiter = 25

    if is_adaptive:
        num_closest_points = st.sidebar.number_input(
            "Closest Points (k):",
            min_value=10,
            max_value=N,
            value=min(100, N),
            help="Number of nearest samples used to refine bounds"
        )
        epsilon = st.sidebar.number_input(
            "Convergence Tolerance (ε):",
            min_value=0.0001,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.4f",
            help="Stop when successive solution change is below epsilon"
        )
        clip = st.sidebar.number_input(
            "Reduction Factor (clip):",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            format="%.2f",
            help="Minimum fraction of the previous range to keep when shrinking bounds"
        )
        maxiter = st.sidebar.number_input(
            "Max Iterations:",
            min_value=1,
            max_value=200,
            value=25,
            help="Maximum adaptive refinement iterations"
        )

    st.sidebar.markdown("---")

    # 6) Visualization
    st.sidebar.subheader("6️⃣ Visualization")
    if n == 2:
        interactive_plot = st.sidebar.checkbox(
            "Interactive 3D Plot (2D functions only)",
            value=False,
            help="Use Plotly for interactive 3D visualization"
        )
    else:
        interactive_plot = False
        st.sidebar.caption("Interactive 3D is available only for 2D functions.")

    return {
        "interactive_plot": interactive_plot,
        "N": int(N),
        "function_name": function_name,
        "basis_function": basis_function,
        "m": int(m),
        "min_val": float(min_val),
        "max_val": float(max_val),
        "random_init": bool(random_init),
        "x0_raw": x0_raw,
        "is_adaptive": bool(is_adaptive),
        "num_closest_points": int(num_closest_points) if num_closest_points is not None else None,
        "epsilon": float(epsilon) if epsilon is not None else None,
        "clip": float(clip) if clip is not None else None,
        "maxiter": int(maxiter),
        "n": n,
    }


# ============================================================================
# OPTIMIZATION EXECUTION
# ============================================================================

def run_optimization(config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Execute optimization and return a payload dict:
      {
        "results": List[OptimizeResult],
        "runtime": float,
        "fig1": Figure|None,
        "fig2": Figure|PlotlyFigure|None,
        "fig3": Figure|None,
        "file_name": str
      }
    """
    # If your legacy main used globals, keep compatibility flags:
    hdmr_engine.is_interactive = bool(config["interactive_plot"])
    hdmr_engine.is_streamlit = True

    try:
        # Parse x0 for non-random init (supports "2.0, 2.0")
        if config["random_init"]:
            x0_for_main = None
        else:
            # use parse_x0 from src.main (you added it)
            x0_for_main = hdmr_engine.parse_x0(config["x0_raw"], config["n"])

        results, runtime, fig1, fig2, fig3, file_name = hdmr_engine.main_function(
            N_=int(config["N"]),
            n_=int(config["n"]),
            function_name_=config["function_name"],
            basis_function_=config["basis_function"],
            m_=int(config["m"]),
            a_=float(config["min_val"]),
            b_=float(config["max_val"]),
            random_init_=bool(config["random_init"]),
            x0_=x0_for_main,
            is_adaptive_=bool(config["is_adaptive"]),
            k_=int(config["num_closest_points"]) if config["is_adaptive"] else 100,
            epsilon_=float(config["epsilon"]) if config["is_adaptive"] else 0.1,
            clip_=float(config["clip"]) if config["is_adaptive"] else 0.9,
            number_of_runs_=1,
            maxiter_=int(config["maxiter"]) if config["is_adaptive"] else 25,
            disp_=False,
            enable_plots_=True,
            seed_=None,
        )

        payload = {
            "results": results,
            "runtime": float(runtime),
            "fig1": fig1,
            "fig2": fig2,
            "fig3": fig3,
            "file_name": file_name,
        }
        return payload, None

    except Exception as e:
        return None, str(e)

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def _safe_render_matplotlib(fig: Any, label: str) -> None:
    if isinstance(fig, matplotlib.figure.Figure):
        st.pyplot(fig, clear_figure=True)
    else:
        st.error(f"❌ {label} could not be rendered (Invalid figure type: {type(fig).__name__})")


def display_results(payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    results = payload["results"]
    runtime = payload["runtime"]
    fig1 = payload["fig1"]
    fig2 = payload["fig2"]
    fig3 = payload["fig3"]

    st.markdown("## 📊 Optimization Results")
    st.markdown("---")

    result = results[0]  # single-run

    if getattr(result, "success", False):
        st.success("### ✅ Optimization Successful!")
    else:
        st.warning("### ⚠️ Optimization Completed with Warnings")
        st.info(f"**Message:** {getattr(result, 'message', '')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎯 Objective Value", f"{float(result.fun):.8f}")
    with c2:
        st.metric("🔢 Function Evals", int(getattr(result, "nfev", 0)))
    with c3:
        st.metric("🔄 Iterations", int(getattr(result, "nit", 1)))
    with c4:
        st.metric("⏱️ Runtime", f"{float(runtime):.3f}s")

    st.markdown("---")

    with st.expander("📋 Detailed Technical Output", expanded=False):
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown("### Solution Vector (x*)")
            st.table(
                {
                    "Variable": [f"x{i+1}" for i in range(len(result.x))],
                    "Value": [f"{float(v):.6f}" for v in result.x],
                }
            )
        with col_res2:
            st.markdown("### Optimizer Metadata")
            st.code(str(result), language="text")

    # Visualizations
    n = int(config["n"])
    interactive_plot = bool(config["interactive_plot"])

    if n == 2:
        st.markdown("## 📈 Visualizations")
        tab1, tab2, tab3 = st.tabs(["📊 HDMR Components", "🗻 3D Surface", "📉 Coefficients"])

        with tab1:
            st.markdown("### HDMR Component Functions")
            _safe_render_matplotlib(fig1, "Component Plot")
            st.caption("**Left**: Raw Data vs Variables | **Right**: HDMR Meta-model vs 1D Components")

        with tab2:
            st.markdown("### 3D Surface Comparison")
            if interactive_plot and PlotlyFigure is not None and isinstance(fig2, PlotlyFigure):
                st.plotly_chart(fig2, use_container_width=True)
            elif fig2 is not None:
                _safe_render_matplotlib(fig2, "3D Surface Plot")
            else:
                st.info("3D visualization is not available for this configuration.")

        with tab3:
            st.markdown("### HDMR Coefficient Magnitudes")
            _safe_render_matplotlib(fig3, "Coefficient Plot")
            st.caption("Variable importance across basis degrees (log scale).")

    else:
        st.markdown("## 📈 High-Dimensional Analysis")
        if fig1 is not None:
            _safe_render_matplotlib(fig1, "High-D Component Plot")
        else:
            st.info("Component analysis plot is not available for this configuration.")

    # Export
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Export Data")
    st.sidebar.download_button(
        label="Download Results as TXT",
        data=str(result),
        file_name=f"hdmr_results_{config['function_name']}.txt",
        mime="text/plain",
    )

# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================

def main() -> None:
    if st.session_state.get("show_intro", True):
        display_header()

    config = get_user_inputs()

    if st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True):
        st.session_state["show_intro"] = False
        st.session_state["optimization_done"] = False
        st.session_state["payload"] = None
        st.session_state["config"] = config

        with st.spinner("⚙️ Running HDMR optimization..."):
            payload, err = run_optimization(config)

        if err:
            st.error(f"❌ Optimization failed: {err}")
        else:
            st.session_state["optimization_done"] = True
            st.session_state["payload"] = payload

    if st.session_state.get("optimization_done", False) and st.session_state.get("payload") is not None:
        display_results(st.session_state["payload"], st.session_state["config"])

    if not st.session_state.get("show_intro", True):
        if st.sidebar.button("⬅️ Back to Introduction"):
            st.session_state["show_intro"] = True
            st.session_state["optimization_done"] = False
            st.session_state["payload"] = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>HDMR Optimization Tool</strong></p>
        <p>Built with Streamlit | Powered by NumPy, SciPy, Matplotlib</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()