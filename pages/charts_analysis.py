import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test import DatasetManager, MarketAnalysisApp, API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('charts_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, hover_data: list = None) -> go.Figure:
    """Create a customizable bar chart"""
    fig = px.bar(
        df, 
        x=x_col, 
        y=y_col,
        color=color_col if color_col != 'None' else None,
        hover_data=hover_data if hover_data else None,
        title=f'{y_col} vs {x_col}' + (f' (colored by {color_col})' if color_col != 'None' else ''),
        height=500
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=True if color_col != 'None' else False
    )
    return fig

def create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, add_trendline: bool = False) -> go.Figure:
    """Create a customizable scatter plot"""
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        color=color_col if color_col != 'None' else None,
        trendline="ols" if add_trendline else None,
        title=f'{y_col} vs {x_col}' + (f' (colored by {color_col})' if color_col != 'None' else ''),
        template="simple_white"
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=True if color_col != 'None' else False
    )
    return fig

def create_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None) -> go.Figure:
    """Create a customizable 3D scatter plot"""
    fig = px.scatter_3d(
        df, 
        x=x_col, 
        y=y_col, 
        z=z_col,
        color=color_col if color_col != 'None' else None,
        title=f'3D Analysis: {x_col}, {y_col}, and {z_col}'
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )
    return fig

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns"""
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    """Get list of categorical columns"""
    return df.select_dtypes(include=['object']).columns.tolist()

def render_charts_page():
    st.title("📊 Interactive Charts Analysis")
    
    st.markdown("""
    Explore your data through interactive visualizations.
    Upload your dataset to get started.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=['csv'],
        help="Upload a CSV file containing your market data"
    )
    
    if uploaded_file is not None:
        try:
            # Initialize the app and dataset
            app = MarketAnalysisApp(API_KEY)
            app.dataset_manager = DatasetManager(uploaded_file)
            df = app.dataset_manager.data
            
            # Display dataset overview
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Overview")
                st.metric("Total Records", len(df))
                st.metric("Features", len(df.columns))
            
            with col2:
                st.subheader("Features")
                st.write(", ".join(df.columns))
            
            # Get column lists
            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            all_cols = numeric_cols + categorical_cols
            
            # Create tabs for different chart types
            tab1, tab2, tab3 = st.tabs([
                "📊 Bar Chart Analysis",
                "📈 Scatter Plot Analysis",
                "🔮 3D Analysis"
            ])
            
            with tab1:
                st.subheader("Customizable Bar Chart")
                st.markdown("Create a bar chart by selecting the features you want to analyze.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("Select X-axis feature", all_cols, key="bar_x")
                with col2:
                    y_col = st.selectbox("Select Y-axis feature", numeric_cols, key="bar_y")
                with col3:
                    color_col = st.selectbox("Select color feature (optional)", ['None'] + all_cols, key="bar_color")
                
                hover_cols = st.multiselect("Select additional hover data (optional)", all_cols)
                
                try:
                    fig1 = create_bar_chart(df, x_col, y_col, color_col, hover_cols if hover_cols else None)
                    st.plotly_chart(fig1, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating bar chart: {str(e)}")
                    logger.error(f"Bar chart error: {str(e)}")
            
            with tab2:
                st.subheader("Customizable Scatter Plot")
                st.markdown("Create a scatter plot by selecting the features you want to analyze.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("Select X-axis feature", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Select Y-axis feature", numeric_cols, key="scatter_y")
                with col3:
                    color_col = st.selectbox("Select color feature (optional)", ['None'] + all_cols, key="scatter_color")
                
                add_trendline = st.checkbox("Add trendline", value=False)
                
                try:
                    fig2 = create_scatter_chart(df, x_col, y_col, color_col, add_trendline)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")
                    logger.error(f"Scatter plot error: {str(e)}")
            
            with tab3:
                st.subheader("Customizable 3D Scatter Plot")
                st.markdown("Create a 3D visualization by selecting three features to analyze.")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    x_col = st.selectbox("Select X-axis feature", numeric_cols, key="3d_x")
                with col2:
                    y_col = st.selectbox("Select Y-axis feature", numeric_cols, key="3d_y")
                with col3:
                    z_col = st.selectbox("Select Z-axis feature", numeric_cols, key="3d_z")
                with col4:
                    color_col = st.selectbox("Select color feature (optional)", ['None'] + all_cols, key="3d_color")
                
                try:
                    fig3 = create_3d_scatter(df, x_col, y_col, z_col, color_col)
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating 3D scatter plot: {str(e)}")
                    logger.error(f"3D scatter plot error: {str(e)}")
            
            # Add chart interaction instructions
            with st.expander("📌 Chart Interaction Tips"):
                st.markdown("""
                ### How to Interact with the Charts
                
                1. **Zoom**: Click and drag to zoom into a specific area
                2. **Pan**: Click and drag in the chart area to pan
                3. **Reset**: Double-click to reset the view
                4. **Hover**: Mouse over data points to see detailed information
                5. **Legend**: Click legend items to show/hide categories
                
                ### 3D Chart Controls
                
                - **Rotate**: Click and drag to rotate the view
                - **Zoom**: Use scroll wheel to zoom in/out
                - **Reset**: Double-click to reset the view
                
                ### Tips for Better Visualization
                
                - For bar charts, categorical data works best on the X-axis
                - Scatter plots work best with numeric features
                - Color coding works well with categorical features
                - The 3D plot is best for exploring relationships between three numeric variables
                """)
            
        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
            logger.error(f"Dataset processing error: {str(e)}")

if __name__ == "__main__":
    render_charts_page() 