import streamlit as st
import pandas as pd
from datetime import datetime
import json
import logging
import google.generativeai as genai
import os
from typing import Optional, List, Dict, Any
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test import DatasetManager, MarketAnalysisApp, API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bi_report.log')
    ]
)
logger = logging.getLogger(__name__)

def render_bi_report_page():
    st.title("📈 BI Report Generator")
    
    st.markdown("""
    This page generates comprehensive Business Intelligence reports from your data.
    Upload your dataset to get started.
    """)
    
    # Initialize session state for BI report
    if 'bi_report' not in st.session_state:
        st.session_state.bi_report = None
    
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
            
            # Sample data preview
            st.subheader("Sample Data")
            st.dataframe(
                df.head(),
                use_container_width=True,
                hide_index=False
            )
            
            # Generate report button
            if st.button("🚀 Generate BI Report", use_container_width=True):
                with st.spinner("🤖 Generating comprehensive BI report..."):
                    try:
                        # Generate the report
                        report = app.generate_bi_report()
                        st.session_state.bi_report = report
                        
                        # Display the report
                        st.success("✅ BI Report generated successfully!")
                        
                        # Create tabs for different sections
                        tab1, tab2, tab3 = st.tabs(["📊 Report", "📈 Visualizations", "⬇️ Download"])
                        
                        with tab1:
                            st.markdown(report)
                        
                        with tab2:
                            st.subheader("Data Visualizations")
                            
                            # Create visualizations using the data
                            try:
                                # Numeric correlations heatmap
                                st.subheader("Feature Correlations")
                                numeric_cols = df.select_dtypes(include=['int64', 'float64'])
                                if not numeric_cols.empty:
                                    corr_matrix = numeric_cols.corr()
                                    st.pyplot(app.plot_correlation_heatmap(corr_matrix))
                                
                                # Distribution plots for numeric columns
                                st.subheader("Feature Distributions")
                                for col in numeric_cols.columns:
                                    st.pyplot(app.plot_distribution(df[col], col))
                                
                                # Category comparisons for categorical columns
                                categorical_cols = df.select_dtypes(include=['object']).columns
                                if not categorical_cols.empty:
                                    st.subheader("Category Comparisons")
                                    for col in categorical_cols:
                                        if df[col].nunique() < 10:  # Only plot if fewer than 10 categories
                                            st.pyplot(app.plot_category_comparison(df, col))
                                
                            except Exception as e:
                                st.warning("Could not generate some visualizations. This might be due to data type incompatibility.")
                                logger.error(f"Visualization error: {str(e)}")
                        
                        with tab3:
                            # Download options
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Download as Markdown
                            st.download_button(
                                label="📥 Download Report (Markdown)",
                                data=report,
                                file_name=f'bi_report_{timestamp}.md',
                                mime='text/markdown',
                                help="Download the BI report in Markdown format"
                            )
                            
                            # Download as PDF (if implemented)
                            st.info("PDF download functionality coming soon!")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating BI report: {str(e)}")
                        logger.error(f"BI report generation error: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
            logger.error(f"Dataset processing error: {str(e)}")

if __name__ == "__main__":
    render_bi_report_page() 