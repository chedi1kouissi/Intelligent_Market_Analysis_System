import streamlit as st
import pandas as pd
from datetime import datetime
import json
import logging
import google.generativeai as genai
import os
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test import DatasetManager, MarketAnalysisApp, API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def render_feature_analysis_page():
    st.title("📈 Feature Analysis")
    
    st.markdown("""
    Analyze relationships between features in your dataset.
    Upload your data to get started.
    """)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
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
            
            # Analysis configuration
            st.subheader("Analysis Configuration")
            default_prompt = """Based on the provided dataset and statistical context, identify and analyze the most significant relationships between features. Focus on:

1. Strong correlations between numeric features
2. Important categorical relationships
3. Time-based patterns (if applicable)
4. Business-relevant insights

Return the relationships in this EXACT format:
- Feature1 | Feature2 | Type | Detailed Description with Statistical Evidence

Example:
- Price | Sales | Positive Correlation | Strong positive correlation (r=0.85, p<0.001) indicating that higher prices are associated with increased sales. Average sales increase by $1,200 for every $100 price increase.

Please provide 3-5 of the most statistically significant and business-relevant relationships."""
            
            custom_prompt = st.text_area(
                "Customize your analysis prompt",
                value=default_prompt,
                height=300,
                help="Customize how the AI should analyze your data"
            )
            
            # Run analysis button
            if st.button("🔍 Run Analysis", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your data..."):
                    try:
                        relationships = app.analyze_feature_relationships(custom_prompt)
                        st.session_state.analysis_results = relationships
                        
                        if relationships:
                            st.success("✅ Analysis completed successfully!")
                            
                            # Create tabs for different views
                            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Visualizations", "⬇️ Download"])
                            
                            with tab1:
                                st.subheader("Analysis Results")
                                results_df = pd.DataFrame(relationships)
                                st.dataframe(
                                    results_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            with tab2:
                                st.subheader("Data Visualizations")
                                
                                # Create visualizations
                                try:
                                    # Numeric correlations heatmap
                                    st.subheader("Feature Correlations")
                                    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
                                    if not numeric_cols.empty:
                                        corr_matrix = numeric_cols.corr()
                                        st.pyplot(app.plot_correlation_heatmap(corr_matrix))
                                    
                                    # Distribution plots
                                    st.subheader("Feature Distributions")
                                    for col in numeric_cols.columns:
                                        st.pyplot(app.plot_distribution(df[col], col))
                                    
                                    # Category comparisons
                                    categorical_cols = df.select_dtypes(include=['object']).columns
                                    if not categorical_cols.empty:
                                        st.subheader("Category Comparisons")
                                        for col in categorical_cols:
                                            if df[col].nunique() < 10:
                                                st.pyplot(app.plot_category_comparison(df, col))
                                    
                                except Exception as e:
                                    st.warning("Could not generate some visualizations. This might be due to data type incompatibility.")
                                    logger.error(f"Visualization error: {str(e)}")
                            
                            with tab3:
                                # Download options
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                results = {
                                    'relationships': relationships,
                                    'timestamp': timestamp,
                                    'dataset_info': {
                                        'rows': len(df),
                                        'columns': len(df.columns),
                                        'features': list(df.columns)
                                    }
                                }
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.download_button(
                                        label="📥 Download Results (JSON)",
                                        data=json.dumps(results, indent=2),
                                        file_name=f'relationship_analysis_{timestamp}.json',
                                        mime='application/json',
                                        help="Download the analysis results as a JSON file"
                                    )
                                
                                with col2:
                                    st.download_button(
                                        label="📥 Download Results (CSV)",
                                        data=results_df.to_csv(index=False),
                                        file_name=f'relationship_analysis_{timestamp}.csv',
                                        mime='text/csv',
                                        help="Download the analysis results as a CSV file"
                                    )
                        else:
                            st.warning("""⚠️ No significant relationships were found. This could mean:
                            1. The data might need preprocessing (e.g., handling missing values, outliers)
                            2. The features might need transformation
                            3. The relationships might be more complex than direct correlations
                            
                            Try:
                            1. Checking your data for quality issues
                            2. Modifying the analysis prompt to be more specific
                            3. Adding more context about the type of relationships you're looking for""")
                            
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
            logger.error(f"Dataset processing error: {str(e)}")

if __name__ == "__main__":
    render_feature_analysis_page() 