import streamlit as st

def render_home_page():
    st.set_page_config(
        page_title="Market Analysis System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Market Analysis System")
    
    st.markdown("""
    Welcome to the Market Analysis System! This tool helps you analyze market data using AI.
    
    ### Available Analysis Tools
    
    1. **📈 Feature Analysis**
       - Analyze relationships between features
       - Generate statistical insights
       - Download analysis results
    
    2. **📊 BI Report Generator**
       - Generate comprehensive business intelligence reports
       - View interactive visualizations
       - Export reports in multiple formats
       
    3. **📊 Interactive Charts**
       - Explore data through interactive visualizations
       - Analyze revenue patterns
       - View 3D market analysis
    
    ### Getting Started
    
    1. Choose your analysis tool from the sidebar
    2. Upload your CSV dataset
    3. Configure analysis parameters
    4. Generate insights!
    
    ### Data Requirements
    
    - File format: CSV
    - Recommended size: Up to 100MB
    - Required columns: Any numeric or categorical data
    
    ### Need Help?
    
    Check out the documentation in the sidebar for detailed instructions and examples.
    """)
    
    # Add sidebar navigation
    st.sidebar.title("⚙️ Navigation")
    
    st.sidebar.markdown("""
    ### Analysis Tools
    - [Feature Analysis](/Feature_Analysis)
    - [BI Report Generator](/BI_Report_Generator)
    - [Interactive Charts](/Charts_Analysis)
    
    ### Documentation
    - [Data Requirements](#data-requirements)
    - [Analysis Types](#available-analysis-tools)
    - [Getting Started](#getting-started)
    """)

if __name__ == "__main__":
    render_home_page() 