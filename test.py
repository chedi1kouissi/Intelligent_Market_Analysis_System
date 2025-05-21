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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Set your API key directly here
API_KEY = "AIzaSyDn0e3pd8ZAOhYl6rS0EUf7_YKxZl0mgYU"  # Replace with your actual API key

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

class DatasetManager:
    def __init__(self, file: Any):
        self.file = file
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.preprocessing_summary: Dict[str, Any] = {}
        self.load_data()
        self.preprocess_data()

    def load_data(self) -> None:
        """Load and validate dataset"""
        try:
            self.data = pd.read_csv(self.file)
            self.original_data = self.data.copy()
            logger.info(f"Dataset loaded successfully with {len(self.data)} rows")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def preprocess_data(self) -> None:
        """Preprocess the dataset with various cleaning and transformation steps"""
        try:
            if self.data is None:
                return

            preprocessing_steps = []
            
            # Handle missing values
            missing_counts = self.data.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0]
            
            for col in columns_with_missing.index:
                if self.data[col].dtype in ['int64', 'float64']:
                    # Fill numeric missing values with median
                    median_value = self.data[col].median()
                    self.data[col].fillna(median_value, inplace=True)
                    preprocessing_steps.append(f"Filled missing values in {col} with median ({median_value:.2f})")
                else:
                    # Fill categorical missing values with mode
                    mode_value = self.data[col].mode()[0]
                    self.data[col].fillna(mode_value, inplace=True)
                    preprocessing_steps.append(f"Filled missing values in {col} with mode ({mode_value})")

            # Handle outliers in numeric columns
            numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Cap outliers at the bounds instead of removing them
                    self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                    preprocessing_steps.append(f"Capped {outliers_count} outliers in {col}")

            # Encode categorical variables
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if self.data[col].nunique() < 10:  # Only encode if fewer than 10 unique values
                    self.data = pd.get_dummies(self.data, columns=[col], prefix=col)
                    preprocessing_steps.append(f"One-hot encoded categorical column {col}")

            # Store preprocessing summary
            self.preprocessing_summary = {
                'original_shape': self.original_data.shape,
                'processed_shape': self.data.shape,
                'steps': preprocessing_steps
            }
            
            logger.info("Data preprocessing completed successfully")
            logger.info(f"Preprocessing steps: {preprocessing_steps}")
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Return summary of preprocessing steps"""
        return self.preprocessing_summary

class MarketAnalysisApp:
    def __init__(self, api_key: str):
        """Initialize the Market Analysis App"""
        self.initialize_ai(api_key)
        self.dataset_manager: Optional[DatasetManager] = None
        plt.style.use('seaborn-v0_8')  # Use the correct seaborn style name
        # Set additional style parameters for better visualization
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def initialize_ai(self, api_key: str) -> None:
        """Initialize the Gemini AI model"""
        if not api_key:
            raise ValueError("API key is required")
            
        genai.configure(api_key=api_key)
        
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=self.generation_config
        )

    def _get_features_description(self, df: pd.DataFrame) -> str:
        """Create a description of the dataset features and their basic statistics"""
        feature_info = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'nulls': int(df[col].isnull().sum()),
                    'type': 'numeric',
                    'sample_values': [float(x) for x in df[col].head(3).tolist()]
                }
            else:
                stats = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in df[col].value_counts().nlargest(3).to_dict().items()},
                    'nulls': int(df[col].isnull().sum()),
                    'type': 'categorical',
                    'sample_values': [str(x) for x in df[col].head(3).tolist()]
                }
            
            feature_info.append({
                'name': str(col),
                'stats': stats
            })
            
        return json.dumps(feature_info, indent=2)

    def analyze_feature_relationships(self, custom_prompt: str) -> List[Dict[str, Any]]:
        """Get feature relationship analysis from AI model"""
        if not self.dataset_manager or self.dataset_manager.data is None:
            logger.error("No dataset available")
            return []

        try:
            df = self.dataset_manager.data
            features_description = self._get_features_description(df)
            
            # Calculate basic correlations for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            correlations = {}
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.1:  # Lower threshold to catch more relationships
                            correlations[f"{col1}_{col2}"] = corr_value

            # Get preprocessing summary
            preprocessing_info = self.dataset_manager.get_preprocessing_summary()
            
            # Enhanced prompt with more context
            enhanced_prompt = f"""Analyze this dataset with statistical rigor. Here's the detailed context:

Dataset Overview:
{df.head().to_string()}

Preprocessing Applied:
{json.dumps(preprocessing_info['steps'], indent=2)}

Features Description:
{features_description}

Statistical Context:
- Total Records: {len(df)}
- Original Features: {', '.join(self.dataset_manager.original_data.columns)}
- Processed Features: {', '.join(df.columns)}
- Numeric Correlations: {json.dumps(correlations, indent=2)}

Additional Statistical Measures:
{self._get_additional_statistics(df)}

{custom_prompt}

IMPORTANT: For each relationship found, you MUST:
1. Provide quantitative evidence with exact numbers
2. Include statistical measures (e.g., correlation coefficients, chi-square values)
3. Describe the strength and direction of relationships
4. Explain practical significance
5. Consider the preprocessing steps applied and their impact

Format each relationship EXACTLY as:
- Feature1 | Feature2 | Type | Detailed Description with Statistical Evidence"""

            logger.info("Sending enhanced prompt to model")
            response = self.model.generate_content(enhanced_prompt)
            
            if not response or not response.text:
                logger.error("Received empty response from model")
                return []
                
            raw_response = response.text.strip()
            logger.info(f"Raw response from model: {raw_response}")
            
            # Parse relationships with more flexible criteria
            relationships = self._parse_relationships_response(raw_response)
            if not relationships:
                relationships = self._parse_unstructured_response(raw_response)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error in relationship analysis: {str(e)}")
            return []

    def _parse_relationships_response(self, response: str) -> List[Dict[str, str]]:
        """Parse structured response into relationships"""
        relationships = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
                
            parts = [part.strip() for part in line.strip('- ').split('|')]
            
            if len(parts) >= 4:
                relationship = {
                    'feature1': parts[0],
                    'feature2': parts[1],
                    'type': parts[2],
                    'description': parts[3]
                }
                relationships.append(relationship)
                logger.info(f"Parsed relationship: {relationship}")
        
        return relationships

    def _parse_unstructured_response(self, response: str) -> List[Dict[str, str]]:
        """Parse unstructured response to extract relationships"""
        relationships = []
        
        if not self.dataset_manager or not self.dataset_manager.data is not None:
            return relationships

        # Get all feature names
        feature_names = list(self.dataset_manager.data.columns)
        
        # Split into sentences and analyze each
        sentences = [s.strip() for s in response.split('.')]
        
        for sentence in sentences:
            # Skip short or empty sentences
            if len(sentence) < 20:
                continue
                
            # Find mentioned features
            mentioned_features = []
            for feature in feature_names:
                if feature.lower() in sentence.lower():
                    mentioned_features.append(feature)
            
            if len(mentioned_features) >= 2:
                # Determine relationship type
                relationship_types = {
                    'correlation': ['correlate', 'correlation', 'associated', 'relationship'],
                    'causation': ['cause', 'effect', 'impact', 'influence', 'affect'],
                    'comparison': ['higher', 'lower', 'more than', 'less than', 'greater'],
                    'pattern': ['pattern', 'trend', 'tendency', 'distribution'],
                    'dependency': ['depends', 'dependent', 'varies with', 'function of']
                }
                
                rel_type = 'relationship'
                for type_name, keywords in relationship_types.items():
                    if any(keyword in sentence.lower() for keyword in keywords):
                        rel_type = type_name
                        break
                
                relationship = {
                    'feature1': mentioned_features[0],
                    'feature2': mentioned_features[1],
                    'type': rel_type.title(),
                    'description': sentence.strip()
                }
                relationships.append(relationship)
                
        return relationships

    def _get_additional_statistics(self, df: pd.DataFrame) -> str:
        """Get additional statistical measures for the dataset"""
        stats = []
        
        # Calculate basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            stats.append(f"{col}:")
            stats.append(f"  - Mean: {df[col].mean():.2f}")
            stats.append(f"  - Median: {df[col].median():.2f}")
            stats.append(f"  - Std Dev: {df[col].std():.2f}")
            stats.append(f"  - Skewness: {df[col].skew():.2f}")
        
        # Calculate frequencies for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 10:
                stats.append(f"{col} (Top 3 categories):")
                value_counts = df[col].value_counts().nlargest(3)
                for val, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    stats.append(f"  - {val}: {count} ({percentage:.1f}%)")
        
        return "\n".join(stats)

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> plt.Figure:
        """Generate a correlation heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            ax=ax,
            fmt='.2f'
        )
        plt.title('Feature Correlations')
        plt.tight_layout()
        return fig

    def plot_distribution(self, data: pd.Series, column_name: str) -> plt.Figure:
        """Generate a distribution plot for a numeric column"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, kde=True, ax=ax)
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.tight_layout()
        return fig

    def plot_category_comparison(self, df: pd.DataFrame, category_column: str) -> plt.Figure:
        """Generate a bar plot for category comparisons"""
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts = df[category_column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        plt.title(f'Distribution of {category_column}')
        plt.xticks(rotation=45)
        plt.xlabel(category_column)
        plt.ylabel('Count')
        plt.tight_layout()
        return fig

    def generate_bi_report(self) -> str:
        """Generate a comprehensive BI report"""
        try:
            if not self.dataset_manager or self.dataset_manager.data is None:
                raise ValueError("No dataset available")

            df = self.dataset_manager.data
            features_description = self._get_features_description(df)
            additional_stats = self._get_additional_statistics(df)

            # Create prompt for the BI report
            prompt = f"""You are a senior Business Intelligence analyst creating a comprehensive report. Using the dataset information below, generate a well-structured BI report.

Dataset Statistics:
{additional_stats}

Features Description:
{features_description}

Sample Data:
{df.head().to_string()}

Please generate a complete BI report in the following Markdown format:

# Market Analysis Report
[Current Date: {datetime.now().strftime('%Y-%m-%d')}]

## Executive Summary
[Provide a concise summary of key findings]

## Dataset Overview
- Total Records: {len(df)}
- Features Analyzed: {', '.join(df.columns)}
- Time Period: [if applicable]

## Key Performance Indicators (KPIs)
[List and explain main KPIs identified in the data]

## Feature Analysis
[For each important feature, provide detailed analysis]

## Market Trends
[Identify and describe key trends]

## Statistical Insights
[Provide detailed statistical analysis]

## Recommendations
[Provide data-driven recommendations]

## Technical Documentation
### Data Quality Assessment
- Missing Values: [analysis]
- Outliers: [analysis]
- Data Distribution: [analysis]

### Methodology
[Explain analysis methods used]

### Limitations and Assumptions
[List key limitations and assumptions]

Generate a comprehensive report following this structure. Focus on actionable insights and clear data-driven conclusions."""

            # Generate the report using AI
            chat = self.model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            if not response or not response.text:
                raise ValueError("Failed to generate BI report")
                
            report = response.text.strip()
            logger.info("Successfully generated BI report")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating BI report: {str(e)}")
            raise

def render_sidebar() -> None:
    """Render the sidebar with configuration options"""
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown("""
    ### Instructions
    1. Upload your CSV dataset
    2. Review the dataset overview
    3. Customize the analysis prompt
    4. Click 'Run Analysis' to start
    """)

def main() -> None:
    st.set_page_config(
        page_title="Market Analysis System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Market Analysis System")
    st.markdown("""
    Welcome to the Market Analysis System. This tool helps you analyze market data using AI.
    Upload your dataset and customize the analysis to get insights about feature relationships.
    """)

    render_sidebar()

    try:
        app = MarketAnalysisApp(API_KEY)
        
        uploaded_file = st.file_uploader(
            "Upload your CSV dataset",
            type=['csv'],
            help="Upload a CSV file containing your market data"
        )
        
        if uploaded_file is not None:
            app.dataset_manager = DatasetManager(uploaded_file)
            df = app.dataset_manager.data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Overview")
                st.metric("Total Records", len(df))
                st.metric("Features", len(df.columns))
            
            with col2:
                st.subheader("Features")
                st.write(", ".join(df.columns))
            
            st.subheader("Sample Data")
            st.dataframe(
                df.head(),
                use_container_width=True,
                hide_index=False
            )
            
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
            
            col1, col2, col3 = st.columns([2,1,1])
            
            with col1:
                analyze_button = st.button(
                    "🔍 Run Analysis",
                    use_container_width=True,
                    help="Start the analysis process"
                )
            
            if analyze_button:
                with st.spinner("🤖 AI is analyzing your data..."):
                    try:
                        relationships = app.analyze_feature_relationships(custom_prompt)
                        st.session_state.analysis_results = relationships
                        
                        if relationships:
                            st.success("✅ Analysis completed successfully!")
                            
                            st.subheader("Analysis Results")
                            results_df = pd.DataFrame(relationships)
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
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
        st.error(f"❌ Error initializing the application: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")

if __name__ == "__main__":
    main()