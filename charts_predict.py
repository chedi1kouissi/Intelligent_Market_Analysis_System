import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Rich console for beautiful output
console = Console()

class MarketAnalysisApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.configure_app()
        self.initialize_ai()
        self.initialize_dataset()
        self.table = None  # Add this line to store the table

    def configure_app(self):
        """Configure Flask application settings"""
        self.app.config.update(
            SESSION_TYPE="filesystem",
            SECRET_KEY=os.urandom(24),
            TEMPLATES_AUTO_RELOAD=True
        )
        Session(self.app)

    def initialize_ai(self):
        """Initialize the Gemini AI model"""
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDn0e3pd8ZAOhYl6rS0EUf7_YKxZl0mgYU')
        genai.configure(api_key=GEMINI_API_KEY)
        
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=self.generation_config
        )

    def initialize_dataset(self):
        """Initialize dataset manager"""
        self.dataset_manager = DatasetManager('market_analysis_dataset.csv')

    def analyze_feature_relationships(self) -> List[Dict[str, Any]]:
        """Get feature relationship analysis from AI model"""
        df = self.dataset_manager.data
        if df is None:
            logger.error("No dataset available")
            return []

        try:
            # Create features description
            features_description = self._get_features_description(df)
            logger.info("Generated features description")
            
            # Create prompt and get response
            prompt = self._create_relationship_prompt(features_description)
            logger.info(f"Sending prompt to model")
            
            # Get AI response
            chat = self.model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            if not response or not response.text:
                logger.error("Received empty response from model")
                return []
                
            raw_response = response.text.strip()
            logger.info(f"Raw response from model: {raw_response}")
            
            # First try structured parsing
            relationships = self._parse_relationships_response(raw_response)
            
            # If no relationships found, try alternative parsing
            if not relationships:
                relationships = self._parse_unstructured_response(raw_response)
                
            return relationships
            
        except Exception as e:
            logger.error(f"Error in relationship analysis: {str(e)}")
            return []

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

    def _create_relationship_prompt(self, features_description: str) -> str:
        """Create prompt for AI relationship analysis"""
        sample_data = self.dataset_manager.data.head().to_string()
        
        return f"""You are a data analyst examining a dataset. Here is the dataset description and first few rows:

Sample Data:
{sample_data}

Please analyze the dataset and list the most important relationships between features. Return ONLY the relationships in this exact format, with one relationship per line:

- Feature1 | Feature2 | Type | Description

Guidelines:
1. Only list strong and significant relationships
2. Each line must start with a hyphen
3. Use pipe symbol (|) to separate the four parts
4. Keep descriptions clear and concise

Example format:
- Price | Size | Positive Correlation | Larger items have higher prices
- Category | Sales | Category Impact | Luxury items have 50% higher sales

Please provide at least 3-5 important relationships you find in the data. Return ONLY the relationship list, no other text."""

    def _parse_relationships_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse structured response into relationships"""
        relationships = []
        
        for line in response.split('\n'):
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
                
            # Remove leading hyphen and split by pipe
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

    def _parse_unstructured_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse less structured responses to extract relationships"""
        relationships = []
        
        # Split into sentences
        sentences = [s.strip() for s in response.replace('\n', ' ').split('.')]
        
        for sentence in sentences:
            # Look for feature names in the sentence
            features = []
            for col in self.dataset_manager.data.columns:
                if str(col) in sentence:
                    features.append(str(col))
                    
            if len(features) >= 2:
                # Try to determine relationship type
                type_keywords = {
                    'correlation': ['correlate', 'correlation', 'associated', 'relationship'],
                    'impact': ['impact', 'affect', 'influence', 'effect'],
                    'difference': ['difference', 'vary', 'variation', 'different'],
                    'pattern': ['pattern', 'trend', 'tendency']
                }
                
                rel_type = 'relationship'
                for type_name, keywords in type_keywords.items():
                    if any(keyword in sentence.lower() for keyword in keywords):
                        rel_type = type_name
                        break
                
                relationship = {
                    'feature1': features[0],
                    'feature2': features[1],
                    'type': rel_type.title(),
                    'description': sentence.strip()
                }
                relationships.append(relationship)
                logger.info(f"Extracted relationship from unstructured text: {relationship}")
                
        
        return relationships

    def display_results(self, relationships: List[Dict[str, Any]]):
        """Display formatted results using Rich"""
        console.print("\n[bold blue]Market Analysis Results[/bold blue]", justify="center")
        console.print("=" * 80, justify="center")

        # Display dataset summary
        self.display_dataset_summary()

        # Display relationships
        console.print("\n[bold green]Key Feature Relationships[/bold green]")
        
        if not relationships:
            console.print("[yellow]No significant relationships were found. This could mean:[/yellow]")
            console.print("- The dataset doesn't contain strong feature relationships")
            console.print("- The AI model needs a different approach to analyze the data")
            console.print("- There might be an issue with the analysis process")
            return

        table = Table(title="Important Relationships Identified")
        table.add_column("Features", style="cyan", width=30)
        table.add_column("Type", style="yellow", width=20)
        table.add_column("Description", style="magenta", width=50)

        for rel in relationships:
            table.add_row(
                f"{rel['feature1']} → {rel['feature2']}",
                rel['type'],
                rel['description']
            )

        self.table = table  # Store the table in the instance
        console.print(table)
        console.print(f"\n[dim]Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    def store_table(self):
        """Store table in a Python file for import"""
        if self.table:
            with open("analysis_results.py", "w") as f:
                f.write("table = '''" + str(self.table) + "'''\n")
            logger.info("Table stored successfully in analysis_results.py")
        else:
            logger.warning("No table to store")

    def display_dataset_summary(self):
        """Display dataset summary statistics"""
        df = self.dataset_manager.data
        
        table = Table(title="Dataset Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Records", str(len(df)))
        table.add_row("Total Features", str(len(df.columns)))
        table.add_row("Feature Names", ", ".join(df.columns.tolist()))
        
        console.print(table)

    def _create_bi_report_prompt(self, features_description: str) -> str:
        """Create prompt for AI to generate a comprehensive BI report"""
        sample_data = self.dataset_manager.data.head().to_string()
        
        return f"""You are a senior Business Intelligence analyst creating a comprehensive report. Using the dataset information below, generate a well-structured BI report.

Sample Data:
{sample_data}

Please generate a complete BI report in the following Markdown format:

# Market Analysis Report
[Current Date]

## Executive Summary
[Provide a concise summary of key findings]

## Dataset Overview
- Total Records: [number]
- Time Period: [start_date to end_date]
- Key Metrics Analyzed: [list]

## Key Performance Indicators (KPIs)
[List and explain main KPIs]

## Feature Analysis
[For each important feature, provide detailed analysis]

## Relationship Analysis
[Previous analysis will be inserted here]

## Market Trends
[Identify and describe key trends]

## [CHARTS_PLACEHOLDER]
The following charts will be generated separately:
1. Feature Correlation Heatmap
2. Time Series Analysis
3. Distribution Analysis
4. Category Comparison Charts

## Recommendations
[Provide data-driven recommendations]

## Technical Documentation
### Data Processing Notes
- Data Quality: [assessment]
- Missing Values: [handling method]
- Outliers: [treatment approach]

### Methodology
[Explain analysis methods used]

### Limitations and Assumptions
[List key limitations and assumptions]

Generate ONLY the report content following this structure exactly. The [CHARTS_PLACEHOLDER] section will be populated with actual visualizations later."""

    def generate_bi_report(self) -> str:
        """Generate a comprehensive BI report"""
        try:
            # Get feature description
            features_description = self._get_features_description(self.dataset_manager.data)
            
            # Create and send prompt
            prompt = self._create_bi_report_prompt(features_description)
            chat = self.model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            if not response or not response.text:
                logger.error("Received empty response for BI report")
                return "Error generating BI report"
                
            report = response.text.strip()
            logger.info("Successfully generated BI report")
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f'bi_report_{timestamp}.md'
            
            with open(report_filename, 'w') as f:
                f.write(report)
            
            logger.info(f"BI report saved to {report_filename}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating BI report: {str(e)}")
            return f"Error generating BI report: {str(e)}"

    def run_analysis(self):
        """Run complete analysis including relationships and BI report"""
        # Get relationships
        relationships = self.analyze_feature_relationships()
        self.display_results(relationships)
        
        # Generate BI report
        console.print("\n[bold blue]Generating BI Report...[/bold blue]")
        report = self.generate_bi_report()
        
        # Display report in console
        console.print("\n[bold green]BI Report Generated[/bold green]")
        console.print(Panel(report, title="BI Report", border_style="blue"))
        
        return relationships, report

class DatasetManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        """Load and validate dataset"""
        try:
            self.data = pd.read_csv(self.file_path, quotechar='"')
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

def main():
    """Main function to run the analysis"""
    try:
        app = MarketAnalysisApp()
        
        console.print("\n[bold yellow]Welcome to Market Analysis System[/bold yellow]", justify="center")
        console.print("[dim]Starting comprehensive analysis...[/dim]\n")

        # Run complete analysis
        relationships, report = app.run_analysis()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'relationships': relationships,
            'timestamp': timestamp
        }
        
        with open(f'relationship_analysis_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to relationship_analysis_{timestamp}.json[/green]")

        # Save table for import in another code
        with open("../table.py", "w") as f:
            f.write("table = '''" + str(app.table) + "'''\n")
        console.print(f"\n[green]Table saved to ../table.py[/green]")
        
        # Store the table correctly
        app.store_table()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print("[yellow]Please check the logs for more details.[/yellow]")

if __name__ == "__main__":
    main()