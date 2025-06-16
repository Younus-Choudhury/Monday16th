import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class RetailAnalytics:
    def __init__(self, features_file=None, sales_file=None, store_file=None):
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
       
        self.features_df = None
        self.sales_df = None
        self.store_df = None
        self.merged_df = None
       
        # Use EXACT filenames as you specified
        self.features_file = features_file or os.path.join(base_dir, 'features_data_set.csv')
        self.sales_file = sales_file or os.path.join(base_dir, 'sales_data_set.csv')
        self.store_file = store_file or os.path.join(base_dir, 'stores_data_set.csv')
       
    def load_data(self):
        """Load and clean the three CSV files"""
        print("📁 Loading CSV files...")
        print(f"Looking for files in: {os.path.dirname(self.features_file)}")
       
        # Check if files exist with exact names
        files = {
            'features': self.features_file,
            'sales': self.sales_file,
            'store': self.store_file
        }
       
        missing_files = []
        for name, path in files.items():
            if os.path.exists(path):
                print(f"  ✅ Found: {os.path.basename(path)}")
            else:
                print(f"  ❌ Not found: {os.path.basename(path)}")
                missing_files.append(name)
       
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            print("Please ensure all CSV files are in the same directory as your script.")
            return False
       
        # Load datasets
        try:
            self.features_df = pd.read_csv(self.features_file)
            self.sales_df = pd.read_csv(self.sales_file)  
            self.store_df = pd.read_csv(self.store_file)
            print("✅ All files loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            return False
           
        # Display basic info
        print(f"\n📊 Dataset Overview:")
        print(f"Features dataset: {self.features_df.shape}")
        print(f"Sales dataset: {self.sales_df.shape}")
        print(f"Store dataset: {self.store_df.shape}")
       
        return True
   
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n🧹 Cleaning data...")
       
        # Clean features data
        features_before = len(self.features_df)
        self.features_df = self.features_df.dropna(subset=['Store', 'Date'])
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'], errors='coerce')
        self.features_df = self.features_df.dropna(subset=['Date'])
        features_after = len(self.features_df)
       
        # Clean sales data
        sales_before = len(self.sales_df)
        self.sales_df = self.sales_df.dropna(subset=['Store', 'Date', 'Weekly_Sales'])
        self.sales_df = self.sales_df[self.sales_df['Weekly_Sales'] > 0]  # Remove negative/zero sales
        self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date'], errors='coerce')
        self.sales_df = self.sales_df.dropna(subset=['Date'])
        sales_after = len(self.sales_df)
       
        # Clean store data
        store_before = len(self.store_df)
        self.store_df = self.store_df.dropna(subset=['Store', 'Type', 'Size'])
        store_after = len(self.store_df)
       
        print(f"Features: {features_before} → {features_after} records")
        print(f"Sales: {sales_before} → {sales_after} records")
        print(f"Store: {store_before} → {store_after} records")
       
    def merge_datasets(self):
        """Merge all three datasets with proper error handling"""
        print("\n🔗 Merging datasets...")
       
        if self.sales_df is None or self.store_df is None or self.features_df is None:
            print("❌ Cannot merge: One or more datasets not loaded properly")
            return False
       
        try:
            # Start with sales data as base
            self.merged_df = self.sales_df.copy()
           
            # Merge with store data
            self.merged_df = pd.merge(self.merged_df, self.store_df, on='Store', how='left')
           
            # Merge with features data
            self.merged_df = pd.merge(
                self.merged_df,
                self.features_df,
                on=['Store', 'Date'],
                how='left'
            )
           
            # Add time-based columns for analysis
            self.merged_df['Month'] = self.merged_df['Date'].dt.month
            self.merged_df['Year'] = self.merged_df['Date'].dt.year
           
            print(f"✅ Merged dataset: {self.merged_df.shape}")
            print(f"Date range: {self.merged_df['Date'].min().date()} to {self.merged_df['Date'].max().date()}")
            print(f"Available columns: {list(self.merged_df.columns)}")
            return True
           
        except Exception as e:
            print(f"❌ Error during merge: {e}")
            return False
       
    def create_pandas_chart(self):
        """Chart 1: Using Pandas - Store Performance Analysis (2010-2012)"""
        print("\n📊 Creating Pandas Chart - Store Performance Analysis (2010-2012)")
       
        try:
            # Filter for 2010-2012
            df_2010_2012 = self.merged_df[self.merged_df['Year'].between(2010, 2012)]
           
            # Calculate store performance metrics
            store_performance = df_2010_2012.groupby('Store').agg({
                'Weekly_Sales': 'sum',
                'Type': 'first',
                'Size': 'first'
            }).reset_index()
            store_performance.columns = ['Store', 'Total_Sales', 'Type', 'Size']
           
            # Get top 15 stores by total sales
            top_stores = store_performance.nlargest(15, 'Total_Sales')
           
            # Create pandas plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Store Performance Analysis (2010-2012)', fontsize=16, fontweight='bold')
           
            # Plot 1: Total Sales by Store
            top_stores.plot(kind='bar', x='Store', y='Total_Sales',
                           ax=ax1, color='skyblue', alpha=0.8, legend=False)
            ax1.set_title('Top 15 Stores - Total Sales Performance', fontsize=14)
            ax1.set_xlabel('Store ID')
            ax1.set_ylabel('Total Sales ($)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
           
            # Add value labels
            for i, v in enumerate(top_stores['Total_Sales']):
                ax1.text(i, v, f'${v/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
           
            # Plot 2: Average Sales by Store Type
            type_performance = df_2010_2012.groupby('Type')['Weekly_Sales'].mean()
            type_performance.plot(kind='bar', ax=ax2, color=['coral', 'lightgreen', 'gold'])
            ax2.set_title('Average Weekly Sales by Store Type', fontsize=14)
            ax2.set_xlabel('Store Type')
            ax2.set_ylabel('Average Weekly Sales ($)')
            ax2.grid(True, alpha=0.3)
           
            # Add value labels
            for i, v in enumerate(type_performance.values):
                ax2.text(i, v, f'${v:,.0f}', ha='center', va='bottom')
           
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.savefig('pandas_store_performance_2010_2012.png', dpi=120)
            print("✅ Saved pandas_store_performance_2010_2012.png")
           
            # Print insights
            print(f"\n💡 Pandas Chart Insights (2010-2012):")
            print(f"• Top performing store: Store {top_stores.iloc[0]['Store']} (${top_stores.iloc[0]['Total_Sales']/1e6:.2f}M)")
            if 'Type' in type_performance:
                print(f"• Store type performance: Type {type_performance.idxmax()} highest (${type_performance.max():,.0f})")
           
        except Exception as e:
            print(f"❌ Error creating pandas chart: {e}")
       
    def create_matplotlib_chart(self):
        """Chart 2: Using Matplotlib - Sales Trends and Analysis (2010-2012)"""
        print("\n📊 Creating Matplotlib Chart - Sales Trends Analysis (2010-2012)")
       
        try:
            # Filter for 2010-2012
            df_2010_2012 = self.merged_df[self.merged_df['Year'].between(2010, 2012)]
           
            # Prepare data
            monthly_sales = df_2010_2012.groupby(df_2010_2012['Date'].dt.to_period('M'))['Weekly_Sales'].sum().reset_index()
            monthly_sales['Date'] = monthly_sales['Date'].astype(str)
           
            # Create figure with improved layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Sales Trends Analysis (2010-2012)', fontsize=16, fontweight='bold')
           
            # Subplot 1: Sales trend over time
            ax1.plot(monthly_sales['Date'], monthly_sales['Weekly_Sales'],
                    marker='o', linewidth=2, color='#2E86AB')
            ax1.fill_between(monthly_sales['Date'], monthly_sales['Weekly_Sales'],
                            alpha=0.3, color='#2E86AB')
            ax1.set_title('Monthly Sales Trend', fontsize=14)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Total Sales ($)')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
           
            # Subplot 2: Sales distribution
            ax2.hist(df_2010_2012['Weekly_Sales'], bins=50,
                    alpha=0.7, color='orange', edgecolor='black')
            ax2.set_title('Weekly Sales Distribution', fontsize=14)
            ax2.set_xlabel('Weekly Sales ($)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
           
            # Add statistics
            mean_sales = df_2010_2012['Weekly_Sales'].mean()
            median_sales = df_2010_2012['Weekly_Sales'].median()
            ax2.axvline(mean_sales, color='red', linestyle='--', label=f'Mean: ${mean_sales:,.0f}')
            ax2.axvline(median_sales, color='green', linestyle='--', label=f'Median: ${median_sales:,.0f}')
            ax2.legend()
           
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig('matplotlib_trends_analysis_2010_2012.png', dpi=120)
            print("✅ Saved matplotlib_trends_analysis_2010_2012.png")
           
            print(f"\n💡 Matplotlib Chart Insights (2010-2012):")
            print(f"• Sales range: ${df_2010_2012['Weekly_Sales'].min():,.0f} - ${df_2010_2012['Weekly_Sales'].max():,.0f}")
            print(f"• Mean sales: ${mean_sales:,.0f}, Median sales: ${median_sales:,.0f}")
           
        except Exception as e:
            print(f"❌ Error creating matplotlib chart: {e}")
       
    def create_seaborn_interactive_chart(self):
        """Chart 3: Using Seaborn + Plotly - Interactive Analysis (2010-2012)"""
        print("\n📊 Creating Seaborn + Plotly Interactive Chart (2010-2012)")
       
        try:
            # Filter for 2010-2012
            df_2010_2012 = self.merged_df[self.merged_df['Year'].between(2010, 2012)]
           
            # Seaborn static analysis
            plt.figure(figsize=(12, 6))
           
            # Check if holiday data is available
            if 'IsHoliday' in df_2010_2012:
                sns.boxplot(data=df_2010_2012, x='Type', y='Weekly_Sales', hue='IsHoliday')
                plt.title('Sales Distribution by Store Type and Holiday Status (2010-2012)', fontsize=14)
            else:
                sns.boxplot(data=df_2010_2012, x='Type', y='Weekly_Sales')
                plt.title('Sales Distribution by Store Type (2010-2012)', fontsize=14)
               
            plt.xlabel('Store Type')
            plt.ylabel('Weekly Sales ($)')
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig('seaborn_analysis_2010_2012.png', dpi=120)
            print("✅ Saved seaborn_analysis_2010_2012.png")
            plt.show()
           
            # Create focused interactive Plotly chart
            print("🚀 Creating Interactive Plotly Visualization (2010-2012)...")
           
            # Prepare data for interactive chart
            store_performance = df_2010_2012.groupby('Store').agg({
                'Weekly_Sales': 'sum',
                'Type': 'first',
                'Size': 'first'
            }).reset_index()
           
            # Create interactive scatter plot
            fig = px.scatter(
                store_performance,
                x='Size',
                y='Weekly_Sales',
                color='Type',
                size='Weekly_Sales',
                hover_data=['Store', 'Type', 'Size'],
                title='Store Performance: Size vs Total Sales (2010-2012)'
            )
           
            fig.update_layout(
                height=500,
                xaxis_title='Store Size (sq ft)',
                yaxis_title='Total Sales ($)',
                template='plotly_white'
            )
           
            # Show interactive chart
            fig.show()
           
            # Calculate insights
            print(f"\n💡 Interactive Chart Insights (2010-2012):")
            print(f"• Total stores analyzed: {len(store_performance)}")
            if 'Type' in store_performance:
                type_perf = store_performance.groupby('Type')['Weekly_Sales'].mean()
                print(f"• Best performing type: {type_perf.idxmax()} (${type_perf.max():,.0f} avg sales)")
           
        except Exception as e:
            print(f"❌ Error creating seaborn/plotly charts: {e}")
           
    def create_heatmap(self):
        """Create an interactive heat map of weekly sales by store and week of year (2010-2012)"""
        print("\n🔥 Creating Interactive Heatmap: Weekly Sales by Store and Week of Year (2010-2012)")
       
        try:
            # Filter for 2010-2012
            df_2010_2012 = self.merged_df[self.merged_df['Year'].between(2010, 2012)]
           
            # Extract week of year from date (ISO week: Monday-Sunday)
            df_2010_2012['Week'] = df_2010_2012['Date'].dt.isocalendar().week
           
            # Group by store and week, calculate average sales
            grouped = df_2010_2012.groupby(['Store', 'Week'])['Weekly_Sales'].mean().reset_index()
           
            # Pivot to create matrix format for heatmap
            pivot_table = grouped.pivot(index='Store', columns='Week', values='Weekly_Sales')
           
            # Sort stores and weeks
            pivot_table = pivot_table.sort_index(axis=0)  # sort stores
            pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)  # sort weeks
           
            # Create hover text
            hover_text = []
            for store in pivot_table.index:
                store_text = []
                for week in pivot_table.columns:
                    sales = pivot_table.loc[store, week]
                    store_text.append(
                        f"Store: {store}<br>Week: {week}<br>Sales: ${sales:,.0f}"
                    )
                hover_text.append(store_text)
           
            # Create the heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='Viridis',
                hoverongaps=False,
                hoverinfo='text',
                text=hover_text
            ))
           
            # Customize layout
            fig.update_layout(
                title='Average Weekly Sales by Store and Week of Year (2010-2012)',
                xaxis_title='Week of Year',
                yaxis_title='Store',
                height=600,
                width=900,
                xaxis=dict(tickmode='linear', dtick=4),
                yaxis=dict(tickmode='linear', dtick=1),
                template='plotly_white'
            )
           
            # Add insights to the chart
            max_week = pivot_table.max().idxmax()
            min_week = pivot_table.min().idxmin()
            fig.add_annotation(
                text=f"Peak sales typically in week {max_week} | Lowest sales in week {min_week}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15, showarrow=False,
                font=dict(size=12)
            )
           
            # Show the figure
            fig.show()
            print("✅ Interactive heatmap created and displayed.")
           
            # Print insights
            print("\n💡 Heatmap Insights (2010-2012):")
            print(f"• Strongest sales typically in week {max_week} (holiday season)")
            print(f"• Weakest sales typically in week {min_week}")
           
        except Exception as e:
            print(f"❌ Error creating heatmap: {e}")
       
    def run_complete_analysis(self):
        """Run the complete analytics pipeline"""
        print("🚀 Starting Retail Analytics Pipeline...\n")
       
        # Step 1: Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting analysis.")
            return False
           
        # Step 2: Clean data
        self.clean_data()
       
        # Step 3: Merge datasets
        if not self.merge_datasets():
            print("❌ Failed to merge datasets. Exiting analysis.")
            return False
       
        # Check data availability
        if self.merged_df is None or len(self.merged_df) == 0:
            print("❌ No data available after cleaning and merging. Exiting analysis.")
            return False
       
        # Step 4: Create visualizations
        print("\n" + "="*60)
        print("📈 CREATING VISUALIZATIONS (2010-2012)")
        print("="*60)
       
        self.create_pandas_chart()
        print("\n" + "-"*40)
       
        self.create_matplotlib_chart()
        print("\n" + "-"*40)
       
        self.create_seaborn_interactive_chart()
        print("\n" + "-"*40)
       
        self.create_heatmap()
       
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("📁 Generated files:")
        print("  • pandas_store_performance_2010_2012.png")
        print("  • matplotlib_trends_analysis_2010_2012.png")
        print("  • seaborn_analysis_2010_2012.png")
        print("  • Interactive Plotly chart (displayed in browser)")
        print("  • Interactive Heatmap (displayed in browser)")
       
        return True

# Execute the analysis
if __name__ == "__main__":
    analytics = RetailAnalytics()
    analytics.run_complete_analysis()
