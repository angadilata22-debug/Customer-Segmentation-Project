"""
=============================================================================
                CUSTOMER SEGMENTATION USING PYTHON & K-MEANS
                     Internship Project Master Runner
=============================================================================

Project Overview:
-----------------
This project demonstrates an end-to-end Machine Learning customer segmentation
pipeline using Python, Pandas, NumPy, Scikit-Learn, Matplotlib, and Seaborn.

Workflow:
---------
1. Data Generation:
   Creates a synthetic customer dataset with demographics and purchase behaviors.
   (Clearly labeled as synthetic data, not real company customer records).
2. Exploratory Data Analysis (EDA):
   Analyzes distributions, correlations, and spending patterns, saving visual plots.
3. Feature Scaling:
   Standardizes Age, Annual Income, Spending Score, and Purchase Frequency with StandardScaler.
4. Optimal Cluster Determination:
   Evaluates 2 to 10 clusters using the Elbow Method (WCSS) and Silhouette Score.
5. K-Means Clustering:
   Applies K-Means clustering algorithm (k=5) to group customers into 5 distinct personas.
6. Segment Profiling & Business Labeling:
   Interprets centroids in original units and names segments (e.g. Elite Loyal, Young Trendsetters).
7. Visualization & Export:
   Generates 2D cluster maps, pairplots, and comparative profile charts.
   Exports segmented customer data and summary analysis reports as CSV files.

Author: Python Data Science Intern
License: MIT (Educational Project)
=============================================================================
"""

import os
import sys
import time

# Ensure non-interactive backend is active before any plotting library is loaded
import matplotlib
matplotlib.use("Agg")

from src.generate_data import save_raw_dataset
from src.eda import run_exploratory_data_analysis
from src.segmentation import run_customer_segmentation_pipeline
from src.utils import print_section_header


def main():
    start_time = time.time()

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║        CUSTOMER SEGMENTATION USING PYTHON & K-MEANS              ║
    ║                 Internship Machine Learning Project              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # -------------------------------------------------------------
    # Step 1: Create Synthetic Customer Dataset
    # -------------------------------------------------------------
    print_section_header("Step 0: Generating Synthetic Customer Dataset")
    raw_csv_path = "data/raw_customers.csv"
    save_raw_dataset(output_path=raw_csv_path, num_records=250)

    # -------------------------------------------------------------
    # Step 2: Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    run_exploratory_data_analysis(data_path=raw_csv_path, output_dir="output/figures")

    # -------------------------------------------------------------
    # Step 3: Segmentation Pipeline (Scaling, Elbow, Silhouette, K-Means)
    # -------------------------------------------------------------
    df_segmented, interpretation_df = run_customer_segmentation_pipeline(data_path=raw_csv_path)

    # -------------------------------------------------------------
    # Execution Summary
    # -------------------------------------------------------------
    elapsed_time = time.time() - start_time
    print_section_header("Project Execution Completed Successfully")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
    print("Project Deliverables Generated:")
    print("  [Data Files]")
    print("    - data/raw_customers.csv (250 synthetic customer records)")
    print("    - data/segmented_customers.csv (Customers with assigned clusters & persona names)")
    print("\n  [Visualizations (output/figures/)]")
    print("    - 01_age_distribution.png")
    print("    - 02_income_distribution.png")
    print("    - 03_spending_score_distribution.png")
    print("    - 04_purchase_frequency_distribution.png")
    print("    - 05_correlation_heatmap.png")
    print("    - 06_income_vs_spending_scatter.png")
    print("    - 07_elbow_method.png")
    print("    - 08_silhouette_scores.png")
    print("    - 09_customer_clusters_2d.png")
    print("    - 10_customer_clusters_pairplot.png")
    print("    - 11_cluster_profile_comparison.png")
    print("\n  [Reports (output/reports/)]")
    print("    - cluster_summary_statistics.csv (Mean/median/std metrics per cluster)")
    print("    - cluster_profiles_interpretation.csv (Target marketing strategies)")
    print("\n  [Documentation]")
    print("    - README.md (Setup instructions, technical architecture, beginner guide)")
    print("    - PROJECT_REPORT.md (Comprehensive internship project report)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
