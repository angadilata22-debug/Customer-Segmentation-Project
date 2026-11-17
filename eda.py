"""
=============================================================================
Module: eda.py
Project: Customer Segmentation Using Python and K-Means Clustering
Description: Performs comprehensive Exploratory Data Analysis (EDA) on customer
             demographics and purchasing behaviour, saving descriptive stats
             and visualization figures.
=============================================================================
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils import setup_plotting_style, save_figure, print_section_header


def run_exploratory_data_analysis(data_path: str = "data/raw_customers.csv",
                                  output_dir: str = "output/figures") -> pd.DataFrame:
    """
    Perform Exploratory Data Analysis (EDA) on the raw customer dataset.

    Steps performed:
    1. Load customer data into a Pandas DataFrame.
    2. Inspect data structure, data types, and check for missing values.
    3. Compute descriptive statistical summaries.
    4. Generate and save exploratory distribution and correlation plots.

    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing customer data.
    output_dir : str
        Folder where generated figure images will be saved.

    Returns:
    --------
    pd.DataFrame: The loaded DataFrame ready for modeling.
    """
    print_section_header("Step 1: Exploratory Data Analysis (EDA)")

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please run data generation first.")

    df = pd.read_csv(data_path)
    print(f"[EDA] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # 2. Inspect first few rows
    print("[EDA] First 5 records of customer dataset:")
    print(df.head())
    print("\n" + "-" * 60)

    # 3. Check for missing values and data types
    print("[EDA] Missing values check:")
    missing_vals = df.isnull().sum()
    print(missing_vals)
    print("\n" + "-" * 60)

    # 4. Summary statistics
    print("[EDA] Descriptive Statistical Summary:")
    stats_df = df.describe().round(2)
    print(stats_df)
    print("\n" + "-" * 60)

    # Configure visual style
    setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Plot 1: Age Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Age"], kde=True, bins=20, color="#2b5c8f", ax=ax)
    ax.set_title("Customer Age Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Age (Years)", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Mean Age: {df['Age'].mean():.1f}")
    ax.legend()
    save_figure(fig, os.path.join(output_dir, "01_age_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 2: Annual Income Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Annual_Income_k$"], kde=True, bins=20, color="#2a9d8f", ax=ax)
    ax.set_title("Annual Income Distribution (in $k)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Annual Income (Thousands of USD)", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.axvline(df["Annual_Income_k$"].mean(), color="red", linestyle="--",
               label=f"Mean Income: ${df['Annual_Income_k$'].mean():.1f}k")
    ax.legend()
    save_figure(fig, os.path.join(output_dir, "02_income_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 3: Spending Score Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Spending_Score"], kde=True, bins=20, color="#e76f51", ax=ax)
    ax.set_title("Customer Spending Score Distribution (1-100)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Spending Score (1 - 100)", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.axvline(df["Spending_Score"].mean(), color="black", linestyle="--",
               label=f"Mean Score: {df['Spending_Score'].mean():.1f}")
    ax.legend()
    save_figure(fig, os.path.join(output_dir, "03_spending_score_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 4: Purchase Frequency Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Purchase_Frequency_Per_Year"], kde=True, bins=20, color="#9b5de5", ax=ax)
    ax.set_title("Annual Purchase Frequency Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Purchases Per Year", fontsize=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.axvline(df["Purchase_Frequency_Per_Year"].mean(), color="red", linestyle="--",
               label=f"Mean Frequency: {df['Purchase_Frequency_Per_Year'].mean():.1f}")
    ax.legend()
    save_figure(fig, os.path.join(output_dir, "04_purchase_frequency_distribution.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 5: Correlation Heatmap
    # -------------------------------------------------------------
    numeric_cols = ["Age", "Annual_Income_k$", "Spending_Score",
                    "Purchase_Frequency_Per_Year", "Average_Transaction_Value_$"]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues",
                square=True, cbar_kws={"shrink": 0.8}, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    save_figure(fig, os.path.join(output_dir, "05_correlation_heatmap.png"))
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 6: Spending Patterns (Income vs. Spending Score)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Annual_Income_k$",
        y="Spending_Score",
        hue="Gender",
        size="Purchase_Frequency_Per_Year",
        sizes=(40, 200),
        palette={"Female": "#e63946", "Male": "#1d3557"},
        alpha=0.85,
        ax=ax
    )
    ax.set_title("Customer Spending Patterns: Annual Income vs. Spending Score",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Annual Income ($k)", fontsize=12)
    ax.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    save_figure(fig, os.path.join(output_dir, "06_income_vs_spending_scatter.png"))
    plt.close(fig)

    print("[EDA] Exploratory data analysis completed successfully!\n")
    return df


if __name__ == "__main__":
    run_exploratory_data_analysis()
