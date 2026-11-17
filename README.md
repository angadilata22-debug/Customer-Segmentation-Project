# Customer Segmentation Using Python and K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning internship project demonstrating unsupervised customer segmentation using **Python**, **Scikit-Learn**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**.

---

## 📌 Important Disclaimer

> [!IMPORTANT]
> **Synthetic Sample Data Notice:**
> The customer dataset utilized in this project is **completely synthetic** and was algorithmically generated using statistical probability distributions for educational, academic, and internship portfolio demonstration purposes. **It does NOT represent real customer records, transactions, or proprietary information of any real-world company or commercial entity.**

---

## 📖 Project Overview

In competitive retail and e-commerce markets, understanding distinct customer groups is vital for targeted marketing, personalized product recommendations, and customer retention. 

This project implements an unsupervised Machine Learning pipeline to group customers into behavioral clusters using:
- **Demographics:** Customer Age
- **Financial Status:** Annual Income ($k)
- **Engagement Habits:** Spending Score (1 - 100)
- **Purchase Behavior:** Annual Purchase Frequency & Average Transaction Value ($)

By applying **Feature Standardization (StandardScaler)**, **The Elbow Method (WCSS)**, and **Silhouette Score Analysis**, we determine that **5 distinct customer clusters** provide the optimal behavioral segmentation.

---

## 📂 Project Directory Structure

```text
Customer_Segmentation_Project/
│
├── data/
│   ├── raw_customers.csv                   # Synthetic raw customer dataset (250 records)
│   └── segmented_customers.csv             # Customer records with assigned Cluster & Persona labels
│
├── src/
│   ├── __init__.py                         # Python package initializer
│   ├── generate_data.py                    # Synthetic dataset generator with reproducible distributions
│   ├── eda.py                              # Exploratory data analysis & statistical distribution plots
│   ├── segmentation.py                     # StandardScaler, Elbow Method, Silhouette Analysis & K-Means
│   └── utils.py                            # Plot styling, figure export helpers, and banner formatters
│
├── output/
│   ├── figures/                            # High-resolution (300 DPI) visualization graphs
│   │   ├── 01_age_distribution.png
│   │   ├── 02_income_distribution.png
│   │   ├── 03_spending_score_distribution.png
│   │   ├── 04_purchase_frequency_distribution.png
│   │   ├── 05_correlation_heatmap.png
│   │   ├── 06_income_vs_spending_scatter.png
│   │   ├── 07_elbow_method.png
│   │   ├── 08_silhouette_scores.png
│   │   ├── 09_customer_clusters_2d.png
│   │   ├── 10_customer_clusters_pairplot.png
│   │   └── 11_cluster_profile_comparison.png
│   │
│   └── reports/                            # Tabular analytical reports
│       ├── cluster_summary_statistics.csv  # Mean, median, std, min, max across all features
│       └── cluster_profiles_interpretation.csv # Persona descriptions & targeted marketing strategies
│
├── main.py                                 # Master pipeline execution script
├── requirements.txt                        # Python library dependencies
├── README.md                               # Project documentation & setup instructions
└── PROJECT_REPORT.md                       # Comprehensive internship academic project report
```

---

## 🧠 Core Machine Learning Concepts (Beginner Guide)

### 1. What is Customer Segmentation?
Customer segmentation is the process of partitioning a customer base into mutually exclusive subgroups based on shared characteristics (such as purchasing frequency, budget, or age). Because the data does not come with pre-existing labels, this is an **Unsupervised Learning** problem.

### 2. Why Feature Scaling (`StandardScaler`) is Essential
K-Means calculates similarity using **Euclidean Distance**:
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

If Annual Income ranges from \$15,000 to \$140,000, while Age ranges from 18 to 70, a difference of \$5,000 in income would completely overpower a difference of 20 years in age. `StandardScaler` standardizes each feature to have a **mean of 0** and a **standard deviation of 1**:
$$z = \frac{x - \mu}{\sigma}$$
This ensures all dimensions contribute fairly to cluster formation.

### 3. How K-Means Clustering Works
1. **Initialization:** Randomly or strategically (using `k-means++`) select $k$ initial cluster centroids.
2. **Assignment:** Assign each customer to the nearest centroid based on Euclidean distance.
3. **Update:** Recalculate each centroid as the mean of all points assigned to that cluster.
4. **Convergence:** Repeat assignment and update steps until centroids stop shifting or reach the maximum iterations.

### 4. Selecting the Optimal Number of Clusters ($k$)
We apply two complementary diagnostic techniques:
- **The Elbow Method (WCSS / Inertia):** Measures the sum of squared distances of samples to their closest cluster center. As $k$ increases, WCSS decreases. We identify the "elbow point" where the rate of decrease abruptly slows down (found at $k = 5$).
- **Silhouette Score:** Evaluates how well-separated and cohesive clusters are, on a scale from $-1$ to $+1$. A higher positive score indicates that samples are close to their own cluster members and far from neighboring clusters. Peak silhouette is achieved at $k = 5$.

---

## 👥 Identified Customer Segments (5 Personas)

| Cluster ID | Segment Name | Income Level | Spending Habits | Purchase Frequency | Target Marketing Strategy |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | **Elite Loyal Shoppers** | High (\$90k+) | High (75-99) | High (40+ / yr) | VIP loyalty tiers, luxury concierge services, exclusive new collection pre-orders. |
| **1** | **Conservative High Earners** | High (\$85k+) | Low (10-35) | Low (10-15 / yr) | Quality assurance guarantees, long-term durability value, high-utility premium products. |
| **2** | **Young Trendsetters** | Low/Mid (\$25k-\$45k) | High (70-95) | High (35-50 / yr) | Social media campaigns (TikTok/Instagram), student discounts, Buy-Now-Pay-Later (BNPL). |
| **3** | **Budget Conscious / At-Risk** | Low (\$15k-\$35k) | Low (10-35) | Low (5-12 / yr) | Clearance flash discounts, economy bundle pricing, essential value packs. |
| **4** | **Balanced Mainstream** | Moderate (\$50k-\$70k) | Moderate (40-60) | Moderate (20-30 / yr) | Personalized cross-selling, seasonal promotional coupons, customer loyalty point multipliers. |

---

## ⚙️ Installation & Setup Instructions

### Prerequisites
- Python 3.10 or higher installed on your computer.
- `pip` package manager.

### Step 1: Open Terminal / Command Prompt
Navigate to the project root folder:
```bash
cd "c:\Users\HP ULTRA 5\Desktop\Customer_Segmentation_Project"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

The installed packages include:
- `pandas` (Data manipulation and table handling)
- `numpy` (Numerical operations and random distributions)
- `scikit-learn` (StandardScaler, KMeans, silhouette_score)
- `matplotlib` (Core charting engine)
- `seaborn` (Statistical data visualizations)

---

## 🚀 Execution Guide

### Option A: Run the Complete End-to-End Pipeline (Recommended)
Execute `main.py` to generate the dataset, run exploratory analysis, evaluate optimal clusters, fit the model, render all 11 plots, and export the final reports:
```bash
python main.py
```

### Option B: Run Individual Modules Step-by-Step
For educational learning, you can execute individual scripts independently:

1. **Generate the Synthetic Dataset:**
   ```bash
   python src/generate_data.py
   ```
2. **Run Exploratory Data Analysis (EDA):**
   ```bash
   python src/eda.py
   ```
3. **Execute Clustering & Segmentation:**
   ```bash
   python src/segmentation.py
   ```

---

## 📊 Summary of Generated Visualizations

All graphs are rendered at **300 DPI** and saved into the `output/figures/` folder:

1. **`01_age_distribution.png`**: Histogram and KDE showing age spread across customers.
2. **`02_income_distribution.png`**: Distribution of annual earnings in thousands of dollars.
3. **`03_spending_score_distribution.png`**: Spread of retail spending score from 1 to 100.
4. **`04_purchase_frequency_distribution.png`**: Distribution of yearly shopping transactions.
5. **`05_correlation_heatmap.png`**: Correlation matrix highlighting feature dependencies.
6. **`06_income_vs_spending_scatter.png`**: Visual inspection of natural customer spending patterns.
7. **`07_elbow_method.png`**: Within-Cluster Sum of Squares (WCSS) curve showing the $k=5$ elbow.
8. **`08_silhouette_scores.png`**: Silhouette coefficient curve confirming $k=5$ cluster quality.
9. **`09_customer_clusters_2d.png`**: 2D scatter plot of clusters with centroids marked as distinct **X** markers.
10. **`10_customer_clusters_pairplot.png`**: High-dimensional pairwise relationship matrix across all 4 segmentation features.
11. **`11_cluster_profile_comparison.png`**: Normalized comparative bar chart highlighting the distinct traits of each customer segment.

---

## 🎓 Internship Project Presentation Tips

If presenting this project to a hiring manager, academic mentor, or internship evaluator, follow this structure:
1. **Explain the Business Problem:** Companies cannot use a "one-size-fits-all" marketing strategy; segmenting customers drives higher ROI and retention.
2. **Emphasize Unsupervised ML:** Explain why clustering was chosen (we do not have ground truth labels).
3. **Highlight Feature Preprocessing:** Explain that Euclidean distance requires `StandardScaler` so that high-dollar incomes don't dwarf age or frequency.
4. **Defend Model Parameters:** Point to the **Elbow Curve** and **Silhouette Analysis** to mathematically justify choosing $k=5$ clusters instead of guessing.
5. **Deliver Actionable Value:** Connect each cluster back to commercial strategy (e.g., how to treat High-Income/Low-Spenders differently from High-Income/High-Spenders).

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE). Educational demonstration only.
