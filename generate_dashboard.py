"""
Script to generate a rich, modern, responsive HTML dashboard for the Customer Segmentation project.
Embeds data and links all 11 high-res visualization images.
"""

import json
import pandas as pd

# Load datasets
df_seg = pd.read_csv('data/segmented_customers.csv')
df_interp = pd.read_csv('output/reports/cluster_profiles_interpretation.csv')
df_stats = pd.read_csv('output/reports/cluster_summary_statistics.csv')

customers_json = json.dumps(df_seg.to_dict(orient='records'))
personas_json = json.dumps(df_interp.to_dict(orient='records'))

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Segmentation Dashboard | Python & K-Means</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #0f766e;
            --accent: #f59e0b;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding-bottom: 60px;
        }}
        /* Header */
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
            color: white;
            padding: 30px 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .header-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 8px;
        }}
        .subtitle {{
            font-size: 1.05rem;
            opacity: 0.9;
            max-width: 800px;
        }}
        .disclaimer-banner {{
            margin-top: 15px;
            background: rgba(245, 158, 11, 0.2);
            border-left: 4px solid #f59e0b;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 0.88rem;
        }}
        /* Navigation */
        nav {{
            background: white;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .nav-container {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            gap: 20px;
            overflow-x: auto;
            padding: 0 20px;
        }}
        .nav-link {{
            padding: 16px 4px;
            text-decoration: none;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.95rem;
            border-bottom: 2px solid transparent;
            white-space: nowrap;
            transition: all 0.2s;
        }}
        .nav-link:hover, .nav-link.active {{
            color: var(--primary-light);
            border-bottom-color: var(--primary-light);
        }}
        /* Main Layout */
        main {{
            max-width: 1200px;
            margin: 30px auto;
            padding: 0 20px;
        }}
        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 35px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .kpi-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary);
            margin: 6px 0;
        }}
        .kpi-subtext {{
            font-size: 0.82rem;
            color: var(--success);
            font-weight: 500;
        }}
        /* Section styling */
        section {{
            margin-bottom: 45px;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 10px;
        }}
        .section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        .section-desc {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 20px;
        }}
        /* Persona Cards */
        .persona-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }}
        .persona-card {{
            background: white;
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 22px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            position: relative;
            overflow: hidden;
        }}
        .persona-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
        }}
        .cluster-0::before {{ background: #1f77b4; }}
        .cluster-1::before {{ background: #ff7f0e; }}
        .cluster-2::before {{ background: #2ca02c; }}
        .cluster-3::before {{ background: #d62728; }}
        .cluster-4::before {{ background: #9467bd; }}
        .persona-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        .persona-title {{
            font-size: 1.2rem;
            font-weight: 700;
        }}
        .cluster-tag {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            background: #eff6ff;
            color: #1e40af;
        }}
        .persona-desc {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 15px;
        }}
        .strategy-box {{
            background: #f1f5f9;
            border-radius: 8px;
            padding: 12px;
            font-size: 0.88rem;
        }}
        .strategy-title {{
            font-weight: 700;
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--secondary);
            margin-bottom: 4px;
        }}
        /* Image Grid */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .gallery-item {{
            background: white;
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s;
        }}
        .gallery-item:hover {{
            box-shadow: 0 6px 12px -2px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }}
        .gallery-img {{
            width: 100%;
            height: 240px;
            object-fit: cover;
            cursor: pointer;
            border-bottom: 1px solid var(--border);
            background: #f1f5f9;
        }}
        .gallery-info {{
            padding: 15px;
        }}
        .gallery-title {{
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 6px;
        }}
        .gallery-desc {{
            font-size: 0.85rem;
            color: var(--text-muted);
        }}
        /* Table Controls & Styling */
        .table-card {{
            background: white;
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .table-toolbar {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
            justify-content: space-between;
            background: #fafafa;
        }}
        .search-box {{
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 0.9rem;
            width: 260px;
        }}
        .filter-select {{
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 0.9rem;
            background: white;
        }}
        .table-responsive {{
            max-height: 450px;
            overflow-y: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.88rem;
        }}
        th {{
            background: #f8fafc;
            position: sticky;
            top: 0;
            padding: 12px 16px;
            font-weight: 700;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover td {{
            background-color: #f1f5f9;
        }}
        /* Lightbox Modal */
        .modal {{
            display: none;
            position: fixed;
            z-index: 999;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .modal-content {{
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            background: white;
        }}
        .close-btn {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="header-container">
            <span class="badge">Machine Learning Internship Project</span>
            <h1>Customer Segmentation Dashboard</h1>
            <p class="subtitle">Interactive visualization of customer behavioral clustering using Python, Pandas, StandardScaler, and K-Means algorithm.</p>
            <div class="disclaimer-banner">
                <strong>Important Notice:</strong> This dataset is completely synthetic sample data generated using statistical distributions for educational demonstration. It does not represent real customer records of any actual commercial entity.
            </div>
        </div>
    </header>

    <!-- Sticky Navigation -->
    <nav>
        <div class="nav-container">
            <a href="#overview" class="nav-link active">Overview & KPIs</a>
            <a href="#personas" class="nav-link">Customer Personas (5 Segments)</a>
            <a href="#clustering" class="nav-link">K-Means Visualizations</a>
            <a href="#model-validation" class="nav-link">Elbow & Silhouette Evaluation</a>
            <a href="#eda" class="nav-link">Exploratory Data Analysis</a>
            <a href="#data-explorer" class="nav-link">Customer Data Explorer</a>
        </div>
    </nav>

    <main>
        <!-- KPI Metrics -->
        <section id="overview">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-title">Total Customers</div>
                    <div class="kpi-value">250</div>
                    <div class="kpi-subtext">Synthetic Records</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Optimal Clusters (k)</div>
                    <div class="kpi-value">5</div>
                    <div class="kpi-subtext">Elbow & Silhouette Validated</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Peak Silhouette Score</div>
                    <div class="kpi-value">0.483</div>
                    <div class="kpi-subtext">High Cluster Cohesion</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">WCSS (Inertia)</div>
                    <div class="kpi-value">188.1</div>
                    <div class="kpi-subtext">Clear Elbow Inflection</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Avg Annual Income</div>
                    <div class="kpi-value">$60.6k</div>
                    <div class="kpi-subtext">Range: $15k - $140k</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Avg Spending Score</div>
                    <div class="kpi-value">52.6</div>
                    <div class="kpi-subtext">Range: 1 - 99</div>
                </div>
            </div>
        </section>

        <!-- Customer Personas -->
        <section id="personas">
            <div class="section-header">
                <h2 class="section-title">Discovered Customer Personas (5 Clusters)</h2>
            </div>
            <p class="section-desc">K-Means grouped the 250 customer profiles into five distinct behavioral segments with actionable commercial strategies.</p>
            
            <div class="persona-grid" id="personaCardsContainer">
                <!-- Injected via JavaScript -->
            </div>
        </section>

        <!-- K-Means Visualizations -->
        <section id="clustering">
            <div class="section-header">
                <h2 class="section-title">Customer Cluster Visualizations</h2>
            </div>
            <p class="section-desc">Visual representations of customer clusters and their centroid locations. Click any image to view in full resolution.</p>

            <div class="gallery-grid">
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/09_customer_clusters_2d.png" alt="2D Customer Clusters" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">2D Customer Clusters & Centroids</div>
                        <div class="gallery-desc">Annual Income vs. Spending Score with cluster centroids marked as bold X icons.</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/11_cluster_profile_comparison.png" alt="Cluster Profile Comparison" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Feature Profile Comparison</div>
                        <div class="gallery-desc">Normalized comparative bar chart showing relative Age, Income, Spending, and Frequency per segment.</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/10_customer_clusters_pairplot.png" alt="Cluster Pairplot" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Multi-Dimensional Cluster Pairplot</div>
                        <div class="gallery-desc">High-dimensional scatterplot matrix across all 4 customer segmentation features.</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Model Evaluation -->
        <section id="model-validation">
            <div class="section-header">
                <h2 class="section-title">Optimal Cluster Selection: Elbow & Silhouette Analysis</h2>
            </div>
            <p class="section-desc">Diagnostic curves justifying the mathematical selection of k = 5 clusters.</p>

            <div class="gallery-grid">
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/07_elbow_method.png" alt="Elbow Method" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">The Elbow Method (WCSS / Inertia)</div>
                        <div class="gallery-desc">Shows sharp drop in inertia from k=2 to k=5, reaching the optimal elbow point at k=5.</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/08_silhouette_scores.png" alt="Silhouette Scores" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Silhouette Score Curve</div>
                        <div class="gallery-desc">Confirms highest cluster cohesion and separation peak at k = 5 (score: 0.4829).</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Exploratory Data Analysis -->
        <section id="eda">
            <div class="section-header">
                <h2 class="section-title">Exploratory Data Analysis (EDA)</h2>
            </div>
            <p class="section-desc">Statistical distributions and correlation patterns prior to model training.</p>

            <div class="gallery-grid">
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/01_age_distribution.png" alt="Age Distribution" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Customer Age Distribution</div>
                        <div class="gallery-desc">Histogram and KDE curve showing customer ages (Mean: 38.8 years).</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/02_income_distribution.png" alt="Income Distribution" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Annual Income Distribution</div>
                        <div class="gallery-desc">Multi-modal spread of customer earnings (Mean: $60.6k).</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/03_spending_score_distribution.png" alt="Spending Score Distribution" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Spending Score Distribution</div>
                        <div class="gallery-desc">Retail engagement score spread from 1 to 100 (Mean: 52.6).</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/04_purchase_frequency_distribution.png" alt="Purchase Frequency" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Purchase Frequency Distribution</div>
                        <div class="gallery-desc">Annual transaction count distribution (Mean: 25.4 purchases/year).</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/05_correlation_heatmap.png" alt="Correlation Heatmap" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Feature Correlation Matrix</div>
                        <div class="gallery-desc">Heatmap quantifying linear relationships between customer features.</div>
                    </div>
                </div>
                <div class="gallery-item">
                    <img class="gallery-img" src="output/figures/06_income_vs_spending_scatter.png" alt="Income vs Spending Scatter" onclick="openModal(this.src)">
                    <div class="gallery-info">
                        <div class="gallery-title">Income vs. Spending Score Scatter</div>
                        <div class="gallery-desc">Scatterplot revealing naturally emerging consumer clusters.</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Customer Data Explorer -->
        <section id="data-explorer">
            <div class="section-header">
                <h2 class="section-title">Interactive Customer Data Explorer</h2>
                <div>
                    <span id="recordCountBadge" style="font-size: 0.9rem; font-weight: 600; color: var(--primary);">250 Customers</span>
                </div>
            </div>
            <p class="section-desc">Filter and search through the segmented customer dataset with cluster labels and assigned persona names.</p>

            <div class="table-card">
                <div class="table-toolbar">
                    <input type="text" id="searchInput" class="search-box" placeholder="Search by ID, Segment, Gender...">
                    <select id="clusterFilter" class="filter-select">
                        <option value="ALL">All Clusters</option>
                        <option value="0">Cluster 0: Young Trendsetters</option>
                        <option value="1">Cluster 1: Conservative High Earners</option>
                        <option value="2">Cluster 2: Elite Loyal Shoppers</option>
                        <option value="3">Cluster 3: Budget Conscious / At-Risk</option>
                        <option value="4">Cluster 4: Balanced Mainstream</option>
                    </select>
                </div>
                <div class="table-responsive">
                    <table id="customerTable">
                        <thead>
                            <tr>
                                <th>Customer ID</th>
                                <th>Gender</th>
                                <th>Age</th>
                                <th>Income ($k)</th>
                                <th>Spending Score</th>
                                <th>Frequency (yr)</th>
                                <th>Avg Cart ($)</th>
                                <th>Cluster</th>
                                <th>Persona Name</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                            <!-- Populated via JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>

    <!-- Modal for full size images -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close-btn">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>

    <script>
        const customers = {customers_json};
        const personas = {personas_json};

        // Render Persona Cards
        const personaContainer = document.getElementById('personaCardsContainer');
        personas.forEach((p, index) => {{
            const card = document.createElement('div');
            card.className = `persona-card cluster-${{p.Cluster_ID}}`;
            card.innerHTML = `
                <div class="persona-header">
                    <div>
                        <span class="cluster-tag">Cluster ${{p.Cluster_ID}}</span>
                        <h3 class="persona-title" style="margin-top:6px;">${{p.Segment_Name}}</h3>
                    </div>
                    <span style="font-weight:700; color:#0f766e; font-size:0.95rem;">${{p.Customer_Count}} (${{((p.Customer_Count/250)*100).toFixed(1)}}%)</span>
                </div>
                <p class="persona-desc">${{p.Description}}</p>
                <div class="strategy-box">
                    <div class="strategy-title">Target Commercial Strategy</div>
                    <div>${{p.Marketing_Strategy}}</div>
                </div>
            `;
            personaContainer.appendChild(card);
        }});

        // Render Customer Table
        const tbody = document.getElementById('tableBody');
        const searchInput = document.getElementById('searchInput');
        const clusterFilter = document.getElementById('clusterFilter');
        const countBadge = document.getElementById('recordCountBadge');

        function renderTable(data) {{
            tbody.innerHTML = '';
            countBadge.innerText = `${{data.length}} Customers`;
            data.forEach(c => {{
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td style="font-weight:600;">${{c.CustomerID}}</td>
                    <td>${{c.Gender}}</td>
                    <td>${{c.Age}}</td>
                    <td>$${{c["Annual_Income_k$"]}}k</td>
                    <td><span style="font-weight:600;">${{c.Spending_Score}}</span> / 100</td>
                    <td>${{c.Purchase_Frequency_Per_Year}}</td>
                    <td>$${{c["Average_Transaction_Value_$"]}}</td>
                    <td><span class="cluster-tag" style="background:#f1f5f9; color:#1e293b;">#${{c.Cluster}}</span></td>
                    <td style="font-weight:600; color:var(--primary);">${{c.Segment_Name}}</td>
                `;
                tbody.appendChild(row);
            }});
        }}

        function filterData() {{
            const query = searchInput.value.toLowerCase();
            const selectedCluster = clusterFilter.value;

            const filtered = customers.filter(c => {{
                const matchesCluster = selectedCluster === 'ALL' || c.Cluster.toString() === selectedCluster;
                const matchesSearch = 
                    c.CustomerID.toString().includes(query) ||
                    c.Segment_Name.toLowerCase().includes(query) ||
                    c.Gender.toLowerCase().includes(query);
                return matchesCluster && matchesSearch;
            }});

            renderTable(filtered);
        }}

        searchInput.addEventListener('input', filterData);
        clusterFilter.addEventListener('change', filterData);

        // Initial render
        renderTable(customers);

        // Image Modal
        function openModal(src) {{
            const modal = document.getElementById('imageModal');
            const img = document.getElementById('modalImg');
            img.src = src;
            modal.style.display = 'flex';
        }}
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}
    </script>
</body>
</html>
"""

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("[DASHBOARD] Successfully created index.html!")
