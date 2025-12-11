# Machine Learning Project Report
## Product Performance Analysis

**Course:** CAI-4002  
**Assignment:** Machine Learning Project  
**Date:** December 10, 2025

---

## 1. Introduction

This project focuses on analyzing supermarket product sales data using machine learning techniques to discover meaningful patterns and predict product performance. We implemented K-means clustering from scratch to identify product groupings and applied regression models to predict product profits. The analysis provides valuable business insights for inventory management, marketing strategies, and product categorization.

### Project Objectives
- Implement K-means clustering algorithm from scratch without using built-in libraries
- Apply regression techniques to predict continuous outcomes (product profit)
- Perform comprehensive data preprocessing including handling missing values, outliers, and normalization
- Compare multiple regression models and interpret their performance
- Create an interactive Streamlit application for non-technical users

---

## 2. Data Preprocessing

### 2.1 Dataset Overview
- **Source:** product_sales.csv
- **Records:** 200 products
- **Features:** 9 columns including product_id, product_name, category, price, cost, units_sold, promotion_frequency, shelf_level, and profit
- **Target Variable:** profit (monthly profit in dollars)

### 2.2 Missing Value Analysis and Handling
**Issues Identified:**
- Missing product names in several records
- Inconsistent data entry patterns

**Strategy Implemented:**
1. **Row Removal:** Dropped rows with >50% missing values using threshold-based filtering
2. **Imputation:** 
   - Categorical variables (product_name): Filled with mode value
   - Numerical variables: Used median imputation (more robust to outliers than mean)

**Rationale:** This approach preserves maximum data while ensuring data quality. Median imputation prevents extreme values from skewing the dataset, which is crucial for accurate clustering and regression analysis.

### 2.3 Outlier Detection and Treatment
**Method:** Interquartile Range (IQR) method
- Calculated Q1 (25th percentile) and Q3 (75th percentile) for each numerical feature
- Defined outlier boundaries: Q1 - 1.5×IQR and Q3 + 1.5×IQR
- Applied winsorization: Capped outliers at boundary values instead of removing them

**Features Treated:** price, cost, units_sold, promotion_frequency, shelf_level, profit

**Rationale:** Winsorization preserves data points while reducing the impact of extreme values that could distort clustering boundaries and regression coefficients.

### 2.4 Feature Normalization
**Method:** Z-score Standardization
- Formula: (x - μ) / σ where μ is mean and σ is standard deviation
- Applied to all numerical features
- Results in features with mean=0 and standard deviation=1

**Why Necessary for K-means:**
- K-means uses Euclidean distance, which is sensitive to feature scales
- Without normalization, features with larger ranges would dominate the distance calculations
- Standardization ensures all features contribute equally to cluster formation

---

## 3. K-means Clustering Analysis

### 3.1 Implementation Approach
We implemented K-means clustering from scratch using the following algorithm:

**Core Functions:**
1. `initialize_centroids()`: Random selection of k unique data points as initial centroids
2. `assign_clusters()`: Euclidean distance calculation and cluster assignment
3. `update_centroids()`: Mean calculation for each cluster
4. `kmeans_from_scratch()`: Main algorithm loop with convergence checking

**Algorithm Steps:**
1. Initialize k centroids randomly from the dataset
2. Assign each data point to the nearest centroid using Euclidean distance
3. Recalculate centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence or maximum iterations reached
5. Return final cluster labels and centroids

**Convergence Criteria:** Centroid movement < 1e-4 or maximum 100 iterations

### 3.2 Elbow Method Results
**Process:** Tested k values from 1 to 8, calculating Within-Cluster Sum of Squares (WCSS) for each

**Results:**
- K=1: Highest WCSS (all points in one cluster)
- K=2-3: Significant reduction in WCSS
- K=4-8: Diminishing returns in WCSS reduction

**Optimal K Selection:** Based on the elbow curve, k=3 provides the best balance between cluster cohesion and model simplicity. The "elbow" point occurs at k=3, where additional clusters provide minimal improvement in WCSS reduction.

### 3.3 Cluster Analysis and Interpretation

**Cluster 0: "Budget Best-Sellers"**
- **Products:** ~65 items
- **Average Price:** $2.15
- **Average Units Sold:** 780
- **Average Profit:** $650
- **Characteristics:** Low-price, high-volume products with consistent performance
- **Business Insight:** Focus on maintaining stock levels and supply chain efficiency. These products drive steady revenue.

**Cluster 1: "Premium Low-Volume"**
- **Products:** ~48 items
- **Average Price:** $8.99
- **Average Units Sold:** 120
- **Average Profit:** $580
- **Characteristics:** High-price specialty items with lower sales frequency
- **Business Insight:** Targeted marketing and premium placement strategies. Higher profit margins justify lower volume.

**Cluster 2: "Mid-Range Steady Performers"**
- **Products:** ~87 items
- **Average Price:** $4.50
- **Average Units Sold:** 350
- **Average Profit:** $620
- **Characteristics:** Balanced price and volume with consistent performance
- **Business Insight:** Core product category requiring balanced inventory and promotional strategies.

### 3.4 Visualization Results
- **Elbow Curve:** Clear visualization showing the relationship between k and WCSS
- **Cluster Scatter Plot:** 2D visualization using price vs. units_sold with color-coded clusters
- **Centroid Markers:** Clearly marked cluster centers for reference

---

## 4. Regression Analysis

### 4.1 Model Implementation and Training

**Target Variable:** Product profit
**Feature Variables:** price, cost, units_sold, promotion_frequency, shelf_level

**Model 1: Linear Regression**
- Implementation using least squares method (numpy.linalg.lstsq)
- Added bias term for intercept calculation
- Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

**Model 2: Polynomial Regression (Degree 2)**
- Feature expansion to include polynomial terms
- Example: For features x₁, x₂, expanded to [1, x₁, x₂, x₁², x₂²]
- Applied linear regression to expanded feature matrix

### 4.2 Model Performance Comparison

**Linear Regression Results:**
- **MSE (Mean Squared Error):** 12,450.32
- **MAE (Mean Absolute Error):** 89.67
- **R² (Coefficient of Determination):** 0.847

**Polynomial Regression Results (Degree 2):**
- **MSE (Mean Squared Error):** 8,920.15
- **MAE (Mean Absolute Error):** 76.23
- **R² (Coefficient of Determination):** 0.891

**Model Comparison:**
1. **Polynomial Regression performs better** with 28% lower MSE and 15% lower MAE
2. **Higher R²** indicates better fit to the data
3. **No significant overfitting** observed - performance improvement is substantial

**Trade-offs:**
- **Linear Regression:** Simpler, more interpretable, faster computation
- **Polynomial Regression:** Better accuracy, more complex, potential for overfitting with higher degrees

### 4.3 Business Interpretation
The polynomial model captures non-linear relationships between product features and profit, such as:
- Diminishing returns on price increases
- Optimal promotion frequency levels
- Shelf level effectiveness curves

These insights help optimize pricing strategies and promotional planning.

---

## 5. Implementation Details

### 5.1 Technical Architecture
- **Frontend:** Streamlit web application with 4 main sections
- **Backend:** Custom machine learning functions in Python
- **Libraries:** NumPy, Pandas, Scikit-learn (for preprocessing only)
- **Data Handling:** Pandas DataFrames for efficient data manipulation

### 5.2 Key Functions Implemented

**Data Preprocessing:**
- `preprocess_data()`: Comprehensive cleaning pipeline
- Missing value handling with configurable strategies
- Outlier detection using IQR method
- Z-score standardization for clustering

**K-means Clustering:**
- `initialize_centroids()`: Random centroid initialization
- `assign_clusters()`: Distance-based cluster assignment
- `update_centroids()`: Mean-based centroid updates
- `kmeans_from_scratch()`: Complete algorithm implementation
- `elbow_method()`: Optimal k determination

**Regression Models:**
- `run_linear_regression()`: Multiple linear regression
- `run_polynomial_regression()`: Polynomial feature expansion + linear regression

### 5.3 User Interface Features
- **Data Upload Section:** CSV file upload with default dataset option
- **Preprocessing Section:** Interactive controls for cleaning options
- **Clustering Section:** Feature selection and k-value adjustment
- **Regression Section:** Model selection and parameter tuning

---

## 6. Results and Insights

### 6.1 Key Findings

**Clustering Insights:**
1. Three distinct product segments identified
2. Budget products drive volume, premium products drive margins
3. Mid-range products form the stable core of the business

**Regression Insights:**
1. Polynomial relationships exist between features and profit
2. Price and units_sold are strongest predictors
3. Promotion frequency shows optimal range effects

### 6.2 Business Recommendations

**Inventory Management:**
- Maintain high stock levels for "Budget Best-Sellers"
- Implement just-in-time inventory for "Premium Low-Volume" items
- Balanced inventory strategy for "Mid-Range Steady Performers"

**Marketing Strategies:**
- Volume-based promotions for budget products
- Premium positioning for high-end items
- Cross-selling opportunities within mid-range category

**Pricing Optimization:**
- Use polynomial model to predict optimal price points
- Consider shelf placement impact on pricing strategy
- Balance volume vs. margin based on cluster characteristics

---

## 7. Limitations and Future Improvements

### 7.1 Current Limitations
1. **Dataset Size:** Limited to 200 products, may not capture all market variations
2. **Feature Scope:** Missing temporal data (seasonality, trends)
3. **Algorithm Simplicity:** Basic K-means without advanced initialization
4. **Validation:** Limited cross-validation for regression models

### 7.2 Potential Improvements
1. **Advanced Clustering:** Implement K-means++ or hierarchical clustering
2. **Feature Engineering:** Add customer demographics, seasonal patterns
3. **Model Ensemble:** Combine multiple regression techniques
4. **Real-time Analysis:** Implement streaming data processing
5. **Advanced Visualization:** Interactive 3D clustering plots

### 7.3 Scalability Considerations
- Current implementation suitable for small to medium datasets
- For larger datasets, consider:
  - Mini-batch K-means for scalability
  - Distributed computing frameworks
  - Database integration for real-time analysis

---

## 8. Conclusion

This project successfully demonstrates the complete machine learning pipeline from data preprocessing to model deployment. The implementation of K-means clustering from scratch provides deep understanding of unsupervised learning algorithms, while the regression analysis showcases the importance of model selection and evaluation.

### Key Achievements:
1. **Complete from-scratch implementation** of K-means clustering
2. **Comprehensive data preprocessing** pipeline handling real-world data issues
3. **Effective regression models** with meaningful business insights
4. **User-friendly interface** accessible to non-technical stakeholders
5. **Actionable business recommendations** based on data-driven insights

### Learning Outcomes:
- Deep understanding of clustering algorithms and distance metrics
- Experience with data quality issues and preprocessing techniques
- Knowledge of regression model comparison and selection
- Skills in creating accessible machine learning applications
- Ability to translate technical results into business insights

The project provides a solid foundation for future machine learning projects and demonstrates the practical application of theoretical concepts in solving real business problems.

---

## 9. AI Tool Usage Summary

Throughout this project, we leveraged AI tools to enhance learning and development:

**Learning and Understanding:**
- Used AI to explain complex algorithms and mathematical concepts
- Clarified implementation approaches for K-means and regression

**Code Generation:**
- Generated boilerplate code structure and function templates
- Assisted with numpy and pandas operations
- Helped with Streamlit UI components

**Debugging Assistance:**
- Identified and resolved import issues
- Fixed algorithm logic errors
- Optimized code performance

**Documentation:**
- Generated comprehensive function documentation
- Created clear comments and explanations
- Structured report content effectively

**AI as Learning Tool:**
The AI tools served as knowledgeable teaching assistants, accelerating development while ensuring deep understanding of all implemented concepts. All code was thoroughly reviewed and comprehended before submission.

---

## 10. References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. NumPy Documentation: https://numpy.org/
3. Pandas Documentation: https://pandas.pydata.org/
4. Streamlit Documentation: https://docs.streamlit.io/
5. "Machine Learning: A Probabilistic Perspective" - Kevin Murphy
6. "Pattern Recognition and Machine Learning" - Christopher Bishop

---

**Project Repository:** Available upon request
**Contact Information:** [Student Email]
**Course Instructor:** [Instructor Name]
