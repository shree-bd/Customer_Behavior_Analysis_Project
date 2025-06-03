# 🛒 Customer Behavior Analysis & Purchase Prediction

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.1.2-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Overview

This project analyzes customer behavior patterns in e-commerce environments and predicts whether customers will complete purchases for items they've added to their shopping carts. Using big data processing with Apache Spark and machine learning with XGBoost, we achieve high-accuracy predictions that can help businesses optimize their conversion strategies.

## 🚀 Key Features

- **Big Data Processing**: Handles 67M+ customer interaction records using Apache Spark
- **Behavioral Analysis**: Comprehensive analysis of customer shopping patterns
- **Purchase Prediction**: ML model to predict cart-to-purchase conversion
- **Feature Engineering**: Advanced feature extraction from temporal and categorical data
- **Data Visualization**: Interactive charts showing customer behavior insights
- **Scalable Architecture**: Designed to handle enterprise-scale datasets

## 🛠️ Technologies Used

### Data Processing
- **Apache Spark (PySpark)** - Distributed data processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Machine Learning
- **XGBoost** - Gradient boosting classifier
- **scikit-learn** - Data preprocessing and model evaluation
- **Feature Engineering** - Custom pipeline for data transformation

### Visualization
- **Matplotlib** - Statistical visualizations
- **Seaborn** - Advanced plotting (potential upgrade)

## 📊 Dataset

- **Source**: [Kaggle - eCommerce Behavior Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- **Size**: 67,501,979 records
- **Time Period**: November 2019
- **Features**: Event time, event type, product details, user information, pricing

### Data Schema
```
├── event_time: Timestamp of the event
├── event_type: view, cart, purchase
├── product_id: Unique product identifier
├── category_id: Category identifier
├── category_code: Category hierarchy
├── brand: Product brand
├── price: Product price in USD
├── user_id: Unique user identifier
└── user_session: Session identifier
```

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.9+
Java 8+ (for Spark)
```

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Customer_Behavior_Analysis_Project.git
cd Customer_Behavior_Analysis_Project

# Install required packages
pip install pyspark pandas numpy scikit-learn xgboost matplotlib jupyter

# For conda users
conda install -c conda-forge pyspark xgboost
```

### Data Preparation
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
2. Place the CSV file in your project directory
3. Update the file path in the notebook

## 📈 Usage

### Running the Analysis
```bash
# Start Jupyter notebook
jupyter notebook "Project Code.ipynb"

# Or run with Python
python -m jupyter notebook
```

### Key Analysis Steps

1. **Data Loading & Exploration**
   ```python
   # Initialize Spark session
   spark = SparkSession.builder.master("local").appName("Market Analysis").getOrCreate()
   
   # Load dataset
   df_market = spark.read.option('header','true').csv("path/to/data.csv", inferSchema=True)
   ```

2. **Feature Engineering**
   - Extract temporal features (weekday, week number)
   - Create category hierarchies
   - Calculate user activity metrics
   - Engineer purchase probability features

3. **Model Training**
   ```python
   # Train XGBoost classifier
   model = XGBClassifier(learning_rate=0.1)
   model.fit(X_train, y_train)
   ```

## 📊 Results & Insights

### Customer Behavior Distribution
- **Views**: 94.1% of all interactions
- **Cart Additions**: 4.9% of interactions
- **Purchases**: 1.6% of interactions

### Model Performance
- **Accuracy**: ~85-90% (varies with feature selection)
- **Precision**: High precision in identifying likely purchasers
- **Recall**: Balanced recall for both purchase and non-purchase cases

### Key Insights
1. **Conversion Funnel**: Only ~32% of cart additions result in purchases
2. **Temporal Patterns**: Clear weekly and daily patterns in purchase behavior
3. **Brand Influence**: Significant impact of brand on purchase probability
4. **Price Sensitivity**: Complex relationship between price and purchase likelihood

## 🔮 Future Enhancements

### Advanced Analytics
- [ ] **Real-time Streaming**: Implement Spark Streaming for live predictions
- [ ] **Deep Learning**: Add LSTM models for sequential behavior analysis
- [ ] **Recommendation Engine**: Build collaborative filtering system
- [ ] **Customer Segmentation**: Advanced clustering analysis

### Model Improvements
- [ ] **Ensemble Methods**: Combine multiple algorithms
- [ ] **Hyperparameter Tuning**: Automated optimization with Optuna
- [ ] **Feature Selection**: Advanced feature importance analysis
- [ ] **Cross-validation**: Implement time-series aware validation

### Infrastructure
- [ ] **Cloud Deployment**: AWS/Azure deployment with auto-scaling
- [ ] **API Development**: REST API for real-time predictions
- [ ] **Dashboard**: Interactive web dashboard with Dash/Streamlit
- [ ] **MLOps Pipeline**: Complete CI/CD for model deployment

### Data Engineering
- [ ] **Data Pipeline**: Automated ETL with Apache Airflow
- [ ] **Data Quality**: Comprehensive data validation framework
- [ ] **Feature Store**: Centralized feature management
- [ ] **A/B Testing**: Framework for model performance testing

## 📁 Project Structure

```
Customer_Behavior_Analysis_Project/
│
├── Project Code.ipynb          # Main analysis notebook
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── data/                      # Data directory (create locally)
│   └── 2019-Nov.csv          # Dataset file
├── models/                    # Saved models (auto-generated)
├── visualizations/            # Generated plots and charts
└── src/                      # Source code modules (future)
    ├── data_processing.py
    ├── feature_engineering.py
    ├── model_training.py
    └── evaluation.py
```

## 📧 Contact & Contribution

- **Issues**: Please report bugs or request features via GitHub Issues
- **Contributions**: Pull requests are welcome! Please read our contributing guidelines
- **Discussions**: Join our community discussions for questions and ideas

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [REES46 Marketing Platform](https://rees46.com/)
- Kaggle community for hosting the dataset
- Apache Spark and XGBoost development teams
- Open source community contributors

---

⭐ **Star this repository if you find it helpful!** ⭐
