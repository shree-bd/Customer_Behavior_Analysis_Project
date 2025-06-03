"""
Data Processing Module for Customer Behavior Analysis

This module contains functions for loading, cleaning, and preprocessing
the e-commerce behavior dataset.
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main class for data processing operations"""
    
    def __init__(self, app_name="CustomerBehaviorAnalysis"):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .master("local[*]") \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        logger.info(f"Spark session initialized: {app_name}")
    
    def load_data(self, file_path, file_format="csv"):
        """
        Load data from file into Spark DataFrame
        
        Args:
            file_path (str): Path to the data file
            file_format (str): Format of the file (csv, parquet, json)
            
        Returns:
            pyspark.sql.DataFrame: Loaded data
        """
        try:
            if file_format.lower() == "csv":
                df = self.spark.read.option('header', 'true') \
                    .csv(file_path, inferSchema=True)
            elif file_format.lower() == "parquet":
                df = self.spark.read.parquet(file_path)
            elif file_format.lower() == "json":
                df = self.spark.read.json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Data loaded successfully. Shape: ({df.count()}, {len(df.columns)})")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values and data quality issues
        
        Args:
            df (pyspark.sql.DataFrame): Input dataframe
            
        Returns:
            pyspark.sql.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        # Remove rows with null values in critical columns
        df_clean = df.filter(
            col("user_id").isNotNull() & 
            col("product_id").isNotNull() & 
            col("event_type").isNotNull()
        )
        
        # Fill missing brand values with "unknown"
        df_clean = df_clean.fillna({"brand": "unknown"})
        
        # Fill missing category_code with "other"
        df_clean = df_clean.fillna({"category_code": "other"})
        
        # Remove outliers in price (negative values or extreme outliers)
        df_clean = df_clean.filter(
            (col("price") > 0) & (col("price") < 10000)
        )
        
        logger.info(f"Data cleaning completed. Rows remaining: {df_clean.count()}")
        return df_clean
    
    def feature_engineering(self, df):
        """
        Create new features for machine learning
        
        Args:
            df (pyspark.sql.DataFrame): Input dataframe
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Convert event_time to timestamp
        df_features = df.withColumn(
            "event_timestamp", 
            to_timestamp(col("event_time"), "yyyy-MM-dd HH:mm:ss")
        )
        
        # Extract temporal features
        df_features = df_features.withColumn("hour", hour("event_timestamp")) \
            .withColumn("day_of_week", dayofweek("event_timestamp")) \
            .withColumn("day_of_month", dayofmonth("event_timestamp")) \
            .withColumn("week_of_year", weekofyear("event_timestamp"))
        
        # Create category hierarchy features
        df_features = df_features.withColumn(
            "category_level1",
            split(col("category_code"), "\\.").getItem(0)
        ).withColumn(
            "category_level2",
            split(col("category_code"), "\\.").getItem(1)
        )
        
        # Create price bins
        df_features = df_features.withColumn(
            "price_range",
            when(col("price") < 10, "low")
            .when(col("price") < 100, "medium")
            .when(col("price") < 500, "high")
            .otherwise("premium")
        )
        
        # Calculate user activity metrics
        user_activity = df_features.groupBy("user_id") \
            .agg(
                count("*").alias("total_events"),
                countDistinct("product_id").alias("unique_products"),
                avg("price").alias("avg_price"),
                sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count")
            )
        
        # Join user activity back to main dataframe
        df_features = df_features.join(user_activity, "user_id", "left")
        
        logger.info("Feature engineering completed")
        return df_features
    
    def create_target_variable(self, df):
        """
        Create target variable for machine learning
        
        Args:
            df (pyspark.sql.DataFrame): Input dataframe
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with target variable
        """
        logger.info("Creating target variable...")
        
        # Get purchase events
        purchases = df.filter(col("event_type") == "purchase") \
            .select("user_id", "product_id", "user_session") \
            .distinct()
        
        # Get cart events
        cart_events = df.filter(col("event_type") == "cart")
        
        # Create target: 1 if purchased after cart, 0 otherwise
        df_target = cart_events.join(
            purchases,
            ["user_id", "product_id", "user_session"],
            "left"
        ).withColumn(
            "is_purchased",
            when(purchases.user_id.isNotNull(), 1).otherwise(0)
        )
        
        logger.info("Target variable created")
        return df_target
    
    def get_data_summary(self, df):
        """
        Get summary statistics of the dataset
        
        Args:
            df (pyspark.sql.DataFrame): Input dataframe
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            "total_records": df.count(),
            "total_columns": len(df.columns),
            "event_type_distribution": df.groupBy("event_type").count().collect(),
            "unique_users": df.select("user_id").distinct().count(),
            "unique_products": df.select("product_id").distinct().count(),
            "date_range": {
                "min": df.agg(min("event_time")).collect()[0][0],
                "max": df.agg(max("event_time")).collect()[0][0]
            }
        }
        
        return summary
    
    def save_data(self, df, output_path, format="parquet", mode="overwrite"):
        """
        Save DataFrame to file
        
        Args:
            df (pyspark.sql.DataFrame): DataFrame to save
            output_path (str): Output file path
            format (str): Output format (parquet, csv, json)
            mode (str): Write mode (overwrite, append)
        """
        try:
            if format.lower() == "parquet":
                df.write.mode(mode).parquet(output_path)
            elif format.lower() == "csv":
                df.write.mode(mode).option("header", "true").csv(output_path)
            elif format.lower() == "json":
                df.write.mode(mode).json(output_path)
            
            logger.info(f"Data saved to {output_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def close_session(self):
        """Close Spark session"""
        self.spark.stop()
        logger.info("Spark session closed")


# Utility functions
def validate_data_quality(df):
    """
    Validate data quality and return quality metrics
    
    Args:
        df (pyspark.sql.DataFrame): Input dataframe
        
    Returns:
        dict: Data quality metrics
    """
    total_rows = df.count()
    
    quality_metrics = {}
    
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        null_percentage = (null_count / total_rows) * 100
        
        quality_metrics[column] = {
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2)
        }
    
    return quality_metrics


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Load data
    df = processor.load_data("data/2019-Nov.csv")
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Feature engineering
    df_features = processor.feature_engineering(df_clean)
    
    # Create target variable
    df_target = processor.create_target_variable(df_features)
    
    # Get summary
    summary = processor.get_data_summary(df_target)
    print("Data Summary:", summary)
    
    # Close session
    processor.close_session() 