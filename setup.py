"""
Setup script for PurchaseIQ - Intelligence-Driven Purchase Prediction Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="purchaseiq",
    version="1.0.0",
    author="Shree BD",
    author_email="your.email@example.com",
    description="Enterprise-grade ML platform for real-time purchase prediction and customer behavior analytics using big data and advanced algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shree-bd/PurchaseIQ",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Point-Of-Sale",
        "Framework :: Apache Spark",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "pre-commit>=2.15.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
            "streamlit>=1.0.0",
        ],
        "advanced": [
            "optuna>=2.10.0",
            "shap>=0.40.0",
            "mlflow>=1.20.0",
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
        ],
        "cloud": [
            "boto3>=1.20.0",
            "azure-storage-blob>=12.9.0",
            "google-cloud-storage>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "purchaseiq=src.main:main",
            "purchaseiq-train=src.model_training:main",
            "purchaseiq-predict=src.prediction:main",
        ],
    },
    keywords="machine-learning, customer-behavior, e-commerce, big-data, spark, xgboost, purchase-prediction, conversion-analytics, real-time-ml, enterprise-ai",
    project_urls={
        "Bug Reports": "https://github.com/shree-bd/PurchaseIQ/issues",
        "Source": "https://github.com/shree-bd/PurchaseIQ",
        "Documentation": "https://github.com/shree-bd/PurchaseIQ/wiki",
        "Roadmap": "https://github.com/shree-bd/PurchaseIQ/blob/main/ENHANCEMENT_ROADMAP.md",
    },
) 