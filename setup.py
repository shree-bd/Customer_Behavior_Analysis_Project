"""
Setup script for Customer Behavior Analysis Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="customer-behavior-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced customer behavior analysis and purchase prediction using big data and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Customer_Behavior_Analysis_Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "advanced": [
            "optuna>=2.10.0",
            "shap>=0.40.0",
            "mlflow>=1.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "customer-analysis=src.main:main",
        ],
    },
    keywords="machine-learning, customer-behavior, e-commerce, big-data, spark, xgboost",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Customer_Behavior_Analysis_Project/issues",
        "Source": "https://github.com/yourusername/Customer_Behavior_Analysis_Project",
        "Documentation": "https://github.com/yourusername/Customer_Behavior_Analysis_Project/wiki",
    },
) 