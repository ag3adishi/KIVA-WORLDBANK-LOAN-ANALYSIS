# scripts/visualization.py

import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_categories(data, column, top_n=10, save_as=None):
    """
    Plot top N categories in a column.
    """
    plt.figure(figsize=(10, 6))
    order = data[column].value_counts().head(top_n).index
    sns.countplot(data=data, y=column, order=order)
    plt.title(f"Top {top_n} Categories in {column}")
    plt.tight_layout()
    
    if save_as:
        plt.savefig(f'../outputs/{save_as}')
    plt.show()

def plot_box_by_category(data, numeric_col, category_col, log_scale=False, save_as=None):
    """
    Boxplot of numeric_col by category_col.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x=numeric_col, y=category_col)
    
    if log_scale:
        plt.xscale('log')
    
    plt.title(f'{numeric_col} by {category_col}')
    plt.tight_layout()
    
    if save_as:
        plt.savefig(f'../outputs/{save_as}')
    plt.show()
