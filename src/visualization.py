import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pie_chart(data_array, title, labels=['Class 0', 'Class 1']):
    """Vẽ biểu đồ tròn từ mảng NumPy"""
    unique, counts = np.unique(data_array, return_counts=True)
    stats = dict(zip(unique, counts))
    values = [stats.get(0, 0), stats.get(1, 0)]
    
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_bar_chart(x_labels, y_values, title, xlabel, ylabel):
    """Vẽ biểu đồ cột"""
    plt.figure(figsize=(10, 5))
    sns.barplot(x=x_labels, y=y_values, palette='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()

def plot_histograms(data_0, data_1, feature_name):
    """Vẽ histogram so sánh 2 lớp"""
    plt.figure(figsize=(10, 5))
    plt.hist(data_0, bins=30, alpha=0.5, label='Not Change (0)', color='blue', density=True)
    plt.hist(data_1, bins=30, alpha=0.5, label='Change (1)', color='red', density=True)
    plt.title(f'Distribution of {feature_name} by Target')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_learning_curve_and_confusion_matrix(loss_history, cm, save_path=None):
    """Vẽ Learning Curve và Confusion Matrix chung 1 hình và lưu ảnh"""
    plt.figure(figsize=(14, 6))

    # 1. Learning Curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Learning Curve (Loss over Iterations)')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)

    # 2. Confusion Matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Change', 'Change'], 
                yticklabels=['Not Change', 'Change'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    
    # Lưu ảnh nếu có đường dẫn
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Đã lưu ảnh kết quả tại: {save_path}")
        
    plt.show()

def plot_scatter(x_data, y_data, target_data, xlabel, ylabel, title):
    """Vẽ biểu đồ phân tán (Scatter Plot)"""
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, c=target_data, cmap='coolwarm', alpha=0.6, edgecolors='w')
    plt.colorbar(label='Target (0: Stay, 1: Leave)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()