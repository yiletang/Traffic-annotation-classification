import pandas as pd
import matplotlib.pyplot as plt


def visualize_data_distribution(train_data_path, val_data_path):
    # Load the data from the CSV files
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    # Calculate the distribution of categories in both sets
    train_distribution = train_data['category'].value_counts(normalize=True).sort_index()
    val_distribution = val_data['category'].value_counts(normalize=True).sort_index()

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.bar(train_distribution.index - 0.2, train_distribution.values, width=0.4, label='Train')
    plt.bar(val_distribution.index + 0.2, val_distribution.values, width=0.4, label='Validation')
    plt.xlabel('Category')
    plt.ylabel('Proportion')
    plt.title('Category Distribution in Train and Validation Sets')
    plt.legend()
    plt.xticks(ticks=range(len(train_distribution)), labels=train_distribution.index, rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_data =r'C:\Users\H2250\Desktop\Traffic annotation classification\data\train_data.csv'
    val_data = r'C:\Users\H2250\Desktop\Traffic annotation classification\data\val_data.csv'
    visualize_data_distribution(train_data, val_data)
