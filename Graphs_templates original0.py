# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Import necessary libraries
    import matplotlib.pyplot as plt
    import numpy as np

    # Dummy data for demonstration
    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']
    methods = ['Method 1', 'Method 2', 'Method 3']
    psnr_values = np.array([
        [32.5, 34.0, 33.5],  # PSNR values for Dataset 1
        [33.0, 34.5, 34.0],  # PSNR values for Dataset 2
        [31.5, 33.5, 32.0],  # PSNR values for Dataset 3
        [34.0, 35.0, 34.5]  # PSNR values for Dataset 4
    ])
    psnr_std = np.array([
        [0.5, 0.4, 0.3],  # Standard deviations for Dataset 1
        [0.6, 0.5, 0.4],  # Standard deviations for Dataset 2
        [0.4, 0.6, 0.5],  # Standard deviations for Dataset 3
        [0.3, 0.4, 0.5]  # Standard deviations for Dataset 4
    ])

    # Settings for bar chart
    x = np.arange(len(datasets))  # X positions for datasets
    bar_width = 0.2  # Width of each bar
    colors = ['steelblue', 'darkorange', 'forestgreen']  # Colors for different methods

    # Create the bar chart
    plt.figure(figsize=(12, 7))
    for i, method in enumerate(methods):
        plt.bar(x + i * bar_width, psnr_values[:, i], bar_width, yerr=psnr_std[:, i],
                capsize=5, label=method, color=colors[i], edgecolor='black')

    # Add labels, title, and legend
    plt.xlabel('Datasets', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.title('Comparison of PSNR Across Datasets and Methods', fontsize=18, fontweight='bold')
    plt.xticks(x + bar_width, datasets, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(30, 36.5)  # Adjust y-axis limits for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Methods", fontsize=12, title_fontsize=14)

    # Enhance layout
    plt.tight_layout()
    plt.show()


