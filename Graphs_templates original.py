# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # Improved bar chart with error bars and formatting adjustments
    # Dummy data for demonstration
    psnr_std = [0.5, 0.4, 0.6, 0.3]  # Example standard deviations for error bars
    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']
    methods = ['Method 1', 'Method 2', 'Method 3']

    psnr_values = np.array([
        [32.5, 34.0, 33.5],  # PSNR values for Dataset 1
        [33.0, 34.5, 34.0],  # PSNR values for Dataset 2
        [31.5, 33.5, 32.0],  # PSNR values for Dataset 3
        [34.0, 35.0, 34.5]   # PSNR values for Dataset 4
    ])

    # Create the improved bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.6  # Adjusted bar width
    bars = plt.bar(methods, [np.average(i) for i in psnr_values], yerr=psnr_std, capsize=5, color='steelblue', edgecolor='black', width=bar_width)

    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom', fontsize=12)

    # Update labels, title, and grid with increased font sizes
    plt.xlabel('Methods', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.title('Comparison of PSNR Across Methods', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(30, 36.5)  # Adjust y-axis limits for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Enhance layout
    plt.tight_layout()
    plt.show()


# # Dummy data for demonstration
#
#
# psnr_std = np.array([
#     [0.5, 0.4, 0.3],  # Standard deviations for Dataset 1
#     [0.6, 0.5, 0.4],  # Standard deviations for Dataset 2
#     [0.4, 0.6, 0.5],  # Standard deviations for Dataset 3
#     [0.3, 0.4, 0.5]   # Standard deviations for Dataset 4
# ])
#
# # Settings for bar chart
# x = np.arange(len(datasets))  # X positions for datasets
# bar_width = 0.2  # Width of each bar
# colors = ['steelblue', 'darkorange', 'forestgreen']  # Colors for different methods
#
# # Create the bar chart
# plt.figure(figsize=(12, 7))
# for i, method in enumerate(methods):
#     plt.bar(x + i * bar_width, psnr_values[:, i], bar_width, yerr=psnr_std[:, i],
#             capsize=5, label=method, color=colors[i], edgecolor='black')
#
# # Add labels, title, and legend
# plt.xlabel('Datasets', fontsize=16)
# plt.ylabel('PSNR (dB)', fontsize=16)
# plt.title('Comparison of PSNR Across Datasets and Methods', fontsize=18, fontweight='bold')
# plt.xticks(x + bar_width, datasets, fontsize=14)
# plt.yticks(fontsize=14)
# plt.ylim(30, 36.5)  # Adjust y-axis limits for better visibility
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.legend(title="Methods", fontsize=12, title_fontsize=14)
#
# # Enhance layout
# plt.tight_layout()
# plt.show()
#
#
# # Create line plot with larger markers
# plt.figure(figsize=(12, 7))
# colors = ['steelblue', 'darkorange', 'forestgreen']
# markers = ['o', 's', '^']
#
# for i, method in enumerate(methods):
#     plt.errorbar(datasets, psnr_values[:, i], yerr=psnr_std[:, i], label=method,
#                  color=colors[i], marker=markers[i], markersize=10, linestyle='-', linewidth=2, capsize=5)
#
# # Add labels, title, and legend
# plt.xlabel('Datasets', fontsize=16)
# plt.ylabel('PSNR (dB)', fontsize=16)
# plt.title('Comparison of PSNR Across Datasets and Methods', fontsize=18, fontweight='bold')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.ylim(30, 36.5)  # Adjust y-axis limits for better visibility
# plt.grid(linestyle='--', alpha=0.7)
# plt.legend(title="Methods", fontsize=12, title_fontsize=14, loc='lower right')
#
# # Enhance layout
# plt.tight_layout()
# plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate dummy data
# frame_indices = np.arange(1, 1001)  # Frame indices from 1 to 1000
# psnr_method_1 = 35 + np.sin(frame_indices * 0.01) + np.random.normal(0, 0.2, len(frame_indices))
# psnr_method_2 = 34.5 + np.cos(frame_indices * 0.01) + np.random.normal(0, 0.2, len(frame_indices))
#
# # Create line plot
# plt.figure(figsize=(14, 7))
# plt.plot(frame_indices, psnr_method_1, label='Method 1', color='steelblue', linewidth=2)
# plt.plot(frame_indices, psnr_method_2, label='Method 2', color='darkorange', linewidth=2)
#
# # Add labels, title, and legend
# plt.xlabel('Frame Index', fontsize=16)
# plt.ylabel('PSNR (dB)', fontsize=16)
# plt.title('PSNR Over Video Frames for Two Methods', fontsize=18, fontweight='bold')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(linestyle='--', alpha=0.7)
# plt.legend(title="Methods", fontsize=12, title_fontsize=14, loc='lower right')
#
# # Enhance layout
# plt.tight_layout()
# plt.show()


