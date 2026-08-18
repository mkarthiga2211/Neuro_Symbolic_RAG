import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def save_table_as_image(csv_path: str, output_image_path: str):
    """
    Reads the final results CSV and renders it as a professional academic table image.
    """
    print(f"📄 Reading result table from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean up column names for display if needed
    df.columns = [c.replace('_', ' ') for c in df.columns]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        cellLoc='center', 
        loc='center',
        colColours=['#f2f2f2'] * len(df.columns)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 2.2) # Scale for better readability

    # Bold the headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.get_text().set_weight('bold')
            cell.set_facecolor('#d9ead3') # Light green for header

    plt.title('Table 1: Comparative Analysis of Demand Forecasting Models', 
              fontsize=16, fontweight='bold', pad=20)

    # Save the figure
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison table saved as image: {output_image_path}")

if __name__ == "__main__":
    csv_file = "results/final_results_table.csv"
    output_img = "results/figures/final_results_table.png"
    
    # Ensure fonts and style are high quality for publication
    save_table_as_image(csv_file, output_img)
