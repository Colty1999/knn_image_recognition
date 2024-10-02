import os
from datetime import datetime
from matplotlib import pyplot as plt

def plot_accuracy(history_df):
    # Create the plots folder if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Generate the plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot with current datetime in filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"plots/plot-{current_time}.png"
    plt.savefig(plot_filename)
    
    # Show the plot
    plt.show()

    print(f"Plot saved as {plot_filename}")
