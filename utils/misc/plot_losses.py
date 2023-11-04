import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(csv_path):
    df = pd.read_csv(csv_path)

    filtered_df = df.dropna(subset=['epoch', 'train/loss_vlb_epoch'])

    global_step = filtered_df['epoch'].values
    train_loss_step = filtered_df['train/loss_vlb_epoch'].values

    # Create a plot if there is data to plot
    if len(global_step) > 0:
        plt.plot(global_step, train_loss_step, marker='o', linestyle='-')
        plt.xlabel('epoch')
        plt.ylabel('train/loss_vlb_epoch')
        plt.title('epoch vs. train/loss_vlb_epoch')
        plt.grid(True)
        plt.savefig("loss_epoch_trainLossVlbEpoch.png")
        
    else:
        print("No data to plot in the specified CSV file.")

    filtered_df = df.dropna(subset=['global_step', 'train/loss_simple_step'])

    global_step = filtered_df['global_step'].values
    train_loss_step = filtered_df['train/loss_simple_step'].values

    # Create a plot if there is data to plot
    if len(global_step) > 0:
        plt.clf()
        plt.plot(global_step, train_loss_step, marker='o', linestyle='-')
        plt.xlabel('global_step')
        plt.ylabel('train/loss_simple_step')
        plt.title('global_step vs. train/loss_simple_step')
        plt.grid(True)
        plt.savefig("loss_globalStep_trainLossSimpleStep.png")
    else:
        print("No data to plot in the specified CSV file.")

    filtered_df = df.dropna(subset=['global_step', 'train/loss_vlb_step'])

    global_step = filtered_df['global_step'].values
    train_loss_step = filtered_df['train/loss_vlb_step'].values

    # Create a plot if there is data to plot
    if len(global_step) > 0:
        plt.clf()
        plt.plot(global_step, train_loss_step, marker='o', linestyle='-')
        plt.xlabel('global_step')
        plt.ylabel('train/loss_vlb_step')
        plt.title('global_step vs. train/loss_vlb_step')
        plt.grid(True)
        plt.savefig("loss_globalStep_trainLossVlbStep.png")
    else:
        print("No data to plot in the specified CSV file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from a CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    args = parser.parse_args()

    plot_csv(args.csv_path)