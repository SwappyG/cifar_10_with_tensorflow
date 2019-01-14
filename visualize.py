import os
from pathlib import Path
from data_logging import load_log
import json
import matplotlib.pyplot as plt
import argparse

def plot_logs(logs, title):
    fig, axes = plt.subplots(nrows=2, ncols=1)

    fig.suptitle(title)

    for index, log in enumerate(logs):

        # Let the user know which rep is being plotted
        print(f"plotting log number {index+1} of {len(logs)}")
        print(f"{log.loss_log  == None}")
        # Plot the loss versus iteration
        axes[0].plot(log.loss_log, 'b', label='rep_'+str(index))
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Losses')
        axes[0].legend()


        # Create labels and mark locations for epoch number
        epoch_labels = list(range(log.epochs))
        epoch_xticks = [i * log.iterations for i in epoch_labels]

        # Create the secondary x-axis for epochs
        axes_0_2 = axes[0].twiny()
        axes_0_2.set_xticks(epoch_xticks)
        axes_0_2.set_xticklabels(epoch_labels)

        # Place the epoch axis somewhat below the iteration axis
        axes_0_2.xaxis.set_ticks_position('bottom')
        axes_0_2.xaxis.set_label_position('bottom')
        axes_0_2.spines['bottom'].set_position(('outward', 36))
        axes_0_2.set_xlabel('Epoch')
        axes_0_2.set_xlim(axes[0].get_xlim())

        # Move to subplot two for validation accuracy axis
        axes[1].plot(log.val_log, 'r', label='rep_'+str(index))
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Validation Accuracies')
        axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def main(**kwargs):

    print("\nThis script will import log files, visualize them, and save generated plots")
    print("Ensure that all logs are located in the './logs' folder")
    print("The logs can be in subfolders inside the logs folder, like './logs/fldA/fldB'")
    print("If this is the case, when entering the folder name, prepend all subfolders")
    print("In other words, provide relative path from inside the logs folder\n")

    # path = input("Enter file or folder name, or relative path from logs folder: \n")
    # path = "logs/" + path
    do_save = kwargs.get('save')
    do_display = kwargs.get('display')

    if not do_save:
        print("\nPlots will not be saved")

    figs = []

    for path in kwargs.get('path'):

        path = "logs/" + path
        print(f"Loading from path: {path}\n")

        # Load all the logs, and remove the summary log
        log_obj = load_log(path, "Training_log")
        summary_log = log_obj[-1]
        data_obj = log_obj[0:-1]
        print(f"{len(data_obj)} logs loaded\n")

        # Remove {path} from "{path}/{name}.txt" to get "{name}.txt"
        title = path.split('/')[-1]

        this_fig = plot_logs(data_obj, title)
        if do_save:

            # Create the path and file names
            save_path = "plots/" + title + "/"
            file_name = title + "_plot"

            # Make a folder in the plots folder if required
            Path(save_path).mkdir(parents=True, exist_ok=True)

            # Save the figure
            print(f"Saving to {save_path}\n")
            this_fig.savefig(save_path + file_name, dpi=600)

            figs.append(this_fig)
            this_fig = None

    print("\nDONE PLOTTING\n")

    if do_display:
        print("Close figure to exit...\n")
        plt.show()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-p', '--path', action='append', required=True)
    input_args = parser.parse_args()
    main(**vars(input_args))
