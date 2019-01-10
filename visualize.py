import os
from data_logging import load_log
import json
import matplotlib

def plot_logs(logs, save=False, path=None)
    pass

def main():

    print("This script will import log files, visualize them, and save generated plots")
    print("Ensure that all logs are located in the './logs' folder")
    print("The logs can be in subfolders inside the logs folder, like './logs/fldA/fldB'")
    print("If this is the case, when entering the folder name, prepend all subfolders")
    print("In other words, provide relative path from inside the logs folder")

    path = input("Enter file or folder name, or relative path from logs folder: \n")

    do_save = True if (input("\nWould you like to save generated plots (y,n): \n") == 'y') else False

    if !do_save:
        print("\nPlots will not be saved")

    log_obj = load_log(path, "Training_log")

    for index, log in enumerate(log_obj):
        print(f"plotting log number {index+1} of {len(log_obj)}")
        plot_logs(log_obj, save=do_save, path=path)

    print("DONE PLOTTING")

if __name__== "__main__":
    main()
