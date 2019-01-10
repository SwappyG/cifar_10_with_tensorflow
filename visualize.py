import os
from data_logging import load_log
import json

def main():

    path = input("Enter the full path of the log file or folder: \n")

    log_obj = load_log(path, "Training_log")

    for index, log in enumerate(log_obj):
        plot_logs(log_obj, save=False)



if __name__== "__main__":
    main()
