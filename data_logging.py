import os
import json
from pathlib import Path
from cifar_classes import Training_log

def check_overwrite(file_name):
    # If the file already exists, confirm with user that they want to overwrite
    if os.path.isfile(file_name):
        user_input = input("File already exists, are you sure you want to overwrite it? (y,n) : ")
        if user_input == 'y' or user_input == 'Y':
            do_write = True
    else:
        do_write = True

    return do_write

def save_logs(logs_list):

    # Get a filename from user to save to
    input_name = input("\nEnter a file name to log the data, or leave blank to discard it: \n")
    print()
    if (input_name == ""):
        print("Skipped logging, no records will be kept\n")

    # If the name isn't blank
    else:
        val_acc_list = []

        folder_name = "logs/" + input_name + "/"
        for num, log in enumerate(logs_list):
            # Place the file in the logs folder, with .txt extension
            file_name = folder_name + input_name + "_num_" + str(num) + ".txt"
            do_write = check_overwrite(file_name)

            # If file doesn't exist, or user ok'ed overwriting, write the log data
            if do_write:
                log_dict = vars(log)
                val_acc_list.append(log_dict["final_val_acc"])
                Path(folder_name).mkdir(parents=True, exist_ok=True)
                with open(file_name, 'w') as log_file:
                    json.dump(log_dict, log_file, indent=4)
                    print(f"Log written to {file_name} as json")



        if len(logs_list) > 1:
            summary_dict = vars(logs_list[0])
            summary_dict.pop("loss_log")
            summary_dict.pop("val_log")
            summary_dict.pop("rep_ID")
            summary_dict["final_val_acc"] = val_acc_list
            summary_dict["avg_val_acc"] = sum(val_acc_list) / float(len(val_acc_list))

            file_name = folder_name + input_name + "_summary.txt"
            do_write = check_overwrite(file_name)

            # If file doesn't exist, or user ok'ed overwriting, write the log data
            if do_write:
                with open(file_name, 'w') as log_file:
                    json.dump(summary_dict, log_file, indent=4)
                    print(f"Summary written to {file_name} as json")


        print(f"All logs written to folder {folder_name}")

def load_log(path, return_type):

    # If the supplied path is a folder, parse through all files in folder
    # This does NOT go through sub folders, only files in the root of folder
    if os.path.isdir(path):

        # Create a list to hold imported data from each file
        logs = []

        # For every file in directory
        for file in os.listdir(path):

            # Open the file as a read only
            with open(file, 'r') as fl:

                # Load the file and parse it as a json
                this_log_dict = json.load(fl)

                # If the required output is a Training_log obj,
                # create and append it to the list
                if return_type == "Training_log":
                    logs.append(Training_log(this_log_dict))
                # Otherwise, just append the dictionary to the list
                else:
                    logs.append(this_log_dict)

                # Clear the var for our imported log to be reused
                this_log_dict = None

            # Return the full list of logs (obj or dict as specified)
            return logs

    # If the path is a file
    elif os.path.isfile(path):

        # open the file as a read only
        with open(path, 'r') as fl:

            # load the file and parse it as a json
            log_dict = json.load(fl)

            # If the required output is a Training_log obj,
            # create and return it
            if return_type == "Training_log":
                return Training_log(log_dict)
            # Otherwise, just return the dictionary
            else:
                return log_dict

    # if the path is neither a folder nor file, raise an error
    else:
        raise FileNotFoundError
