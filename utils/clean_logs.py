# delete logs that are empty
import os


def is_log_file_empty(log_file):
    with open(log_file, "r") as f:
        return not f.read().strip()  # True if empty or only whitespace

def delete_empty_logs(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith(".log"):
            file_path = os.path.join(dir_path, filename)
            if is_log_file_empty(file_path):
                os.remove(file_path)
                print(f"Deleted empty log file: {file_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(base_dir, "../logs")
    delete_empty_logs(dir_path)