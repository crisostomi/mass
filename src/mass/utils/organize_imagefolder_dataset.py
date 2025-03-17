import os
import shutil


def list_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders


def list_files(directory):
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            files.append(item_path)
    return files


def move_classes_to_upper_folder(directory):
    print(directory)

    folders = list_folders(directory)
    folders.remove("outliers")
    folders.remove("misc")
    folders = sorted(folders)

    print(folders)

    for folder in folders:
        letter_directory = os.path.join(directory, folder)

        # print(letter_directory)

        classes_directories = list_folders(letter_directory)

        # print(classes_directories)

        for class_name in classes_directories:
            src = os.path.join(letter_directory, class_name)
            dst = src.replace(f"/{folder}/", "/")

            try:
                shutil.move(src, dst)
                print(f"{src} -> {dst}")
            except shutil.Error as err:
                print(f"Error moving folder: {err}")

        try:
            os.rmdir(letter_directory)
            print(f"{letter_directory} directory removed successfully!")
        except OSError as e:
            print(f"Error removing directory: {e}")


if __name__ == "__main__":
    DATASET_NAME = "SUN397"
    directory = f"./data/{DATASET_NAME}/train_test"
    train_dir = f"./data/{DATASET_NAME}/train"
    os.makedirs(train_dir, exist_ok=True)
    test_dir = f"./data/{DATASET_NAME}/test"
    os.makedirs(test_dir, exist_ok=True)

    folders = list_folders(directory)

    for folder in folders:
        train_folder = os.path.join(train_dir, folder)
        test_folder = os.path.join(test_dir, folder)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

    for folder in folders:
        class_directory = os.path.join(directory, folder)

        files = list_files(class_directory)

        tot_files = len(files)
        num_train_file = int(0.8 * tot_files)
        num_test_file = tot_files - num_train_file
        train_files = files[0:num_train_file]
        test_files = files[num_test_file:]

        for src in train_files:
            dst = src.replace("train_test", "train")
            print(src, dst)

            try:
                shutil.copy(src, dst)
                print(f"{src} -> {dst}")
            except shutil.Error as err:
                print(f"Error copying file: {err}")

        for src in test_files:
            dst = src.replace("train_test", "test")

            try:
                shutil.copy(src, dst)
                print(f"{src} -> {dst}")
            except shutil.Error as err:
                print(f"Error copying file: {err}")
