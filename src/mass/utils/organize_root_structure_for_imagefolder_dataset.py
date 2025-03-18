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
    print(f"directory: {directory}")

    folders = list_folders(directory)
    try:
        folders.remove("outliers")
        folders.remove("misc")
    except ValueError:
        pass
    folders = sorted(folders)

    print(f"folders: {folders}")

    for folder in folders:
        letter_directory = os.path.join(directory, folder)

        print(f"letter_directory: {letter_directory}")

        classes_directories = list_folders(letter_directory)

        print(f"classes_directories: {classes_directories}")

        for class_name in classes_directories:
            src = os.path.join(letter_directory, class_name)
            dst = src.replace(f"/{folder}/", "/")

            print(f"src: {src}")

            subdirectories = list_folders(src)

            if len(subdirectories) > 0:
                print(f"Subdirectories found in {src}: {subdirectories}")

                for subdir in subdirectories:
                    # print(subdir)

                    subdir_src = os.path.join(src, subdir)

                    for file in list_files(subdir_src):
                        try:
                            shutil.move(file, src)
                            # time.sleep(1)
                            print(f"{subdir_src} -> {src}")
                        except shutil.Error as err:
                            print(f"Error moving folder: {err}")

                    try:
                        os.rmdir(subdir_src)
                        print(f"{subdir_src} directory removed successfully!")
                    except OSError as e:
                        print(f"[73] Error removing directory: {e}")

            try:
                shutil.move(src, dst)
                # time.sleep(1)
                print(f"{src} -> {dst}")
            except shutil.Error as err:
                print(f"Error moving folder: {err}")

        try:
            os.rmdir(letter_directory)
            print(f"{letter_directory} directory removed successfully!")
        except OSError as e:
            print(f"[87] Error removing directory: {e}")


if __name__ == "__main__":
    DATASET_NAME = "SUN397"
    directory = f"./data/{DATASET_NAME}"
    move_classes_to_upper_folder(directory)
