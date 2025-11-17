import os

def get_filenames_in_folder(folder_path):
    """
    Retrieves a list of filenames within a specified folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list containing the names of files in the folder.
              Returns an empty list if the folder does not exist or is empty.
    """
    try:
        filenames = os.listdir(folder_path)
        filenames = [f for f in filenames if f.startswith("walk") and os.path.isfile(os.path.join(folder_path, f))]
        return filenames
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
if __name__ == "__main__":
    folder_path = "LAFAN1_Retargeting_Dataset/g1/"
    filenames = get_filenames_in_folder(folder_path)
    print(f"Filenames in folder '{folder_path}':")
    for filename in filenames:  
        print(filename)

    for j in range(len(filenames)):
        motions = filenames[:j+1]
        print(motions)