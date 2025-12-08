import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

def download_slide_data(slide_name, tech, output_dir):
    """
    Downloads gene expression, coordinates, and image data for a specific slide
    from the STimage-1K4M dataset and saves them to a target directory.

    Args:
        slide_name (str): The unique identifier of the slide (e.g., '151673' or 'GSM4284316').
        tech (str): The technology used ('Visium', 'ST', or 'VisiumHD').
        output_dir (str): The local path where files should be saved.
    """
    
    # Define the repository ID
    repo_id = "jiawennnn/STimage-1K4M"
    repo_type = "dataset"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing slide: {slide_name} | Technology: {tech}")

    # List of file types to download with their specific folder paths and likely extensions
    # Note: Filename patterns might vary slightly based on the specific dataset upload structure.
    # The patterns below are based on the common structure of STimage-1K4M.
    files_to_download = [
        {
            "type": "Gene Expression",
            "repo_path": f"{tech}/gene_exp/{slide_name}_count.csv",
            "save_name": f"gene_exp/{slide_name}_count.csv"
        },
        {
            "type": "Coordinates",
            # Often coordinate files have a '_coord' suffix or similar. 
            # If this fails, check if the dataset uses just '{slide_name}.csv' in the coord folder.
            "repo_path": f"{tech}/coord/{slide_name}_coord.csv", 
            "save_name": f"coord/{slide_name}_coord.csv"
        },
        {
            "type": "Image",
            "repo_path": f"{tech}/image/{slide_name}.png", # Defaulting to .jpg, might be .tif or .png
            "save_name": f"image/{slide_name}_image.png"
        }
    ]

    # Flag to prevent listing files multiple times if multiple downloads fail
    has_listed_slide = False

    for file_info in files_to_download:
        try:
            print(f"Downloading {file_info['type']}...")
            
            # Download file to local cache
            cached_path = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=file_info['repo_path']
            )
            
            # Copy file from cache to the specified output directory
            destination_path = os.path.join(output_dir, file_info['save_name'])
            shutil.copy(cached_path, destination_path)
            
            has_listed_slide = True
            print(f"Successfully saved to: {destination_path}")

        except Exception as e:
            print(f"Error downloading {file_info['type']} ({file_info['repo_path']}): {e}")
            
            # Only list available files once, and only if the error suggests the file wasn't found
            if not has_listed_slide:
                print(f"\n--- ID '{slide_name}' not found. Listing available files in '{tech}/{file_info['type']}' ---")
                try:
                    # Get the directory prefix (e.g., 'Visium/gene_exp')
                    dir_prefix = os.path.dirname(file_info['repo_path'])
                    
                    # Fetch all files from the repo
                    all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
                    
                    # Filter files that match the directory prefix
                    available_files = [f for f in all_files if f.startswith(dir_prefix)]
                    
                    if available_files:
                        print(f"Found {len(available_files)} files. Showing first 20:")
                        for f in available_files[:20]:
                            print(f" - {f}")
                        if len(available_files) > 20:
                            print(f"... and {len(available_files) - 20} more.")
                    else:
                        print("No files found in this directory. Check if the 'tech' name is correct.")
                        
                except Exception as list_err:
                    print(f"Could not list files: {list_err}")
                
                print("------------------------------------------------------------------\n")
                

    print(f"\n--- Download process for {slide_name} completed ---")

# --- Example Usage ---
if __name__ == "__main__":
    # Example parameters (Replace these with the actual IDs from the paper)
    my_slide = "Human_Breast_Andersson_10142021_ST_A1" 
    my_tech = "ST"       
    my_output_folder = r"C:/mnt/data/SpaGFT-Research/data"

    download_slide_data(slide_name=my_slide, tech=my_tech, output_dir=my_output_folder)
