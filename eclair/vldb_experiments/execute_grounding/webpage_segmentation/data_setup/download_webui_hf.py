from huggingface_hub import snapshot_download
import zipfile
import os
from tqdm import tqdm
import shutil
import gzip
from itertools import groupby
import argparse

def download_webui(version):

    def combine_zip_files(parts, output_file):
        with open(output_file, 'wb') as wfd:
            for part in parts:
                with open(part, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10)

    def extract_zip(zip_file_path, extract_to_path):
        if not os.path.exists(extract_to_path):
            os.makedirs(extract_to_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)

    def combine_all_zips_in_folder(folder_path):
        # List all zip file parts
        zip_parts = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if '.zip.' in file]

        # Group parts by their base filename
        grouped_parts = groupby(sorted(zip_parts), key=lambda x: x.rsplit('.', 2)[0])

        # Combine each group of parts into a single ZIP file
        for base_name, parts in grouped_parts:
            parts_list = list(parts)
            if len(parts_list) > 1:
                combined_zip_file = base_name + '.zip'
                combine_zip_files(parts_list, combined_zip_file)
            else:
                print(f"Single-part zip found: {parts_list[0]}")
        
        return combined_zip_file

    # Download train set
    snapshot_download(repo_id=f"biglab/webui-{version}", repo_type="dataset")

    # Download val set
    snapshot_download(repo_id="biglab/webui-val", repo_type="dataset")

    # Download test set
    snapshot_download(repo_id="biglab/webui-test", repo_type="dataset")

    train_webui_dir = os.path.join(os.getenv("HF_HOME"), f"hub/datasets--biglab--webui-{version}")
    val_webui_dir = os.path.join(os.getenv("HF_HOME"), "hub/datasets--biglab--webui-test")
    test_webui_dir = os.path.join(os.getenv("HF_HOME"), "hub/datasets--biglab--webui-val")

    all_webui_dirs = [train_webui_dir, val_webui_dir, test_webui_dir]

    # Make a new datapath for to move unzipped files to
    destination_dir = os.path.join(os.getenv("HF_HOME"), f"hub/webui-{version}")
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Go into each snapshot directory and combine zips in files to one zip, then unzip 
    for webui_dir in all_webui_dirs:
        snapshot_dir = os.path.join(webui_dir, "snapshots")

        # Identify the only subfolder inside the snapshot directory
        subfolders = [f.path for f in os.scandir(snapshot_dir) if f.is_dir()]
        if len(subfolders) == 1:
            snapshot_subfolder_path = subfolders[0]
            combined_zip_file = combine_all_zips_in_folder(snapshot_subfolder_path)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            extract_zip(combined_zip_file, destination_dir)
        else:
            print("Error: There should be only one subfolder inside 'snapshot'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download webui dataset from huggingface.')
    parser.add_argument('--version', type=str, help='Name of the experiment configuration file', choices=["7k","70k","350k"])

    args = parser.parse_args()

    path_to_output_dir: str = os.getenv("HF_HOME")
    print("HF_HOME=", path_to_output_dir)
    if path_to_output_dir is None:
        raise Exception("HF_HOME not set")

    download_webui(args.version)


