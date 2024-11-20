import os
import shutil

# 소스와 대상 디렉토리 경로
source_dir = "/mnt/rv1/althome/escho/training_dataset/random_pdb_include_dockQ"
destination_dir = "/home2/escho/GNN_DOVE_PEPRANK/inf_results"

# 대상 디렉토리 하위 폴더 리스트 가져오기
destination_subfolders = {folder for folder in os.listdir(destination_dir) if os.path.isdir(os.path.join(destination_dir, folder))}

# 소스 디렉토리 하위 폴더를 순회
for folder_name in os.listdir(source_dir):
    source_subfolder = os.path.join(source_dir, folder_name)
    
    # 소스 하위 폴더인지 확인
    if os.path.isdir(source_subfolder) and folder_name in destination_subfolders:
        source_file = os.path.join(source_subfolder, "dockQ_results_sorted.csv")
        dest_folder = os.path.join(destination_dir, folder_name)
        dest_file = os.path.join(dest_folder, "dockQ_results_sorted.csv")
        
        # 소스 파일이 존재하면 복사
        if os.path.exists(source_file):
            shutil.copy(source_file, dest_file)
            print(f"Copied: {source_file} -> {dest_file}")
        else:
            print(f"File not found: {source_file}")
