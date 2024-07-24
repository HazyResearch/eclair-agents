"""Runs end-to-end pipeline for automating a task from a screen recording"""
import os

path_to_data_folder: str = '../../data/hospital_data/' # path to the folder containing the raw demonstration data
path_to_demo_folder: str = '../../data/hospital_demo/' # path to the folder containing the finished ECLAIR execution
path_to_sop: str = os.path.join(path_to_data_folder, 'manual_sitter.txt')
task_descrip: str = 'Place a sitter order in Epic as a nurse'

is_record_new_demo: bool = False

# Step 0: Record the nurse's demonstration 
# NOTE: This has already been done for you, so this is skipped by default)
if is_record_new_demo:
    # Step 0a: Record demonstration
    os.system(f"""
        python record/record.py --path_to_output_dir {path_to_data_folder} \
                            --is_desktop_only \
                            --valid_application_name "Citrix Viewer" \
                            --name sitter
    """)

    # Step 0b: Postprocess demonstration files
    os.system(f"""
        python record/postprocess.py "{path_to_data_folder}" \
                            --task_descrip "{task_descrip}" \
                            --buffer_seconds 0 \
                            --valid_application_name "Citrix Viewer"
    """)

# Step 1: Generate SOP from demonstration
os.system(f"""
    python demonstrate/gen_sop.py {path_to_data_folder} \
                        --task_descrip "{task_descrip}" \
                        --is_act \
                        --is_pairwise \
                        --is_crop_to_action
""")

# Step 2: Run ECLAIR on Epic
os.system(f"""
    python execute/main.py {path_to_data_folder} \
                    --task "{task_descrip}" \
                    --path_to_sop "{path_to_sop}" \
                    --max_calls 50 \
                    --env_type desktop \
                    --executor uniagent \
                    --path_to_sop "{path_to_sop}"
""")

# Step 3: Validate that the workflow was successfully completed
os.system(f"""
    python validate/check_task_completion.py {path_to_demo_folder} \
                                            --is_include_sop \
                                            --is_td \
                                            --is_kf \
                                            --model GPT4 \
                                            --demo_name sitter
""")