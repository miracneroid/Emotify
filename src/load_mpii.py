import scipy.io
import os

# Determine dataset path relative to current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'mpii_human_pose_v1_u12_1.mat')

if not os.path.exists(mat_file_path):
    print(f"Warning: MPII dataset file not found at {mat_file_path}")
else:
    print(f"Loading MPII dataset from: {mat_file_path}")
    mpii_data = scipy.io.loadmat(mat_file_path)

    # Extract 'RELEASE' data
    release_data = mpii_data['RELEASE'][0, 0]

    # Extract 'annolist' and 'act' fields
    annolist = release_data['annolist']
    act_data = release_data['act']

    print(f"'annolist' found with shape: {annolist.shape}")
    print(f"'act' found with shape: {act_data.shape}")

    # Extract and clean valid actions
    valid_actions = set()
    for i in range(len(act_data)):
        try:
            action_entry = act_data[i][0]
            act_name = action_entry[1]
            if act_name.size > 0:
                valid_actions.add(str(act_name[0]))
        except Exception:
            pass

    valid_actions = sorted(valid_actions)
    output_txt = os.path.join(script_dir, "actions.txt")
    with open(output_txt, "w") as f:
        for action in valid_actions:
            f.write(action + "\n")

    print(f"\nTotal unique actions found: {len(valid_actions)}")

    output_json = os.path.join(script_dir, "mpii_actions_summary.json")
    import json
    action_records = [{"id": idx, "action": act} for idx, act in enumerate(valid_actions)]
    with open(output_json, "w") as f:
        json.dump(action_records[:500], f, indent=2)
    print(f"Exported summary to '{output_json}'.")
