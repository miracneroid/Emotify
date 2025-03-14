import scipy.io

# Load MPII dataset
mpii_data = scipy.io.loadmat('/Users/miracneroid/Developer/Projects/Emotify/src/mpii_human_pose_v1_u12_1.mat')

# Extract 'RELEASE' data
release_data = mpii_data['RELEASE'][0, 0]

# Extract 'annolist' and 'act' fields
annolist = release_data['annolist']
act_data = release_data['act']

print(f"'annolist' found with shape: {annolist.shape}")
print(f"'act' found with shape: {act_data.shape}")

# Extract and clean valid actions
valid_actions = set()  # Use a set to avoid duplicates
for i in range(len(act_data)):
    try:
        action_entry = act_data[i][0]
        act_name = action_entry[1]  # Extract action name
        
        if act_name.size > 0:
            valid_actions.add(str(act_name[0]))  # Convert np.str_ to Python string

    except Exception:
        pass  # Ignore errors

# Convert set to sorted list
valid_actions = sorted(valid_actions)

# Save cleaned actions to a file
with open("actions.txt", "w") as f:
    for action in valid_actions:
        f.write(action + "\n")

# Print summary
print(f"\nTotal unique actions found: {len(valid_actions)}")
print("Sample unique actions:", valid_actions[:10])  # Show first 10
