import os

def extract_scores_from_txt(directory, output_file, label):
    with open(output_file, 'a') as outfile:
        for txt_file in os.listdir(directory):
            if txt_file.endswith(".txt"):
                file_path = os.path.join(directory, txt_file)
                with open(file_path, 'r') as f:
                    for line in f:
                        data = line.strip().split()
                        if len(data) == 6:  # Ensure confidence score is present
                            score = float(data[5])
                            outfile.write(f"{label},{score}\n")  # Write label (regular/split) and score

# Define directories
regular_labels_dir = r"C:\Users\GenCyber\Documents\Yolov8\runs\detect\val42\labels"
split_labels_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New\output_labels"
output_file = "detection_scores.csv"

# Write headers
with open(output_file, 'w') as f:
    f.write("label,score\n")

# Extract scores
extract_scores_from_txt(regular_labels_dir, output_file, "regular")
extract_scores_from_txt(split_labels_dir, output_file, "split")

print(f"Scores extracted and saved in {output_file}")
