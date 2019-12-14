'''
Script to find a specific mosaic and move to desired directory

'''
import os
import shutil

# NIO_numbers_test = {
# "ependymoma": [],
# "glioblastoma": [],
# "greymatter": [],
# "lowgradeglioma": [],
# "lymphoma": [],
# "medulloblastoma": [],
# "meningioma": [],
# "metastasis": [],
# "nondiagnostic": [],
# "pilocyticastrocytoma": [],
# "pituitaryadenoma": [],
# "pseudoprogression": [],
# "schwannoma": [], 
# "whitematter": []
# }


source_dir = "/home/todd/Desktop/SRH_genetics/srh_patches/patches/training_patches/training/IDHmut"
dest_dir = "/home/todd/Desktop/SRH_genetics/srh_patches/patches/training_patches/validation/IDHmut"
file_to_move = "NIO472"

source_dir_len = len(os.listdir(source_dir))
dest_dir_len = len(os.listdir(dest_dir))

os.chdir(source_dir)
file_list = os.listdir(source_dir)
found_files = []

def compare_dir_size():
	source_post_move = len(os.listdir(source_dir))
	dest_dir_move = len(os.listdir(dest_dir))

	print("Source directory diff: " + str(source_post_move - source_dir_len))
	print("Dest. directory diff: " + str(dest_dir_move - dest_dir_len))
	if (dest_dir_move - dest_dir_len) == -(source_post_move - source_dir_len):
		print("Successful move!")

	else:
		print("Something went wrong! Please review directories.")

def main():
	for file in file_list:
		if ("tif" in file or "png" in file) and (file_to_move in file):
			found_files.append(file)

	accum = 0
	for file in found_files:
		shutil.move(source_dir + "/" + file, dst = dest_dir + "/" + file)
		accum += 1
		print(file)

	print("Total files moved: " + str(accum))


if __name__ == '__main__':
	main()
	compare_dir_size()
