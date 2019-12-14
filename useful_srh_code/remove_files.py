
'''
Delete files from directory based on string matching

'''
import os

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
# "whitematter": []
# }

source_dir = "/home/todd/Desktop/CNN_Images/nio_training_tiles/whitematter"
file_to_find = "040-0032_1"

source_dir_len = len(os.listdir(source_dir))

os.chdir(source_dir)
file_list = os.listdir(source_dir)
found_files = []

def compare_dir_size():
	source_post_move = len(os.listdir(source_dir))
	print("Source directory diff: " + str(source_post_move - source_dir_len))

def main():
	accum = 0
	for file in file_list:
		if (file_to_find in file):
			found_files.append(file)
			os.remove(file)
			print(file)
			accum += 1

	print("Total files deleted: " + str(accum))

if __name__ == '__main__':
	main()
	compare_dir_size()
