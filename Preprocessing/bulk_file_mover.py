'''
Will dump all images back into specificed directory

MUST USE ONLY for full 13 category mover. 

'''

import os
import shutil  
import sys

source_dir = "/home/orringer-lab/Desktop/Training_Images/tester"
dest_dir = "/media/orringer-lab/UNTITLED/nio_training_images"

source_dict = {
"ependymoma": [],
"glioblastoma": [],
"greymatter": [],
"lowgradeglioma": [],
"lymphoma": [],
"medulloblastoma": [],
"meningioma": [],
"metastasis": [],
"nondiagnostic": [],
"pilocyticastrocytoma": [],
"pituitaryadenoma": [],
"pseudoprogression": [],
"whitematter": []
}

def main():
	for root, dirs, files in os.walk(source_dir):
		for file in files:
			for key, val in source_dict.items():
				if key in root:
					if ("tif" in file) and ("NIO" in file): 
					# if ("png" in file): 
						source_dict[key].append(root + "/" + file)


	for tumor, files in source_dict.items():
		for file in files:
			shutil.copy(file, dst = dest_dir + "/" + tumor)

if __name__ == '__main__':
	main()