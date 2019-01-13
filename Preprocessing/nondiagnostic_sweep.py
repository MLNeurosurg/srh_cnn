'''
Script to check nondiagnostic category for incorrect nondiagnostic labelling

'''

import os 

NIO_dict = {}
file_list = []
for roots, dirs, files in os.walk("/home/orringer-lab/Desktop/Training_Images/cnn_tiles_validation/nondiagnostic"):
	for file in files:
		# if "NIO058" in file:
		file_list.append(file[0:11])

# Find unique NIO cases
unique_set = sorted(list(set(file_list)))
for NIO_case in unique_set:
	if NIO_case not in NIO_dict.keys():
		NIO_dict[NIO_case] = 0

# Increment dictionary with number of files for each NIO case
for NIO_case, val in NIO_dict.items():
	for file in file_list:
		if NIO_case in file:
			NIO_dict[NIO_case] += 1

ordered_list = []
for key, val in NIO_dict.items():
	ordered_list.append((key, val))

ordered_list = sorted(ordered_list, key = lambda x: (int(x[1])))
for item in ordered_list:
	print(item)

