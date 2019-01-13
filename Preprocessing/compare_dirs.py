

import os

def diff(list1, list2):
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    return list(c - d)

def remove_ext(filelist):
	files = []
	for item in filelist:
		files.append(item[0:-4])
	return files

filelist1 = remove_ext(sorted(os.listdir("/home/todd/Desktop/CNN_Images/inv_training_tiles/greymatter")))
filelist2 = remove_ext(sorted(os.listdir("/home/todd/Desktop/CNN_Images/nio_training_tiles/greymatter")))

for file in filelist1

print("New files: " + str(len(filelist1)))
print("Old files: " + str(len(filelist2)))

if filelist1 == filelist2:
	print("Have all files!")

diff_lists = diff(filelist1, filelist2)
trimmed = []
for file in diff_lists:
	trimmed.append(file[0:3])

else:
	# print(diff(filelist1, filelist2))
	print(sorted(set(trimmed)))
	print(len(diff(filelist1, filelist2)))
