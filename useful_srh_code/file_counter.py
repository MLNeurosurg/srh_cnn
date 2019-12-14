

import os
import argparse

parser = argparse.ArgumentParser(description="Find files within a directory using string matching")
parser.add_argument("-dir", "--directory", type=str, help="Directory to be searched.")
parser.add_argument("-str", "--string_match", type=str, help="String to search in each file name.")
parser.add_argument("-nchar", "--number_characters", type=int, help="Number of characters to include to list unique files.")
args = parser.parse_args()

def main():
	file_list = []
	for root, dirs, files in os.walk(args.directory):
		for file in files:
			if (("tif" in file) or ("png" in file)) and (args.string_match in file):
				file_list.append(file[0:args.number_characters])

	sorted_list = sorted(list(set(file_list)))
	for file in sorted_list:
		print(file)

	print(len(sorted(list(set(file_list)))))

if __name__ == '__main__':
	main()
