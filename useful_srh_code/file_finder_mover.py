#!/usr/bin/env python3

'''
Script to quickly find of files in a source directory with or without string matching, and move to destination directory 
'''
import os
import sys
import shutil
import argparse

parser = argparse.ArgumentParser(description="Find files within a source directory using string matching and move to destination directory")
parser.add_argument("-src", "--source_directory", type=str, help="Source directory to be searched.")
parser.add_argument("-dest", "--destination_directory", type=int, help="Destination directory for files.")
parser.add_argument("-str", "--string_match", type=str, help="String to search in each file name.")
parser.add_argument("-move_or_copy", "--move_or_copy",  nargs='?', type=str, const="copy", help="Argument to specify if the files are to be moved ('move') or copied ('copy'). Defaults to copy to avoid losing files.")
args = parser.parse_args()

source_dir = args.source_directory
dest_dir = args.destination_directory
string_match = args.string_match
move_or_copy = args.move_or_copy

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
		if ("tif" in file or "png" in file) and (string_match in file):
			found_files.append(file)

	accum = 0
	for file in found_files:
		if args.move_or_copy == "move":
			shutil.move(source_dir + "/" + file, dst = dest_dir + "/" + file)
			accum += 1
			print(file)
		
		elif args.move_or_copy == "copy":
			shutil.copy(source_dir + "/" + file, dst = dest_dir + "/" + file)
			accum += 1
			print(file)

		else:
			"Incorrect "
		
	print("Total files moved: " + str(accum))


if __name__ == '__main__':

	usr_input = input("You around about " + move_or_copy.upper() + " the selected files. Is this correct (y/yes): ")
	
	if usr_input == "y" or usr_input == "yes":
		main()
		compare_dir_size()
	else:
		sys.exit(1)
