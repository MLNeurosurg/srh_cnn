#!/usr/bin/env python3

'''
Simple script to delete files from directory based on string matching
'''
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Simple script to delete files from directory based on string matching")
parser.add_argument("-src", "--source_directory", type=str, help="Source directory to be searched.")
parser.add_argument("-str", "--string_match", type=str, help="String to find in files to be deleted.")
args = parser.parse_args()

source_dir = args.source_directory
string_match = args.string_match

source_dir_len = len(os.listdir(source_dir))

os.chdir(source_dir)
file_list = os.listdir(source_dir)

def compare_dir_size():
	source_post_move = len(os.listdir(source_dir))
	print("Source directory diff: " + str(source_post_move - source_dir_len))

def main():

	found_files = [file for file in file_list if string_match in file]
	usr_input = input("You are about to permanently delete " + str(len(found_files)) + ". Do you want to proceeed? (y/n)")

	if usr_input == "y":
		for file in found_files:
			os.remove(file)
			print(file)
	else: 
		sys.exit(1)

if __name__ == '__main__':
	main()
	compare_dir_size()
