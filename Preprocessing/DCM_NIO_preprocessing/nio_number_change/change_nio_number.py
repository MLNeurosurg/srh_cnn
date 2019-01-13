

import pydicom as dicom
import os
import argparse

parser = argparse.ArgumentParser(description='Change the accesssion number in .DCM files.')
parser.add_argument("-nionum", metavar='', type=str, required=True, help="NIO number to assign to files.")
parser.add_argument("-dir", metavar='', type=str, required=True, help="File directory with .dcm files.")
args = parser.parse_args()

def main(file_list):
	for file in file_list:
		dcm_file = dicom.read_file(args.dir + "/" + file)
		dcm_file.AccessionNumber = args.nionum
		dicom.write_file(file, dcm_file, write_like_original=True)

if __name__ == '__main__':
	
	file_list = os.listdir(args.dir)
	main(file_list)