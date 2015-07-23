import csv
import numpy as np
from matplotlib import pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import pprint
import random as r
import itertools as it

row_num = 0

retina_radius = 10
image_width = 27 # It's 28x28, but 27 is the last index for the width
left_buffer = retina_radius / 2
right_buffer = left_buffer + (0 if retina_radius % 2 == 0 else 1) # If radius is 5, then right_buffer = 2, but we need 3

def print_num(num_row, n=28, active_char = "*", inactive_char = " "):
	counter = 1
	output_string = ''
	for x in num_row:
		output_string += active_char if int(x) > 0 else inactive_char
		if counter % n == 0:
			output_string += '\n'
		counter += 1


	print output_string

def generate_picture(num_encoding):
	num_matrix = np.array(num_encoding).reshape(28,28)
	plt.imshow(num_matrix)
	plt.gray()
	plt.show()

def spatioretinal_encoder(encoding, x, y):
	retina = []
	location = []
	location_matrix = np.zeros((28,28), dtype = np.int)
	num_matrix = np.array(encoding).reshape(28,28)

	for coordinate in it.product(range(x-left_buffer,x+right_buffer),range(y-left_buffer, y+right_buffer)):
		retina.append(num_matrix[coordinate[0]][coordinate[1]])
		location_matrix[coordinate[0]][coordinate[1]] = 1

	location = location_matrix.flatten().tolist()

	# print_num(retina, retina_radius)
	# print_num(location, inactive_char = "-")


	return retina + location

with open('train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row_num == 1:
        	first_num_line = ['1' if int(x) >= 1 else '0' for x in row]

        row_num = row_num + 1

    first_num_name = first_num_line.pop(0)
    # generate_picture(first_num_line)

print_num(first_num_line)
print first_num_line
generate_picture([float(x) for x in first_num_line])
saccades = [(r.randrange(left_buffer, image_width - right_buffer), r.randrange(left_buffer, image_width - right_buffer)) for x in range(10000)]

# with open('1_100.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',')
#     counter = 1
#     for saccade in saccades:
#     	if counter <= 100: 
#     		spamwriter.writerow(spatioretinal_encoder(first_num_line,saccade[0],saccade[1]))
#     		counter += 1






