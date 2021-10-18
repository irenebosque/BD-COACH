#Use the extended slicing syntax list[start:stop:step] to get a new list containing every nth element. Leave start and stop empty, and set step to the desired n.

a_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

first_part = a_list[0:5:1]
second_part = a_list[5::2]

complete = first_part + second_part
print(complete)