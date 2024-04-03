import pandas
import imageio
import glob
import imutils

# Initialize an empty DataFrame to store the final dataset
final_dataset = pandas.DataFrame()

# Iterate through each digit (from 0 to 9)
for i in range(10):
	print(i)
	result = []
	result_extended = [] # Initialize a list to store extended images

	# Iterate through each image file in the directory corresponding to the current digit
	for file_path in glob.glob("dataset_raw/" + str(i) + "/*.jpg"):
		# Read the image using imageio
		imimage = imageio.v2.imread(file_path)

		# Rotate the image by positive and negative angles and resize it
		imimage_rotate_positive_5 = imutils.rotate(imimage, angle = 5)
		imimage_rotate_negative_5 = imutils.rotate(imimage, angle = -5)
		imimage_enlarge = imutils.resize(imimage, width = 30)

		# Crop the resized image to 28x28 pixels
		imimage_enlarge = imimage_enlarge[1:29, 1:29]


		# Flatten the images into 1D arrays
		imimage = imimage.flatten()
		imimage_rotate_positive_5 = imimage_rotate_positive_5.flatten()
		imimage_rotate_negative_5 = imimage_rotate_negative_5.flatten()
		imimage_enlarge = imimage_enlarge.flatten()

		# Append the original and extended images to the result list
		result.append(imimage)
		result_extended.append(imimage_rotate_positive_5)
		result_extended.append(imimage_rotate_negative_5)
		result_extended.append(imimage_enlarge)

	# Convert the list of flattened images into a DataFrame for original images
	result_dataframe = pandas.DataFrame(result)
	result_dataframe = result_dataframe.add_prefix('pixel_')
	result_dataframe['digit'] = i
	result_dataframe['extended'] = 0 # Add a column to indicate original images

	# Convert the list of flattened images into a DataFrame for extended images
	result_extended_dataframe = pandas.DataFrame(result_extended)
	result_extended_dataframe = result_extended_dataframe.add_prefix('pixel_')
	result_extended_dataframe['digit'] = i
	result_extended_dataframe['extended'] = 1 # Add a column to indicate extended images 

	# Concatenate the current digit's DataFrame with the final dataset
	final_dataset = pandas.concat([result_dataframe, final_dataset], axis=0)

# Save the final dataset to a CSV file
final_dataset.to_csv("dataset.csv", index=False)

#This code reads image files from directories corresponding to each digit (0 to 9) 
#in a "dataset_raw" directory. For each image, it performs several transformations 
#(rotation by positive and negative angles, resizing, and cropping) to generate 
#augmented images. It then flattens all images into 1D arrays and appends them 
#to the result list. After processing all images for a given digit, 
#it converts the lists of flattened images into DataFrames and adds columns to 
#indicate whether the images are original or extended. Finally, it concatenates 
#all digit DataFrames into one final dataset and saves it to a CSV file named "dataset.csv".

















