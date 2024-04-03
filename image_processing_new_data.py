import pandas
import imageio
import glob
import imutils


# Initialize an empty list to store flattened images and a list to store filenames
result = []
filename_list = []

# Iterate through each image file in the directory "new_data"
for file_path in glob.glob("new_data/*.jpg"):
  # Extract the filename from the file path
  filename = file_path.split("/")[-1]
  filename_list.append(filename)

  # Read the image using imageio and flatten it into a 1D array
  imimage = imageio.v2.imread(file_path)
  imimage = imimage.flatten()

  # Append the flattened image to the result list
  result.append(imimage)

# Convert the list of flattened images into a DataFrame
new_data = pandas.DataFrame(result)

# Add a prefix to the column names to indicate pixel values
new_data = new_data.add_prefix('pixel_')
# Add a column 'filename' to specify the filename for each image
new_data['filename'] = filename_list

# Save the new data DataFrame to a CSV file
new_data.to_csv("new_data.csv", index=False)

#This code reads image files from the "new_data" directory, flattens each image into a 1D 
#array, and stores them in a list along with their corresponding filenames. 
#Then, it converts the list of flattened images into a DataFrame, 
#adds a prefix to the column names to indicate pixel values, and adds a column 'filename' 
#to specify the filename for each image. Finally, it saves the new data DataFrame to a 
#CSV file named "new_data.csv".





