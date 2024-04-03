import pandas
import imageio
import glob

# Initialize an empty DataFrame to store the final dataset
final_dataset = pandas.DataFrame()

# Iterate through each digit (from 0 to 9)
for i in range(10):
  print(i)
  result = []

  # Iterate through each image file in the directory corresponding to the current digit
  for file_path in glob.glob("dataset_raw/" + str(i) + "/*.jpg"):
    # Read the image using imageio and flatten it into a 1D array
    imimage = imageio.v2.imread(file_path)
    imimage = imimage.flatten()
    result.append(imimage)
    
  # Convert the list of flattened images into a DataFrame
  result_dataframe = pandas.DataFrame(result)
  # Add a prefix to the column names to indicate pixel values
  result_dataframe = result_dataframe.add_prefix('pixel_')
  # Add a column 'digit' to specify the digit label for each image
  result_dataframe['digit'] = i

  # Concatenate the current digit's DataFrame with the final dataset
  final_dataset = pandas.concat([result_dataframe, final_dataset], axis=0)
 
# Save the final dataset to a CSV file 
final_dataset.to_csv("dataset.csv", index=False)

# file_path = "dataset_raw/0/img_1.jpg"
# imimage = imageio.v2.imread(file_path)
# imimage = imimage.flatten()

# print(imimage)

#This code reads image files from directories corresponding to each digit (0 to 9) 
#in a "dataset_raw" directory. It flattens each image into a 1D array, 
#creates a DataFrame containing these pixel values, 
#adds a prefix to the column names indicating pixel values, 
#and adds a column 'digit' to specify the digit label for each image. 
#Finally, it concatenates all digit DataFrames into one final dataset 
#and saves it to a CSV file named "dataset.csv".