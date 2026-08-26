#Will remove any data points which exist from x < 0.5 cm

import os
import sys
import pandas as pd
import shutil

file_path = sys.argv[1]
#apparatus_name = str(sys.argv[2])
#print(apparatus_name)

absolute_file_path = sys.argv[1]
absolute_file_dir = os.path.dirname(absolute_file_path)

dataframe = pd.read_csv(file_path)
#print(dataframe)
dataframe = dataframe.drop(dataframe[dataframe.x_cm < 0.5000].index) 

#Optionally will remove anything with confidence lower than 0.04000
dataframe = dataframe.drop(dataframe[dataframe.confidence < 0.04000].index)

dataframe.to_csv('predictions.cleaned.csv', index=False)

cleaned_dataframe = pd.read_csv("predictions.cleaned.csv")
#print(cleaned_dataframe)

#Averaged data sets based on photo name
dataframe2 = cleaned_dataframe.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
print(dataframe2)
dataframe2.to_csv("average_temp_photo.csv", index=False)

n_lane1 = []
n_lane2 = []
n_lane3 = []
n_lane4 = []
n_lane5 = []
n_lane6 = []

#Will bin and create csv files based on lanes
#Hansolo

# apparatus_name == "hansolo" == "chuba" == "r2d2" == "anakin" == "obiwan":
    
lane1_df = cleaned_dataframe[(cleaned_dataframe['y_cm'] < 3.0)]
n_lane1 = len(lane1_df)
lane1_df.to_csv("Lane_1_predictions.csv", index=False)
print(n_lane1)

lane2_df = cleaned_dataframe.loc[(cleaned_dataframe['y_cm'] < 5.5) & (cleaned_dataframe['y_cm'] > 3.0)]
n_lane2 = len(lane2_df)
lane2_df.to_csv("Lane_2_predictions.csv", index=False)

lane3_df = cleaned_dataframe.loc[(cleaned_dataframe['y_cm'] < 8.1) & (cleaned_dataframe['y_cm'] > 5.5)]
n_lane3 = len(lane3_df)
lane3_df.to_csv("Lane_3_predictions.csv", index=False)

lane4_df = cleaned_dataframe.loc[(cleaned_dataframe['y_cm'] < 10.6) &(cleaned_dataframe['y_cm'] > 8.1)]
n_lane4 = len(lane4_df)
lane4_df.to_csv("Lane_4_predictions.csv", index=False)

lane5_df = cleaned_dataframe.loc[(cleaned_dataframe['y_cm'] < 13.1) & (cleaned_dataframe['y_cm'] > 10.6)]
n_lane5 = len(lane5_df)
lane5_df.to_csv("Lane_5_predictions.csv", index=False)

lane6_df = cleaned_dataframe.loc[(cleaned_dataframe['y_cm'] < 15.6) & (cleaned_dataframe['y_cm'] > 13.1)]
n_lane6 = len(lane6_df)
lane6_df.to_csv("Lane_6_predictions.csv", index=False)


#Averaged data sets based on photo name for each separate lane along with max and min
dataframe2 = cleaned_dataframe.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)


#Lane 1
dataframeL1 = pd.read_csv("Lane_1_predictions.csv")
dataframeATL1 = dataframeL1.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
dataframeATL1.to_csv("average_temp_photo_lane1.csv", index=False)

#Lane 2
dataframeL2 = pd.read_csv("Lane_2_predictions.csv")
dataframeATL2 = dataframeL2.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
dataframeATL2.to_csv("average_temp_photo_lane2.csv", index=False)

#Lane 3
dataframeL3 = pd.read_csv("Lane_3_predictions.csv")
dataframeATL3 = dataframeL3.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
dataframeATL3.to_csv("average_temp_photo_lane3.csv", index=False)

#Lane 4
dataframeL4 = pd.read_csv("Lane_4_predictions.csv")
dataframeATL4 = dataframeL4.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
dataframeATL4.to_csv("average_temp_photo_lane4.csv", index=False)

#Lane 5
dataframeL5 = pd.read_csv("Lane_5_predictions.csv")
dataframeATL5 = dataframeL5.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
dataframeATL5.to_csv("average_temp_photo_lane5.csv", index=False)

#Lane 6
dataframeL6 = pd.read_csv("Lane_6_predictions.csv")
dataframeATL6 = dataframeL6.groupby(['path'], as_index=False)['temperature'].mean().rename({'temperature': 'AVGTEMP'}, axis=1)
dataframeATL6.to_csv("average_temp_photo_lane6.csv", index=False)



#Moves all Files into the Parent Directory
shutil.move("Lane_1_predictions.csv", absolute_file_dir)
shutil.move("Lane_2_predictions.csv", absolute_file_dir)
shutil.move("Lane_3_predictions.csv", absolute_file_dir)
shutil.move("Lane_4_predictions.csv", absolute_file_dir)
shutil.move("Lane_5_predictions.csv", absolute_file_dir)
shutil.move("Lane_6_predictions.csv", absolute_file_dir)
shutil.move("average_temp_photo_lane1.csv", absolute_file_dir)
shutil.move("average_temp_photo_lane2.csv", absolute_file_dir)
shutil.move("average_temp_photo_lane3.csv", absolute_file_dir)
shutil.move("average_temp_photo_lane4.csv", absolute_file_dir)
shutil.move("average_temp_photo_lane5.csv", absolute_file_dir)
shutil.move("average_temp_photo_lane6.csv", absolute_file_dir)
shutil.move("average_temp_photo.csv", absolute_file_dir)
shutil.move("predictions.cleaned.csv", absolute_file_dir)



