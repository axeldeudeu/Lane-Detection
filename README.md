# Lane-Detection
Lane Detection for Self Driving Cars

## ENPM 673 Perception for Autonomous Robots - Project 2

## Lane Detection
<p align="center">
<img src="data/test_video.gif"/>
</p>
### Steps to run
	with images: end folder path containing images needs to be given
	pyhton3 lane.py -dataset 1 -path /home/siddharth/Downloads/673/Project2/data/data_1/data/

	Challenge video: Please ensure the video file is present in the current working directory or provide the complete file path
	python3 lane.py -dataset 2
	python3 lane.py -dataset 2 -path /home/siddharth/Downloads/673/Project2/data/data_2/challenge_video.mp4
	(output vides will be rendered in the present working directory)

### Parameters:
	-dataset: 1: For the images ; 2: for challenge video
	-path: path of input folder/file

### Outputs from run:
<p align="center">
<img src="data/LaneDetection_DataSet_1.gif"/>
</p>
<p align="center">
<img src="data/Lane_Detection_video_challenge.gif"/>
</p>


## Enhancing the quality of video from night drive using Histogram equalization method
<p align="center">
<img src="data/Night_Drive_Gamma_Correction.gif"/>
</p>
### Steps to run
	Please ensure that the video file is in the present working directory or provide the path argument having complete video file path
	python3 problem1.py -method  histogram
	python3 problem1.py -method  histogram -path /home/siddharth/Downloads/673/Project2/Night\ Drive\ -\ 2689.mp4
	python3 problem1.py -method  gamma
        (please note that the execution using the histogram method will take consume CPU, time and memory)

### Parameters:
	-method: method to be used - either histogram or gamma
	-path: path of input file

### Google drive link to output videos: 
  https://drive.google.com/drive/folders/1ivJRFKzYbeRKER1eEn-jbTcZUcTrM7-g?usp=sharing
