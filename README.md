All credits goes to:
https://core-electronics.com.au/guides/raspberry-pi/face-recognition-with-raspberry-pi-and-opencv/

Step by step process on how to train the model

	1. image_capture.py
	• edit PERSON_NAME with the student name then save
	• run the program and take as many image as you want (ideally > 5)
	• hit "Q" when done to exit the program
	• the new dataset for the new student is under the dataset folder
	
	2. model_training.py
	• run this to train the model with what we have inside the dataset folder
	• it should create a file named encodings.pickle
	• the pickle file is the trained model for that particular dataset

	3. facial_recognition_hardware.py
	• main program
	• any logic we want to add in the system (this is where we will mostly add them)

READ BEFORE RUNNING THE PROGRAM

Make sure to setup Thonny under the virtual environment for face_rec whenever you run the program.
If you are running the code under terminal, make sure you are in the virtual environment.

FOLDER/FILE CHECKLIST
All of these must be under attendance-Log_faceRecognition folder

	[Python files]
	image_capture.py
	model_training.py
	facial_recognition.py
	facial_recognition_hardware.py <--- what we edit

	[Folders]
	dataset <--- holds the images that came from image_capture.py, also used to generate encodings.pickle
	licenses

	[Google API to connect to google drive] 
	token.pickle <--- contact me if this is missing. Necessary to connect to Google Drive that host all the attendance log
	credentials.json <--- API from google cloud to connect Google's API and sheet, contact Marc if missing.

	[CSV files that holds student list]
	csv files are the list of students that we use to generate the spreadsheet. First line is the folder address.
	We only have classA.csv for now. If we need another set of student list, create another .csv file IE: classB.csv
	and edit line 42 under facial_recognition_hardware.py
