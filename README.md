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
