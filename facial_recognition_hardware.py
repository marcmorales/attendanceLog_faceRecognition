import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
import json
from datetime import datetime
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from apiclient import discovery
import datetime
from google.oauth2 import service_account
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle, os
import csv

# 1. Connect to Google Sheets
scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as f:
        creds = pickle.load(f)
else:
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scope)
    creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as f:
        pickle.dump(creds, f)
client = gspread.authorize(creds)
destFolderId = ''
# List of names that will trigger the GPIO pin
authorized_names = []
ids = []
with open('information.csv', mode ='r', encoding='utf-8-sig')as file:
  csvFile = csv.reader(file)
  for i, lines in enumerate(csvFile):
        if i == 0:
            destFolderId = lines[0]
            continue
        authorized_names.append(lines[0])
        ids.append(lines[1])
# Ask whether you want to create new log or open previous file
def createNewFile():
    x = datetime.datetime.now()
    title = "Attendance Tracker " + x.strftime("%x") + ", "+ x.strftime("%X")
    drive_service = discovery.build('drive', 'v3', credentials=creds)  # Use "credentials" of "gspread.authorize(credentials)".
    file_metadata = {
        'name': title,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
        'parents': [destFolderId]
    }
    file = drive_service.files().create(body=file_metadata).execute()
    print(file)
    sheet = client.open(title).sheet1
    sheet.update_cell(1,1,"Name")
    sheet.update_cell(1,2,"ID")
    sheet.update_cell(1,3,"Status")
    sheet.update_cell(1,4,"Time")
    for i in range(len(authorized_names)):
        sheet.update_cell(i + 2, 1, authorized_names[i])
        sheet.update_cell(i + 2, 2, ids[i])
        sheet.update_cell(i + 2, 3, "Unknown")
    return sheet
def openPreviousFile():
    filename = input("Please input name or ID of spreadsheet you wish to open")
    sheet = client.open(filename).sheet1
    return sheet


while True:
    openPreviousFileOrCreateNewLog = input("Please input 1 to create new log, and input 2 to open previous file:\n")
    if(openPreviousFileOrCreateNewLog == "1"):
        sheet = createNewFile()
        break
    elif(openPreviousFileOrCreateNewLog == "2"):
        sheet = openPreviousFile()
        break
    else:
        print("Invalid input. Please try again")



# Load roster from Google Sheets
def load_roster():
    roster = {}
    records = sheet.get_all_values()
    for i, row in enumerate(records[1:], start=2):
        if len(row) >= 2:
            name = row[0].strip()
            student_id = row[1].strip()
            if name:
                roster[name] = {'row': i, 'id': student_id, 'name': name}
    return roster

roster = load_roster()
print(f"[INFO] Loaded {len(roster)} students: {[v['name'] for v in roster.values()]}")


today_str = datetime.now().strftime("%Y-%m-%d")
local_log_file = f"attendance_{today_str}.json"
logged_times = {}

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})) # lower resolution from 1920x1080 to 640x480
picam2.start()

# Initialize GPIO
output = LED(14)

# Initialize our variables
cv_scaler = 8 # changed from 4 to 8 to reduce mem strain and boost FPS

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0



def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
            #JSON logic for creating JSON log for face that are recognized
            
            #CHECK 1: Is this a known student?
            #Check 2: have we already logged them in this session?
            if name != "Unknown" and name not in already_logged:
                
                # create timestamp variable (to match time between local and cloud data)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                time_only = datetime.now().strftime("%H:%M:%S")  # time-only string for the sheet's Time column
                
                # check the student exists in the roster we loaded from Google Sheets at startup
                if name in roster:  # removed name_lower, now compares directly against roster
                    row = roster[name]['row']  # NEW: get the exact sheet row number for this student
                    try:
                        # updates the student's existing row instead of appending a new row at the bottom
                        sheet.update(f'C{row}:D{row}', [["Present", time_only]])
                        print(f"[CLOUD] Updated row {row} → {name}: Present at {time_only}")
                    except Exception as e:
                        print(f"[ERROR] Could not update Google Sheet for {name}: {e}")
                else:
                    print(f"[WARNING] {name} recognised but not found in roster.")  # NEW: warns if face is known but missing from the sheet

                # add the name to our 'memory' list so they are not logged again
                already_logged.append(name)  # Prevents duplicates
                logged_times[name] = timestamp  # NEW: store timestamp so export_to_xlsx() can reference it when R is pressed

                # replaces the old append logic — instead of adding one line per detection,
                # we rewrite the entire file each time as a clean snapshot (one entry per student)
                snapshot = []
                for student in roster.values():
                    n = student['name']
                    snapshot.append({
                        "student_name": n,
                        "student_id": student['id'],  # includes student ID
                        "status": "Present" if n in already_logged else "Unknown",  # default is "Unknown" instead of "Absent"
                        "time_logged": logged_times.get(n, ""),
                        "date": today_str
                    })
                # "w" overwrites the file each time instead of "a" which appended
                with open(local_log_file, "w") as f:
                    json.dump(snapshot, f, indent=2)
                print(f"[LOCAL] Snapshot updated: {name} marked Present at {timestamp}")
                
            # --- END OF JSON logic block ---
            
            # Check if the detected face is in our authorized list
            if name in authorized_names:
                authorized_face_detected = True
        face_names.append(name)
    
    # ============================================
    # Control the GPIO pin based on face detection
    # ============================================
    if authorized_face_detected:
        output.on()  # Turn on Pin
    else:
        output.off()  # Turn off Pin
    
    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        
        # Add an indicator if the person is authorized
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
    
    return frame

def export_to_xlsx():
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"

    header_font  = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    header_fill  = PatternFill("solid", start_color="2E75B6")
    present_fill = PatternFill("solid", start_color="C6EFCE")
    absent_fill  = PatternFill("solid", start_color="FFCCCC")
    center_align = Alignment(horizontal="center", vertical="center")
    left_align   = Alignment(horizontal="left", vertical="center")
    thin         = Side(style="thin", color="CCCCCC")
    border       = Border(top=thin, bottom=thin, left=thin, right=thin)

    headers    = ["Name", "ID", "Status", "Time"]
    col_widths = [20, 10, 12, 20]

    for col, (header, width) in enumerate(zip(headers, col_widths), start=1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align
        cell.border = border
        ws.column_dimensions[cell.column_letter].width = width
    ws.row_dimensions[1].height = 22

    for row_idx, student in enumerate(roster.values(), start=2):
        name       = student['name']
        student_id = student['id']
        if name in already_logged:
            status, time_val, row_fill = "Present", logged_times.get(name, ""), present_fill
        else:
            status, time_val, row_fill = "Absent", "", absent_fill

        for col, (val, align) in enumerate(zip([name, student_id, status, time_val],
                                                [left_align, center_align, center_align, center_align]), start=1):
            cell = ws.cell(row=row_idx, column=col, value=val)
            cell.fill = row_fill
            cell.alignment = align
            cell.border = border
            cell.font = Font(name="Arial", size=10)
        ws.row_dimensions[row_idx].height = 18

    filename = f"attendance_report_{today_str}.xlsx"
    wb.save(filename)
    print(f"[EXPORT] Spreadsheet saved as '{filename}'")

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# attendance tracker. Keeps track of who is already in the room so they are not logged twice
already_logged = [] 

while True:
    # Capture a frame from camera
    frame = picam2.capture_array()
    
    # Process the frame with the function
    processed_frame = process_frame(frame)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # ADD hint on screen
    cv2.putText(display_frame, "R: Export Report | Q: Quit",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)

    # UPDATED key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        export_to_xlsx()

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()
output.off()  # Make sure to turn off the GPIO pin when exiting
