from datetime import datetime
import sqlite3
import cv2
import pandas as pd
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
import face_recognition
import numpy as np


# Load the KV file
Builder.load_file('main_app.kv')

class MainScreen(Screen):
    
    def go_to_register_screen(self):
        app = App.get_running_app()
        app.root.current = 'register'

    def go_to_punchin_screen(self):
        app = App.get_running_app()
        app.root.current = 'punchin'

    def go_to_reports_screen(self):
        app = App.get_running_app()
        app.root.current = 'reports'

    def go_to_SynchTocloud_screen(self):
        app = App.get_running_app()
        app.root.current = 'synctocloud'

class SyncToCloudScreen(Screen):
    pass

class RegisterScreen(Screen):
    
    def __init__(self, **kwargs):
        super(RegisterScreen, self).__init__(**kwargs)
        
        
        #self.name_label = Label(text="Name: Unknown", font_size=20)
        
        #layout.add_widget(self.name_label)
      
        self.captured_photo = None
        self.capture = None
        self.img = self.ids.my_image
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
    def register(self):
        
        # Access the TextInput widgets and get user input
        
        name = self.ids.fullname_input.text
        emp_id = self.ids.EmpId_input.text
        role= self.ids.role_spinner.text
            
        if self.captured_photo is not None:
            
            fetchallrows = self.fetch_from_DB()
            print(f"result_blobs '{len(fetchallrows)}'")
            for row in fetchallrows: 
                blob_data = row[1]
                print(f"blob_data '{blob_data}'")
                bytes_data = bytes(blob_data)

                # Convert bytes to NumPy array
                np_array = np.frombuffer(bytes_data, dtype=np.uint8)

                # Decode NumPy array to OpenCV frame
                cv_frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                Db_detected_face = self.detect_face(cv_frame)
                captured_detected_face = self.detect_face(self.captured_photo)
                if Db_detected_face is not None :
                    detected_face_rgb = cv2.cvtColor(captured_detected_face, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    self.registered_face_encoding = self.encode_face(Db_detected_face)
                    if self.match_face(detected_face_rgb, self.registered_face_encoding):
                        self.ids.name_label.text = f"{row[0]} ,you are already Registered !"
                        print(f"{row[0]} ,you are already Registered !")
                        self.display_image(detected_face_rgb)  # Display the recognized face
                        break
                    else :
                        self.name_label.text = "Name: Unknown"
                        print("No registered face found.")
                        break
                else :
                    if captured_detected_face is not None and name is not None and role is not None and emp_id is not None:
                        self.save_face_to_db(name, captured_detected_face,role,emp_id)
                        self.ids.name_label.text = f" Welcome, {row[0]}"
                        print(".")
                        break     
                
        else:
            self.ids.name_label.text = "Name: Unknown"
            print("Capture a photo first.")
            

    def create_table_if_not_exists(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          ( ID INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT, encoding BLOB, photo BLOB,Role TEXT,Emp_ID TEXT)''')
        self.db_connection.commit() 
        cursor.execute('''CREATE TABLE IF NOT EXISTS attendancelog
                          ( ID INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT, emp_id TEXT, role TEXT,logintime DATETIME)''') 
        self.db_connection.commit() 
        # Perform registration logic (you can customize this part)
        #print(f"Username: {username}, Password: {password}")
    def capture_photo(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Camera not found or cannot be opened.")
            return
        ret, frame = self.capture.read()
        if ret:
            self.captured_photo = frame
            self.display_image(frame)
            print("Photo captured" )

    def save_face_to_db(self, name, detected_face,role,emp_id):
        encoding_str = self.encode_face(detected_face)
        _, encoded_frame = cv2.imencode('.jpg', self.captured_photo)

        # Convert the NumPy array to bytes
        bytes_frame = encoded_frame.tobytes()
        if encoded_frame is not None:
            cursor = self.db_connection.cursor()
            cursor.execute("INSERT INTO faces (name, encoding, photo, role, emp_id) VALUES (?, ?, ?, ?, ?)", (name, sqlite3.Binary(encoded_frame), sqlite3.Binary(bytes_frame),role,emp_id))
            self.db_connection.commit()
        
    
    def display_image(self, frame):
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            detected_face = gray[y:y + h, x:x + w]
            return detected_face
        return None
    
    def encode_face(self, detected_face):
        # Convert the detected face to RGB format
        detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        
        # Encode the detected face using face_recognition library
        detected_face_encoding = face_recognition.face_encodings(detected_face_rgb)
        
        if len(detected_face_encoding) > 0:
            # Convert the encoding to a string
            encoding_str = detected_face_encoding[0].tobytes()
            return encoding_str
        
        return None
    
    def fetch_from_DB(self):
        self.db_connection = sqlite3.connect("face_recognition.db")
        self.create_table_if_not_exists()
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT name,photo,role,emp_id FROM faces")
        results = cursor.fetchall()
        return results
    
    def match_face(self, detected_face, registered_face_encoding):
        # Convert the detected face to an encoding using face_recognition library
        detected_face_encoding = face_recognition.face_encodings(detected_face)
        if len(detected_face_encoding) > 0:
            # Calculate the distance using numpy's linalg.norm function
            distance = np.linalg.norm(np.frombuffer(registered_face_encoding, dtype=np.float64) - detected_face_encoding[0])
            
            # Define a threshold for considering it a match (you can adjust this threshold as needed)
            threshold = 0.6  # Adjust this threshold as needed
            
            # Check if the distance is below the threshold
            if distance < threshold:
                return True
        
        return False
    def on_stop(self):
        if self.capture is not None:
            self.capture.release()
        if self.db_connection is not None:
            self.db_connection.close()
        cv2.destroyAllWindows()
        self.ids.my_image.texture=None
        self.ids.name_label.text = None

    def go_to_main_screen(self):
        self.on_stop()
        app = App.get_running_app()
        app.root.current = 'main'
class PunchinScreen(Screen):
    def __init__(self, **kwargs):
        super(PunchinScreen, self).__init__(**kwargs)
        self.capture = None
        self.img = self.ids.punch_image
        self.captured_photo = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        #self.name_label = Label(text="Name: Unknown", font_size=20)
        
        #layout.add_widget(self.name_label)
        

    def login_with_face(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Camera not found or cannot be opened.")
            return
        ret, frame = self.capture.read()
        if ret:
            self.captured_photo = frame
            self.display_image(frame)
            print("Photo captured" )
        if self.captured_photo is not None:
            fetchallrows = self.fetch_from_DB()
            print(f"result_blobs '{len(fetchallrows)}'")
            for row in fetchallrows: 
                blob_data = row[1]
                print(f"blob_data '{blob_data}'")
                bytes_data = bytes(blob_data)

                # Convert bytes to NumPy array
                np_array = np.frombuffer(bytes_data, dtype=np.uint8)

                # Decode NumPy array to OpenCV frame
                cv_frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                
                Db_detected_face = self.detect_face(cv_frame)
                captured_detected_face = self.detect_face(self.captured_photo)
                if Db_detected_face is not None :
                    detected_face_rgb = cv2.cvtColor(captured_detected_face, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    self.registered_face_encoding = self.encode_face(Db_detected_face)
                    if self.match_face(detected_face_rgb, self.registered_face_encoding):
                        self.ids.punchname.text = f"Welcome,{row[0]}!"
                        print(f"Welcome,{row[0]}")
                        self.display_image(detected_face_rgb)  # Display the recognized face
                        name = row[0]
                        role = row[2]
                        emp_id = row[3]
                        current_time = datetime.now().strftime("%H:%M:%S")
                        self.save_Attendancelogs_to_db(name,role,emp_id,current_time)
                        print("Attendance logged Successfully")
                        break
                    else :
                        self.ids.punchname.text = "Name: Unknown"
                        print("Face not recognized. Please try again.")
                        break
                else :
                        self.ids.punchname.text = "Name: Unknown"
                        print("No registered face found.")
                        break  
        else:
            self.ids.name_label.text = "Name: Unknown"
            print("camera not recognized.")   
    def Downloadattendance(self):
        self.db_connection = sqlite3.connect("face_recognition.db")
        self.create_table_if_not_exists()
        query = 'SELECT * FROM attendancelog'
        attendance_data = pd.read_sql_query(query, self.db_connection)

        # Export the DataFrame to a CSV file
        attendance_data.to_csv('attendance_sheet.csv', index=False)      

    def create_table_if_not_exists(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          ( ID INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT, encoding BLOB, photo BLOB,Role TEXT,Emp_ID TEXT)''')
        self.db_connection.commit() 

    def display_image(self, frame):
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            detected_face = gray[y:y + h, x:x + w]
            return detected_face
        return None
    
    def fetch_from_DB(self):
        self.db_connection = sqlite3.connect("face_recognition.db")
        self.create_table_if_not_exists()
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT name,photo,role,emp_id FROM faces")
        results = cursor.fetchall()
        return results
    def encode_face(self, detected_face):
        # Convert the dete  cted face to RGB format
        detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        
        # Encode the detected face using face_recognition library
        detected_face_encoding = face_recognition.face_encodings(detected_face_rgb)
        
        if len(detected_face_encoding) > 0:
            # Convert the encoding to a string
            encoding_str = detected_face_encoding[0].tobytes()
            return encoding_str
        
        return None
    def save_Attendancelogs_to_db(self, name,role,emp_id,current_time):
        if name is not None and role is not None and emp_id is not None and current_time is not None:
            cursor = self.db_connection.cursor()
            cursor.execute("INSERT INTO attendancelog (name,emp_id,role,logintime) VALUES (?, ?, ?, ?)", (name, emp_id,role,current_time))
            self.db_connection.commit()

    def match_face(self, detected_face, registered_face_encoding):
        # Convert the detected face to an encoding using face_recognition library
        detected_face_encoding = face_recognition.face_encodings(detected_face)
        if len(detected_face_encoding) > 0:
            # Calculate the distance using numpy's linalg.norm function
            distance = np.linalg.norm(np.frombuffer(registered_face_encoding, dtype=np.float64) - detected_face_encoding[0])
            
            # Define a threshold for considering it a match (you can adjust this threshold as needed)
            threshold = 0.6  # Adjust this threshold as needed
            
            # Check if the distance is below the threshold
            if distance < threshold:
                return True
        
        return False
    def on_stop(self):
        if self.capture is not None:
            self.capture.release()
        if self.db_connection is not None:
            self.db_connection.close()
        cv2.destroyAllWindows()
        self.ids.punch_image.texture= None
        self.ids.punchname.text = 'Name: Unknown'

    def go_to_main_screen(self):
        self.on_stop()
        app = App.get_running_app()
        app.root.current = 'main'
class ReportsScreen(Screen):
    def go_to_main_screen(self):
        app = App.get_running_app()
        app.root.current = 'main'

class MyApp(App):
    def build(self):
        
        sm = ScreenManager()

        main_screen = MainScreen(name='main')
        register_screen = RegisterScreen(name='register')
        punchin_screen = PunchinScreen(name='punchin')
        reports_screen = ReportsScreen(name='reports')
        synctocloud_screen = ReportsScreen(name='synctocloud')

        sm.add_widget(main_screen)
        sm.add_widget(register_screen)
        sm.add_widget(punchin_screen)
        sm.add_widget(reports_screen)
        sm.add_widget(synctocloud_screen)

        sm.current = 'main'

        return sm

if __name__ == '__main__':
    MyApp().run()