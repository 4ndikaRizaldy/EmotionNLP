# Load Database Pkg
import sqlite3
import threading

# Function to create a new connection for each thread
def get_connection():
    return sqlite3.connect('data.db')

# Fxn
def create_page_visited_table():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT, timeOfvisit TEXT)')

def add_page_visited_details(pagename, timeOfvisit):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', (pagename, timeOfvisit))
        conn.commit()

def view_all_page_visited_details():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM pageTrackTable')
        data = c.fetchall()
    return data

# Fxn To Track Input & Prediction
def create_emotionclf_table():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TEXT)')

def add_prediction_details(rawtext, prediction, probability, timeOfvisit):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit) VALUES (?, ?, ?, ?)', (rawtext, prediction, probability, timeOfvisit))
        conn.commit()

def view_all_prediction_details():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM emotionclfTable')
        data = c.fetchall()
    return data