import os
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify, flash
from openai import OpenAI
from livepeer_ai import Livepeer
from dotenv import load_dotenv
import json
import time
import sqlite3
from sqlite3 import Error
from functools import wraps
from datetime import datetime  # Import datetime
from reportlab.lib.utils import ImageReader
from PIL import Image
from io import BytesIO
import bcrypt
import logging
from threading import Lock
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session management

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Livepeer SDK
livepeer = Livepeer(http_bearer=os.getenv("LIVEPEER_API_KEY"))


# Setup logger
logging.basicConfig(filename='app.log', level=logging.ERROR)


# Database setup
DB_FILE = "stories.db"
stories = []

def create_connection():
    conn = None
    try:
        # Connect to the database with a timeout of 10 seconds
        conn = sqlite3.connect(DB_FILE, timeout=10)
        # Set the journal mode to WAL (Write-Ahead Logging)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Error as e:
        print(e)
    return conn

def create_tables(conn):
    # Create tables if they do not exist
    try:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            authcode TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT NOT NULL,
            num_pages INTEGER NOT NULL,
            style_prompt TEXT,
            character_descriptions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS story_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            page_number INTEGER,
            text TEXT,
            image_url TEXT,
            FOREIGN KEY (story_id) REFERENCES stories (id)
        )""")
        conn.commit()
    except Error as e:
        print(e)

def fetch_stories():
    conn = create_connection()
   
    if conn:
        try:
            cursor = conn.cursor()
            # Assuming you want to join story details with image URLs
            cursor.execute("""
                SELECT s.id, s.prompt, s.num_pages, s.created_at, s.user_id, sp.image_url 
                FROM stories s
                LEFT JOIN story_pages sp ON s.id = sp.story_id
                ORDER BY s.created_at DESC
            """)
            rows = cursor.fetchall()
            for row in rows:
                stories.append({
                    'id': row[0],
                    'prompt': row[1],
                    'num_pages': row[2],
                    'created_at': row[3],
                    'user_id': row[4],
                    'image_url': row[5]  # Now fetching from story_pages
                })
        except Error as e:
            print(f"Database error: {e}")
        finally:
            conn.close()
    return stories

# Initialize database
conn = create_connection()
if conn is not None:
    create_tables(conn)
    conn.close()
else:
    print("Error! Cannot create the database connection.")



class StoryGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.livepeer = Livepeer(http_bearer=os.getenv("LIVEPEER_API_KEY"))

    def generate_story(self, prompt, num_pages):
        try:
            # Generate the story text
            story_text = self._generate_text(prompt, num_pages)
            yield json.dumps({"progress": 25})  # Indicate that text generation is complete

            # Generate a consistent style prompt
            style_prompt = self._generate_style_prompt(prompt)
            yield json.dumps({"progress": 50})  # Indicate that style prompt generation is complete

            # Generate character descriptions
            character_descriptions = self._generate_character_descriptions(story_text)
            yield json.dumps({"progress": 75})  # Indicate that character descriptions are complete

            # Generate images for the story using Livepeer with the consistent style and characters
            image_paths = self._generate_images_livepeer(story_text, style_prompt, character_descriptions)
            yield json.dumps({"progress": 100})  # Indicate that image generation is complete

            # Yield the final story and image paths as a serializable object
            yield json.dumps({
                "complete": True,
                "story_data": {
                    "story_text": story_text,
                    "image_paths": image_paths,
                    "prompt": prompt,
                    "num_pages": len(story_text),
                    "style_prompt": style_prompt,
                    "character_descriptions": character_descriptions
                }
            })
        except Exception as e:
            yield json.dumps({"error": f"Error generating story: {str(e)}"})

    
    
    def _generate_text(self, prompt, num_pages):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a children's story writer."},
                    {"role": "user", "content": f"Write a {num_pages}-page children's story about {prompt}. Each page should be a short paragraph. Separate pages with '---'."}
                ]
            )
            # Split the story into pages
            pages = response.choices[0].message.content.split('---')
            return pages[:num_pages]  # Ensure we have exactly num_pages
        except Exception as e:
            print(f"Error in _generate_text: {e}")
            raise

    def _generate_style_prompt(self, story_prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in creating consistent art styles for children's books."},
                    {"role": "user", "content": f"Create a detailed art style description for a children's story about {story_prompt}. The description should ensure consistency across multiple illustrations."}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in _generate_style_prompt: {e}")
            raise

    def _generate_character_descriptions(self, story_text):
        try:
            full_story = " ".join(story_text)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing stories and describing characters consistently."},
                    {"role": "user", "content": f"Based on this story, provide brief, consistent descriptions for the main characters. Include their appearance, key features, and any distinguishing characteristics:\n\n{full_story}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in _generate_character_descriptions: {e}")
            raise

    def _generate_images_livepeer(self, story_text, style_prompt, character_descriptions):
        image_paths = []
        for index, page in enumerate(story_text):
            retries = 3
            for attempt in range(retries):
                try:
                    res = self.livepeer.generate.text_to_image(request={
                        "prompt": f"{style_prompt}\n{character_descriptions}\nIllustration for: {page[:100]}...",
                        "model_id": "black-forest-labs/FLUX.1-dev",
                        "width": 1024,
                        "height": 1024,
                    })

                    if res.image_response is not None and res.image_response.images:
                        image_paths.append(res.image_response.images[0].url)
                    else:
                        print(f"Error generating image for page {index}: No image response")
                        image_paths.append(None)
                    break
                except Exception as e:
                    print(f"Error generating image for page {index} on attempt {attempt + 1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(2)
                    else:
                        image_paths.append(None)
        return image_paths

story_generator = StoryGenerator()

# Custom login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        authcode = request.form['authcode']
        
        conn = create_connection()
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id, authcode FROM users WHERE username = ?", (username,))
                user = cursor.fetchone()
                
                if user and bcrypt.checkpw(authcode.encode('utf-8'), user[1]):  # Verify the hashed authcode
                    session['user_id'] = user[0]
                    session['username'] = username
                    return redirect(url_for('index'))
                else:
                    flash('Invalid username or authcode')
            finally:
                conn.close()
        
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        authcode = request.form['authcode']

        # Basic validation for inputs
        if not username or not email or not authcode:
            flash('All fields are required')
            return render_template('register.html')
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long')
            return render_template('register.html')

        # Example email validation
        if '@' not in email or '.' not in email.split('@')[-1]:
            flash('Invalid email format')
            return render_template('register.html')

        # Only allow valid codes
        if authcode not in ['sg12@', 'sg99#']:
            flash('Invalid authcode. Contact admin for valid code')
            return render_template('register.html')

        hashed_authcode = bcrypt.hashpw(authcode.encode('utf-8'), bcrypt.gensalt())  # Hash the authcode
        
        conn = create_connection()
        if conn is not None:
            try:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, email, authcode) VALUES (?, ?, ?)",
                               (username, email, hashed_authcode))
                conn.commit()
                flash('Registration successful. Please log in.')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username or email already exists')
            finally:
                conn.close()
        
    return render_template('register.html')

@app.route('/generate_story')
def generate_story():
    prompt = request.args.get('prompt')
    num_pages = int(request.args.get('num_pages'))
    user_id = session.get('user_id')  # Get the user ID from the session
    print(f"Generating story with prompt: {prompt}, num_pages: {num_pages}")

    def generate():
        yield "data: " + json.dumps({"progress": 0}) + "\n\n"
        generator = story_generator.generate_story(prompt, num_pages)

        story_data = None  # To hold the final story data

        for update in generator:
            print(f"Generator update: {update}")
            yield "data: " + update + "\n\n"
            if "complete" in json.loads(update):
                story_data = json.loads(update)

        # Save the generated story to the database
        if story_data and user_id:
            conn = create_connection()
            if conn is not None:
                try:
                    cursor = conn.cursor()
                    # Insert the story into the stories table
                    cursor.execute("INSERT INTO stories (user_id, prompt, num_pages, style_prompt, character_descriptions) VALUES (?, ?, ?, ?, ?)",

                                   (user_id, prompt, len(story_data['story_data']['story_text']), story_data['story_data']['style_prompt'], story_data['story_data']['character_descriptions']))

                    story_id = cursor.lastrowid  # Get the ID of the newly inserted story

                    # Insert each page into the story_pages table
                    for page_number, text in enumerate(story_data['story_data']['story_text'], start=1):
                        image_url = story_data['story_data']['image_paths'][page_number - 1] if page_number - 1 < len(story_data['story_data']['image_paths']) else None
                        cursor.execute("INSERT INTO story_pages (story_id, page_number, text, image_url) VALUES (?, ?, ?, ?)",

                                       (story_id, page_number, text, image_url))

                    conn.commit()
                except Error as e:
                    print(f"Error saving story to database: {e}")
                finally:
                    conn.close()

    return Response(generate(), mimetype='text/event-stream')


@app.route('/save_story', methods=['POST'])
def save_story():
    story_data = request.json
    stories.append(story_data)  # Save the story to the in-memory list
    story_id = len(stories) - 1  # Get the index of the newly saved story
    print(f"Story saved: {story_data}")
    return jsonify({"success": True, "redirect": url_for('view_story', story_id=story_id)})


@app.route('/view_story/<int:story_id>')
def view_story(story_id):
    # Fetch the story from the database
    conn = create_connection()
    story = None
    creator_name = None  # Initialize creator_name variable
    if conn:
        try:
            cursor = conn.cursor()
            # Fetch the main story details
            cursor.execute("SELECT prompt, style_prompt, character_descriptions, user_id FROM stories WHERE id = ?", (story_id,))
            story_data = cursor.fetchone()
            if story_data:
                story = {
                    'prompt': story_data[0],
                    'style_prompt': story_data[1],
                    'character_descriptions': story_data[2],
                    'user_id': story_data[3],  # Get user_id from the story
                    'story_text': [],  # Initialize empty list for story text
                    'image_paths': []   # Initialize empty list for image paths
                }

                # Fetch the pages for this story
                cursor.execute("SELECT text, image_url FROM story_pages WHERE story_id = ? ORDER BY page_number", (story_id,))
                pages = cursor.fetchall()
                for page in pages:
                    story['story_text'].append(page[0])  # Append the text of the page
                    story['image_paths'].append(page[1])  # Append the image URL of the page

                # Fetch the creator's name based on user_id
                cursor.execute("SELECT username FROM users WHERE id = ?", (story['user_id'],))
                user_data = cursor.fetchone()
                if user_data:
                    creator_name = user_data[0]  # Get the creator's username

        except Error as e:
            print(f"Error fetching story from database: {e}")
        finally:
            conn.close()

    if not story:
        return redirect(url_for('index'))

    # Pre-zip the story text and image paths
    zipped_story = list(zip(story['story_text'], story['image_paths']))

    return render_template('view_story.html', 
                           story_id=story_id, 
                           zipped_story=zipped_story,
                           prompt=story['prompt'], 
                           style_prompt=story['style_prompt'],
                           character_descriptions=story['character_descriptions'],
                           num_pages=len(story['story_text']),
                           creator_name=creator_name)  # Pass the creator's name to the template

@app.route('/storyboard')
def storyboard():
    stories = fetch_stories()  # Assuming this function fetches stories from the database

    # Format the `created_at` date using `strftime`
    for story in stories:
        if isinstance(story['created_at'], str):  # Check if it's a string
            # Convert string to datetime object
            try:
                story['created_at'] = datetime.strptime(story['created_at'], '%Y-%m-%d %H:%M')
            except ValueError:
                # Handle cases where the string format is unexpected
                continue
        
        # Format the datetime object
        story['created_at'] = story['created_at'].strftime('%Y-%m-%d %H:%M')

        # Fetch the creator's name based on user_id
        conn = create_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT username FROM users WHERE id = ?", (story['user_id'],))
                user_data = cursor.fetchone()
                if user_data:
                    story['creator_name'] = user_data[0]  # Add creator's name to the story dictionary
                else:
                    story['creator_name'] = "Unknown"  # Fallback if no user is found
            except Error as e:
                print(f"Error fetching creator name: {e}")
            finally:
                conn.close()

    return render_template('storyboard.html', stories=stories)


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')  # Get the search query from the request
    conn = create_connection()  # Function to create a database connection
    stories = []

    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, prompt, num_pages, style_prompt, character_descriptions FROM stories WHERE prompt LIKE ?", ('%' + query + '%',))
            stories = cursor.fetchall()

            stories_data = [{
                "id": story[0],
                "prompt": story[1],
                "num_pages": story[2],
                "style_prompt": story[3],
                "character_descriptions": story[4]
            } for story in stories]
            
            return render_template('search_results.html', query=query, stories=stories_data)
        except Error as e:
            logging.error(f"Error fetching stories from database: {e}")
        finally:
            conn.close()

    return render_template('search_results.html', query=query, stories=stories)



if __name__ == '__main__':
    app.run(debug=True)
