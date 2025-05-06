from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from database import create_connection
import mysql.connector
from mysql.connector import Error
import torch
import io
import os
from datetime import datetime
from PIL import Image
import traceback
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1600 * 1024 * 1024

@app.before_request
def require_login():
    allowed_routes = ['login', 'signup', 'static', 'uploaded_file']
    if request.endpoint not in allowed_routes and not session.get('logged_in'):
        return redirect(url_for('login'))

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3)

    def forward(self, x):
        return self.conv(x)

def get_model_from_db(model_id):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT model_data FROM models WHERE id = %s", (model_id,))
    model_data = cursor.fetchone()[0]
    cursor.close()
    connection.close()
    
    model = UNet()
    model.load_state_dict(torch.load(io.BytesIO(model_data)))
    return model

def save_uploaded_file(file):
    if file.filename == '':
        return None
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard' if session.get('is_admin') else 'user_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Both fields are required', 'danger')
            return redirect(url_for('login'))

        try:
            connection = create_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Debug: Print actual SQL query
            print(f"Executing: SELECT * FROM users WHERE username = '{username}'")
            
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            # Debug: Print database response
            print("Database returned:", user)

            if not user:
                flash('Username not found', 'danger')
                return redirect(url_for('login'))

            # Debug: Compare passwords
            print(f"Input password: {password}")
            print(f"Stored hash: {user['password']}")
            print(f"Check result: {check_password_hash(user['password'], password)}")

            if check_password_hash(user['password'], password):
                session.update({
                    'logged_in': True,
                    'user_id': user['id'],
                    'username': user['username'],
                    'is_admin': user['is_admin']
                })
                return redirect(url_for('admin_dashboard' if user['is_admin'] else 'user_dashboard'))
            
            flash('Incorrect password', 'danger')
            return redirect(url_for('login'))

        except Exception as e:
            flash(f'Login error: {str(e)}', 'danger')
            return redirect(url_for('login'))
        finally:
            if cursor: cursor.close()
            if connection: connection.close()

    return render_template('auth/sign-in.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('logged_in'):
        return redirect(url_for('user_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Both fields are required', 'danger')
            return redirect(url_for('signup'))
        
        try:
            connection = create_connection()
            cursor = connection.cursor()
            hashed_pw = generate_password_hash(password)
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
            connection.commit()
            flash('Account created! Please login', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Username exists', 'danger')
        except Exception as e:
            flash('Registration failed', 'danger')
        finally:
            cursor.close()
            connection.close()
    
    return render_template('auth/sign-up.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        flash('Unauthorized', 'danger')
        return redirect(url_for('login'))
    
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        search_seg = request.args.get('search_seg', '')
        search_cls = request.args.get('search_cls', '')
        search_user = request.args.get('search_user', '')

        # Segmentation models
        cursor.execute("""
            SELECT * FROM models 
            WHERE model_type = 'segmentation'
            AND model_name LIKE %s 
            ORDER BY upload_date DESC
        """, (f'%{search_seg}%',))
        segmentation_models = cursor.fetchall()

        # Classification models
        cursor.execute("""
            SELECT * FROM models 
            WHERE model_type = 'classification'
            AND model_name LIKE %s 
            ORDER BY upload_date DESC
        """, (f'%{search_cls}%',))
        classification_models = cursor.fetchall()

        # Users
        cursor.execute("""
            SELECT * FROM users 
            WHERE username LIKE %s 
            ORDER BY created_at DESC
        """, (f'%{search_user}%',))
        users = cursor.fetchall()

        return render_template('admin/dashboard.html',
                             segmentation_models=segmentation_models,
                             classification_models=classification_models,
                             users=users)
    finally:
        cursor.close()
        connection.close()

@app.route('/admin/upload_model', methods=['POST'])
def upload_model():
    if not session.get('is_admin'):
        flash('Unauthorized access', 'danger')
        return redirect(url_for('login'))

    connection = None
    cursor = None
    try:
        model_name = request.form.get('model_name')
        model_file = request.files.get('model_file')
        model_type = request.form.get('model_type', 'segmentation')

        if not model_name or not model_file:
            flash('Missing required fields', 'danger')
            return redirect(url_for('admin_dashboard'))

        if model_file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('admin_dashboard'))

        allowed_extensions = {'pth', 'pt'}
        file_ext = model_file.filename.rsplit('.', 1)[1].lower() if '.' in model_file.filename else ''
        if file_ext not in allowed_extensions:
            flash(f'Invalid file type: .{file_ext}. Allowed: .pth, .pt', 'danger')
            return redirect(url_for('admin_dashboard'))

        MAX_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
        model_file.seek(0, os.SEEK_END)
        file_size = model_file.tell()
        model_file.seek(0)
        
        if file_size > MAX_SIZE:
            flash(f'File too large ({file_size/1024/1024:.2f}MB > {MAX_SIZE/1024/1024}MB)', 'danger')
            return redirect(url_for('admin_dashboard'))

        connection = create_connection()
        cursor = connection.cursor()
        
        model_data = model_file.read()
        cursor.execute(
            "INSERT INTO models (model_name, model_type, model_data) VALUES (%s, %s, %s)",
            (model_name, model_type, model_data)
        )
        connection.commit()
        flash('Model uploaded successfully', 'success')

    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'danger')
        if connection:
            connection.rollback()
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'danger')
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_model/<int:model_id>')
def delete_model(model_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))
        connection.commit()
        flash('Model deleted', 'success')
    except Exception as e:
        flash('Delete failed', 'danger')
    finally:
        cursor.close()
        connection.close()
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if not session.get('is_admin'):
        flash('Unauthorized', 'danger')
        return redirect(url_for('login'))
    
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        connection.commit()
        flash('User deleted successfully', 'success')
    except Exception as e:
        flash(f'Delete failed: {str(e)}', 'danger')
    finally:
        cursor.close()
        connection.close()
    
    return redirect(url_for('admin_dashboard'))

@app.route('/user/dashboard', methods=['GET', 'POST'])
def user_dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                flash('No file', 'danger')
                return redirect(url_for('user_dashboard'))
            
            image = request.files['image']
            filename = save_uploaded_file(image)
            
            if not filename:
                flash('Invalid file', 'danger')
                return redirect(url_for('user_dashboard'))
            
            cursor.execute("SELECT id FROM models WHERE model_type = 'segmentation' ORDER BY upload_date DESC LIMIT 1")
            model_result = cursor.fetchone()
            
            if not model_result:
                flash('No models available', 'danger')
                return redirect(url_for('user_dashboard'))
            
            model = get_model_from_db(model_result['id'])
            input_tensor = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            with torch.no_grad():
                output = model(input_tensor)
            
            result_filename = f"result_{filename}"
            save_segmentation_result(output, os.path.join(app.config['UPLOAD_FOLDER'], result_filename))
            
            cursor.execute("""
                INSERT INTO processing_history (user_id, original_image, result_image)
                VALUES (%s, %s, %s)
            """, (session['user_id'], filename, result_filename))
            connection.commit()
            
            return render_template('user/dashboard.html',
                                 original_image=filename,
                                 result_image=result_filename)
        
        cursor.execute("""
            SELECT * FROM processing_history 
            WHERE user_id = %s 
            ORDER BY processed_at DESC LIMIT 5
        """, (session['user_id'],))
        history = cursor.fetchall()
        return render_template('user/dashboard.html', history=history)
    
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('user_dashboard'))
    finally:
        cursor.close()
        connection.close()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = torch.nn.Sequential(
        torch.transforms.Resize((256, 256)),
        torch.transforms.ToTensor()
    )
    return transform(image).unsqueeze(0)

def save_segmentation_result(output, save_path):
    output = output.squeeze().cpu().numpy()
    output = (output * 255).astype('uint8')
    Image.fromarray(output).save(save_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)