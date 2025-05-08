from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, send_from_directory
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

MODEL_BASE_DIR = "stored_models"
SEGMENTATION_DIR = os.path.join(MODEL_BASE_DIR, "segmentation")
CLASSIFICATION_DIR = os.path.join(MODEL_BASE_DIR, "classification")

# Create directories if they don't exist
os.makedirs(SEGMENTATION_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)


@app.before_request
def require_login():
    allowed_routes = ['login', 'signup', 'static', 'uploaded_file']
    if request.endpoint not in allowed_routes and not session.get('logged_in'):
        return redirect(url_for('login'))

# Simple UNet definition (for state_dict .pth files)
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3)  # Minimal example

    def forward(self, x):
        return self.conv(x)

def get_model_from_db(model_id):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT model_path, model_format FROM models WHERE id = %s", (model_id,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()

    # Construct full path
    model_type = result['model_path'].split('/')[0]  # 'segmentation' or 'classification'
    full_path = os.path.join(MODEL_BASE_DIR, result['model_path'])

    if result['model_format'] == 'torchscript':
        return torch.jit.load(full_path)
    else:
        model = UNet()  # Your predefined architecture
        model.load_state_dict(torch.load(full_path))
        return model

def save_uploaded_file(file):
    if file.filename == '':
        return None
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename

def save_classification_result(output, save_path):
    # Temporary implementation until classification is ready
    raise NotImplementedError("Classification processing not implemented yet")
    
    # This will prevent plt errors while maintaining the interface
    # When ready, implement with:
    # 1. Import matplotlib.pyplot as plt
    # 2. Add classification visualization logic
    # 3. Install matplotlib if needed

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
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()

            if not user:
                flash('Username not found', 'danger')
                return redirect(url_for('login'))

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
        # Validate form data
        model_name = request.form.get('model_name')
        model_file = request.files.get('model_file')
        model_type = request.form.get('model_type', 'segmentation')

        if not model_name or not model_file:
            flash('Missing required fields', 'danger')
            return redirect(url_for('admin_dashboard'))

        if model_file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('admin_dashboard'))

        # Validate file extension and determine model format
        file_ext = model_file.filename.rsplit('.', 1)[1].lower() if '.' in model_file.filename else ''
        allowed_extensions = {'pt', 'pth'}
        
        if file_ext not in allowed_extensions:
            flash(f'Invalid file type: .{file_ext}. Allowed: .pth, .pt', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        model_format = 'torchscript' if file_ext == 'pt' else 'state_dict'

        # Validate file size (300MB limit)
        MAX_SIZE = 300 * 1024 * 1024
        model_file.seek(0, os.SEEK_END)
        file_size = model_file.tell()
        model_file.seek(0)
        
        if file_size > MAX_SIZE:
            flash(f'File too large ({file_size/1024/1024:.2f}MB > {MAX_SIZE/1024/1024}MB)', 'danger')
            return redirect(url_for('admin_dashboard'))

        # Create model storage directories if needed
        save_dir = os.path.join(MODEL_BASE_DIR, model_type)
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{model_file.filename}"
        filepath = os.path.join(save_dir, filename)

        # Save file to filesystem
        model_file.save(filepath)

        # Store metadata in database
        connection = create_connection()
        cursor = connection.cursor()
        
        relative_path = os.path.join(model_type, filename)
        cursor.execute(
            "INSERT INTO models (model_name, model_type, model_format, model_path) VALUES (%s, %s, %s, %s)",
            (model_name, model_type, model_format, relative_path)
        )
        connection.commit()
        
        flash(f'Model {model_name} uploaded successfully!', 'success')

    except mysql.connector.Error as err:
        flash(f'Database error: {err}', 'danger')
        # Cleanup failed upload
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if connection:
            connection.rollback()

    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'danger')
        # Cleanup failed upload
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if connection:
            connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

    return redirect(url_for('admin_dashboard'))

# ... [Keep all other routes (delete_model, delete_user, user_dashboard, etc.) unchanged] ...
@app.route('/admin/delete_model/<int:model_id>')
def delete_model(model_id):
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get file path before deletion
        cursor.execute("SELECT model_path FROM models WHERE id = %s", (model_id,))
        model_path = cursor.fetchone()['model_path']
        
        # Delete from database
        cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))
        connection.commit()
        
        # Delete physical file
        full_path = os.path.join(MODEL_BASE_DIR, model_path)
        if os.path.exists(full_path):
            os.remove(full_path)
            
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

    # Initialize all template variables with default values
    models = []
    history = []
    original_image = None
    result_image = None
    error = None
    connection = None
    cursor = None

    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        # Get available models
        cursor.execute("""
            SELECT id, model_name, model_type 
            FROM models 
            WHERE model_type IN ('segmentation', 'classification')
            ORDER BY model_type, upload_date DESC
        """)
        models = cursor.fetchall()

        # Process image upload if POST request
        if request.method == 'POST':
            if 'image' not in request.files:
                raise ValueError("No file uploaded")

            image_file = request.files['image']
            filename = save_uploaded_file(image_file)
            
            if not filename:
                raise ValueError("Invalid file format")

            # Get selected model or use latest segmentation model
            selected_model_id = request.form.get('model_id')
            if selected_model_id:
                cursor.execute("""
                    SELECT id, model_path, model_format, model_type 
                    FROM models 
                    WHERE id = %s
                """, (selected_model_id,))
            else:
                cursor.execute("""
                    SELECT id, model_path, model_format, model_type 
                    FROM models 
                    WHERE model_type = 'segmentation'
                    ORDER BY upload_date DESC 
                    LIMIT 1
                """)

            model_data = cursor.fetchone()
            if not model_data:
                raise ValueError("No available models for processing")

            # Process image with selected model
            model = get_model_from_db(model_data['id'])
            input_tensor = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            with torch.no_grad():
                output = model(input_tensor)

            # Save results based on model type
            result_filename = f"result_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            
            if model_data['model_type'] == 'segmentation':
                save_segmentation_result(output, save_path)
            else:
                save_classification_result(output, save_path)

            # Store processing history
            cursor.execute("""
                INSERT INTO processing_history (user_id, original_image, result_image, model_id)
                VALUES (%s, %s, %s, %s)
            """, (session['user_id'], filename, result_filename, model_data['id']))
            connection.commit()

            original_image = filename
            result_image = result_filename

        # Always get processing history (for both GET and POST)
        cursor.execute("""
            SELECT ph.*, m.model_name 
            FROM processing_history ph
            JOIN models m ON ph.model_id = m.id
            WHERE user_id = %s 
            ORDER BY processed_at DESC 
            LIMIT 5
        """, (session['user_id'],))
        history = cursor.fetchall()

    except Exception as e:
        if connection:
            connection.rollback()
        error = str(e)
        if "not implemented" in error.lower():
            error = "Classification processing is not available yet"
        flash(error, 'danger')
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

    return render_template('user/dashboard.html',
                         models=models,
                         history=history,
                         original_image=original_image,
                         result_image=result_image,
                         error=error)

@app.route('/get-models')
def get_models():
    category = request.args.get('category', 'eyes')
    process_type = request.args.get('type', 'segmentation')
    
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, model_name 
            FROM models 
            WHERE model_type = %s 
            ORDER BY model_name ASC
        """, (process_type,))
        
        models = cursor.fetchall()
        return jsonify(models)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
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