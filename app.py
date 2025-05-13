from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, send_from_directory
from database import create_connection
import mysql.connector
from mysql.connector import Error
import torch
import os
from datetime import datetime
from PIL import Image
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
import torch.nn as nn

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1600 * 1024 * 1024

MODEL_BASE_DIR = "stored_models"
SEGMENTATION_DIR = os.path.join(MODEL_BASE_DIR, "segmentation")
CLASSIFICATION_DIR = os.path.join(MODEL_BASE_DIR, "classification")

os.makedirs(SEGMENTATION_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)

#Unet model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Downsampling path
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling path
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # Final convolution
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.down1(x)
        x = self.pool(conv1)
        conv2 = self.down2(x)
        x = self.pool(conv2)
        x = self.down3(x)
        
        # Decoder
        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)
        
        return torch.sigmoid(self.conv_last(x))
    
@app.before_request
def require_login():
    allowed_routes = ['login', 'signup', 'static', 'uploaded_file', 'home']  
    if request.endpoint not in allowed_routes and not session.get('logged_in'):
        return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('home/home.html') 

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
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        cin = request.form.get('cin', '').strip()
        dob_str = request.form.get('date_of_birth')  # Format: dd/mm/yyyy
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not all([first_name, last_name, cin, dob_str, username, password]):
            flash('Tous les champs sont obligatoires', 'danger')
            return redirect(url_for('signup'))
        
        try:
            connection = create_connection()
            cursor = connection.cursor()

            cursor.execute("SELECT * FROM users WHERE cin = %s OR username = %s", (cin, username))
            if cursor.fetchone():
                flash('CIN ou nom d\'utilisateur déjà utilisé', 'danger')
                return redirect(url_for('signup'))
            hashed_pw = generate_password_hash(password)
            cursor.execute("""
                INSERT INTO users 
                (first_name, last_name, cin, date_of_birth, username, password)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (first_name, last_name, cin, dob_str, username, hashed_pw))
            connection.commit()
            flash('Compte créé avec succès! Connectez-vous', 'success')
            return redirect(url_for('login'))

        except mysql.connector.IntegrityError:
            flash('Erreur de base de données', 'danger')
        except Exception as e:
            flash(f'Erreur: {str(e)}', 'danger')
        finally:
            if cursor: cursor.close()
            if connection: connection.close()
    
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

        cursor.execute("""
            SELECT * FROM models 
            WHERE model_type = 'segmentation'
            AND model_name LIKE %s 
            ORDER BY upload_date DESC
        """, (f'%{search_seg}%',))
        segmentation_models = cursor.fetchall()

        cursor.execute("""
            SELECT * FROM models 
            WHERE model_type = 'classification'
            AND model_name LIKE %s 
            ORDER BY upload_date DESC
        """, (f'%{search_cls}%',))
        classification_models = cursor.fetchall()

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

        file_ext = model_file.filename.rsplit('.', 1)[1].lower() if '.' in model_file.filename else ''
        allowed_extensions = {'pt', 'pth'}
        
        if file_ext not in allowed_extensions:
            flash(f'Invalid file type: .{file_ext}. Allowed: .pth, .pt', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        model_format = 'torchscript' if file_ext == 'pt' else 'state_dict'

        MAX_SIZE = 300 * 1024 * 1024
        model_file.seek(0, os.SEEK_END)
        file_size = model_file.tell()
        model_file.seek(0)
        
        if file_size > MAX_SIZE:
            flash(f'File too large ({file_size/1024/1024:.2f}MB > {MAX_SIZE/1024/1024}MB)', 'danger')
            return redirect(url_for('admin_dashboard'))

        save_dir = os.path.join(MODEL_BASE_DIR, model_type)
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{model_file.filename}"
        filepath = os.path.join(save_dir, filename)

        model_file.save(filepath)

        connection = create_connection()
        cursor = connection.cursor()
        
        relative_path = os.path.join(model_type, filename)
        cursor.execute(
            "INSERT INTO models (model_name, model_type, model_format, model_path) VALUES (%s, %s, %s, %s)",
            (model_name, model_type, model_format, relative_path)
        )
        connection.commit()
        
        flash(f'Model {model_name} uploaded successfully!', 'success')

    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if connection:
            connection.rollback()
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected():
            connection.close()

    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_model/<int:model_id>')
def delete_model(model_id):
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT model_path FROM models WHERE id = %s", (model_id,))
        model_path = cursor.fetchone()['model_path']
        
        cursor.execute("DELETE FROM models WHERE id = %s", (model_id,))
        connection.commit()
        
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

    models = []
    history = []
    error = None
    connection = None
    cursor = None

    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT id, model_name, model_type 
            FROM models 
            WHERE model_type IN ('segmentation', 'classification')
            ORDER BY model_type, upload_date DESC
        """)
        models = cursor.fetchall()

        if request.method == 'POST':
            if 'image' not in request.files:
                raise ValueError("No file uploaded")

            image_file = request.files['image']
            filename = save_uploaded_file(image_file)
            
            if not filename:
                raise ValueError("Invalid file format")

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

            model = get_model_from_db(model_data['id'])
            input_tensor = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            with torch.no_grad():
                output = model(input_tensor)

            result_filename = f"result_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            
            if model_data['model_type'] == 'segmentation':
                save_segmentation_result(output, save_path)
            else:
                raise NotImplementedError("Classification processing not implemented")

            cursor.execute("""
                INSERT INTO processing_history (user_id, original_image, result_image, model_id)
                VALUES (%s, %s, %s, %s)
            """, (session['user_id'], filename, result_filename, model_data['id']))
            connection.commit()

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
        if connection: connection.rollback()
        error = str(e)
        if "not implemented" in error.lower():
            error = "Classification processing is not available yet"
        flash(error, 'danger')
    finally:
        if cursor: cursor.close()
        if connection: connection.close()

    return render_template('user/dashboard.html',
                         models=models,
                         history=history,
                         error=error)

@app.route('/process-image', methods=['POST'])
def process_image():
    connection = None  # Add this
    cursor = None    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        model_id = request.form.get('model_id')
        
        if not model_id or not file:
            return jsonify({'error': 'Missing required fields'}), 400

        filename = save_uploaded_file(file)
        if not filename:
            return jsonify({'error': 'Invalid file format'}), 400
        
        if file.filename == '':
            return jsonify({'error': 'Empty file submitted'}), 400

        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
        model_data = cursor.fetchone()
        
        if not model_data:
            return jsonify({'error': 'Model not found'}), 404

        model = get_model_from_db(model_data['id'])
        model.eval()
        
        input_tensor = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        #######
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            output = model(input_tensor)
        
        result_filename = f"result_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        if model_data['model_type'] == 'segmentation':
            save_segmentation_result(output, save_path)
        else:
            return jsonify({'error': 'Classification not implemented'}), 501

        cursor.execute("""
            INSERT INTO processing_history (user_id, original_image, result_image, model_id)
            VALUES (%s, %s, %s, %s)
        """, (session['user_id'], filename, result_filename, model_id))
        connection.commit()

        return jsonify({
            'original': url_for('uploaded_file', filename=filename),
            'processed': url_for('uploaded_file', filename=result_filename)
        })

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        app.logger.error(traceback.format_exc())  
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected(): 
            connection.close()

def get_model_from_db(model_id):
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT model_path, model_format, model_type FROM models WHERE id = %s", (model_id,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError("Model not found")

        full_path = os.path.join(MODEL_BASE_DIR, result['model_path'])

        if result['model_format'] == 'torchscript':
            return torch.jit.load(full_path)
        else:
            # Initialize model with correct architecture
            if result['model_type'] == 'segmentation':
                model = UNet()
                # Handle state_dict loading with error details
                state_dict = torch.load(full_path)
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError as e:
                    print(f"Error loading state_dict: {str(e)}")
                    print("Missing keys:", [k for k in state_dict.keys() if k not in model.state_dict()])
                    print("Unexpected keys:", [k for k in model.state_dict() if k not in state_dict.keys()])
                    raise
            else:
                raise ValueError("Unsupported model type")
            
            model.eval()
            return model
            
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected():
            connection.close()

def save_uploaded_file(file):
    if file.filename == '':
        return None
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("Invalid image mode")
            
        transform = torch.nn.Sequential(
            torch.nn.Resize((256, 256)),
            torch.nn.ToTensor()
        )
        return transform(img).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

def save_segmentation_result(output, save_path):
    output = output.squeeze().cpu().numpy()
    output = (output * 255).astype('uint8')
    Image.fromarray(output).save(save_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=False)