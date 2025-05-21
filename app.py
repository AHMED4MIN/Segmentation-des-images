from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, send_from_directory
from database import create_connection
import mysql.connector
from mysql.connector import Error
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from PIL import Image
import traceback
import cv2
from PIL import Image
from torchvision import transforms
from werkzeug.security import generate_password_hash, check_password_hash
from model import build_unet 
import time
import traceback
from typing import Union


app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1600 * 1024 * 1024

MODEL_BASE_DIR = "stored_models"
SEGMENTATION_DIR = os.path.join(MODEL_BASE_DIR, "segmentation")
CLASSIFICATION_DIR = os.path.join(MODEL_BASE_DIR, "classification")

os.makedirs(SEGMENTATION_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)




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
    #session.clear()  # À retirer après les tests : pour supprimer la session
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

            connection.commit()

        

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
                         error=error)

def create_overlay(original: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create overlay visualization of mask on original image"""
    try:
        # Ensure mask is single-channel and properly sized
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        if original.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        # Create red mask overlay
        mask_color = np.zeros_like(original)
        mask_color[mask > 0] = [0, 0, 255]  # Red color for mask
        
        # Blend with original image
        overlay = cv2.addWeighted(original, 1 - alpha, mask_color, alpha, 0)
        return overlay
        
    except Exception as e:
        print(f"⚠️ Overlay creation error: {str(e)}")
        return original  # Fallback to original image
    
@app.route('/process-image', methods=['POST'])
def process_image():
    connection = None
    cursor = None
    try:
        # Initial validation
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        model_id = request.form.get('model_id')
        
        if not model_id or not file:
            return jsonify({'error': 'Missing required fields'}), 400

        # Save uploaded file
        filename = save_uploaded_file(file)
        if not filename:
            return jsonify({'error': 'Invalid file format'}), 400

        # Database connection
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
        model_data = cursor.fetchone()
        
        if not model_data:
            return jsonify({'error': 'Model not found'}), 404

        # Load model with debugging
        start_load = time.time()
        model = get_model_from_db(model_data['id'], create_connection, MODEL_BASE_DIR)
        print(f"🕒 Model loading took: {time.time() - start_load:.2f}s")

        # Preprocessing with validation
        input_tensor = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(f"🔵 Input tensor shape: {input_tensor.shape}")
        print(f"🔵 Input range - Min: {input_tensor.min().item():.4f}, Max: {input_tensor.max().item():.4f}")

        # Model inference
        start_infer = time.time()
        with torch.no_grad():
            model.eval()
            output = model(input_tensor)
            
            # Raw output statistics
            print(f"🔴 Raw output stats:")
            print(f"Shape: {output.shape}")
            print(f"Min: {output.min().item():.4f}")
            print(f"Max: {output.max().item():.4f}")
            print(f"Mean: {output.mean().item():.4f}")
            print(f"Std: {output.std().item():.4f}")

            # Apply sigmoid
            probabilities = torch.sigmoid(output)
            print(f"🟢 Post-sigmoid stats:")
            print(f"Min: {probabilities.min().item():.4f}")
            print(f"Max: {probabilities.max().item():.4f}")
            print(f"Mean: {probabilities.mean().item():.4f}")

        print(f"🕒 Inference took: {time.time() - start_infer:.2f}s")

        # Save results
        result_filename = f"result_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Process and save mask
        try:
            # Convert to numpy and threshold
            mask_np = probabilities.squeeze().cpu().numpy()
            print(f"🟠 Numpy mask stats:")
            print(f"Shape: {mask_np.shape}")
            print(f"Min: {mask_np.min():.4f}")
            print(f"Max: {mask_np.max():.4f}")
            
            mask = (mask_np > 0.5).astype(np.uint8) * 255
            cv2.imwrite(save_path, mask)
            
            # Calculate metrics (NEW ADDITION)
            total_pixels = mask.size
            foreground_pixels = np.count_nonzero(mask)
            foreground_percent = (foreground_pixels / total_pixels) * 100
            print(f"✅ Mask saved with {foreground_percent:.2f}% foreground")
        except Exception as save_error:
            print(f"❌ Mask saving failed: {str(save_error)}")
            raise

        # Create overlay
        try:
            original = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            overlay = create_overlay(original, mask)
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], f"overlay_{filename}")
            cv2.imwrite(overlay_path, overlay)
        except Exception as overlay_error:
            print(f"⚠️ Overlay creation failed: {str(overlay_error)}")
            overlay_path = None

        # Prepare response (MODIFIED TO INCLUDE METRICS)
        response = {
            'original': url_for('uploaded_file', filename=filename),
            'processed': url_for('uploaded_file', filename=result_filename),
            'metrics': {
                'foreground_percent': round(foreground_percent, 2),
                'total_pixels': int(total_pixels),
                'foreground_pixels': int(foreground_pixels)
            }
        }
        if overlay_path:
            response['overlay'] = url_for('uploaded_file', filename=f"overlay_{filename}")

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"🔥 Processing error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected(): 
            connection.close()
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_model_from_db(model_id, connection_creator, model_base_dir):
    """Load trained U-Net model from database with rigorous checks"""
    connection = None
    cursor = None
    try:
        # 1. Database connection
        connection = connection_creator()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT model_path, model_format, model_type FROM models WHERE id = %s",
            (model_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"❌ Model ID {model_id} not found in database")

        # 2. Validate model file existence
        full_path = os.path.join(model_base_dir, result['model_path'])
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"🚨 Model file missing at {full_path}")

        print(f"🔍 Loading model from: {full_path}")

        # 3. Initialize fresh model (must match training architecture)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_unet().to(device)  # Your actual model class

        # 4. Advanced state dict handling
        try:
            checkpoint = torch.load(full_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, nn.Module):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint

            # Remove module prefix if present (for DDP trained models)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # Load with strict=False to handle potential mismatches
            load_result = model.load_state_dict(state_dict, strict=False)
            
            # Print missing/unexpected keys for debugging
            if load_result.missing_keys:
                print(f"⚠️ Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"⚠️ Unexpected keys: {load_result.unexpected_keys}")

        except Exception as e:
            raise RuntimeError(f"🔥 Failed to load weights: {str(e)}") from e

        # 5. Validate model initialization
        print("✅ Model loaded successfully")
        print(f"📐 Model architecture: {model.__class__.__name__}")
        
        # Debug: Print first conv layer weights
        first_conv = model.e1.conv.conv1.weight
        print(f"⚙️ First conv layer weights (mean±std): {first_conv.mean().item():.4f} ± {first_conv.std().item():.4f}")
        
        # Debug: Check for NaN/inf values
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"🤯 NaN values detected in {name}")
            if torch.isinf(param).any():
                raise ValueError(f"🤯 Inf values detected in {name}")

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Critical error loading model: {str(e)}")
        print("🛠️ Debugging info:")
        print(f"- Model path: {full_path}")
        print(f"- Checkpoint keys: {list(checkpoint.keys()) if 'checkpoint' in locals() else 'N/A'}")
        print(f"- Device: {device}")
        print(traceback.format_exc())
        raise RuntimeError("Model loading failed") from e

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def detect_architecture_from_state_dict(state_dict):
    """
    Analyze state_dict keys to determine the most likely architecture
    """
    keys = list(state_dict.keys())
    
    # Basic UNet detection
    if any('up' in k and 'conv' in k for k in keys) or any('down' in k for k in keys):
        return "UNet"
    
    # Default unknown
    return "unknown"


def save_uploaded_file(file):
    if file.filename == '':
        return None
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filename

def preprocess_image(image_path):
    """EXACT replica of training preprocessing"""
    # Load like DriveDataset
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize and normalize
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0  # No mean/std subtraction
    
    # Channel order
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    
    # Debug check
    print(f"\n🔵 Input Stats:")
    print(f"Min: {image.min():.4f}")
    print(f"Max: {image.max():.4f}")
    print(f"Mean: {image.mean():.4f}")
    
    return torch.from_numpy(image).unsqueeze(0)

def save_segmentation_result(output, save_path):
    """Process and save segmentation results with proper tensor handling"""
    try:
        # Ensure output is a tensor and apply sigmoid
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
            
        # Process output tensor
        output = output.detach().cpu()
        probabilities = torch.sigmoid(output)
        
        # Convert to numpy array for OpenCV
        mask_np = probabilities.numpy().squeeze()  # Remove batch and channel dims
        
        # Threshold and scale
        mask = (mask_np > 0.5).astype(np.uint8) * 255
        
        # Save result
        cv2.imwrite(save_path, mask)
        
        # Debug output
        print(f"Processed mask - Foreground %: {(mask > 0).mean() * 100:.2f}%")

    except Exception as e:
        print(f"❌ Error saving segmentation result: {str(e)}")
        # Create error image
        error_img = np.zeros((512, 512), dtype=np.uint8)
        cv2.putText(error_img, "Processing Error", (50, 256), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imwrite(save_path, error_img)
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