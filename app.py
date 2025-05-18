from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, send_from_directory
from database import create_connection
import mysql.connector
from mysql.connector import Error
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
from PIL import Image
import traceback
import cv2
from werkzeug.security import generate_password_hash, check_password_hash

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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class DownWithResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_residual = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_residual(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.noise_estimator = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x, skip):
        # Implémentation simplifiée de l'attention
        batch, channels, _, _ = x.size()
        noise = self.noise_estimator(x.mean([2,3]))
        combined = torch.cat([skip, x], dim=1)
        attention = torch.sigmoid(self.fc2(F.relu(self.fc1(combined))))
        return x * attention

class UpWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.attention = AttentionBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x, skip)
        return x

class ASE_Res_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownWithResidual(64, 128)
        self.down2 = DownWithResidual(128, 256)
        self.down3 = DownWithResidual(256, 512)
        self.down4 = DownWithResidual(512, 1024)
        
        # Decoder
        self.up3 = UpWithAttention(1024, 512)
        self.up2 = UpWithAttention(512, 256)
        self.up1 = UpWithAttention(256, 128)
        self.up0 = UpWithAttention(128, 64)
        
        # Final layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up3(x5, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)
        
        return torch.sigmoid(self.outc(x))

class RobustAdaptiveUNet(nn.Module):
    def __init__(self, encoder_channels=None, decoder_channels=None, in_channels=3, out_channels=1):
        super().__init__()

        # Default channel structure if not inferred
        self.encoder_channels = encoder_channels or [64, 128, 256, 512]
        self.decoder_channels = decoder_channels or list(reversed(self.encoder_channels))

        # Initial conv
        self.inc = DoubleConv(in_channels, self.encoder_channels[0])

        # Down path
        self.downs = nn.ModuleList()
        for i in range(len(self.encoder_channels) - 1):
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(self.encoder_channels[i], self.encoder_channels[i + 1])
            ))

        # Up path
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(len(self.decoder_channels) - 1):
            self.ups.append(nn.ConvTranspose2d(self.decoder_channels[i], self.decoder_channels[i + 1], kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(self.decoder_channels[i + 1] + self.encoder_channels[-(i + 2)], self.decoder_channels[i + 1]))

        # Final conv
        self.outc = nn.Conv2d(self.decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))

        x = features[-1]

        for i in range(len(self.ups)):
            x = self.ups[i](x)

            # Resize if necessary (handle mismatched feature map sizes)
            skip = features[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([x, skip], dim=1)
            x = self.up_convs[i](x)

        return torch.sigmoid(self.outc(x))


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


@app.route('/process-image', methods=['POST'])
def process_image():
    connection = None
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

        # Load the model with improved loading logic
        model = get_model_from_db(model_data['id'])
        model.eval()
        
        # Preprocess image and prepare for model
        input_tensor = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Free GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process image with model
        try:
            with torch.no_grad():
                output = model(input_tensor)
                
                # Log output information for debugging
                if isinstance(output, dict):
                    app.logger.info(f"Model returned dictionary with keys: {output.keys()}")
                    if 'out' in output:
                        output_tensor = output['out']
                    else:
                        output_tensor = list(output.values())[0]  # Take first value
                else:
                    output_tensor = output
                
                app.logger.info(f"Output shape: {output_tensor.shape}")
                app.logger.info(f"Output min: {output_tensor.min().item()}, max: {output_tensor.max().item()}, mean: {output_tensor.mean().item()}")
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                # Fallback to CPU if CUDA OOM
                torch.cuda.empty_cache()
                device = torch.device('cpu')
                model = model.to(device)
                input_tensor = input_tensor.to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
            else:
                raise e

        # Save results
        result_filename = f"result_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Save result based on model type
        if model_data['model_type'] == 'segmentation':
            save_segmentation_result(output, save_path)
        else:
            return jsonify({'error': 'Classification not implemented'}), 501

        # Create overlay version for better visualization
        overlay_filename = f"overlay_{filename}"
        try:
            # Load original and mask
            original = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            mask = cv2.imread(save_path)
            
            # Resize if needed
            if original.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
            
            # Create overlay
            alpha = 0.6
            overlay = cv2.addWeighted(original, 1-alpha, mask, alpha, 0)
            
            # Save overlay
            overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
            cv2.imwrite(overlay_path, overlay)
        except Exception as overlay_error:
            app.logger.warning(f"Could not create overlay: {str(overlay_error)}")
            overlay_filename = None

        # Log the processing in the database
        try:
            cursor.execute("""
                INSERT INTO user_history (user_id, model_id, image_path, result_path)
                VALUES (%s, %s, %s, %s)
            """, (session.get('user_id'), model_id, filename, result_filename))
            connection.commit()
        except Exception as db_error:
            app.logger.warning(f"Could not log to history: {str(db_error)}")

        # Return paths to the processed images
        response = {
            'original': url_for('uploaded_file', filename=filename),
            'processed': url_for('uploaded_file', filename=result_filename)
        }
        
        # Add overlay if available
        if overlay_filename:
            response['overlay'] = url_for('uploaded_file', filename=overlay_filename)

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        app.logger.error(traceback.format_exc())  
        if connection: connection.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected(): 
            connection.close()

def get_model_from_db(model_id):
    """
    Load model from database with improved architecture detection and error handling
    """
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT model_path, model_format, model_type FROM models WHERE id = %s", (model_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError("Model not found")

        full_path = os.path.join(MODEL_BASE_DIR, result['model_path'])
        print(f"Loading model from: {full_path}")

        # Check file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found at {full_path}")

        # Load the state dict or model
        try:
            state_dict = torch.load(full_path, map_location='cpu')
            print(f"Successfully loaded state_dict with {len(state_dict)} keys")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Check if it's a complete model or state_dict
        if isinstance(state_dict, nn.Module):
            print("✅ Loaded complete model")
            return state_dict

        # Extract model architecture info from state_dict
        architecture_type = detect_architecture_from_state_dict(state_dict)
        print(f"Detected architecture: {architecture_type}")

        if architecture_type == "deeplabv3":
            from torchvision.models.segmentation import deeplabv3_resnet101
    
            # Try to infer the number of classes
            num_classes = 1
            for key, value in state_dict.items():
                if 'classifier.4.weight' in key:
                    num_classes = value.shape[0]
                    print(f"✅ Inferred num_classes: {num_classes}")
                    break

            model = deeplabv3_resnet101(weights=None, num_classes=num_classes)
            strict = False


        elif architecture_type == "ASE_Res_UNet":
            model = ASE_Res_UNet()
            strict = True
        elif architecture_type == "UNet":
            model = UNet()
            strict = True
        else:
            print("⚠ Using RobustAdaptiveUNet for unknown architecture")
            model = create_robust_adaptive_unet_from_state_dict(state_dict)
            strict = False

        # Try to load with adjusted keys if needed
        try:
            incompatible = model.load_state_dict(state_dict, strict=strict)
            print(f"Model loaded with {'no' if not incompatible else 'some'} incompatible keys")
            if not strict and incompatible:
                print(f"Missing keys: {len(incompatible.missing_keys)}, Unexpected keys: {len(incompatible.unexpected_keys)}")
        except Exception as e:
            print(f"Error during state_dict loading: {str(e)}")
            print("🔄 Trying adjusted key mapping...")
            
            # Try with adjusted keys
            mapped_state_dict = map_state_dict_keys(state_dict, model)
            incompatible = model.load_state_dict(mapped_state_dict, strict=False)
            print(f"Model loaded with adjusted keys. Missing: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}")

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        print(traceback.format_exc())
        print("⚠ Falling back to SuperSafeUNet")
        return create_super_safe_unet()
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected():
            connection.close()

def detect_architecture_from_state_dict(state_dict):
    """
    Analyze state_dict keys to determine the most likely architecture
    """
    keys = list(state_dict.keys())
    
    # DeepLabV3 detection
    if any('classifier.4' in k for k in keys) or any('classifier.0' in k for k in keys):
        return "deeplabv3"
    
    # ASE_Res_UNet detection - has attention blocks
    if any('attention' in k for k in keys):
        return "ASE_Res_UNet"
    
    # Basic UNet detection
    if any('up' in k and 'conv' in k for k in keys) or any('down' in k for k in keys):
        return "UNet"
    
    # Default unknown
    return "unknown"

def create_adaptive_unet():
    """
    Create a flexible UNet model that can adapt to different architectures
    """
    class AdaptiveUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            
            # Encoder path with different channel sizes
            self.inc = DoubleConv(in_channels, 64)
            
            self.down1 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(64, 128)
            )
            
            self.down2 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(128, 256)
            )
            
            self.down3 = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(256, 512)
            )
            
            # Decoder path
            self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.conv_up3 = DoubleConv(512, 256)  # 512 = 256 + 256 (skip connection)
            
            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(256, 128)  # 256 = 128 + 128 (skip connection)
            
            self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(128, 64)  # 128 = 64 + 64 (skip connection)
            
            # Final layer
            self.outc = nn.Conv2d(64, out_channels, 1)

        def forward(self, x):
            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            
            # Decoder with skip connections
            x = self.up3(x4)
            # Handle different sizes for skip connections
            if x.shape[2:] != x3.shape[2:]:
                x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x3, x], dim=1)
            x = self.conv_up3(x)
            
            x = self.up2(x)
            if x.shape[2:] != x2.shape[2:]:
                x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x2, x], dim=1)
            x = self.conv_up2(x)
            
            x = self.up1(x)
            if x.shape[2:] != x1.shape[2:]:
                x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x1, x], dim=1)
            x = self.conv_up1(x)
            
            return torch.sigmoid(self.outc(x))
    
    return AdaptiveUNet()

def map_state_dict_keys(state_dict, model):
    """
    Attempt to map keys from state_dict to the target model structure
    """
    new_state_dict = {}
    model_keys = dict(model.named_parameters())
    
    # Common patterns to try mapping
    for old_key, value in state_dict.items():
        # Try direct match first
        if old_key in model_keys:
            new_state_dict[old_key] = value
            continue
        
        # Try to handle different encoder naming
        if old_key.startswith('inc.') and 'inc.' in str(model_keys):
            new_key = old_key
            new_state_dict[new_key] = value
        elif old_key.startswith('down1.'):
            if 'down1.' in str(model_keys):
                new_key = old_key
            else:
                new_key = old_key.replace('down1.', 'down1.1.')
            new_state_dict[new_key] = value
        elif old_key.startswith('down2.'):
            if 'down2.' in str(model_keys):
                new_key = old_key
            else:
                new_key = old_key.replace('down2.', 'down2.1.')
            new_state_dict[new_key] = value
        elif old_key.startswith('down3.'):
            if 'down3.' in str(model_keys):
                new_key = old_key
            else:
                new_key = old_key.replace('down3.', 'down3.1.')
            new_state_dict[new_key] = value
        
        # Try to handle different decoder naming
        elif old_key.startswith('up1.'):
            if old_key.startswith('up1.up.'):
                new_key = old_key.replace('up1.up.', 'up1.')
            elif old_key.startswith('up1.conv.'):
                new_key = old_key.replace('up1.conv.', 'conv_up1.')
            else:
                new_key = old_key
            new_state_dict[new_key] = value
        elif old_key.startswith('up2.'):
            if old_key.startswith('up2.up.'):
                new_key = old_key.replace('up2.up.', 'up2.')
            elif old_key.startswith('up2.conv.'):
                new_key = old_key.replace('up2.conv.', 'conv_up2.')
            else:
                new_key = old_key
            new_state_dict[new_key] = value
        elif old_key.startswith('up3.'):
            if old_key.startswith('up3.up.'):
                new_key = old_key.replace('up3.up.', 'up3.')
            elif old_key.startswith('up3.conv.'):
                new_key = old_key.replace('up3.conv.', 'conv_up3.')
            else:
                new_key = old_key
            new_state_dict[new_key] = value
        elif old_key.startswith('outc.'):
            new_state_dict[old_key] = value
        else:
            # Add without modification as fallback
            new_state_dict[old_key] = value
    
    return new_state_dict

def create_super_safe_unet():
    """Create an extremely simple and safe UNet model that is guaranteed to work with any input"""
    class SuperSafeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Very simple encoder - just map to features
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            
            # Very simple decoder - directly to output (no skip connections to worry about)
            self.decoder = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(8, 1, 1)  # Direct 1-channel output
            )
            
        def forward(self, x):
            # Super simple forward pass without any complex channel logic
            x = self.encoder(x)
            x = self.decoder(x)
            return torch.sigmoid(x)
    
    return SuperSafeUNet()



    """Create a model that adapts to the state_dict's structure by analyzing it first"""
    # Analyze state_dict to determine architecture
    up_layers = {}
    in_channels = 3  # Default input channels
    out_channels = 1  # Default output channels
    
    # Extract key info from state dict
    for key in state_dict.keys():
        if 'up' in key and '.weight' in key and 'conv' not in key:
            parts = key.split('.')
            layer_name = parts[0]  # e.g., 'up1', 'up2'
            if layer_name not in up_layers:
                up_layers[layer_name] = {}
                
            if parts[-1] == 'weight':
                shape = state_dict[key].shape
                up_layers[layer_name]['shape'] = shape
                
                # For transposed conv layers
                if len(shape) == 4:  # Conv or ConvTranspose weights
                    in_ch, out_ch = shape[1], shape[0]  # For ConvTranspose: out_channels, in_channels/groups, kH, kW
                    up_layers[layer_name]['in_ch'] = in_ch
                    up_layers[layer_name]['out_ch'] = out_ch
    
    # Define an adaptive UNet class based on the analysis
    class AdaptiveUNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Determine encoder-decoder structure based on analysis
            max_layer_num = max([int(layer_name.replace('up', '')) for layer_name in up_layers]) if up_layers else 4
            
            # Default channel config if we can't determine from state_dict
            channels = [64, 128, 256, 512, 1024][:max_layer_num+1]
            
            # Override with detected channels if available
            for i in range(1, max_layer_num+1):
                layer_name = f'up{i}'
                if layer_name in up_layers and 'out_ch' in up_layers[layer_name]:
                    if i < len(channels):
                        channels[i] = up_layers[layer_name]['out_ch']
            
            # Encoder
            self.inc = DoubleConv(in_channels, channels[0])
            self.downs = nn.ModuleList()
            for i in range(max_layer_num):
                self.downs.append(nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(channels[i], channels[i+1])
                ))
            
            # Decoder
            self.ups = nn.ModuleList()
            for i in range(max_layer_num, 0, -1):
                self.ups.append(nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=2, stride=2),
                    DoubleConv(channels[i], channels[i-1])  # After concatenation
                ))
            
            # Final convolution
            self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)
            
        def forward(self, x):
            # Store encoder outputs for skip connections
            features = [self.inc(x)]
            
            # Encoder path
            for down in self.downs:
                features.append(down(features[-1]))
            
            # Start with the bottleneck
            x = features[-1]
            
            # Decoder path with skip connections
            for i, up in enumerate(self.ups):
                # Apply transposed convolution
                x = up[0](x)
                
                # Get corresponding encoder feature map
                skip_connection = features[-(i+2)]
                
                # Handle potential size mismatches
                if x.shape[2:] != skip_connection.shape[2:]:
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
                
                # Concatenate skip connection
                x = torch.cat([skip_connection, x], dim=1)
                
                # Apply convolutions after concatenation
                x = up[1](x)
            
            return torch.sigmoid(self.outc(x))
    
    # Create model and load state dict with flexible settings
    model = AdaptiveUNet()
    
    # Modify state_dict keys if necessary to match our model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('up') and '.up.' in key:
            # Handle transposed convolution weights
            parts = key.split('.')
            layer_num = int(parts[0].replace('up', ''))
            new_key = f'ups.{len(model.ups) - layer_num}.0.{".".join(parts[2:])}'
            new_state_dict[new_key] = value
        elif key.startswith('up') and '.conv.' in key:
            # Handle conv weights after skip connections
            parts = key.split('.')
            layer_num = int(parts[0].replace('up', ''))
            new_key = f'ups.{len(model.ups) - layer_num}.1.{".".join(parts[2:])}'
            new_state_dict[new_key] = value
        elif key.startswith('inc'):
            new_state_dict[key] = value
        elif key.startswith('down'):
            parts = key.split('.')
            layer_num = int(parts[0].replace('down', ''))
            if len(parts) > 2:
                new_key = f'downs.{layer_num-1}.1.{".".join(parts[2:])}'
            else:
                new_key = f'downs.{layer_num-1}.{".".join(parts[1:])}'
            new_state_dict[new_key] = value
        elif key.startswith('outc'):
            new_state_dict[key] = value
    
    # Try to load with the new state dict, fall back to original if needed
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except:
        model.load_state_dict(state_dict, strict=False)
    
    return model

def create_adaptive_unet(state_dict):
    """Create a model that adapts to the state_dict's structure by analyzing it first"""
    # Analyze state_dict to determine architecture
    up_layers = {}
    in_channels = 3  # Default input channels
    out_channels = 1  # Default output channels
    
    # Analyze output layer first
    out_layer_key = None
    for key in state_dict.keys():
        if 'outc' in key and '.weight' in key:
            out_layer_key = key
            out_layer_shape = state_dict[key].shape
            if len(out_layer_shape) == 4:  # Conv weights shape is [out_channels, in_channels, kH, kW]
                in_channels_for_out = out_layer_shape[1]  # This will be our final encoder channel count
                out_channels = out_layer_shape[0]  # This is our output channel count
    
    # Extract key info from state dict
    for key in state_dict.keys():
        if 'up' in key and '.weight' in key and 'conv' not in key:
            parts = key.split('.')
            layer_name = parts[0]  # e.g., 'up1', 'up2'
            if layer_name not in up_layers:
                up_layers[layer_name] = {}
                
            if parts[-1] == 'weight':
                shape = state_dict[key].shape
                up_layers[layer_name]['shape'] = shape
                
                # For transposed conv layers
                if len(shape) == 4:  # Conv or ConvTranspose weights
                    in_ch, out_ch = shape[1], shape[0]  # For ConvTranspose: out_channels, in_channels/groups, kH, kW
                    up_layers[layer_name]['in_ch'] = in_ch
                    up_layers[layer_name]['out_ch'] = out_ch
    
    # Define an adaptive UNet class based on the analysis
    class AdaptiveUNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Determine encoder-decoder structure based on analysis
            max_layer_num = max([int(layer_name.replace('up', '')) for layer_name in up_layers]) if up_layers else 4
            
            # Default channel config if we can't determine from state_dict
            channels = [64, 128, 256, 512, 1024][:max_layer_num+1]
            
            # Override with detected channels if available
            for i in range(1, max_layer_num+1):
                layer_name = f'up{i}'
                if layer_name in up_layers and 'out_ch' in up_layers[layer_name]:
                    if i < len(channels):
                        channels[i] = up_layers[layer_name]['out_ch']
            
            # Ensure proper channels for output layer if detected
            if 'in_channels_for_out' in locals():
                channels[0] = in_channels_for_out
            
            # Encoder
            self.inc = DoubleConv(in_channels, channels[0])
            self.downs = nn.ModuleList()
            for i in range(max_layer_num):
                self.downs.append(nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(channels[i], channels[i+1])
                ))
            
            # Decoder with special handling for the "concat" channel issue
            self.ups = nn.ModuleList()
            for i in range(max_layer_num, 0, -1):
                # For upsampling we use standard channels
                self.ups.append(nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=2, stride=2),
                    # Critical: we use channels[i-1]*2 as input to handle the concatenation
                    DoubleConv(channels[i-1]*2, channels[i-1])
                ))
            
            # Final convolution
            self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)
            
        def forward(self, x):
            # Store encoder outputs for skip connections
            features = [self.inc(x)]
            
            # Encoder path
            for down in self.downs:
                features.append(down(features[-1]))
            
            # Start with the bottleneck
            x = features[-1]
            
            # Decoder path with skip connections
            for i, up in enumerate(self.ups):
                # Apply transposed convolution
                x = up[0](x)
                
                # Get corresponding encoder feature map
                skip_connection = features[-(i+2)]
                
                # Handle potential size mismatches
                if x.shape[2:] != skip_connection.shape[2:]:
                    x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
                
                # Concatenate skip connection
                x = torch.cat([skip_connection, x], dim=1)
                
                # Apply convolutions after concatenation
                x = up[1](x)
            
            return torch.sigmoid(self.outc(x))
    
    # Create model and load state dict with flexible settings
    model = AdaptiveUNet()
    
    # Modify state_dict keys if necessary to match our model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('up') and '.up.' in key:
            # Handle transposed convolution weights
            parts = key.split('.')
            layer_num = int(parts[0].replace('up', ''))
            new_key = f'ups.{len(model.ups) - layer_num}.0.{".".join(parts[2:])}'
            new_state_dict[new_key] = value
        elif key.startswith('up') and '.conv.' in key:
            # Handle conv weights after skip connections
            parts = key.split('.')
            layer_num = int(parts[0].replace('up', ''))
            new_key = f'ups.{len(model.ups) - layer_num}.1.{".".join(parts[2:])}'
            new_state_dict[new_key] = value
        elif key.startswith('inc'):
            new_state_dict[key] = value
        elif key.startswith('down'):
            parts = key.split('.')
            layer_num = int(parts[0].replace('down', ''))
            if len(parts) > 2:
                new_key = f'downs.{layer_num-1}.1.{".".join(parts[2:])}'
            else:
                new_key = f'downs.{layer_num-1}.{".".join(parts[1:])}'
            new_state_dict[new_key] = value
        elif key.startswith('outc'):
            new_state_dict[key] = value
    
    # Try to load with the new state dict, fall back to original if needed
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except:
        model.load_state_dict(state_dict, strict=False)
    
    return model

def create_compatible_unet(state_dict):
    """Create a model with architecture matching the state_dict's dimensions"""
    # Inspect the state dictionary to determine dimensions
    channels = {}
    
    # Try to extract dimensions from the state dict
    for key, param in state_dict.items():
        if 'up4.up.weight' in key:
            channels['up4_in'] = param.shape[0]
            channels['up4_out'] = param.shape[1]
        elif 'up3.up.weight' in key:
            channels['up3_in'] = param.shape[0]
            channels['up3_out'] = param.shape[1]
        elif 'up2.up.weight' in key:
            channels['up2_in'] = param.shape[0]
            channels['up2_out'] = param.shape[1]
        elif 'up1.up.weight' in key:
            channels['up1_in'] = param.shape[0]
            channels['up1_out'] = param.shape[1]
    
    # Create a flexible model class
    class FlexibleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # We'll use the dimensions extracted from state_dict
            self.down1 = DoubleConv(3, channels.get('up1_out', 64))
            self.down2 = DoubleConv(channels.get('up1_out', 64), channels.get('up2_out', 128))
            self.down3 = DoubleConv(channels.get('up2_out', 128), channels.get('up3_out', 256))
            self.down4 = DoubleConv(channels.get('up3_out', 256), channels.get('up4_out', 512))
            self.pool = nn.MaxPool2d(2)
            
            # Upsampling path
            self.up4 = nn.ConvTranspose2d(channels.get('up4_in', 512), channels.get('up4_out', 512), 2, stride=2)
            self.up3 = nn.ConvTranspose2d(channels.get('up3_in', 256), channels.get('up3_out', 256), 2, stride=2)
            self.up2 = nn.ConvTranspose2d(channels.get('up2_in', 128), channels.get('up2_out', 128), 2, stride=2)
            self.up1 = nn.ConvTranspose2d(channels.get('up1_in', 64), channels.get('up1_out', 64), 2, stride=2)
            
            # Conv paths after upsampling
            self.conv_up4 = DoubleConv(channels.get('up4_out', 512) + channels.get('up3_out', 256), 
                                      channels.get('up3_in', 256))
            self.conv_up3 = DoubleConv(channels.get('up3_out', 256) + channels.get('up2_out', 128), 
                                      channels.get('up2_in', 128))
            self.conv_up2 = DoubleConv(channels.get('up2_out', 128) + channels.get('up1_out', 64), 
                                      channels.get('up1_in', 64))
            self.conv_up1 = DoubleConv(channels.get('up1_out', 64) * 2, channels.get('up1_out', 64))
            
            # Final convolution
            self.conv_last = nn.Conv2d(channels.get('up1_out', 64), 1, 1)

        def forward(self, x):
            # Encoder
            d1 = self.down1(x)
            x = self.pool(d1)
            d2 = self.down2(x)
            x = self.pool(d2)
            d3 = self.down3(x)
            x = self.pool(d3)
            d4 = self.down4(x)
            x = self.pool(d4)
            
            # Bridge and decoder
            x = self.up4(x)
            x = torch.cat([x, d4], dim=1)
            x = self.conv_up4(x)
            
            x = self.up3(x)
            x = torch.cat([x, d3], dim=1)
            x = self.conv_up3(x)
            
            x = self.up2(x)
            x = torch.cat([x, d2], dim=1)
            x = self.conv_up2(x)
            
            x = self.up1(x)
            x = torch.cat([x, d1], dim=1)
            x = self.conv_up1(x)
            
            return torch.sigmoid(self.conv_last(x))
    
    # Create model and load state dict without strict checks
    model = FlexibleUNet()
    model.load_state_dict(state_dict, strict=False)
    return model

def create_simple_unet():
    """Create a simplified UNet model as a fallback option"""
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Minimal encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            
            # Minimal decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(32, 1, 1)
            )
            
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return torch.sigmoid(x)
    
    return SimpleUNet()

def create_legacy_ase_res_unet():
    """Create a legacy version of ASE_Res_UNet that matches the saved model architecture"""
    class LegacyASE_Res_UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            
            # Encoder
            self.inc = DoubleConv(in_channels, 64)
            self.down1 = DownWithResidual(64, 128)
            self.down2 = DownWithResidual(128, 256)
            self.down3 = DownWithResidual(256, 512)
            self.down4 = DownWithResidual(512, 1024)
            
            # Decoder - note the naming difference (up4 instead of up0)
            self.up4 = UpWithAttention(1024, 512)
            self.up3 = UpWithAttention(512, 256)
            self.up2 = UpWithAttention(256, 128)
            self.up1 = UpWithAttention(128, 64)
            
            # Final layer with a different structure
            self.outc = nn.Sequential(
                nn.Conv2d(64, out_channels, 1)
            )

        def forward(self, x):
            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Decoder
            x = self.up4(x5, x4)
            x = self.up3(x, x3)
            x = self.up2(x, x2)
            x = self.up1(x, x1)
            
            return torch.sigmoid(self.outc(x))
            
    return LegacyASE_Res_UNet()

def create_robust_adaptive_unet_from_state_dict(state_dict, in_channels=3, out_channels=1):
    """
    Build a RobustAdaptiveUNet based on the encoder conv weights in the state_dict
    """
    encoder_channels = []
    for k, v in state_dict.items():
        if 'conv' in k and 'weight' in k and len(v.shape) == 4:
            out_ch = v.shape[0]
            if out_ch not in encoder_channels:
                encoder_channels.append(out_ch)
            if len(encoder_channels) >= 4:
                break

    if not encoder_channels:
        encoder_channels = [64, 128, 256, 512]

    decoder_channels = list(reversed(encoder_channels))
    return RobustAdaptiveUNet(encoder_channels, decoder_channels, in_channels, out_channels)


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
            
        # Use torchvision.transforms instead of torch.nn
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Apply transformations and add batch dimension
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor
        
    except Exception as e:
        app.logger.error(f"Image preprocessing error: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise ValueError(f"Image processing failed: {str(e)}")


def save_segmentation_result(output, save_path):
    """
    Process model output and save segmentation result with improved visualization
    """
    import numpy as np
    import cv2
    from PIL import Image
    import os

    try:
        # Load original image to get size
        original_path = save_path.replace('result_', '')
        if not os.path.exists(original_path):
            print(f"[WARNING] Original image not found at {original_path}")
            # Use output size if original not available
            width, height = output.shape[-1], output.shape[-2]
        else:
            original_img = Image.open(original_path).convert('RGB')
            width, height = original_img.size

        # Process output tensor
        # Handle DeepLabV3-style output
        if isinstance(output, dict) and 'out' in output:
            output = output['out']

        output = output.detach().cpu()

        print(f"[INFO] Raw model output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")

        # Handle DeepLabV3 format where output is in a dictionary
        if isinstance(output, dict) and 'out' in output:
            output = output['out']

        # Handle different output structures
        if output.dim() == 4:  # [B, C, H, W]
            if output.shape[1] > 1:  # Multi-class segmentation
                output = output.squeeze(0)  # Remove batch dimension
                output = torch.argmax(output, dim=0)  # Get class indices
            else:  # Binary segmentation
                output = output.squeeze(0).squeeze(0)  # Remove extra dimensions
        elif output.dim() == 3 and output.shape[0] > 1:  # [C, H, W]
            output = torch.argmax(output, dim=0)  # Get class indices
        elif output.dim() == 3 and output.shape[0] == 1:  # [1, H, W]
            output = output.squeeze(0)  # Remove channel dimension

        # Convert to numpy
        output_np = output.numpy()
        
        # Diagnostic info
        print(f"[INFO] Processed output shape: {output_np.shape}")
        print(f"[INFO] Output range: min={output_np.min()}, max={output_np.max()}")
        print(f"[INFO] Unique values: {np.unique(output_np)}")

        # Resize to original image dimensions
        output_resized = cv2.resize(output_np, (width, height), 
                                   interpolation=cv2.INTER_NEAREST)

        # Create colored output for better visualization
        if output_resized.max() <= 1.0 and output_resized.dtype != np.uint8:
            # Likely probabilities - convert to binary
            binary_mask = (output_resized > 0.5).astype(np.uint8) * 255
            
            # Create a colored overlay
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            colored_mask[..., 0] = binary_mask  # Blue channel
            colored_mask[..., 1] = 0  # Green channel
            colored_mask[..., 2] = binary_mask  # Red channel
            
            # Save both binary and colored versions
            Image.fromarray(binary_mask).save(save_path.replace('.', '_binary.'))
            Image.fromarray(colored_mask).save(save_path)
            print("[INFO] Saved binary and colored masks")
        else:
            # Handle multi-class segmentation
            # Define a colormap (up to 20 classes)
            colormap = np.array([
                [0, 0, 0],        # Class 0 - Background (black)
                [255, 0, 0],      # Class 1 - Red
                [0, 255, 0],      # Class 2 - Green
                [0, 0, 255],      # Class 3 - Blue
                [255, 255, 0],    # Class 4 - Yellow
                [255, 0, 255],    # Class 5 - Magenta
                [0, 255, 255],    # Class 6 - Cyan
                [128, 0, 0],      # Class 7 - Dark red
                [0, 128, 0],      # Class 8 - Dark green
                [0, 0, 128],      # Class 9 - Dark blue
                [128, 128, 0],    # Class 10 - Olive
                [128, 0, 128],    # Class 11 - Purple
                [0, 128, 128],    # Class 12 - Teal
                [128, 128, 128],  # Class 13 - Gray
                [64, 0, 0],       # Class 14 - Maroon
                [192, 0, 0],      # Class 15 - Crimson
                [64, 128, 0],     # Class 16 - Forest green
                [192, 128, 0],    # Class 17 - Orange
                [64, 0, 128],     # Class 18 - Indigo
                [192, 0, 128],    # Class 19 - Pink
            ], dtype=np.uint8)

            # Ensure colormap has enough colors
            max_class = int(np.ceil(output_resized.max()))
            if max_class >= len(colormap):
                more_colors = np.random.randint(0, 255, size=(max_class + 1 - len(colormap), 3), dtype=np.uint8)
                colormap = np.vstack([colormap, more_colors])

            # Create color-coded segmentation mask
            output_resized = output_resized.astype(np.int32)
            color_mask = colormap[output_resized]
            
            # If original image exists, create overlay
            if os.path.exists(original_path):
                try:
                    # Read original image
                    original = cv2.imread(original_path)
                    original = cv2.resize(original, (width, height))
                    
                    # Create a 50% blend of original and mask
                    alpha = 0.7  # Transparency factor
                    overlay = cv2.addWeighted(original, 1-alpha, color_mask, alpha, 0)
                    
                    # Save the overlay version
                    cv2.imwrite(save_path.replace('.', '_overlay.'), overlay)
                    print("[INFO] Saved segmentation overlay")
                except Exception as overlay_error:
                    print(f"[WARNING] Overlay creation failed: {str(overlay_error)}")

            # Save color-coded segmentation mask
            Image.fromarray(color_mask).save(save_path)
            print("[INFO] Saved color-coded segmentation mask")

    except Exception as e:
        print(f"[ERROR] Saving segmentation result failed: {str(e)}")
        print(traceback.format_exc())
        
        # Create a simple fallback image in case of errors
        try:
            fallback_img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(fallback_img, "Error processing", (20, 128), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imwrite(save_path, fallback_img)
        except:
            pass


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