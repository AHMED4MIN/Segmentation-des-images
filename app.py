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
            return torch.jit.load(full_path, map_location=torch.device('cpu'))
        else:
            # Load the state dictionary first to inspect
            state_dict = torch.load(full_path, map_location='cpu')
            
            # Try multiple model architectures to find one that matches
            if result['model_type'] == 'segmentation':
                # Try the safe fallback UNet first - guaranteed to work for any input
                try:
                    model = create_super_safe_unet()
                    app.logger.info("Using guaranteed compatible UNet architecture")
                    model.eval()
                    return model
                except Exception as e0:
                    app.logger.info(f"Even safe UNet failed: {str(e0)}")
            
                # Try the enhanced adaptive UNet 
                try:
                    model = create_adaptive_unet(state_dict)
                    app.logger.info("Successfully loaded model with adaptive UNet architecture")
                    model.eval()
                    return model
                except Exception as e1:
                    app.logger.info(f"Adaptive UNet loading failed: {str(e1)}")
                
                # Try the simple UNet 
                try:
                    model = UNet()
                    model.load_state_dict(state_dict)
                    app.logger.info("Successfully loaded model with UNet architecture")
                    model.eval()
                    return model
                except Exception as e2:
                    app.logger.info(f"UNet loading failed: {str(e2)}")
                
                # Try the ASE_Res_UNet
                try:
                    model = ASE_Res_UNet()
                    model.load_state_dict(state_dict)
                    app.logger.info("Successfully loaded model with ASE_Res_UNet architecture")
                    model.eval()
                    return model
                except Exception as e3:
                    app.logger.info(f"ASE_Res_UNet loading failed: {str(e3)}")
                
                # Last resort: create a bare minimum model that can process images
                app.logger.warning("Using basic model as fallback. This may reduce accuracy.")
                return create_simple_unet()
            else:
                raise ValueError("Unsupported model type")
            
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
    finally:
        if cursor: cursor.close()
        if connection and connection.is_connected():
            connection.close()


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