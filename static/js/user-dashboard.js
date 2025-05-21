// user-dashboard.js
document.addEventListener('DOMContentLoaded', () => {
    initializeEventHandlers();
    resetImageDisplays();
    loadProcessingView('segmentation');
});

function initializeEventHandlers() {
    // Gestion du téléchargement d'image
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    
    dropZone.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', handleFileSelect);
    
    // Gestion drag and drop
    ['dragover', 'dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, preventDefaults);
    });

    dropZone.addEventListener('dragover', highlightDropZone);
    dropZone.addEventListener('dragleave', unhighlightDropZone);
    dropZone.addEventListener('drop', handleDrop);

    // Bouton de traitement
    document.getElementById('processBtn').addEventListener('click', processImage);

    // Gestion des boutons de navigation
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.addEventListener('click', switchProcessType);
    });

    // Upload de modèle personnalisé
    document.getElementById('uploadModelBtn').addEventListener('click', () => {
        document.getElementById('modelUpload').click();
    });

    document.getElementById('modelUpload').addEventListener('change', handleModelUpload);
}

function switchProcessType(e) {
    document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    const processType = e.target.dataset.type;
    loadProcessingView(processType);
}

function resetImageDisplays() {
    const transparent = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
    document.getElementById('originalResult').src = transparent;
    document.getElementById('processedResult').src = transparent;
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlightDropZone() {
    document.getElementById('dropZone').classList.add('dragover');
}

function unhighlightDropZone() {
    document.getElementById('dropZone').classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    document.getElementById('imageInput').files = files;
    handleFileSelect({ target: { files } });
}

function handleFileSelect(e) {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];

    if (!file || !file.type.startsWith('image/')) {
        alert('Veuillez télécharger une image valide');
        resetFileInput();
        return;
    }
    showUploadSuccess();
}

function showUploadSuccess() {
    const dropZone = document.getElementById('dropZone');
    dropZone.classList.add('upload-success');
    setTimeout(() => dropZone.classList.remove('upload-success'), 2000);
}

function resetFileInput() {
    document.getElementById('imageInput').value = '';
}

async function processImage() {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];
    const modelId = document.getElementById('modelSelect').value;
    
    if (!file || !modelId) {
        alert('Veuillez sélectionner une image et un modèle');
        return;
    }

    toggleLoading(true);

    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('model_id', modelId);

        const response = await fetch('/process-image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP ! statut : ${response.status}`);
        }

        const result = await response.json();
        console.log('Server Response:', result);
        if (result.error) {
            throw new Error(result.error);
        }

        document.getElementById('originalResult').src = result.original;
        document.getElementById('processedResult').src = result.processed;
        if (result.metrics) {
            document.getElementById('foregroundPercent').textContent = 
                `${result.metrics.foreground_percent}%`;
            document.getElementById('foregroundPixels').textContent = 
                `${result.metrics.foreground_pixels.toLocaleString()} / ${result.metrics.total_pixels.toLocaleString()}`;
        }
    } catch (error) {
        handleProcessingError(error);
    } finally {
        toggleLoading(false);
    }
}

function toggleLoading(isLoading) {
    const btn = document.getElementById('processBtn');
    btn.classList.toggle('loading', isLoading);
    btn.disabled = isLoading;
}

function handleProcessingError(error) {
    console.error('Erreur de traitement:', error);
    alert(`Échec du traitement : ${error.message}`);
    resetImageDisplays();
    resetFileInput();
}

function loadProcessingView(processType) {
    document.querySelector('.current-category').textContent = 
        processType === 'segmentation' ? '👁️ Analyse Oculaire' : '📊 Classification';
    fetchModels(processType);
}

async function fetchModels(processType) {
    try {
        const response = await fetch(`/get-models?type=${processType}`);
        if (!response.ok) throw new Error('Échec du chargement des modèles');
        
        const models = await response.json();
        populateModelDropdown(models);
        
    } catch (error) {
        console.error('Erreur de chargement des modèles:', error);
        alert('Impossible de charger les modèles');
    }
}

function populateModelDropdown(models) {
    const dropdown = document.getElementById('modelSelect');
    dropdown.innerHTML = '<option value="">Sélectionner un modèle</option>';
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.model_name;
        dropdown.appendChild(option);
    });
}

async function handleModelUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('model', file);

    try {
        const response = await fetch('/upload-model', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Échec de l\'upload');
        
        alert('Modèle uploadé avec succès !');
        const activeType = document.querySelector('.sidebar-btn.active').dataset.type;
        fetchModels(activeType);
        
    } catch (error) {
        console.error('Erreur d\'upload:', error);
        alert("Erreur lors de l'upload du modèle: " + error.message);
    } finally {
        e.target.value = '';
    }
}

// Update handleFileSelect function:
function handleFileSelect(e) {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];

    if (!file || !file.type.startsWith('image/')) {
        alert('Veuillez télécharger une image valide');
        resetFileInput();
        return;
    }

    // Read image metadata
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
            document.getElementById('inputDimensions').textContent = 
                `${this.width}px × ${this.height}px`;
            document.getElementById('inputSize').textContent = 
                `${(file.size/1024).toFixed(1)} KB`;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    showUploadSuccess();
}