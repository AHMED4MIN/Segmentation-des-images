document.addEventListener('DOMContentLoaded', () => {
    // Category Selection
    document.querySelectorAll('.option-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const category = btn.closest('.category-card').dataset.category;
            const processType = btn.dataset.type;
            loadProcessingView(category, processType);
        });
    });

    // File Upload Handling
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    
    dropZone.addEventListener('click', () => imageInput.click());
    
    imageInput.addEventListener('change', handleFileSelect);
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if(files.length) handleFileSelect({ target: { files } });
    });

    // Process Button
    document.getElementById('processBtn').addEventListener('click', processImage);
});

function loadProcessingView(category, processType) {
    document.getElementById('categoryView').style.display = 'none';
    document.getElementById('processingView').style.display = 'grid';
    document.getElementById('currentCategory').textContent = 
        `${category === 'eyes' ? '👁️ Eyes' : '🩻 X-Ray'} Analysis`;
    
    // Update active sidebar button
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === processType);
    });
    
    // TODO: Load appropriate models based on category and processType
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if(!file.type.startsWith('image/')) return alert('Please upload an image file');
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('previewImage');
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

async function processImage() {
    const file = document.getElementById('imageInput').files[0];
    const modelId = document.getElementById('modelSelect').value;
    
    if(!file) return alert('Please select an image');
    if(!modelId) return alert('Please select a model');

    const formData = new FormData();
    formData.append('image', file);
    formData.append('model_id', modelId);

    try {
        const response = await fetch('/process-image', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if(result.error) throw new Error(result.error);
        
        document.getElementById('originalResult').src = result.original;
        document.getElementById('processedResult').src = result.processed;
    } catch (error) {
        alert(`Processing failed: ${error.message}`);
    }
}