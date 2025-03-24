document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let uploadedFile = null;
    const modelSelect = document.getElementById('model-select');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const fileUpload = document.getElementById('file-upload');
    const uploadArea = document.getElementById('upload-area');
    
    // Scroll to upload section when "Start Diagnosis" is clicked
    document.getElementById('start-diagnosis').addEventListener('click', function() {
        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
    });
    
    // Scroll to chat section when "Chat Now" is clicked
    document.getElementById('chat-btn').addEventListener('click', function() {
        document.getElementById('chat').scrollIntoView({ behavior: 'smooth' });
    });
    
    // Handle model card selection
    const modelCards = document.querySelectorAll('.model-card');
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            const modelKey = this.getAttribute('data-model');
            
            // Select the model in the dropdown
            modelSelect.value = modelKey;
            
            // Scroll to upload section
            document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
        });
    });
    
    // Handle file upload via browse button
    fileUpload.addEventListener('change', function(e) {
        handleFileUpload(e.target.files[0]);
    });
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('active');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        this.classList.remove('active');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    // Function to handle file upload
    function handleFileUpload(file) {
        if (!file) return;
        
        // Check if file is an image
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (JPG, JPEG, or PNG).');
            return;
        }
        
        uploadedFile = file;
        analyzeBtn.disabled = false;
        
        // Show file name in upload area
        const fileName = document.createElement('p');
        fileName.textContent = `Selected file: ${file.name}`;
        
        // Remove any previous file name
        const previousFileName = uploadArea.querySelector('.file-name');
        if (previousFileName) {
            uploadArea.removeChild(previousFileName);
        }
        
        fileName.classList.add('file-name');
        uploadArea.appendChild(fileName);
        
        // Change the upload icon to a check mark
        const uploadIcon = uploadArea.querySelector('i');
        uploadIcon.className = 'fas fa-check-circle';
        uploadIcon.style.color = '#4CAF50';
    }
    
    // Handle analyze button click
    analyzeBtn.addEventListener('click', function() {
        if (!uploadedFile) return;
        
        // Show loading state
        this.disabled = true;
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        // Create FormData object
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('model', modelSelect.value);
        
        // Display the uploaded image in the results section
        const resultsImg = document.getElementById('results-img');
        const reader = new FileReader();
        reader.onload = function(e) {
            resultsImg.src = e.target.result;
        };
        reader.readAsDataURL(uploadedFile);
        
        // Set the model used text
        document.getElementById('model-used').textContent = modelSelect.options[modelSelect.selectedIndex].text;
        
        // Make actual API call to the backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Show results section
            uploadSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            
            // Update result information with data from the backend
            document.getElementById('prediction-result').textContent = data.prediction;
            document.getElementById('confidence-text').textContent = `${data.confidence.toFixed(1)}%`;
            document.getElementById('confidence-level').style.width = `${data.confidence}%`;
            
            // Update report content with Gemini-generated content
            document.getElementById('report-content').innerHTML = data.report_content;
            
            // Store the report URL for downloading
            const reportUrl = data.report_url;
            
            // Store the URL in a hidden field (create it if it doesn't exist)
            let reportUrlField = document.getElementById('report-url');
            if (!reportUrlField) {
                reportUrlField = document.createElement('input');
                reportUrlField.type = 'hidden';
                reportUrlField.id = 'report-url';
                document.body.appendChild(reportUrlField);
            }
            reportUrlField.value = reportUrl;
            
            // Also store it as a data attribute on the download button
            document.getElementById('download-report').setAttribute('data-report-url', reportUrl);
            
            // Reset analyze button
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Image';
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error during analysis:', error);
            alert('An error occurred during image analysis. Please try again.');
            
            // Reset analyze button
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Image';
        });
    });
    
    // Handle "New Diagnosis" button click
    document.getElementById('new-diagnosis').addEventListener('click', function() {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        
        // Reset the upload area
        uploadedFile = null;
        analyzeBtn.disabled = true;
        fileUpload.value = '';
        
        // Reset the upload icon
        const uploadIcon = uploadArea.querySelector('i');
        uploadIcon.className = 'fas fa-cloud-upload-alt';
        uploadIcon.style.color = '';
        
        // Remove file name if it exists
        const fileName = uploadArea.querySelector('.file-name');
        if (fileName) {
            uploadArea.removeChild(fileName);
        }
        
        // Scroll to upload section
        uploadSection.scrollIntoView({ behavior: 'smooth' });
    });
    
    // Handle "Download Report" button click
    document.getElementById('download-report').addEventListener('click', function() {
        // Get the report URL from the hidden field or data attribute
        const reportUrl = document.getElementById('report-url').value || this.getAttribute('data-report-url');
        
        if (reportUrl) {
            // Create a temporary anchor element to trigger the download
            const downloadLink = document.createElement('a');
            downloadLink.href = reportUrl;
            downloadLink.target = '_blank';
            
            // Add download attribute to force download (if browser supports it)
            downloadLink.setAttribute('download', '');
            
            // Append to body, click and remove
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        } else {
            alert('No report available for download. Please analyze an image first.');
        }
    });
    
    // Add active class to current nav item based on scroll position
    const sections = document.querySelectorAll('section[id]');
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= (sectionTop - sectionHeight / 3)) {
                current = section.getAttribute('id');
            }
        });
        
        document.querySelectorAll('nav ul li a').forEach(a => {
            a.classList.remove('active');
            if (a.getAttribute('href') === `#${current}`) {
                a.classList.add('active');
            }
        });
    });
    
    // Mobile menu toggle (would need to be implemented if a mobile menu button exists)
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', function() {
            const nav = document.querySelector('nav');
            nav.classList.toggle('show');
        });
    }
    
    // Handle appointment button click
    document.querySelector('.appointment-btn').addEventListener('click', function() {
        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
    });
});