# AI-Powered Medical Diagnosis Platform

## Overview
This project is an AI-driven healthcare platform designed to assist in diagnosing various diseases using deep learning models. It provides an intuitive interface where users can upload medical images (such as X-rays, histopathology images, and skin lesion scans) to receive AI-generated diagnostic insights. Additionally, a chatbot powered by Google's Gemini API offers medical explanations and symptom analysis.

![Screenshot 2025-03-22 122813](https://github.com/user-attachments/assets/4879819d-0e3d-4a3c-8d17-c536c2ee94ab)
![Screenshot 2025-03-22 093455](https://github.com/user-attachments/assets/3528c9c6-bbf8-489a-a655-24813e7811a6)
![Screenshot 2025-03-22 093933](https://github.com/user-attachments/assets/f8c8ca07-0eab-4cab-b050-eab91f06bd52)
![Screenshot 2025-03-22 122715](https://github.com/user-attachments/assets/a269db08-00cc-4c1c-a405-8eb8d7a4794b)
![Screenshot 2025-03-22 122729](https://github.com/user-attachments/assets/b94ab14d-7144-4ebe-b2e7-dff4eeb5c25a)




## Features
- **Image-Based Disease Detection**: Supports classification for multiple diseases using deep learning models.
- **AI-Generated Medical Reports**: Generates professional reports with explanations, risk assessments, and recommended next steps.
- **Medical Chatbot**: Provides information about conditions, symptoms, and possible treatments.
- **Symptom Checker**: Assists users in identifying potential conditions based on symptoms.
- **User-Friendly Web Interface**: Designed for accessibility and ease of use.

## Supported Disease Classifications
The platform supports six AI models trained on medical imaging datasets:
1. **Breast Cancer Detection** - Classifies histopathology images as *Healthy* or *Sick*.
2. **COVID-19 Analysis** - Analyzes chest X-rays for *COVID*, *Lung Opacity*, *Normal*, and *Viral Pneumonia*.
3. **Malaria Detection** - Detects malaria in blood smear images (*Parasitized* or *Uninfected*).
4. **Pneumonia Detection** - Identifies pneumonia in chest X-ray images (*Normal* or *Pneumonia*).
5. **Skin Cancer Classification** - Differentiates between *Benign* and *Malignant* skin lesions.
6. **Tuberculosis Screening** - Screens chest X-rays for *Normal* or *Tuberculosis*.


## Model Accuracy
1. Here are the accuracy scores of each model:
2. Skin Cancer Detection: 90%
3. Malaria Detection: 94%
4. Breast Cancer Classification: 95%
5. Tuberculosis Detection: 97%
6. Pneumonia Detection: 95%
7. COVID-19 Detection: 96%


## Technologies Used
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning Framework**: PyTorch
- **Deep Learning Model**: EfficientNetB0 (Transfer Learning)
- **API Integration**: Google's Gemini API (for chat and report generation)
- **Database**: JSON-based storage (for model metadata and reports)

## Team Contributions
- **Frontend Development**: Prachi and Khushi (UI/UX design, web development)
- **Backend Development**: Mohit and Pushkar (Flask API, integration)
- **AI Model Development**: Pushkar (developed all models from scratch)
- **Model Optimization**: Pushkar (used EfficientNetB0 for transfer learning)

## Future Enhancements
- **Doctor-Patient Video Consultation** *(In Progress)*
- **User Authentication & Profile Management**
- **Improved Symptom Checker with NLP**
- **Expanded Disease Database**

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-medical-diagnosis.git
   cd ai-medical-diagnosis
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```sh
   export GEMINI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```sh
   python app.py
   ```
5. Access the web interface at:
   ```
   http://127.0.0.1:5000
   ```

## Usage
- Upload an image to get a disease classification.
- Use the chatbot for medical guidance.
- Generate a medical report for further analysis.

## Disclaimer
This AI system is designed for research and educational purposes. It does **not** provide a definitive medical diagnosis. Always consult a healthcare professional for medical advice and diagnosis.

---

For contributions or issues, please contact the development team.

