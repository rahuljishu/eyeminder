import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, threshold_local
from skimage import exposure, morphology, filters, segmentation, feature, measure
import os
import tempfile
from io import BytesIO
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import warnings
import base64

# Set page configuration
st.set_page_config(
    page_title="Eyeminder - Retinal Hemorrhage Analysis",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4285f4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Eyeminder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retinal Hemorrhage Analysis System</div>', unsafe_allow_html=True)

class RetinalHemorrhageAnalyzer:
    def __init__(self):
        self.CLAHE_CLIP_LIMIT = 4.0
        self.CLAHE_GRID_SIZE = (8, 8)
        self.VESSEL_SIGMAS = [0.5, 1.0, 1.5]
        self.MIN_LESION_SIZE = 100
        self.EPSILON = 1e-10
        
        self.SEVERITY_THRESHOLDS = {
            'Normal': 5.0,
            'Mild': 15.0,
            'Moderate': 30.0,
            'Severe': float('inf')
        }

    def analyze_image(self, image_path, output_dir="results"):
        """Complete analysis pipeline with robust error handling"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            img = self._load_validate_image(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            processed = self._preprocess_image(img)
            
            vessel_free = self._remove_retinal_vessels(processed)
            
            final_mask, features = self._detect_hemorrhages(vessel_free)
            
            results = self._generate_results(img, final_mask, base_name)
            
            self._save_outputs(img, final_mask, results, output_dir, base_name)
            
            return {
                'status': 'success',
                'results': results,
                'output_files': {
                    'image': os.path.join(output_dir, f"{base_name}_result.jpg"),
                    'data': os.path.join(output_dir, f"{base_name}_results.json")
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _load_validate_image(self, path):
        """Load and validate input image"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
            
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Unsupported image format or corrupt file")
            
        img = cv2.resize(img, (800, 800))
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    def _preprocess_image(self, img):
        """Advanced image enhancement pipeline"""
        # LAB space illumination correction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT,
            tileGridSize=self.CLAHE_GRID_SIZE
        )
        lab[:,:,0] = clahe.apply(l_channel)
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Green channel extraction
        green = corrected[:,:,1]
        return cv2.GaussianBlur(green, (5,5), 0)

    def _remove_retinal_vessels(self, img):
        """Numerically stable vessel removal"""
        vessel_mask = np.zeros_like(img, dtype=np.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for sigma in self.VESSEL_SIGMAS:
                H = feature.hessian_matrix(img, sigma=sigma, 
                                         use_gaussian_derivatives=True)
                eigvals = feature.hessian_matrix_eigvals(H)
                
                denominator = np.where(np.abs(eigvals[1]) > self.EPSILON,
                                    eigvals[1],
                                    np.sign(eigvals[1]) * self.EPSILON)
                Rb = (eigvals[0] / denominator)**2
                
                S = np.sqrt(eigvals[0]**2 + eigvals[1]**2 + self.EPSILON)
                vesselness = np.exp(-Rb/0.5) * (1 - np.exp(-S/0.25))
                
                vessel_mask = np.maximum(vessel_mask, vesselness)
        
        vessel_mask = np.nan_to_num(vessel_mask, nan=0.0)
        if np.all(vessel_mask == 0):
            return img
            
        vessel_mask = exposure.rescale_intensity(
            vessel_mask,
            in_range=(np.min(vessel_mask), np.max(vessel_mask)),
            out_range=(0, 255)
        )
        return cv2.subtract(img, vessel_mask.astype(np.uint8))

    def _detect_hemorrhages(self, img):
        """Robust hemorrhage detection"""
        binary = cv2.adaptiveThreshold(
            np.nan_to_num(img),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            101, 5
        )
        
        cleaned = morphology.binary_opening(binary, morphology.disk(2))
        final_mask = morphology.remove_small_objects(
            cleaned.astype(bool),
            min_size=self.MIN_LESION_SIZE
        )
        
        regions = measure.regionprops(measure.label(final_mask))
        features = [{
            'area': r.area,
            'centroid': r.centroid,
            'eccentricity': r.eccentricity
        } for r in regions]
        
        return final_mask, features

    def _generate_results(self, img, mask, base_name):
        """Generate comprehensive clinical results"""
        coverage = np.sum(mask) / mask.size * 100
        hemorrhage_types = self._classify_hemorrhage_types(mask)
        
        return {
            'patient_id': base_name,
            'date_processed': datetime.now().strftime('%Y-%m-%d'),
            'coverage_percent': float(f"{coverage:.2f}"),
            'severity': self._determine_severity(coverage),
            'hemorrhage_types': hemorrhage_types,
            'image_quality': self._assess_image_quality(img),
            'clinical_recommendations': self._get_clinical_recommendations(coverage)
        }

    def _classify_hemorrhage_types(self, mask):
        """Categorize hemorrhage morphology"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        counts = {'dot_blot': 0, 'flame': 0, 'other': 0}
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            if area < 15 and circularity > 0.7:
                counts['dot_blot'] += 1
            elif area > 30 and circularity < 0.3:
                counts['flame'] += 1
            else:
                counts['other'] += 1
                
        return counts

    def _assess_image_quality(self, img):
        """Evaluate image quality metrics"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'quality': 'Good' if sharpness > 100 and contrast > 40 else 'Poor'
        }

    def _determine_severity(self, coverage):
        """Classify clinical severity"""
        for severity, threshold in self.SEVERITY_THRESHOLDS.items():
            if coverage < threshold:
                return severity
        return 'Severe'

    def _get_clinical_recommendations(self, coverage):
        """Evidence-based clinical guidance"""
        recommendations = {
            'Normal': [
                "Routine follow-up in 1 year",
                "Continue annual diabetic eye screening"
            ],
            'Mild': [
                "Follow-up in 6 months",
                "Optimize glycemic control",
                "Monitor for progression"
            ],
            'Moderate': [
                "Refer to ophthalmologist",
                "Consider focal laser treatment",
                "Follow-up in 3 months",
                "Optical Coherence Tomography recommended"
            ],
            'Severe': [
                "Urgent ophthalmology referral",
                "Pan-retinal photocoagulation evaluation",
                "Monthly monitoring required",
                "Assess for macular edema"
            ]
        }
        return recommendations.get(self._determine_severity(coverage), [])

    def _save_outputs(self, img, mask, results, output_dir, base_name):
        """Save all analysis outputs"""
        # Save visual result
        result_img = img.copy()
        result_img[mask, :] = [0, 0, 255]  # Mark hemorrhages in red
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_result.jpg"),
            result_img
        )
        
        with open(os.path.join(output_dir, f"{base_name}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

class PDFReportGenerator:
    def __init__(self):
        self.page_size = letter
        self.styles = getSampleStyleSheet()
        self.report_title = "Eyeminder - Retinal Hemorrhage Analysis Report"
        self.logo_path = None  # No logo in Streamlit app

    def generate_pdf(self, analysis_result, output_dir="reports"):
        """Generate professional PDF report"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_name = analysis_result['results']['patient_id']
            pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
            
            doc = SimpleDocTemplate(pdf_path, pagesize=self.page_size)
            elements = []
            
            if self.logo_path and os.path.exists(self.logo_path):
                logo = Image(self.logo_path, width=2*inch, height=1*inch)
                elements.append(logo)
            
            elements.append(Paragraph(self.report_title, self.styles['Title']))
            elements.append(Spacer(1, 0.25*inch))
            
            patient_info = [
                ["Patient ID:", analysis_result['results']['patient_id']],
                ["Date of Analysis:", analysis_result['results']['date_processed']],
                ["Image Quality:", analysis_result['results']['image_quality']['quality']]
            ]
            elements.append(Table(patient_info, hAlign='LEFT'))
            elements.append(Spacer(1, 0.5*inch))
            
            results_table = [
                ["Parameter", "Value"],
                ["Total Coverage", f"{analysis_result['results']['coverage_percent']}%"],
                ["Severity Level", analysis_result['results']['severity']],
                ["Dot/Blot Hemorrhages", str(analysis_result['results']['hemorrhage_types']['dot_blot'])],
                ["Flame-Shaped Hemorrhages", str(analysis_result['results']['hemorrhage_types']['flame'])],
                ["Other Lesions", str(analysis_result['results']['hemorrhage_types']['other'])]
            ]
            elements.append(Table(results_table, style=[
                ('BACKGROUND', (0,0), (-1,0), '#77aaff'),
                ('TEXTCOLOR', (0,0), (-1,0), '#ffffff'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), '#f2f7ff'),
                ('GRID', (0,0), (-1,-1), 1, '#aaaaaa')
            ]))
            elements.append(Spacer(1, 0.5*inch))
            
            result_img_path = analysis_result['output_files']['image']
            if os.path.exists(result_img_path):
                img = Image(result_img_path, width=5*inch, height=5*inch)
                elements.append(Paragraph("Analysis Results Visualization", self.styles['Heading2']))
                elements.append(img)
                elements.append(Spacer(1, 0.25*inch))
            
            elements.append(Paragraph("Clinical Recommendations", self.styles['Heading2']))
            for rec in analysis_result['results']['clinical_recommendations']:
                elements.append(Paragraph(f"• {rec}", self.styles['Normal']))
            
            doc.build(elements)
            return pdf_path
            
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {str(e)}")

# Helper function for creating download links
def get_download_link(file_path, link_text):
    with open(file_path, "rb") as file:
        contents = file.read()
    b64 = base64.b64encode(contents).decode()
    filename = os.path.basename(file_path)
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'

def run_method1(image_path):
    # Load the fundus image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Frangi filter to enhance blood vessels
    vessel_enhanced = frangi(gray)
    
    # Convert to binary using thresholding
    _, vessel_binary = cv2.threshold((vessel_enhanced * 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    
    # Remove small noise objects - FIX: Use the morphology module prefix
    vessel_binary = morphology.remove_small_objects(vessel_binary.astype(bool), min_size=100).astype(np.uint8) * 255
    
    # Convert image to HSV and extract red channel for hemorrhages
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))  # Detect red lesions
    red_mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))  
    red_areas = red_mask1 | red_mask2  # Combine both red ranges
    
    # Detect damaged blood vessel areas by finding overlap between vessels and red hemorrhages
    damaged_areas = cv2.bitwise_and(vessel_binary, red_areas)
    
    # Overlay detected damaged areas on the original image
    highlighted_damage = image.copy()
    highlighted_damage[damaged_areas == 255] = [0, 0, 255]  # Highlight damage in red
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Fundus Image")
    axes[1].imshow(red_areas, cmap='gray')
    axes[1].set_title("Detected Hemorrhage Areas")
    axes[2].imshow(cv2.cvtColor(highlighted_damage, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Blood Vessel Leakage Highlighted (Red)")
    
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    
    # Save figure to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    return buf

def run_method2(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Resize for standardization
    resized = cv2.resize(img, (512, 512))
    
    # Extract green channel (most informative for retinal images)
    green = resized[:, :, 1]
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_green = clahe.apply(green)
    
    # Detect and mask optic disc (bright circular region)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(green, (9, 9), 2),
        cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )
    
    # Create mask to remove optic disc
    mask = np.ones_like(green) * 255
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(mask, (x, y), r, 0, -1)
    
    # Apply mask to remove optic disc
    masked_img = cv2.bitwise_and(clahe_green, mask)
    
    # Enhance hemorrhage features using Hessian-based filtering
    frangi_result = feature.hessian_matrix(
        masked_img, 
        sigma=1.5, 
        use_gaussian_derivatives=False
    )
    vesselness = feature.hessian_matrix_eigvals(frangi_result)[0]
    vesselness = exposure.rescale_intensity(vesselness, out_range=(0, 255))
    
    # Apply adaptive thresholding for hemorrhage segmentation
    binary = cv2.adaptiveThreshold(
        np.uint8(vesselness), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 2
    )
    
    # Clean up result with morphological operations
    kernel = morphology.disk(3)
    cleaned = morphology.binary_opening(binary, kernel)
    final_mask = morphology.remove_small_objects(cleaned.astype(bool), min_size=50)
    
    # Create visualization with overlay
    overlay = resized.copy()
    overlay[final_mask, :] = [0, 0, 255]  # Mark hemorrhages in red
    
    # Create figure for display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    
    axes[1].imshow(clahe_green, cmap='gray')
    axes[1].set_title("CLAHE Green Channel")
    
    axes[2].imshow(masked_img, cmap='gray')
    axes[2].set_title("Optic Disc Removed")
    
    axes[3].imshow(vesselness, cmap='gray')
    axes[3].set_title("Vessel Enhancement")
    
    axes[4].imshow(binary, cmap='gray')
    axes[4].set_title("Binary Threshold")
    
    axes[5].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[5].set_title("Hemorrhage Detection")
    
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    
    # Save figure to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    return buf

def run_clinical_analysis(image_path, temp_dir):
    analyzer = RetinalHemorrhageAnalyzer()
    pdf_gen = PDFReportGenerator()
    
    # Create results directory within temp_dir
    results_dir = os.path.join(temp_dir, "results")
    reports_dir = os.path.join(temp_dir, "reports")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Run analysis
    result = analyzer.analyze_image(image_path, output_dir=results_dir)
    
    if result['status'] == 'success':
        # Generate PDF report
        try:
            pdf_path = pdf_gen.generate_pdf(result, output_dir=reports_dir)
            
            # Display results
            img = cv2.imread(image_path)
            result_img = cv2.imread(result['output_files']['image'])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            axes[1].title.set_text(f"Result: {result['results']['severity']}")
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save figure to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            
            return result, buf, pdf_path
        except Exception as e:
            return result, None, None
    else:
        return result, None, None

# Main app functionality
uploaded_file = st.file_uploader("Upload a fundus image for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Save uploaded file to temp directory
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption='Uploaded Fundus Image', use_column_width=True)
    
    with col2:
        st.write("### Image Details")
        img = cv2.imread(temp_file_path)
        st.write(f"- **Filename**: {uploaded_file.name}")
        st.write(f"- **Image Size**: {img.shape[1]} x {img.shape[0]} pixels")
        st.write(f"- **File Size**: {round(os.path.getsize(temp_file_path) / 1024, 2)} KB")
    
    # Process button
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Processing image... Please wait"):
            
            # Create tab layout
            method_tabs = st.tabs([
                "Clinical Analysis with PDF Report", 
                "Method 1: Vessel + Hemorrhage", 
                "Method 2: Step-wise Detection"
            ])
            
            # Clinical Analysis Tab
            with method_tabs[0]:
                result, result_img_buf, pdf_path = run_clinical_analysis(temp_file_path, temp_dir)
                
                if result['status'] == 'success':
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success("Analysis completed successfully!")
                        st.write(f"### Severity: {result['results']['severity']}")
                        st.write(f"### Coverage: {result['results']['coverage_percent']}%")
                        
                        # Image quality
                        quality = result['results']['image_quality']
                        st.write("### Image Quality")
                        st.write(f"- **Assessment**: {quality['quality']}")
                        st.write(f"- **Sharpness**: {quality['sharpness']:.2f}")
                        st.write(f"- **Contrast**: {quality['contrast']:.2f}")
                        
                        # Hemorrhage types
                        h_types = result['results']['hemorrhage_types']
                        st.write("### Hemorrhage Classifications")
                        st.write(f"- **Dot/Blot**: {h_types['dot_blot']}")
                        st.write(f"- **Flame-shaped**: {h_types['flame']}")
                        st.write(f"- **Other**: {h_types['other']}")
                    
                    with col2:
                        # Show result image
                        if result_img_buf:
                            st.image(result_img_buf, caption=f"Analysis Result: {result['results']['severity']}", use_column_width=True)
                    
                    # Clinical recommendations
                    st.write("### Clinical Recommendations")
                    for rec in result['results']['clinical_recommendations']:
                        st.write(f"- {rec}")
                    
                    # Download links
                    st.write("### Downloads")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if pdf_path and os.path.exists(pdf_path):
                            st.markdown(get_download_link(pdf_path, "📄 Download PDF Report"), unsafe_allow_html=True)
                    
                    with col2:
                        if os.path.exists(result['output_files']['image']):
                            st.markdown(get_download_link(result['output_files']['image'], "🖼️ Download Result Image"), 
                                        unsafe_allow_html=True)
                        
                        if os.path.exists(result['output_files']['data']):
                            st.markdown(get_download_link(result['output_files']['data'], "📊 Download JSON Data"), 
                                        unsafe_allow_html=True)
                else:
                    st.error(f"Analysis failed: {result['message']}")
            
            # Method 1 Tab
            with method_tabs[1]:
                method1_result = run_method1(temp_file_path)
                st.image(method1_result, caption="Method 1: Vessel and Hemorrhage Detection", use_column_width=True)
            
            # Method 2 Tab
            with method_tabs[2]:
                method2_result = run_method2(temp_file_path)
                st.image(method2_result, caption="Method 2: Step-wise Hemorrhage Detection", use_column_width=True)

    # Clean up temp files when the session ends
    def cleanup():
        import shutil
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
    
    # Register cleanup function
    import atexit
    atexit.register(cleanup)

# About section
st.write("---")
st.write("""
## About Eyeminder
**Eyeminder** is a comprehensive retinal hemorrhage analysis system designed for clinical use. It combines multiple analysis methods to detect and classify hemorrhages in fundus images.

### Features:
- **Clinical Analysis**: Advanced detection with severity classification and clinical recommendations
- **PDF Reports**: Detailed reports suitable for medical documentation
- **Method 1**: Blood vessel analysis with hemorrhage correlation
- **Method 2**: Step-wise image processing for hemorrhage detection
- **Hemorrhage Classification**: Differentiates between dot/blot and flame-shaped hemorrhages

### Usage:
1. Upload a fundus image
2. Click "Analyze Image"
3. View results across different analysis methods
4. Download the PDF report and result images

### Clinical Edition v2.4 with PDF Reports
""")
