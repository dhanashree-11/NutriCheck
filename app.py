from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import cv2
import numpy as np
import speech_recognition as sr
from PIL import Image
import google.generativeai as genai
import re
import base64  # ✅ Import missing base64 module
from io import BytesIO
import io


# Flask app initialization
app = Flask(__name__)

# Set a secret key for session management
app.secret_key = 'supersecretkey'

# Increase file upload size limit to 50MB (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Increase limit to 50MB


app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # ✅ Restrict allowed file types

# Configure Google Gemini AI API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ✅ Function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from PIL import Image
import io
import base64

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure it's in RGB format

    # Resize image
    img = img.resize((640, 480))

    # Convert to bytes
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")  # Save as PNG
    image_bytes = buffered.getvalue()

    # Convert to AI-compatible format
    return [{
        "mime_type": "image/png",
        "data": image_bytes
    }]


def get_gemini_response(image_data, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Ensure `image_data` is always correctly structured as a list of dicts
    if isinstance(image_data, list) and all(isinstance(item, dict) for item in image_data):
        response = model.generate_content(image_data + [prompt])  
    elif isinstance(image_data, dict):
        response = model.generate_content([image_data, prompt])  
    else:
        raise TypeError("Invalid image_data format. Expected a dict or list of dicts.")

    return response.text



# ✅ Function to format AI response for proper HTML rendering
def format_response(response_text):
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response_text)
    formatted_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_text)
    formatted_text = formatted_text.replace("\n", "<br><br>")
    formatted_text = re.sub(r'(?m)^- (.*)', r'<li>\1</li>', formatted_text)
    return formatted_text

# ✅ Route for homepage
@app.route('/')
def home():
    return render_template('index.html', background_image=url_for('static', filename='images/AdobeStock_646559614.jpeg'))

# ✅ Corrected Route for Uploading and Analyzing Images
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')
    camera_image = request.form.get('camera_image')

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        processed_image = preprocess_image(file_path)
        image_url = url_for('uploaded_file', filename=file.filename)
   
    elif camera_image:
        try:
            # Decode Base64 image
            image_data = base64.b64decode(camera_image.split(',')[1])  
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], "captured_image.jpg")

            # Save the image properly
            with open(file_path, "wb") as f:
                f.write(image_data)

            processed_image = preprocess_image(file_path)
            image_url = url_for('uploaded_file', filename="captured_image.jpg")  # ✅ Set correct URL

        except Exception as e:
            return f"Error processing camera image: {str(e)}"

    else:
        flash("Invalid file type. Please upload a PNG, JPG, or JPEG file.")
        return redirect(request.url)

    nutritional_analysis_prompt = """
You are a nutrition expert analyzing the food items in the provided image. Keep the response minimalistic and structured in the following format:

### *1️⃣ Nutrient Breakdown (Amount per Serving)*
   - *Calories:* XX kcal  
   - *Sodium:* XX mg  
   - *Protein:* XX g  
   - *Fat:* XX g  
   - *Carbohydrates:* XX g  
   - *Sugar:* XX g  

---

### *2️⃣ 🔹 Nutrients that are Good in this Food:*
   - List key nutrients that are beneficial along with their amounts.
   - Provide a short reason for each (e.g., *Protein: 10g (Good for muscle growth)*).

---

### *3️⃣ ⚠ Nutrients to Watch Out For:*
   - List nutrients that are *excessive/unhealthy* along with their amounts.
   - Provide a brief explanation for why they should be moderated.

---

### *4️⃣ ⭐ Health Score (Out of 10):*
*X/10*  
A short 1-line explanation about why the food got this rating.

---

### *5️⃣ 📏 Ideal Consumption Amount:*
- *Recommended daily intake* based on health guidelines.

---

### *6️⃣ 🍏 Healthier Alternatives (If Food is Not Ideal):*
   - Suggest better food choices if this food is high in unhealthy nutrients.
   - Keep it *practical and easy to follow* (e.g., "Instead of white bread, opt for whole wheat bread").
   - If the food is already very healthy, state *'This food is already a great choice!'*.

Ensure the response is concise, easy to read, and structured without unnecessary details.
"""

    response_text = get_gemini_response(processed_image, nutritional_analysis_prompt)
    formatted_response = format_response(response_text)  # ✅ Apply formatting

    return render_template('result.html', response=formatted_response, image_url=image_url)

@app.route('/ingredient-analysis', methods=['POST'])
def ingredient_analysis():
    file = request.files.get('file')
    camera_image = request.form.get('camera_image')

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        processed_image = preprocess_image(file_path)
        image_url = url_for('uploaded_file', filename=file.filename)
   
    elif camera_image:
        try:
            # Decode Base64 image
            image_data = base64.b64decode(camera_image.split(',')[1])  
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], "captured_image.jpg")

            # Save the image properly
            with open(file_path, "wb") as f:
                f.write(image_data)

            processed_image = preprocess_image(file_path)
            image_url = url_for('uploaded_file', filename="captured_image.jpg")

        except Exception as e:
            return f"Error processing camera image: {str(e)}"

    else:
        flash("Invalid file type. Please upload a PNG, JPG, or JPEG file.")
        return redirect(request.url)

    # Ingredient Analysis Prompt
    ingredient_analysis_prompt = """
    You are a food safety and quality expert analyzing the ingredient list from the provided image. Keep the response concise and structured as follows:

    ### **1️⃣ 🏷️ Ingredient Breakdown:**
       - List each ingredient found in the image.
       - If an ingredient is unclear, mention it and ask the user for confirmation.

    ---

    ### **2️⃣ ✅ Beneficial Ingredients:**
       - Highlight ingredients that provide health benefits (e.g., fiber, vitamins, minerals).
       - Include a **short reason** why they are good (e.g., "Vitamin C: Boosts immunity").

    ---

    ### **3️⃣ ⚠️ Ingredients to Watch Out For:**
       - List ingredients that may be **harmful, excessive, or allergenic**.
       - Provide a **brief explanation** of their risks (e.g., "High Sodium: May increase blood pressure").

    ---

    ### **4️⃣ ⭐ Overall Health Rating (Out of 10):**
       - **X/10**  
       - Give a **short 1-line explanation** based on the ingredient quality.

    ---

    ### **5️⃣ 📏 Recommended Consumption:**
       - Specify whether this product is **safe for daily use or should be consumed occasionally**.

    ---

    ### **6️⃣ 🍏 Healthier Alternatives (Based on Indian Diet):**
       - If the product has **harmful additives or excessive sugar**, suggest **healthier Indian food substitutes**.
       - Examples:
         - **Too much sugar?** Opt for jaggery-based sweets.
         - **Too many preservatives?** Suggest fresh homemade alternatives.
         - **High in refined oils?** Suggest cold-pressed oils or traditional Indian alternatives.

    Ensure the response is easy to read, structured, and practical for everyday decision-making.
    """

    response_text = get_gemini_response(processed_image, ingredient_analysis_prompt)
    formatted_response = format_response(response_text)

    return render_template('result.html', response=formatted_response, image_url=image_url)

@app.route('/ingredient-analysis')
def ingredient_analysis_page():
    return render_template('ingredient_analysis.html')


# ✅ Route for voice recognition
@app.route('/voice-input', methods=['GET', 'POST'])
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
    except:
        text = "Could not understand audio"
    
    return render_template('voice.html', voice_text=text)

# ✅ Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/nutritional-analysis')
def nutritional_analysis():
    return render_template('nutritional_analysis.html')

@app.route('/compare-ingredients', methods=['GET', 'POST'])
def compare_ingredients():
    response1 = None
    response2 = None
    comparison_data = None
    verdict_table = None

    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)

            file1.save(file_path1)
            file2.save(file_path2)

            # ✅ Preprocess images properly
            processed_image1 = preprocess_image(file_path1) 
            processed_image2 = preprocess_image(file_path2) 

            # ✅ Ingredient Analysis Prompt
            ingredient_analysis_prompt = """
            You are a food safety and nutrition expert analyzing the ingredient list from an image.
            Extract and categorize:
            - Beneficial Ingredients with health benefits
            - Harmful Ingredients with potential risks
            - Overall Health Score (out of 10)
            - Suggested Alternatives for improvement
            """

            # ✅ Analyze each ingredient list separately
            response1 = get_gemini_response(processed_image1, ingredient_analysis_prompt)
            response2 = get_gemini_response(processed_image2, ingredient_analysis_prompt)

            # ✅ Format AI responses properly
            response1 = format_response(response1)
            response2 = format_response(response2)

            # ✅ **Forcing AI to Return a Properly Formatted Comparison and Verdict**
            comparison_prompt = f"""
            You are a food safety and nutrition expert comparing two ingredient lists.

            **Ingredient List 1 Analysis:**
            {response1}

            **Ingredient List 2 Analysis:**
            {response2}

            ---
            ## **⚖ 1️⃣ Beneficial Ingredients Comparison**
            | Category | Ingredient List 1 | Ingredient List 2 |
            |----------|------------------|------------------|
            | **Key Nutrients** | Extract key beneficial ingredients from List 1 | Extract key beneficial ingredients from List 2 |
            | **Vitamins & Minerals** | Extract vitamins & minerals from List 1 | Extract vitamins & minerals from List 2 |
            | **Whole Grains** | Extract whole grains from List 1 | Extract whole grains from List 2 |

            ---
            ## **⚠ 2️⃣ Harmful Ingredients Comparison**
            | Category | Ingredient List 1 | Ingredient List 2 |
            |----------|------------------|------------------|
            | **Added Sugars** | Extract harmful sugars from List 1 | Extract harmful sugars from List 2 |
            | **Unhealthy Fats** | Extract unhealthy fats from List 1 | Extract unhealthy fats from List 2 |
            | **Additives & Preservatives** | Extract artificial additives from List 1 | Extract artificial additives from List 2 |

            ---
            ## **📊 3️⃣ Ingredient List Health Rating**
            | Ingredient List | Health Score (Out of 10) | Explanation |
            |---------------|------------------|--------------|
            | **List 1** | Extract Health Score from List 1 | Short explanation for List 1 |
            | **List 2** | Extract Health Score from List 2 | Short explanation for List 2 |

            ---

            ## **🏆 4️⃣ Final Verdict**
            | Aspect | Verdict |
            |--------|---------|
            | **Which ingredient list is better and why?** | Short explanation comparing the two lists |
            | **Concise Summary** | Brief summary highlighting key differences and main points |
            | **Healthiest Option?** | Clearly state if one is better or if both are equally unhealthy |

            """

            # ✅ Ensure AI returns structured comparison
            full_comparison = get_gemini_response([], comparison_prompt)

            # ✅ Format AI response for HTML
            formatted_comparison = format_response(full_comparison)

            # ✅ Extract **comparison table** and **verdict separately**
            verdict_pattern = re.search(r"🏆 4️⃣ Final Verdict([\s\S]*)", formatted_comparison)

            if verdict_pattern:
                verdict_text = verdict_pattern.group(1).strip()
            else:
                verdict_text = "No clear verdict generated."

            # ✅ **Ensure verdict is properly formatted**
            verdict_lines = [line.strip() for line in verdict_text.split("\n") if line.strip()]
            while len(verdict_lines) < 3:
                verdict_lines.append("Not provided")  # Fill missing lines

            # ✅ **Format the Verdict as a Simple Structured Text Block**
            verdict_text = f"""
            <b>🏆 Overall Verdict</b><br><br>
            <b>Which ingredient list is better and why?</b><br> {verdict_lines[0]}<br><br>
            <b>Concise Summary:</b><br> {verdict_lines[1]}<br><br>
            <b>Healthiest Option?</b><br> {verdict_lines[2]}<br>
            """

            # ✅ Extract only the structured **comparison table** part
            comparison_data = formatted_comparison.replace(verdict_text, "")
        else:
            flash("⚠ Please upload valid images for both ingredient lists.", "error")

    return render_template('compare.html', response1=response1, response2=response2, comparison_data=comparison_data, verdict=verdict_table)







if __name__ == '__main__':
    app.run(debug=True)
