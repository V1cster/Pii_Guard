import streamlit as st
from ultralytics import YOLO
from transformers import pipeline
import PIL.Image
from PIL.ExifTags import TAGS
import random
# Handle iPhone HEIC files since a lot of people are going to be using Iphones
import pillow_heif

# --- 1. INITIALIZE HEIC SUPPORT ---
pillow_heif.register_heif_opener()

# --- 2. PAGE CONFIG & ETHICS ---
st.set_page_config(page_title="PII-Guard Local", layout="wide")

# Top Header with UoG Logo
head1, head2 = st.columns([0.8, 0.2])
with head1:
    st.title("🛡️ PII-Guard: Local Privacy Shield")
    st.write("Analyze your social media posts for private data before you upload.")
with head2:
    try:
        st.image("uog_logo.png", width=150)
    except:
        st.info("🎓 UoG Logo Placeholder")

# Ethics Sidebar
if 'w_code' not in st.session_state:
    st.session_state.w_code = random.randint(1000, 9999)

st.sidebar.title("🔐 Research Session")
st.sidebar.info(f"Your Withdrawal Code: **{st.session_state.w_code}**")
st.sidebar.write("Please keep this code for survey.")

with st.expander("ℹ️ Privacy & Ethics Disclosure"):
    st.write(
        "All processing happens locally on your device. No data is sent to the cloud.")

# --- 3. LOAD MODELS ---
# YOLO for Images (Local .pt file)
image_model = YOLO("yolo11n.pt")

# [NEW] Load BART for Intent Analysis
intent_classifier = pipeline("zero-shot-classification",
                             model="facebook/bart-large-mnli")

# BERT for Text (download to D: drive)
# 'dslim/bert-base-NER' because it is public everything else is private and just won't load.
text_pipe = pipeline("token-classification",
                     model="dslim/bert-base-NER",
                     aggregation_strategy="simple")

# --- IMPROVED RISK SCORING FUNCTION ---


def calculate_advanced_risk(text_results, i_results, user_text):
    score = 0
    alerts = []

    # Analyze Intent (Context)
    intent = intent_classifier(user_text, candidate_labels=[
                               "home/private", "public/social"])
    intent_label = intent['labels'][0]

    # Multiplier: 2x risk if the AI thinks the person is at home
    multiplier = 2.0 if intent_label == "home/private" else 1.0

    # Check Text results
    for res in text_results:
        label = res.get('entity_group', 'Unknown')
        if label == "LOC":
            score += 25 * multiplier
            alerts.append(f"Location found in {intent_label} context.")
        if label == "PER":
            score += 15
            alerts.append("Personal Name detected.")

    # Check Image results
    has_person = any(i_results[0].names[int(b.cls)]
                     == "person" for b in i_results[0].boxes)
    if has_person:
        score += 20 * multiplier
        alerts.append(f"Recognizable face in {intent_label} setting.")

    return min(score, 100), intent_label, alerts


# --- 4. UI LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.header("Step 1: Text Analysis")
    user_text = st.text_area(
        "Enter your caption:", placeholder="e.g., I am Victor and I live in London")

    if st.button("Analyze Text"):
        if user_text:
            # The AI scans the text here
            results = text_pipe(user_text)
            if results:
                for res in results:
                    # 'entity_group' tells us if it's a Person (PER) or Location (LOC)
                    label = res.get('entity_group', 'Unknown')
                    score = res['score']
                    st.warning(
                        f"⚠️ {label} detected (Confidence: {score:.2f})")
            else:
                st.success("✅ No PII detected in text.")
        else:
            st.error("Please enter text first.")

with col2:
    st.header("Step 2: Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload image:", type=["jpg", "jpeg", "png", "heic"])

    if uploaded_file:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="Current Image", use_container_width=True)

        # deep dive logic looks at the EXIF data for coordinates or camera serial numbers
        exif_data = image.getexif()
        if exif_data:
            has_gps = False
            for tag_id in exif_data:
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'GPSInfo':
                    has_gps = True

            if has_gps:
                st.error("🚨 METADATA ALERT: This image contains embedded GPS coordinates. Sharing this file will reveal your exact location (longitude/latitude) even if your face is blurred.")
            else:
                st.info(
                    "ℹ️ Metadata Check: No hidden GPS coordinates found in this file.")
        else:
            st.write("No metadata found in this image.")

        if st.button("Scan Image"):
            img_results = image_model.predict(image)
            st.subheader("Found Objects:")
            for box in img_results[0].boxes:
                name = img_results[0].names[int(box.cls)]
                st.write(f"• Detected: {name}")

# --- 5. THE FUSION LOGIC (giving the software context) ---
st.divider()
st.header("Step 3: Final Privacy Verdict")
if st.button("Generate Risk Report"):
    if user_text and uploaded_file:
        # Re-run scans for the fusion logic
        t_res = text_pipe(user_text)
        i_res = image_model.predict(PIL.Image.open(uploaded_file))

        # Call the Advanced Risk Brain
        final_score, context, reasons = calculate_advanced_risk(
            t_res, i_res, user_text)

        # --- [RISK METER] ---
        st.subheader(f"Total Privacy Risk Score: {final_score}%")

        # Use st.progress to create the visual meter
        if final_score >= 75:
            st.progress(final_score / 100, text="🔴 CRITICAL RISK")
            st.error(
                "Highly sensitive data combined with private context detected.")
        elif final_score >= 35:
            st.progress(final_score / 100, text="🟡 MEDIUM RISK")
            st.warning("Potential PII detected. Review context before sharing.")
        else:
            st.progress(final_score / 100, text="🟢 LOW RISK")
            st.success("No significant contextual leaks found.")

        # Display the "Why" behind the score
        st.write(f"**Detected Intent:** {context.upper()}")
        st.write("**Risk Factors identified:**")
        for r in reasons:
            st.write(f"- {r}")

    else:
        st.info("Please complete Steps 1 & 2 first to generate the report.")
