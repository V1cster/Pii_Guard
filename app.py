import streamlit as st
from ultralytics import YOLO
from transformers import pipeline
import PIL.Image
import random

# --- 1. PAGE CONFIG & ETHICS ---
st.set_page_config(page_title="PII-Guard Local", layout="wide")

# This creates the 4-digit code need for the ethics/feedback form
if 'w_code' not in st.session_state:
    st.session_state.w_code = random.randint(1000, 9999)

st.sidebar.title("🔐 Research Session")
st.sidebar.info(f"Your Withdrawal Code: **{st.session_state.w_code}**")
st.sidebar.write("Please make a note of this Code.")

st.title("🛡️ PII-Guard: Local Privacy Shield")
st.write("Analyze your social media posts for private data before you upload.")

# --- 2. LOAD MODELS ---
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


# --- 3. UI LAYOUT ---
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
        "Upload image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="Current Image", use_container_width=True)

        if st.button("Scan Image"):
            img_results = image_model.predict(image)
            st.subheader("Found Objects:")
            for box in img_results[0].boxes:
                name = img_results[0].names[int(box.cls)]
                st.write(f"• Detected: {name}")

# --- 4. THE FUSION LOGIC (giving the software context) ---
st.divider()
st.header("Step 3: Final Privacy Verdict")
if st.button("Generate Risk Report"):
    if user_text and uploaded_file:
        # Re-run scans for the fusion
        t_res = text_pipe(user_text)
        i_res = image_model.predict(PIL.Image.open(uploaded_file))

        # the Advanced Risk Brain
        final_score, context, reasons = calculate_advanced_risk(
            t_res, i_res, user_text)

        st.subheader(f"Risk Score: {final_score}%")
        st.write(f"**Detected Intent:** {context.upper()}")

        for r in reasons:
            st.write(f"- {r}")

        if final_score > 70:
            st.error("🚨 CRITICAL: High probability of a contextual privacy leak.")
        elif final_score > 30:
            st.warning(
                "⚠️ MEDIUM: Some risks detected. Consider redacting details.")
        else:
            st.success("🟢 LOW: No major contextual leaks found.")
    else:
        st.info("Complete Steps 1 & 2 first.")
