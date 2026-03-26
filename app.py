import streamlit as st
from ultralytics import YOLO
from transformers import pipeline
import PIL.Image
import random

# --- 1. PAGE CONFIG & ETHICS ---
st.set_page_config(page_title="PII-Guard Local", layout="wide")

# This creates the 4-digit code you need for your ethics/feedback form
if 'w_code' not in st.session_state:
    st.session_state.w_code = random.randint(1000, 9999)

st.sidebar.title("🔐 Research Session")
st.sidebar.info(f"Your Withdrawal Code: **{st.session_state.w_code}**")
st.sidebar.write("Please keep this code for your feedback survey.")

st.title("🛡️ PII-Guard: Local Privacy Shield")
st.write("Analyze your social media posts for private data before you upload.")

# --- 2. LOAD MODELS ---
# YOLO for Images (Local .pt file)
image_model = YOLO("yolo11n.pt")

# BERT for Text (Public model - will download to D: drive automatically)
# We use 'dslim/bert-base-NER' because it is public and very stable.
text_pipe = pipeline("token-classification",
                     model="dslim/bert-base-NER",
                     aggregation_strategy="simple")

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

# --- 4. THE FUSION LOGIC (The 'Context' Step) ---
st.divider()
st.header("Step 3: Final Privacy Verdict")
if st.button("Generate Risk Report"):
    if user_text and uploaded_file:
        # Run text check
        t_res = text_pipe(user_text)
        labels = [r.get('entity_group') for r in t_res]

        # Run image check
        i_res = image_model.predict(PIL.Image.open(uploaded_file))
        has_person = any(i_res[0].names[int(b.cls)] ==
                         "person" for b in i_res[0].boxes)

        # Logic: If text has a Location and Image has a Person = High Risk
        if "LOC" in labels and has_person:
            st.error(
                "🚨 CRITICAL: Contextual Leak! You are posting a recognizable face and a specific location.")
        elif labels:
            st.warning("⚠️ MEDIUM: Private identifiers found in text.")
        else:
            st.success("🟢 LOW: No major contextual leaks detected.")
    else:
        st.info("Complete Steps 1 & 2 first.")
