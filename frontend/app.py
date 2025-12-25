import streamlit as st
import time
import base64

st.set_page_config(page_title="RetinaCheck", layout="centered")

# -------- Helper ----------
def get_image_base64(file):
    return base64.b64encode(file.getvalue()).decode()

# -------- Styles ----------
st.markdown("""
<style>
body {background:#ffffff;}
.header {text-align:center;font-size:42px;font-weight:800;color:#00b3b3;}
.sub {text-align:center;color:#555;margin-bottom:30px;}

.upload-box {
border:2px dashed #00b3b3;
border-radius:18px;
padding:20px;
text-align:center;
margin-top:20px;
width:244px;
height:244px;
display:flex;            /* make it flex */
justify-content:center;  /* horizontal center */
align-items:center;      /* vertical center */
margin:20px auto 0 auto; /* center box itself horizontally */
}
        
.upload-box:hover {
box-shadow:0 0 18px rgba(0,179,179,0.4);
transform:scale(1.01);
transition:0.3s;
}
.upload-box img {
max-width:100%;
}

.main-btn {
background:#00b3b3;
color:white;
padding:15px;
border-radius:14px;
font-size:20px;
width:100%;
}

.main-btn:hover {
background:#009999;
transform:scale(1.02);
transition:0.3s;
}

.result-card {
border-radius:16px;
padding:22px;
text-align:center;
font-weight:700;
animation:fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
from {opacity:0; transform:translateY(10px);}
to {opacity:1; transform:translateY(0);}
}

.detected-card {
background:#00b3b3;
color:white;
box-shadow:0 0 25px rgba(0,179,179,0.6);
}

.notdetected-card {
background:#f7f7f7;
border:2px solid #00b3b3;
color:#00b3b3;
}
</style>
""", unsafe_allow_html=True)

# -------- UI ----------
st.markdown("<div class='header'>RetinaCheck 👁</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Check your eye with retinal disease detection.</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg","png","jpeg"])
    
if uploaded:
    img_b64 = get_image_base64(uploaded)
    st.markdown(f"""
        <div class='upload-box'>
            <img src='data:image/png;base64,{img_b64}'>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing retina..."):
            time.sleep(2)

        result = "Cataract"   # fake for now

        st.progress(100)
        st.markdown("### Detection Result")

        col1,col2,col3,col4 = st.columns(4)
        diseases = ["Cataract","Glaucoma","Diabetic Retinopathy","Normal"]

        for col,d in zip([col1,col2,col3,col4], diseases):
            with col:
                if d == result:
                    st.markdown(f"<div class='result-card detected-card'>{d}<br>Detected</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-card notdetected-card'>{d}<br>Not Detected</div>", unsafe_allow_html=True)
