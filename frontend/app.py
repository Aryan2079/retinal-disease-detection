import streamlit as st
import time

st.set_page_config(page_title="RetinaCheck", layout="centered")

st.markdown("""
<style>
body {background:#ffffff;}
.header {text-align:center;font-size:42px;font-weight:800;color:#00b3b3;}
.sub {text-align:center;color:#555;margin-bottom:30px;}

.upload-box {
border:2px dashed #00b3b3;
border-radius:18px;
padding:35px;
text-align:center;
}
            
.upload-box:hover {
box-shadow: 0 0 18px rgba(0,179,179,0.4);
transform: scale(1.01);
transition: 0.3s;
}

.detected {
box-shadow: 0 0 25px rgba(0,179,179,0.6);
}


.main-btn {
background:#00b3b3;
color:white;
padding:15px;
border-radius:14px;
font-size:20px;
width:100%;
}

.result-card {
border-radius:16px;
padding:22px;
text-align:center;
font-weight:700;
}

.detected {
background:#00b3b3;
color:white;
}

.notdetected {
background:#f7f7f7;
border:2px solid #00b3b3;
color:#00b3b3;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>RetinaCheck \U0001F441</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Check you eye with retinal disease detection. </div>", unsafe_allow_html=True)

uploaded = st.file_uploader("")

if uploaded:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.image(uploaded, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing retina..."):
            time.sleep(2)

        result = "Cataract"  # fake now

        st.progress(100)
        st.markdown("### Detection Result")

        col1,col2,col3,col4 = st.columns(4)
        diseases = ["Cataract","Glaucoma","Diabetic Retinopathy","Normal"]

        for col,d in zip([col1,col2,col3,col4], diseases):
            with col:
                if d == result:
                    st.markdown(f"<div class='result-card detected'>{d}<br>Detected</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-card notdetected'>{d}<br>Not Detected</div>", unsafe_allow_html=True)
