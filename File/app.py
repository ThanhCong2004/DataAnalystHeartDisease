import streamlit as st
import pandas as pd
import joblib
from groq import Groq  # Dùng thư viện Groq thay cho Google

# --- 1. CẤU HÌNH AI (SỬ DỤNG GROQ - MIỄN PHÍ & NHANH) ---
# Dán API Key từ Groq vào đây
GROQ_API_KEY = "#"


def call_ai_advisor(user_data, risk_score):
    if not GROQ_API_KEY.startswith("gsk_"):
        return "⚠️ Vui lòng cấu hình API Key của Groq để nhận tư vấn AI."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Tạo nội dung gửi cho AI
        content = f"""
        Bệnh nhân: {user_data['age']} tuổi, BMI: {user_data['bmi']}, Huyết áp: {user_data['systolic_bp']}/{user_data['diastolic_bp']}.
        Lối sống: Bước chân {user_data['daily_steps']}, Ngủ {user_data['sleep_hours']}h, Hút thuốc: {user_data['smoker']}, Rượu: {user_data['alcohol']}.
        Rủi ro dự báo: {risk_score:.1%}.
        Hãy đóng vai bác sĩ, đưa ra nhận xét ngắn gọn và 3 lời khuyên sức khỏe cụ thể bằng tiếng Việt.
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="llama-3.3-70b-versatile",  # Model cực mạnh và miễn phí
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi gọi AI: {str(e)}"


# --- 2. LOAD MODEL ML ---
@st.cache_resource
def load_assets():
    m = joblib.load('health_risk_pipeline.pkl')
    f = joblib.load('feature_columns.pkl')
    return m, f


model, features = load_assets()

# --- 3. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Health Predictor", layout="centered")

# Làm đẹp CSS
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .main-card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
    .stNumberInput, .stSelectbox { border-radius: 8px !important; }
    .btn-predict { background-color: #2E86C1; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Chẩn Đoán Sức Khỏe Thông Minh")
st.write("Vui lòng nhập chính xác các chỉ số để hệ thống phân tích.")

# --- 4. FORM NHẬP LIỆU (Trên cùng 1 trang) ---
with st.form("input_form"):
    st.markdown("### 📝 Chỉ số cơ bản")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Tuổi", 1, 100, 30)
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        bmi = st.number_input("BMI (Chỉ số khối)", 10.0, 50.0, 23.0)
        systolic = st.number_input("Huyết áp tâm thu", 80, 200, 120)
        diastolic = st.number_input("Huyết áp tâm trương", 50, 130, 80)
        cholesterol = st.number_input("Cholesterol", 100, 400, 190)
    with col2:
        steps = st.number_input("Số bước chân/ngày", 0, 30000, 7000)
        sleep = st.number_input("Giờ ngủ/đêm", 3.0, 12.0, 7.5)
        water = st.number_input("Lượng nước (Lít)", 0.5, 5.0, 2.0)
        calories = st.number_input("Calo tiêu thụ", 1000, 5000, 2200)
        resting_hr = st.number_input("Nhịp tim lúc nghỉ", 40, 120, 72)
        family = st.selectbox("Gia đình có tiền sử bệnh?", ["No", "Yes"])

    st.markdown("### 🚬 Thói quen sinh hoạt")
    # Sửa giao diện Hút thuốc & Rượu bia nằm ngang gọn đẹp
    c_smoke, c_alc = st.columns(2)
    with c_smoke:
        smoker = st.radio("Có hút thuốc không?", ["No", "Yes"], horizontal=True)
    with c_alc:
        alcohol = st.radio("Có uống rượu bia không?", ["No", "Yes"], horizontal=True)

    predict_btn = st.form_submit_button("🚀 PHÂN TÍCH & DỰ BÁO")

# --- 5. HIỂN THỊ KẾT QUẢ (Ngay dưới Form) ---
if predict_btn:
    # Xử lý dữ liệu
    input_dict = {
        'age': age, 'bmi': bmi, 'daily_steps': steps, 'sleep_hours': sleep,
        'water_intake_l': water, 'calories_consumed': calories,
        'smoker': 1 if smoker == "Yes" else 0,
        'alcohol': 1 if alcohol == "Yes" else 0,
        'resting_hr': resting_hr, 'systolic_bp': systolic, 'diastolic_bp': diastolic,
        'cholesterol': cholesterol, 'family_history': 1 if family == "Yes" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Female': 1 if gender == "Female" else 0
    }

    df_input = pd.DataFrame([input_dict])[features]
    prob = model.predict_proba(df_input)[0][1]

    # Vùng kết quả
    st.markdown("---")
    st.subheader("📊 Kết quả dự báo")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Xác suất rủi ro", f"{prob:.1%}")
        if prob > 0.35:
            st.error("PHÂN LOẠI: NGUY CƠ CAO")
        else:
            st.success("PHÂN LOẠI: AN TOÀN")

    with res_col2:
        st.progress(prob)
        st.write("Thanh biểu diễn mức độ rủi ro (0% - 100%)")

    # Gọi AI nhận xét
    st.markdown("### 🤖 Tư vấn từ Bác sĩ AI")
    with st.spinner("AI đang phân tích các chỉ số..."):
        # Ghi chú: smoker và alcohol ở đây gửi chuỗi Yes/No cho AI dễ hiểu
        user_info_for_ai = input_dict.copy()
        user_info_for_ai['smoker'] = smoker
        user_info_for_ai['alcohol'] = alcohol

        advice = call_ai_advisor(user_info_for_ai, prob)
        st.info(advice)