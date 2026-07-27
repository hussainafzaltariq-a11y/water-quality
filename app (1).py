import gradio as gr
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import requests
import json

warnings.filterwarnings('ignore')
# -------------------------------------------------------------------------
# Hugging Face Free LLM (AI-Powered Urdu Guide)
# -------------------------------------------------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

# HF_TOKEN ko environment variable se lein
HF_TOKEN = os.getenv("HF_TOKEN")

def get_ai_explanation(prediction, feature_importance_text, cleaned_inputs):
    """Generates detailed Urdu guide for farmers using Hugging Face LLM."""
    
    # --- SMART FALLBACK GUIDE (Always Works) ---
    prediction_advice = {
        'Excellent': 'آپ کا پانی بہترین معیار کا ہے۔ یہ پینے اور کھیتی باڑی دونوں کے لیے محفوظ ہے۔',
        'Good': 'آپ کا پانی اچھے معیار کا ہے۔ پینے سے پہلے ابال لیں تو بہتر ہے۔',
        'Fair': 'آپ کا پانی قابلِ قبول ہے لیکن پینے سے پہلے ضرور ابالیں یا فلٹر کریں۔',
        'Marginal': 'آپ کا پانی معمولی معیار کا ہے۔ اسے پینے کے لیے استعمال نہ کریں، صرف کھیتی باڑی کے لیے استعمال کریں۔',
        'Poor': 'آپ کا پانی ناقص معیار کا ہے۔ اسے کسی بھی استعمال میں نہ لائیں۔ فوری طور پر ماہر سے رابطہ کریں۔'
    }
    
    # Feature-specific tips
    feature_tips = []
    for i, val in enumerate(cleaned_inputs):
        med = MEDIANS[i]
        iqr = IQRS[i]
        if abs(val - med) / (iqr + 0.001) > 0.5:
            feature_name = FEATURE_NAMES[i]
            if 'Ammonia' in feature_name and val > med:
                feature_tips.append('پانی میں امونیا زیادہ ہے۔ ابال کر پئیں۔')
            elif 'Oxygen' in feature_name and val < med:
                feature_tips.append('آکسیجن کم ہے۔ پانی کو ہوا دار رکھیں۔')
            elif 'pH' in feature_name and (val < 6.5 or val > 8.5):
                feature_tips.append('پی ایچ لیول بہتر کریں۔ تھوڑا سا لیموں یا چونا ڈالیں۔')
            elif 'Nitrogen' in feature_name and val > med:
                feature_tips.append('نائٹروجن زیادہ ہے۔ پانی کو فلٹر کریں۔')
            elif 'Nitrate' in feature_name and val > med:
                feature_tips.append('نائیٹریٹ زیادہ ہے۔ بچوں اور بوڑھوں کو نہ پلائیں۔')
            elif 'BOD' in feature_name and val > med:
                feature_tips.append('آکسیجن کی طلب زیادہ ہے۔ پانی کو ہوا دار بنائیں۔')
            elif 'Orthophosphate' in feature_name and val > med:
                feature_tips.append('فاسفیٹ زیادہ ہے۔ کھیتی کے لیے موزوں نہیں۔')
            elif 'Temperature' in feature_name and (val < 5 or val > 30):
                feature_tips.append('پانی کا درجہ حرارت بہت کم یا زیادہ ہے۔')
    
    fallback_message = prediction_advice.get(prediction, 'پانی کی کیفیت جانچیں۔')
    if feature_tips:
        fallback_message += " " + " ".join(feature_tips)
    else:
        fallback_message += " تمام پیرامیٹرز نارمل ہیں۔"
    
    # --- TRY HUGGING FACE LLM ---
    try:
        prompt = f"""You are a water quality assistant for Pakistani farmers. 
Respond ONLY in SIMPLE URDU. 
Prediction: {prediction}. 
Important factors: {feature_importance_text}.
Give a 2-sentence practical guide for the farmer.

Example:
Prediction: Bad (pH high, Ammonia high).
Response: "Aap ke pani mein Ammonia aur pH zyada hai. Is ko ubal kar piyein, ya kisi maahir se contact karein."
"""
        payload = {"inputs": prompt}
        
        # =============================================================
        # 👇 YAHAN HEADERS WALA CODE ADD KARO
        # =============================================================
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        response = requests.post(
            HF_API_URL, 
            json=payload, 
            headers=headers,  # 👈 YEH LINE IMPORTANT HAI
            timeout=5
        )
        # =============================================================
        
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
            if generated_text and len(generated_text) > 10:
                return generated_text.replace(prompt, "").strip() or fallback_message
    except Exception as e:
        print(f"⚠️ LLM API error, using fallback: {e}")
    
    return fallback_message
# -------------------------------------------------------------------------
# Load Model and Scaler
# -------------------------------------------------------------------------
MODEL_PATH = "lightgbm_split2_best_model2 .pkl"
SCALER_PATH = "robust_scaler .pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    raise e

# Feature Definitions
FEATURE_NAMES = [
    'Ammonia (mg/l)', 'Biochemical Oxygen Demand (mg/l)', 
    'Dissolved Oxygen (mg/l)', 'Orthophosphate (mg/l)', 
    'pH (ph units)', 'Temperature (cel)', 'Nitrogen (mg/l)', 
    'Nitrate (mg/l)'
]

# Medians and IQRs from the RobustScaler
MEDIANS = [0.066, 2.7, 10.2, 0.144, 7.78, 11.46, 4.98, 4.5]
IQRS = [0.438, 1.34, 0.93, 0.247, 0.39, 5.1, 6.01, 4.25]

CLASSES = ['Excellent', 'Good', 'Fair', 'Marginal', 'Poor']

# Custom CSS for glassmorphic design
CUSTOM_CSS = """
:root {
    --primary-cyan: #06b6d4;
    --primary-blue: #3b82f6;
    --bg-dark: #0f172a;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', sans-serif;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    backdrop-filter: blur(20px) !important;
}

.top-margin {
    margin-top: 20px !important;
}

.action-btn {
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4) !important;
}

.clear-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
    transition: all 0.3s ease !important;
}

.clear-btn:hover {
    background: rgba(255, 255, 255, 0.08) !important;
}

.badge-excellent {
    background: linear-gradient(135deg, #10b981, #34d399) !important;
    color: white !important;
    padding: 8px 24px !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

.badge-good {
    background: linear-gradient(135deg, #34d399, #6ee7b7) !important;
    color: white !important;
    padding: 8px 24px !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

.badge-fair {
    background: linear-gradient(135deg, #f59e0b, #fbbf24) !important;
    color: white !important;
    padding: 8px 24px !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

.badge-marginal {
    background: linear-gradient(135deg, #f97316, #fb923c) !important;
    color: white !important;
    padding: 8px 24px !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

.badge-poor {
    background: linear-gradient(135deg, #ef4444, #f87171) !important;
    color: white !important;
    padding: 8px 24px !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
}

h1, h2, h3, h4 {
    color: var(--text-primary) !important;
}

label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

input, select, textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
"""

# -------------------------------------------------------------------------
# Explainable AI & Charting Helpers
# -------------------------------------------------------------------------
def plot_probabilities(probs):
    """Generates a vertical bar chart of prediction confidence by class."""
    colors = ['#10b981', '#34d399', '#f59e0b', '#f97316', '#ef4444']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    bars = ax.bar(CLASSES, probs * 100, color=colors, width=0.5, edgecolor='none')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.1, color='#94a3b8')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color='#f8fafc', fontsize=10, fontweight='bold')
        
    ax.set_title("Class Confidence (%)", color='#06b6d4', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Probability (%)", color='#94a3b8', fontsize=11)
    ax.set_ylim(0, 115)
    
    plt.tight_layout()
    return fig

def plot_feature_importance():
    """Generates a horizontal bar chart of global feature importances."""
    try:
        importances = model.feature_importances_
    except:
        # Fallback importances if model doesn't have feature_importances_
        importances = [3614, 3008, 873, 3838, 959, 707, 1371, 602]
    
    sorted_idx = np.argsort(importances)
    sorted_features = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    bars = ax.barh(sorted_features, sorted_importances, color='#3b82f6', edgecolor='none', height=0.6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.1, color='#94a3b8')
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f' {width:.0f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', color='#94a3b8', fontsize=9)
        
    ax.set_title("Global Feature Importance", color='#06b6d4', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Importance Score", color='#94a3b8', fontsize=11)
    
    plt.tight_layout()
    return fig

def explain_prediction(inputs):
    """Calculates local feature contribution rankings based on deviations from normal medians."""
    impacts = []
    
    for i in range(8):
        val = inputs[i]
        med = MEDIANS[i]
        iqr = IQRS[i]
        
        if FEATURE_NAMES[i] == 'Dissolved Oxygen (mg/l)':
            impact = (med - val) / iqr
            if val < med:
                status = f"Low ({val:.2f} vs normal {med:.2f} mg/l)"
                sig = "⚠️ Reduces aquatic life support"
                color = "#ef4444"
            else:
                status = f"Optimal ({val:.2f} mg/l)"
                sig = "✅ High oxygenation, excellent health"
                color = "#10b981"
        elif FEATURE_NAMES[i] == 'pH (ph units)':
            impact = abs(val - med) / iqr
            if val < 6.5:
                status = f"Acidic ({val:.2f} vs neutral {med:.2f})"
                sig = "⚠️ Acidic conditions degrade WQI"
                color = "#ef4444"
            elif val > 8.5:
                status = f"Alkaline ({val:.2f} vs neutral {med:.2f})"
                sig = "⚠️ Alkaline levels can be toxic"
                color = "#ef4444"
            else:
                status = f"Optimal ({val:.2f})"
                sig = "✅ Balanced neutral pH"
                color = "#10b981"
        else:
            impact = (val - med) / iqr
            if val > med:
                status = f"High ({val:.2f} vs normal {med:.2f} mg/l)"
                sig = f"⚠️ Pollutant elevation (x{val/med:.1f})" if med > 0 else "⚠️ Elevated level"
                color = "#ef4444" if impact > 1 else "#f97316"
            else:
                status = f"Low/Normal ({val:.2f} mg/l)"
                sig = "✅ Safe background concentration"
                color = "#10b981"
                
        impacts.append({
            'feature': FEATURE_NAMES[i],
            'val': val,
            'impact': impact,
            'status': status,
            'sig': sig,
            'color': color
        })
        
    sorted_impacts = sorted(impacts, key=lambda x: x['impact'], reverse=True)
    
    explanation_html = """
    <div style='margin-top: 15px;'>
        <h4 style='color: #06b6d4; margin-bottom: 12px; font-size: 1.15rem; font-weight: 600;'>
            🔍 Local Feature Contribution Rankings
        </h4>
        <div style='display: flex; flex-direction: column; gap: 10px;'>
    """
    
    for item in sorted_impacts:
        feat = item['feature']
        status = item['status']
        sig = item['sig']
        color = item['color']
        
        explanation_html += f"""
        <div style='background: rgba(255,255,255,0.02); border-left: 4px solid {color}; padding: 10px 14px; border-radius: 6px;'>
            <div style='display: flex; justify-content: space-between; font-size: 0.95rem;'>
                <span style='font-weight: 500; color: #f8fafc;'>{feat}</span>
                <span style='color: {color}; font-weight: 600;'>{status}</span>
            </div>
            <div style='font-size: 0.85rem; color: #94a3b8; margin-top: 4px;'>{sig}</div>
        </div>
        """
        
    explanation_html += """
        </div>
    </div>
    """
    return explanation_html

# -------------------------------------------------------------------------
# Core Prediction Routine
# -------------------------------------------------------------------------
def predict_water_quality(ammonia, bod, do, orthophosphate, ph, temp, nitrogen, nitrate):
    try:
        inputs = [ammonia, bod, do, orthophosphate, ph, temp, nitrogen, nitrate]
        
        cleaned_inputs = []
        for i, val in enumerate(inputs):
            if val is None or str(val).strip() == "" or (isinstance(val, float) and np.isnan(val)):
                cleaned_inputs.append(MEDIANS[i])
            else:
                try:
                    cleaned_inputs.append(float(val))
                except ValueError:
                    error_html = f"""
                    <div style='text-align: center; padding: 15px; border-radius: 8px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3);'>
                        <h4 style='color: #ef4444; margin: 0;'>⚠️ Input Validation Error</h4>
                        <p style='color: #94a3b8; margin: 5px 0 0 0; font-size: 0.95rem;'>
                            '{val}' is not a valid number for <b>{FEATURE_NAMES[i]}</b>.
                        </p>
                    </div>
                    """
                    return error_html, None, "<div style='color:#ef4444;'>Validation failed.</div>"

        inputs_df = pd.DataFrame([cleaned_inputs], columns=FEATURE_NAMES)
        scaled_inputs = scaler.transform(inputs_df)
        
        probs = model.predict_proba(scaled_inputs)[0]
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class] * 100
        
        class_name = CLASSES[pred_class]
        
        badge_html = f"""
        <div style='text-align: center; padding: 20px; border-radius: 16px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);'>
            <span class='badge-{class_name.lower()}'>{class_name}</span>
            <p style='font-size: 1.3rem; margin-top: 16px; color: #f8fafc; margin-bottom: 0;'>
                Model Confidence: <span style='color: #06b6d4; font-weight: bold;'>{confidence:.2f}%</span>
            </p>
        </div>
        """
        
        prob_chart = plot_probabilities(probs)
        explanation_html = explain_prediction(cleaned_inputs)
        
        # =============================================================
        # URDU EXPLANATION (Hugging Face Free Model)
        # =============================================================
        try:
            feature_text = ""
            for i, val in enumerate(cleaned_inputs):
                med = MEDIANS[i]
                iqr = IQRS[i]
                if abs(val - med) / (iqr + 0.001) > 0.5:
                    feature_text += f"{FEATURE_NAMES[i]} ({val:.2f}), "
            if not feature_text:
                feature_text = "All parameters normal"
            else:
                feature_text = feature_text.rstrip(", ")
            
            urdu_explanation = get_ai_explanation(class_name, feature_text, cleaned_inputs)
            
            urdu_html = f"""
            <div style='margin-top: 20px; padding: 16px; border-radius: 8px; background: rgba(6, 182, 212, 0.05); border: 1px solid rgba(6, 182, 212, 0.2);'>
                <h4 style='color: #06b6d4; margin: 0 0 8px 0;'>🇺🇷 Urdu Summary for Farmers</h4>
                <p style='color: #f8fafc; font-size: 1.1rem; line-height: 1.6; margin: 0;'>
                    {urdu_explanation}
                </p>
            </div>
            """
            full_explanation = explanation_html + urdu_html
            
        except Exception as e:
            print(f"⚠️ Urdu translation error: {e}")
            urdu_html = f"""
            <div style='margin-top: 20px; padding: 16px; border-radius: 8px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3);'>
                <h4 style='color: #ef4444; margin: 0 0 8px 0;'>⚠️ Service Unavailable</h4>
                <p style='color: #f8fafc; font-size: 1rem; line-height: 1.6; margin: 0;'>
                    Prediction: {class_name}. Please consult local water expert.
                </p>
            </div>
            """
            full_explanation = explanation_html + urdu_html
        
        return badge_html, prob_chart, full_explanation

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        err_html = f"""
        <div style='text-align: center; padding: 15px; border-radius: 8px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3);'>
            <h4 style='color: #ef4444; margin: 0;'>⚠️ Prediction Execution Failed</h4>
            <p style='color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9rem;'>{str(e)}</p>
        </div>
        """
        return err_html, None, f"<pre style='color:#ef4444; font-size: 0.85rem;'>{error_details}</pre>"

def predict_hybrid(
    s_ammonia, s_bod, s_do, s_orthophosphate, s_ph, s_temp, s_nitrogen, s_nitrate,
    t_ammonia, t_bod, t_do, t_orthophosphate, t_ph, t_temp, t_nitrogen, t_nitrate
):
    def resolve_val(text_val, slider_val):
        if text_val is not None and str(text_val).strip() != "":
            return text_val.strip()
        return slider_val

    ammonia = resolve_val(t_ammonia, s_ammonia)
    bod = resolve_val(t_bod, s_bod)
    do = resolve_val(t_do, s_do)
    orthophosphate = resolve_val(t_orthophosphate, s_orthophosphate)
    ph = resolve_val(t_ph, s_ph)
    temp = resolve_val(t_temp, s_temp)
    nitrogen = resolve_val(t_nitrogen, s_nitrogen)
    nitrate = resolve_val(t_nitrate, s_nitrate)
    
    return predict_water_quality(ammonia, bod, do, orthophosphate, ph, temp, nitrogen, nitrate)

# -------------------------------------------------------------------------
# Build Gradio Interface
# -------------------------------------------------------------------------
with gr.Blocks(title="🌊 Water Quality Prediction Dashboard", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='margin: 0; font-size: 2.5rem; background: linear-gradient(90deg, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🌊 Water Quality Prediction Dashboard
        </h1>
        <p style='margin: 10px 0 0 0; color: #94a3b8; font-size: 1.1rem;'>
            AI-Powered Water Quality Assessment using LightGBM Machine Learning
        </p>
    </div>
    """)

    # Main Grid
    with gr.Row():
        
        # Left Panel - Inputs
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.HTML("<h3 style='color: #06b6d4; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px; margin-top: 0;'>💧 Water Metrics Configurator</h3>")
            
            with gr.Tabs():
                with gr.TabItem("🎛️ Guided Sliders"):
                    s_ammonia = gr.Slider(minimum=0.0, maximum=10.0, value=0.066, step=0.001, label="Ammonia (mg/l)")
                    s_bod = gr.Slider(minimum=0.0, maximum=30.0, value=2.7, step=0.01, label="Biochemical Oxygen Demand (mg/l)")
                    s_do = gr.Slider(minimum=0.0, maximum=20.0, value=10.2, step=0.01, label="Dissolved Oxygen (mg/l)")
                    s_orthophosphate = gr.Slider(minimum=0.0, maximum=5.0, value=0.144, step=0.001, label="Orthophosphate (mg/l)")
                    s_ph = gr.Slider(minimum=0.0, maximum=14.0, value=7.78, step=0.01, label="pH (ph units)")
                    s_temp = gr.Slider(minimum=-5.0, maximum=45.0, value=11.46, step=0.01, label="Temperature (°C)")
                    s_nitrogen = gr.Slider(minimum=0.0, maximum=25.0, value=4.98, step=0.01, label="Nitrogen (mg/l)")
                    s_nitrate = gr.Slider(minimum=0.0, maximum=20.0, value=4.5, step=0.01, label="Nitrate (mg/l)")

                with gr.TabItem("📝 Precision Inputs"):
                    gr.HTML("<p style='font-size:0.85rem; color:#64748b; margin-bottom: 12px;'>Enter precise values. Non-empty fields override sliders.</p>")
                    t_ammonia = gr.Textbox(placeholder="e.g., 0.05", label="Ammonia (mg/l)")
                    t_bod = gr.Textbox(placeholder="e.g., 2.50", label="Biochemical Oxygen Demand (mg/l)")
                    t_do = gr.Textbox(placeholder="e.g., 10.5", label="Dissolved Oxygen (mg/l)")
                    t_orthophosphate = gr.Textbox(placeholder="e.g., 0.12", label="Orthophosphate (mg/l)")
                    t_ph = gr.Textbox(placeholder="e.g., 7.6", label="pH (ph units)")
                    t_temp = gr.Textbox(placeholder="e.g., 12.0", label="Temperature (°C)")
                    t_nitrogen = gr.Textbox(placeholder="e.g., 4.5", label="Nitrogen (mg/l)")
                    t_nitrate = gr.Textbox(placeholder="e.g., 4.0", label="Nitrate (mg/l)")

        # Right Panel - Outputs
        with gr.Column(scale=1):
            
            with gr.Column(elem_classes=["glass-panel"]):
                gr.HTML("<h3 style='color: #06b6d4; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 12px;'>📊 Prediction Results</h3>")
                
                out_badge = gr.HTML(value="""
                <div style='text-align: center; padding: 20px; border-radius: 16px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08);'>
                    <span style='color: #64748b; font-size: 1.1rem;'>👆 Adjust parameters and click Predict</span>
                </div>
                """)

            with gr.Column(elem_classes=["glass-panel", "top-margin"]):
                with gr.Tabs():
                    with gr.TabItem("📈 Probability Distribution"):
                        out_chart = gr.Plot(label="Confidence Distribution")
                        
                    with gr.TabItem("🔍 XAI Analysis"):
                        out_explanation = gr.HTML(value="<p style='color:#64748b; font-size: 0.95rem;'>Click predict to run the explainable AI analysis.</p>")
                        
                    with gr.TabItem("🌐 Global Importance"):
                        gr.Plot(value=plot_feature_importance(), label="Feature Importance")

    # Buttons
    with gr.Row(elem_classes=["top-margin"]):
        btn_clear = gr.Button("🔄 Reset to Defaults", elem_classes=["clear-btn"], scale=1)
        btn_predict = gr.Button("⚡ Predict Water Quality", elem_classes=["action-btn"], scale=2)

    # Events
    input_list_sliders = [s_ammonia, s_bod, s_do, s_orthophosphate, s_ph, s_temp, s_nitrogen, s_nitrate]
    input_list_text = [t_ammonia, t_bod, t_do, t_orthophosphate, t_ph, t_temp, t_nitrogen, t_nitrate]
    output_list = [out_badge, out_chart, out_explanation]

    btn_predict.click(
        fn=predict_hybrid,
        inputs=input_list_sliders + input_list_text,
        outputs=output_list,
        api_name="predict"
    )

    def reset_inputs():
        return [0.066, 2.7, 10.2, 0.144, 7.78, 11.46, 4.98, 4.5] + [""] * 8

    btn_clear.click(
        fn=reset_inputs,
        inputs=[],
        outputs=input_list_sliders + input_list_text
    )

# -------------------------------------------------------------------------
# Launch App
# -------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)