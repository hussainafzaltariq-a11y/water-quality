# 🌊 Water Quality Prediction Dashboard

## App Name & What It Does

**Water Quality Prediction Dashboard** is an AI-powered web application that predicts water quality based on 8 key water parameters. It classifies water into five categories: **Excellent, Good, Fair, Marginal, or Poor** — and provides an **Explainable AI (XAI)** breakdown of which parameters most influenced the prediction.

## The Real Problem It Solves

**For whom:** Farmers, villagers, and rural communities in Pakistan who lack access to laboratory water testing.

**The problem:** Millions of people in rural Pakistan drink unsafe water without knowing it. Laboratory testing is expensive, time-consuming, and often unavailable in remote areas. This leads to waterborne diseases, crop failure, and loss of livelihood.

**How this app solves it:** This app provides **instant, free, and accessible** water quality assessment — anyone with a smartphone or computer can check their water quality in seconds. The built-in Explainable AI helps users understand *why* their water is classified in a certain way, empowering them to take informed action.

## Live Deployed URL

🌐 **[https://huggingface.co/spaces/Tariq349/water-quality-xai](https://huggingface.co/spaces/Tariq349/water-quality-xai)**

> *Click the link above to use the app directly — no installation required.*

## Features List

| Feature | Description |
|---------|-------------|
| ✅ **Water Quality Prediction** | Predicts water quality (Excellent, Good, Fair, Marginal, Poor) from 8 input parameters |
| ✅ **Model Confidence Score** | Shows how confident the AI is about its prediction |
| ✅ **Probability Distribution Chart** | Visualizes the probability of each class |
| ✅ **Explainable AI (XAI) Analysis** | Breaks down which features most influenced the prediction |
| ✅ **Global Feature Importance** | Shows which parameters are most important overall |
| ✅ **Dual Input Modes** | Use sliders for quick adjustments or precision inputs for exact values |
| ✅ **Reset Values** | One-click reset to default values |

## The AI Feature — How It Works

**Model:** LightGBM Classifier trained on a water quality dataset.

**What it does:** The model takes 8 water quality parameters as input and outputs a classification (Excellent, Good, Fair, Marginal, or Poor) along with a confidence score.

**Explainable AI (XAI):** The app uses feature contribution analysis to show which parameters had the most impact on the prediction. This helps users understand *why* their water is classified in a certain way.

I uses Hugging Face's `google/flan-t5-small` model via the Hugging Face Inference API
**System Prompt :**
> *"You are a water quality assistant for Pakistani farmers. Respond in SIMPLE URDU. Given the prediction and feature importance, give practical advice on what the user should do next."*


**The 8 Parameters Used:**

1. Ammonia (mg/l)
2. Biochemical Oxygen Demand (BOD) (mg/l)
3. Dissolved Oxygen (mg/l)
4. Orthophosphate (mg/l)
5. pH (ph units)
6. Temperature (°C)
7. Nitrogen (mg/l)
8. Nitrate (mg/l)

## Tools & Services Used

| Tool / Service | Purpose |
|----------------|---------|
| **Python** | Programming Language |
| **Gradio** | Web App Framework |
| **LightGBM** | Machine Learning Model |
| **Scikit-learn** | Data Preprocessing (RobustScaler) |
| **Matplotlib & Seaborn** | Data Visualization |
| **Pandas & NumPy** | Data Manipulation |
| **Hugging Face Spaces** | Free Cloud Hosting |
| **GitHub** | Version Control |

## Screenshots



![Screenshot 1 - Prediction Dashboard]
<img width="959" height="473" alt="image" src="https://github.com/user-attachments/assets/ab78a162-42a3-4710-adef-0b76b1be0041" />


![Screenshot 2 - XAI Analysis]
<img width="947" height="416" alt="image" src="https://github.com/user-attachments/assets/804aa921-08ac-42aa-951f-0d5bd483ebdc" />

![Screenshot 3 - Probability Distribution]
<img width="958" height="476" alt="image" src="https://github.com/user-attachments/assets/09bfb1ec-4076-4e20-ba65-6d0d4eaca389" />

## How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/hussainafzaltariq-a11y/water-quality.git
cd water-quality
