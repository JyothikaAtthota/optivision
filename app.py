"""
OPTIVISION -- Gradio Web Interface (Gradio v5 compatible)
Real-time DR detection and blindness prediction UI
Run: python app.py
"""

import os
import gradio as gr
import pandas as pd
from PIL import Image

from predict import predict, DR_CLASSES, BLINDNESS_RISK

CHECKPOINT = "outputs/best_model.pth"

RISK_EMOJI = {
    "Low":       "🟢",
    "Moderate":  "🟡",
    "High":      "🟠",
    "Very High": "🔴",
    "Critical":  "🚨",
}

# ── Prediction logic ──────────────────────────────────────────────────────────
def analyze_image(image: Image.Image):
    """
    Returns:
        summary   (str)       – Markdown result card
        df        (DataFrame) – probability table for BarPlot
    """
    if image is None:
        empty_df = pd.DataFrame({"Stage": DR_CLASSES,
                                 "Probability (%)": [0.0] * len(DR_CLASSES)})
        return "**No image provided. Please upload a retinal fundus image.**", empty_df

    if not os.path.exists(CHECKPOINT):
        empty_df = pd.DataFrame({"Stage": DR_CLASSES,
                                 "Probability (%)": [0.0] * len(DR_CLASSES)})
        return (
            "**Model checkpoint not found.**\n\n"
            "Run `python train.py` first, or place `best_model.pth` in `outputs/`.",
            empty_df,
        )

    result = predict(image, checkpoint_path=CHECKPOINT)

    emoji   = RISK_EMOJI.get(result["blindness_risk"], "⚪")
    summary = (
        f"## {emoji} Diagnosis Result\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **DR Stage** | {result['predicted_class']} |\n"
        f"| **Confidence** | {result['confidence']*100:.1f}% |\n"
        f"| **Blindness Risk** | {result['blindness_risk']} |\n\n"
        "---\n\n"
        "**Stage Probabilities:**\n\n"
    )
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        summary += f"- `{cls:<20}` {bar} {prob*100:.1f}%\n"

    df = pd.DataFrame({
        "Stage":           list(result["probabilities"].keys()),
        "Probability (%)": [v * 100 for v in result["probabilities"].values()],
    })

    return summary, df


# ── UI layout ─────────────────────────────────────────────────────────────────
def build_interface():
    empty_df = pd.DataFrame({"Stage": DR_CLASSES,
                              "Probability (%)": [0.0] * len(DR_CLASSES)})

    with gr.Blocks(
        title="OPTIVISION -- Diabetic Retinopathy Detection",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown(
            """
            # 👁 OPTIVISION
            ### AI-Powered Diabetic Retinopathy Detection & Blindness Risk Prediction
            *Upload a retinal fundus image for instant DR stage classification.*
            """
        )

        with gr.Row():
            # Left column — input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Retinal Fundus Image",
                    type="pil",
                    sources=["upload", "webcam"],
                    height=300,
                )
                analyze_btn = gr.Button("Analyze Image", variant="primary")

            # Right column — outputs
            with gr.Column(scale=1):
                result_text = gr.Markdown(value="*Upload an image and click Analyze.*")
                prob_bar = gr.BarPlot(
                    value=empty_df,
                    x="Stage",
                    y="Probability (%)",
                    title="Stage Probabilities",
                    height=250,
                    y_lim=[0, 100],
                )

        # DR severity reference table
        gr.Markdown("### DR Severity Reference")
        gr.DataFrame(
            value=[
                ["No DR",             "No visible retinal damage",             "Low"],
                ["Mild DR",           "Microaneurysms present",                "Moderate"],
                ["Moderate DR",       "Hemorrhages & hard exudates",           "High"],
                ["Severe DR",         "Extensive hemorrhages, venous beading", "Very High"],
                ["Proliferative DR",  "Neovascularisation, risk of blindness", "Critical"],
            ],
            headers=["Stage", "Key Features", "Blindness Risk"],
            interactive=False,
        )

        gr.Markdown(
            "*OPTIVISION is a screening aid only. "
            "Always consult a qualified ophthalmologist for diagnosis.*"
        )

        # Wire up the button
        analyze_btn.click(
            fn=analyze_image,
            inputs=image_input,
            outputs=[result_text, prob_bar],
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        inbrowser=True,
    )
