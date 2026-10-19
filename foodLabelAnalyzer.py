import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# 1. CONFIGURATION
# ============================================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY", None)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OUTPUT_DIR = Path("output")

if API_KEY is None:
    # Keep the app renderable; show a clear setup message when Analyze is clicked.
    llm = None
else:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=API_KEY,
        temperature=0.2,
    )

st.set_page_config(
    page_title="Food Label Analyzer",
    page_icon="🥬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f8fbf8; }
    [data-testid="stSidebar"] { background-color: #edf5ef; }
    .hero { padding: .5rem 0 1rem 0; }
    .hero h1 { color: #173b2b; }
    .hero p { color: #607568; font-size: 1.05rem; }
    .panel {
        background: white; border: 1px solid #dbe8de; border-radius: 14px;
        padding: 1rem 1.15rem; margin-bottom: 1rem;
    }
    div.stButton > button[kind="primary"] {
        background: #19734c; border: none; border-radius: 9px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. READ INPUT
# ============================================================
def read_uploaded_label(uploaded_file):
    """
    Return the uploaded image as a data URL for a vision-capable model.
    """
    if uploaded_file is None:
        return None

    mime_type = uploaded_file.type or "image/png"
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def prepare_label_input(label_text, image_data_url):
    """
    Prepare label text and/or image content for the model.
    """
    content = []

    if label_text.strip():
        content.append({
            "type": "text",
            "text": "User-provided nutrition label text:\n" + label_text.strip(),
        })

    if image_data_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url},
        })

    return content


# ============================================================
# 3. SEND A PROMPT TO GEMINI
# ============================================================
def ask_gemini(prompt):
    """
    Send a prompt to the configured Google Gemini model.
    """
    if llm is None:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to your .env file and restart Streamlit."
        )

    response = llm.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)


def ask_gemini_with_image(prompt, image_data_url=None):
    """
    Send text plus an optional label image to Gemini.
    """
    if llm is None:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to your .env file and restart Streamlit."
        )

    if not image_data_url:
        return ask_gemini(prompt)

    message = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    response = llm.invoke([("human", message)])
    return response.content if isinstance(response.content, str) else str(response.content)


# ============================================================
# 4. ANALYZE THE FOOD LABEL
# ============================================================
def analyze_food_label(label_text, image_data_url, preferences):
    """
    Interpret the ingredient list and nutrition facts.
    Use only facts visible in the supplied label.
    """
    prompt = f"""
You are a careful food-label analysis assistant.

Analyze the provided nutrition label and ingredient list for a general audience.
The user's optional preferences are: {preferences or "None provided"}.

Tasks:
1. Identify the serving size and servings per package if visible.
2. Summarize calories and key nutrients per serving.
3. Interpret total sugar and added sugar separately when shown.
4. Comment on sodium, saturated fat, fiber, protein, and other listed nutrients
   when relevant to the label.
5. Explain ingredient-list observations in accessible language. Ingredients are
   generally listed by weight, but do not infer exact amounts from order.
6. Identify useful details that cannot be determined from the label.
7. Suggest practical alternative foods or product types and explain why they
   may be worth comparing. Do not claim any option is universally healthier.
8. Tell the user which label values to compare when choosing an alternative.

Rules:
- Use only information supplied or clearly readable in the label.
- Never invent ingredients, nutrient values, serving sizes, or health effects.
- If the image is blurry or a value is uncertain, mark it as uncertain and ask
  the user to verify it on the package.
- Avoid diagnosing conditions or prescribing a diet.
- Do not call foods "toxic" or moralize food choices.
- Context depends on serving size, overall dietary pattern, and individual needs.
- Return a clear report with headings, not JSON yet.

LABEL TEXT:
{label_text.strip() if label_text.strip() else "[No pasted text; inspect the image if provided.]"}
"""
    return ask_gemini_with_image(prompt, image_data_url)


# ============================================================
# 5. STRUCTURED JSON EXTRACTION
# ============================================================
def create_json_analysis(final_analysis):
    """
    Convert the narrative analysis into valid JSON.
    """
    prompt = f"""
Convert the food-label analysis below into valid JSON.

Use exactly this structure:
{{
  "product_summary": "",
  "serving_size": "",
  "nutrients": [
    {{
      "name": "",
      "amount_per_serving": "",
      "interpretation": "",
      "context_or_caveat": ""
    }}
  ],
  "ingredient_observations": [],
  "positive_features": [],
  "things_to_compare": [],
  "healthier_alternatives": [
    {{
      "alternative": "",
      "why_consider_it": "",
      "what_to_compare_on_label": ""
    }}
  ],
  "uncertainties_to_verify": [],
  "general_disclaimer": ""
}}

Return ONLY valid JSON. Do not use Markdown fences.
Do not invent missing information. Use empty strings or empty arrays when needed.
Alternative suggestions must be practical examples, not medical prescriptions.

ANALYSIS:
{final_analysis}
"""
    return ask_gemini(prompt)


# ============================================================
# 6. SAVE RESULTS
# ============================================================
def save_results(final_analysis, json_analysis):
    """
    Save a readable report and structured JSON to the output directory.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "analysis.txt", "w", encoding="utf-8") as file:
        file.write(final_analysis)

    try:
        data = json.loads(json_analysis)
        with open(OUTPUT_DIR / "analysis.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return data
    except json.JSONDecodeError:
        with open(OUTPUT_DIR / "analysis_raw.json", "w", encoding="utf-8") as file:
            file.write(json_analysis)
        return None


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================
def display_results(final_analysis, data):
    """
    Display narrative and structured analysis in Streamlit.
    """
    st.markdown("## 🌿 Your Food Label Analysis")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Simple explanation")
    st.markdown(final_analysis)
    st.markdown("</div>", unsafe_allow_html=True)

    if data:
        st.markdown("### 🥑 Healthier alternatives to consider")
        alternatives = data.get("healthier_alternatives", [])
        if alternatives:
            columns = st.columns(min(3, len(alternatives)))
            for index, alternative in enumerate(alternatives):
                with columns[index % len(columns)]:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown(f"**{alternative.get('alternative', 'Alternative')}**")
                    st.write(alternative.get("why_consider_it", ""))
                    compare = alternative.get("what_to_compare_on_label", "")
                    if compare:
                        st.caption("Compare: " + compare)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No alternative suggestions were returned.")

        st.markdown("### 🔎 Nutrients to notice")
        for nutrient in data.get("nutrients", []):
            name = nutrient.get("name", "Nutrient")
            amount = nutrient.get("amount_per_serving", "Not provided")
            with st.expander(f"{name} — {amount}"):
                st.write(nutrient.get("interpretation", ""))
                caveat = nutrient.get("context_or_caveat", "")
                if caveat:
                    st.caption(caveat)

        with st.expander("Ingredient observations"):
            for item in data.get("ingredient_observations", []):
                st.markdown(f"- {item}")

        with st.expander("Uncertainties to verify"):
            uncertainties = data.get("uncertainties_to_verify", [])
            if uncertainties:
                for item in uncertainties:
                    st.markdown(f"- {item}")
            else:
                st.write("No additional uncertainties were listed.")

        st.download_button(
            "Download structured JSON",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name="analysis.json",
            mime="application/json",
        )

    st.download_button(
        "Download readable report",
        data=final_analysis,
        file_name="analysis.txt",
        mime="text/plain",
    )


# ============================================================
# 8. STREAMLIT USER INTERFACE
# ============================================================
def main():
    with st.sidebar:
        st.markdown("## 🥬 Food Label Analyzer")
        st.caption("Understand ingredients and nutrients.")
        st.divider()
        st.markdown("**How to use**")
        st.markdown("1. Upload a label image or paste label text.\n2. Add optional preferences.\n3. Select Analyze label.")
        st.caption("Label content is sent to Google Gemini for analysis.")

    st.markdown(
        """
        <div class="hero">
          <h1>Food Label Analyzer</h1>
          <p>Interpret ingredients and nutrients, then explore practical alternatives with clear explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns([0.9, 1.1], gap="large")

    with input_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 1. Upload nutrition label")
        uploaded_file = st.file_uploader(
            "Choose a label image",
            type=["png", "jpg", "jpeg", "webp"],
        )
        image_data_url = read_uploaded_label(uploaded_file)
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded label", use_container_width=True)

        label_text = st.text_area(
            "Or paste nutrition facts and ingredients",
            height=220,
            placeholder=(
                "Serving size: 1 cup (55 g)\n"
                "Calories: 210\n"
                "Sodium: 320 mg\n"
                "Dietary fiber: 3 g\n"
                "Added sugars: 4 g\n"
                "Protein: 5 g\n\n"
                "Ingredients: whole grain oats, sugar, ..."
            ),
        )
        preferences = st.text_input(
            "Optional preferences",
            placeholder="e.g. vegetarian, budget-friendly, nut-free",
        )
        analyze_button = st.button(
            "✨ Analyze label",
            type="primary",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 2. Analysis results")
        st.write("Your ingredient insights, nutrient interpretation, and possible alternatives will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_button:
        if not label_text.strip() and not image_data_url:
            st.warning("Upload an image or paste nutrition label text first.")
            return

        try:
            with st.spinner("Step 1/3: Interpreting the label..."):
                final_analysis = analyze_food_label(
                    label_text, image_data_url, preferences
                )

            with st.spinner("Step 2/3: Creating structured JSON..."):
                json_analysis = create_json_analysis(final_analysis)

            with st.spinner("Step 3/3: Saving results..."):
                data = save_results(final_analysis, json_analysis)

            display_results(final_analysis, data)

            if data is None:
                st.warning(
                    "The narrative report was saved, but the model's JSON was invalid. "
                    "A raw JSON response was saved for inspection."
                )
            else:
                st.success("Analysis complete. Files saved in the output/ directory.")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check your GOOGLE_API_KEY, model access, internet connection, "
                "and whether your selected model supports image inputs."
            )


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
