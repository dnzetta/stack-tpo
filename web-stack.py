import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from tensorflow.keras.models import load_model

# --- Page Background and Author Section ---
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background-color: #eef6fa;
        color: #002244; /* Darker but softer text */
    }

    /* Author section styling */
    .author {
        background-color: #cce0ff;
        color: #003366;
        font-style: italic;
        font-size: 16px;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Tab label button text */
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button {
    color: #002244 !important;       /* Dark navy text */
    font-weight: 600;
    font-size: 14px;
    background-color: #ffffff !important;  /* White background */
}

/* Active and inactive tabs */
[data-testid="stTabs"] div[role="tab"][aria-selected="true"] button,
[data-testid="stTabs"] div[role="tab"][aria-selected="false"] button {
    background-color: #ffffff !important;  /* Keep flat white background */
    color: #002244 !important;             /* Ensure text stays readable */
}

/* Tab content text */
[data-testid="stTabs"] div[data-baseweb="tab-panel"] * {
    color: #002244 !important;  /* Keep content readable */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style text input box */
div[data-baseweb="input"] input {
    background-color: #ffffff !important; /* White background */
    color: #002244 !important;            /* Dark navy text */
    border: 1px solid #4da6ff !important; /* Light blue border */
    border-radius: 8px !important;
    padding: 6px 10px !important;
}

/* Placeholder text color */
div[data-baseweb="input"] input::placeholder {
    color: #666666 !important; /* Gray placeholder for readability */
}

/* File uploader box */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px dashed #4da6ff !important;
    border-radius: 8px !important;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Primary buttons created by st.button */
div.stButton > button {
    background-color: #ffffff !important;  /* White background */
    color: #000000 !important;             /* Black text */
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;  /* Black border for contrast */
    padding: 8px 16px !important;
}

/* Hover/focus state */
div.stButton > button:hover,
div.stButton > button:focus {
    background-color: #f0f0f0 !important;  /* Slight gray on hover */
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* All buttons (Predict, Download, Browse files) */
div.stButton > button,
div[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;  /* White background */
    color: #000000 !important;             /* Black text */
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;  /* Optional: black border for contrast */
    padding: 8px 16px !important;
}

/* Hover/focus state */
div.stButton > button:hover,
div.stButton > button:focus,
div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] button:focus {
    background-color: #f0f0f0 !important;  /* Slight gray on hover */
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Load baseline models ---
cnn_model = load_model("baseline_model_cnn_ma.keras")
bilstm_model = load_model("baseline_model_bilstm_ma.keras")
satt_model = load_model("baseline_model_satt_ma.keras")

# --- Load meta model ---
meta_model = load_model("meta_att_stacked_model.keras")

# --- Functions ---
def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fp_array = np.array(list(fp), dtype=float).reshape(1, -1)
    return fp_array

def predict_baselines(fp_array):
    cnn_prob = cnn_model.predict(fp_array, verbose=0)
    fp_bilstm = fp_array.reshape(fp_array.shape[0], 1, fp_array.shape[1])
    bilstm_prob = bilstm_model.predict(fp_bilstm, verbose=0).squeeze(axis=1)
    satt_prob = satt_model.predict(fp_array, verbose=0)
    stacked_probs = np.hstack([cnn_prob, bilstm_prob, satt_prob])
    return stacked_probs, cnn_prob, bilstm_prob, satt_prob

def predict_meta(stacked_probs):
    final_prob = meta_model.predict(stacked_probs, verbose=0)
    return final_prob

# --- Streamlit Interface ---
st.set_page_config(page_title="SMILES Probability Predictor", layout="centered")
st.title("🧪 Thyroid Peroxidase Toxicity Screening")

st.markdown("""
### Prediction Server
Predict **thyroid peroxidase (TPO) toxicity** from SMILES using **MACCS fingerprints** and **stacked models**.\n
**Baselines included:** CNN, BiLSTM, and attention models.\n
**Meta model:** Attention-based stacked model.
""")

# --- Tabs for SMILES vs CSV input ---
tab1, tab2 = st.tabs(["Single SMILES", "CSV Batch Prediction"])

# --- Single SMILES Input ---
with tab1:
    smiles_input = st.text_input("🔹 Enter SMILES:")
    if st.button("Predict SMILES"):
        if not smiles_input:
            st.warning("Please enter a SMILES string.")
        else:
            fp_array = smiles_to_maccs(smiles_input)
            if fp_array is None:
                st.error("Invalid SMILES string.")
            else:
                stacked_probs, cnn_prob, bilstm_prob, satt_prob = predict_baselines(fp_array)
                final_prob = predict_meta(stacked_probs)
                prob_value = final_prob[0][0]

                
                # Display metrics
                st.metric(label="Final Prediction", value=f"{prob_value:.4f}")
                st.write("**Baseline Predictions:**")
                st.write(f"CNN: {cnn_prob[0][0]:.4f}")
                st.write(f"BiLSTM: {bilstm_prob[0][0]:.4f}")
                st.write(f"Attention: {satt_prob[0][0]:.4f}")

                # Textual interpretation
                st.markdown("**Interpretation:**")
                if prob_value < 0.5:
                    st.success("Non-toxic")
                elif prob_value == 0.5:
                    st.warning("Uncertain")
                else:
                    st.error("Toxic")

                # Heatmap visualization below metrics
                st.write("**Prediction Heatmap:**")
                fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=500)  # smaller figure

                sns.heatmap(
                    [[prob_value]],
                    vmin=0, vmax=1,
                    cmap="RdYlGn_r",
                    annot=True,
                    fmt=".3f",
                    cbar=True,
                    annot_kws={"size": 5},  # smaller font
                    ax=ax
                )

                # Make colorbar tick labels smaller
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=4)

                # Remove axis labels/ticks
                ax.set_xticks([])
                ax.set_yticks([])

                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)  # prevent Streamlit from stretching

                # Explanation just below heatmap
                st.markdown(
                    "**Heatmap Interpretation:**  \n"
                    "0–0.5 → Non-toxic  \n"
                    "0.5 → Uncertain  \n"
                    "0.5–1.0 → Toxic"
                )



# --- CSV Batch Prediction ---
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a CSV with 'SMILES' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "SMILES" not in df.columns:
            st.error("The CSV must contain a 'SMILES' column.")
        else:
            predictions = []
            for smiles in df["SMILES"]:
                fp_array = smiles_to_maccs(smiles)
                if fp_array is not None:
                    stacked_probs, _, _, _ = predict_baselines(fp_array)
                    final_prob = predict_meta(stacked_probs)
                    predictions.append(final_prob[0][0])
                else:
                    predictions.append(None)

            df["Predicted_Probability"] = predictions

            # Add Toxicity Label column
            def toxicity_label(prob):
                if prob is None:
                    return "Invalid SMILES"
                elif prob < 0.5:
                    return "Non-toxic"
                elif prob == 0.5:
                    return "Uncertain"
                else:
                    return "Toxic"

            df["Toxicity_Label"] = df["Predicted_Probability"].apply(toxicity_label)

            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

            # Heatmap legend
            st.write("**Prediction Heatmap:**")
            st.markdown(
                "**Heatmap Interpretation:**  \n"
                "0–0.5 → Non-toxic  \n"
                "0.5 → Uncertain  \n"
                "0.5–1.0 → Toxic"
            )

# --- Spacer before author section ---
st.markdown("<br><br><br>", unsafe_allow_html=True)

# --- Author Section ---
st.markdown("""
<div class="author">
Authors\n
Darlene Nabila Zetta<sup>1</sup>, Tarapong Srisongkram<sup>2</sup>  

<sup>1</sup>*Graduate School in the Program of Pharmaceutical Sciences, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*  
<sup>2</sup>*Division of Pharmaceutical Chemistry, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*
</div>
""", unsafe_allow_html=True)
