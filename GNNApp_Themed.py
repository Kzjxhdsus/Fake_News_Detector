import streamlit as st
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

# --- Set Page Config ---
st.set_page_config(
    page_title="Fake News Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Encode and Set Background Image ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
      background-attachment: fixed;
      background-position: center;
      background-repeat: no-repeat;
    }}
    .main > div {{
      background-color: rgba(255, 255, 255, 0.85);
      padding: 2rem;
      margin: 2rem;
      border-radius: 15px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    .stTextArea textarea, .stFileUploader, .stSelectbox, .stButton {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .stButton>button {{
        background-color: #0E79B2;
        color: white;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #065a82;
    }}
    h1, h2, h3 {{
        color: #111;
        text-align: center;
    }}
    body {{
        font-family: 'Segoe UI', sans-serif;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("background.png")  # ← make sure this file exists in same directory

# --- Sidebar Info ---
st.sidebar.markdown("## 🧠 About")
st.sidebar.markdown("This app uses a GraphSAGE GNN model to detect fake news articles by analyzing article content and similarity patterns with existing data.")

# --- Define GraphSAGE Model ---
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Vectorizer and Model ---
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

@st.cache_resource
def load_model_and_data():
    data_x = torch.load("gnn_train_x.pt")
    in_channels = data_x.shape[1]
    model = GraphSAGE(in_channels, hidden_channels=64, out_channels=2)
    model.load_state_dict(torch.load("gnn_model.pt", map_location=device))
    model.to(device).eval()
    return model, data_x.to(device)

vectorizer = load_vectorizer()
model, data_x = load_model_and_data()

# --- UI Header ---
st.title("🧠  Detecting fake news on social media Using GNNss")
st.markdown("This tool uses a GraphSAGE GNN model to detect fake news based on article content and social sharing patterns.")

# --- Input Section ---
example_articles = [
    "The president signed a new bill to improve rural education systems across the country.",
    "A Martian spaceship landed in Canada according to an anonymous Twitter user."
]
example_choice = st.selectbox("Or choose an example article:", [""] + example_articles)
article = ""

if example_choice:
    article = example_choice
else:
    article = st.text_area("✏️ Paste your news article here", height=400)

# --- Classify Button ---
if st.button("Classify"):
    if not article.strip():
        st.warning("Please enter a valid article.")
    else:
        with st.spinner("Analyzing the article..."):
            try:
                input_vec = vectorizer.transform([article]).toarray()
                x_input = torch.tensor(input_vec, dtype=torch.float).to(device)

                sims = cosine_similarity(input_vec, data_x.cpu().numpy())[0]
                top_k_idx = np.argsort(sims)[-5:]

                edge_list = [(0, i + 1) for i in range(5)] + [(i + 1, 0) for i in range(5)]
                for i in range(5):
                    for j in range(i + 1, 5):
                        sim_ij = cosine_similarity(
                            [data_x[top_k_idx[i]].cpu().numpy()],
                            [data_x[top_k_idx[j]].cpu().numpy()]
                        )[0][0]
                        if sim_ij > 0.7:
                            edge_list += [(i + 1, j + 1), (j + 1, i + 1)]

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
                neighbor_feats = data_x[top_k_idx]
                x_combined = torch.cat([x_input, neighbor_feats], dim=0)
                data_infer = Data(x=x_combined, edge_index=edge_index).to(device)

                with torch.no_grad():
                    output = model(data_infer)
                    probs = torch.softmax(output[0], dim=0)
                    pred = probs.argmax().item()

                # --- Display Result ---
                st.markdown("## 🧮 Classification Results")
                if pred == 0:
                    st.success("✅ This news is **REAL**")
                else:
                    st.error("❌ This news is **FAKE**")

                st.write(f"Confidence Score: {probs.max().item() * 100:.2f}%")

                # --- View Article ---
                with st.expander("📄 View Submitted Article"):
                    st.markdown(f"```{article.strip()}```")

            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
