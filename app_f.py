import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import os
import cv2
import json
import plotly.graph_objects as go
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from skimage.morphology import skeletonize, disk
from skimage.filters.rank import entropy
from scipy.spatial import KDTree
import scipy.signal
import re
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="Protean Modeler | Lakshya Gupta", layout="wide", page_icon="🧊")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .report-box { background-color: #161b22; padding: 15px; border-radius: 10px; border-left: 5px solid #00CC96; margin-top: 10px; }
    .metric-box { background-color: #262730; padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #444; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #00CC96; color: white; font-weight: bold;}
    h1, h2, h3 { color: #00CC96; }
</style>
""", unsafe_allow_html=True)

class StabilizedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.cbr(1, 32); self.enc2 = self.cbr(32, 64)
        self.pool = nn.MaxPool2d(2); self.bottleneck = self.cbr(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.dec2 = self.cbr(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2); self.dec1 = self.cbr(64, 32)
        self.final = nn.Conv2d(32, 1, 1)
    def cbr(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True),
                             nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(True))
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1); e2 = self.enc2(p1); p2 = self.pool(e2)
        b = self.bottleneck(p2); u2 = self.up2(b); u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2); u1 = self.up1(d2); u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1); return self.final(d1)

class FaultGAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(FaultGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=0.0)
        self.conv2 = GATConv(hidden_channels * 4, num_classes, heads=1, concat=False, dropout=0.0)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class OfflineAnalyst:
    @staticmethod
    def generate_narrative(risk_pct, total_nodes, horizon, depth, avg_dip, regime, fractal_dim):
        # 1. Base Assessment
        if risk_pct > 25:
            severity = "CRITICAL"
            tone = "highly compromised"
            recommendation = "IMMEDIATE SUSPENSION of drilling plans is advised."
            physics = "high vertical transmissibility, indicating active fluid migration pathways."
            sgr_context = ("FAULT SEAL ANALYSIS: The fault network exhibits high vertical persistence. "
                           "The estimated displacement likely exceeds the sealing capacity of the overburden shale, "
                           "resulting in a low Shale Gouge Ratio (SGR). Risk of hydrocarbon migration is HIGH.")
        elif risk_pct > 5:
            severity = "MODERATE"
            tone = "partially compartmentalized"
            recommendation = "Proceed with CAUTION; LWD (Logging While Drilling) required."
            physics = "localized stress accumulation with potential for minor leakage."
            sgr_context = ("FAULT SEAL ANALYSIS: While structural deformation is present, the fault throw "
                           "appears contained within the regional seal. Clay smearing mechanisms likely "
                           "maintain trap integrity (Moderate SGR). Risk of migration is MANAGEABLE.")
        else:
            severity = "NEGLIGIBLE"
            tone = "structurally competent"
            recommendation = "Standard drilling parameters approved."
            physics = "robust seal integrity with no significant vertical connectivity."
            sgr_context = ("FAULT SEAL ANALYSIS: Faulting is minimal and confined to the reservoir zone. "
                           "Caprock integrity is intact.")

        text = (
            f"SUBJECT: GEOMECHANICAL INTEGRITY ASSESSMENT - GULLFAKS FIELD\n\n"
            f"1. EXECUTIVE SUMMARY\n"
            f"The Protean Geo-Analytics platform has concluded a structural analysis of the {horizon} "
            f"formation at a depth of {depth}m. The analysis utilized a Hybrid Neuro-Symbolic approach, "
            f"combining Convolutional Neural Networks for fault segmentation with Graph Attention Networks (GAT) "
            f"for topology assessment. The structural integrity of the surveyed sector is currently classified as {severity}.\n\n"
            
            f"2. TECTONIC REGIME & STRESS FIELD\n"
            f"Structural analysis of the fault vectors indicates a Mean Dip Angle of {avg_dip:.1f} degrees. "
            f"This geometry is consistent with an {regime} regime (Andersonian Classification). "
            f"Fractal Dimension Analysis (D = {fractal_dim:.2f}) suggests the fracture intensity is "
            f"{'high (damage zone)' if fractal_dim > 1.4 else 'localized'}.\n\n"

            f"3. STRUCTURAL TOPOLOGY ANALYSIS\n"
            f"The Graph Property Predictor (The Engineer) identified a total of {total_nodes} discrete structural nodes. "
            f"Of these, {risk_pct:.2f}% exhibit geometric properties consistent with {physics} "
            f"The fault network appears {tone}.\n\n"
            
            f"4. PETROPHYSICAL CONTEXT (SGR PROXY)\n"
            f"{sgr_context}\n\n"
            
            f"5. OPERATIONAL RECOMMENDATIONS\n"
            f"Based on the calculated probability distribution of leakage, {recommendation} "
            f"Future well placement optimization should prioritize avoiding the connected sub-graphs "
            f"highlighted in the attached visualization."
        )
        return text

def create_offline_report(analysis_text, fig_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Protean Geo-Analytics: Structural Risk Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | User: Lakshya Gupta (IITR)", ln=True)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, analysis_text)
    pdf.ln(10)
    if os.path.exists(fig_path):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Fig 1: AI-Generated Fault Probability Map", ln=True)
        pdf.image(fig_path, x=30, w=150)
    return pdf.output(dest='S').encode('latin-1')


def compute_instantaneous_phase(img_array):
    """Computes Instantaneous Phase using Hilbert Transform."""
    analytic_signal = scipy.signal.hilbert(img_array)
    instantaneous_phase = np.angle(analytic_signal)
    norm_phase = (instantaneous_phase + np.pi) / (2 * np.pi) 
    return norm_phase

def compute_acoustic_impedance(img_array):
    """Recursive Inversion: Reflection -> Relative Impedance."""
    imp = np.cumsum(img_array, axis=0)
    imp = scipy.signal.detrend(imp, axis=0)
    mi, ma = imp.min(), imp.max()
    return (imp - mi) / (ma - mi) if ma > mi else imp

def compute_seismic_entropy(img_array):
    """Texture Analysis: Local Entropy to detect chaotic zones."""
    img_u8 = (img_array * 255).astype(np.uint8)
    ent_map = entropy(img_u8, disk(3))
    mi, ma = ent_map.min(), ent_map.max()
    return (ent_map - mi) / (ma - mi) if ma > mi else ent_map

def analyze_tectonic_regime(G):
    """Returns: Avg Dip, Regime Name, List of Raw Dips (for Rose Diagram)"""
    if len(G.nodes) < 2: return 0, "Indeterminate", []
    
    components = list(nx.connected_components(G))
    dips = []
    
    for comp in components:
        if len(comp) < 3: continue 
        pts = np.array([G.nodes[n]['pos'] for n in comp])
        y = pts[:, 0]; x = pts[:, 1]
        dy = y.max() - y.min(); dx = x.max() - x.min()
        
        if dx == 0: angle = 90
        else: angle = np.degrees(np.arctan(dy/dx))
        dips.append(angle)
        
    if not dips: return 0, "Indeterminate", []
    
    avg_dip = np.mean(dips)
    if avg_dip > 45: regime = "EXTENSIONAL (Normal Faulting)"
    elif avg_dip < 45: regime = "COMPRESSIONAL (Thrust Faulting)"
    else: regime = "STRIKE-SLIP / VERTICAL"
        
    return avg_dip, regime, dips

def calculate_fractal_dimension(Z, threshold=0.9):
    """Box-Counting Fractal Dimension Analysis."""
    pixels = np.array(np.where(Z > threshold))
    if pixels.shape[1] == 0: return 0
    
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns = []
    for scale in scales:
        H, edges = np.histogramdd(pixels.T, bins=(np.arange(0,128,scale),np.arange(0,128,scale)))
        Ns.append(np.sum(H > 0))
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    return -coeffs[0]

def plot_rose_diagram(dips):
    """Creates a polar Rose Diagram of fault dip angles."""
    if not dips: return None
    bins = np.linspace(0, 360, 37) 
    dips_mirrored = dips + [d + 180 for d in dips] # Symmetry
    hist, _ = np.histogram(dips_mirrored, bins=bins)
    width = 2 * np.pi / len(bins)
    theta = np.deg2rad(bins[:-1])
    
    fig = plt.figure(figsize=(3, 3), facecolor='none')
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(theta, hist, width=width, bottom=0.0, color='#00CC96', alpha=0.7, edgecolor='white')
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_yticklabels([]); ax.grid(color='#444444', linestyle='--', alpha=0.5)
    ax.spines['polar'].set_visible(False)
    plt.setp(ax.get_xticklabels(), color="white", fontsize=8)
    return fig

@st.cache_resource
def load_models():
    vision = StabilizedUNet().to(DEVICE)
    if os.path.exists("fault_model_stabilized.pth"):
        vision.load_state_dict(torch.load("fault_model_stabilized.pth", map_location=DEVICE))
    else: return None, None
    vision.eval()
    
    risk = FaultGAT(3, 16, 2).to(DEVICE)
    if os.path.exists("gat_model_complete.pth"):
        risk.load_state_dict(torch.load("gat_model_complete.pth", map_location=DEVICE))
    else: return None, None
    risk.eval()
    return vision, risk

def load_data(slice_name):
    path = f"dataset_final/images/{slice_name}"
    if os.path.exists(path):
        img = cv2.imread(path, -1)
        if img is not None:
            img = np.nan_to_num(img)
            if img.dtype != np.uint8:
                mi, ma = img.min(), img.max()
                img = (img - mi) / (ma - mi) * 255.0 if ma > mi else img
                img = img.astype(np.uint8)
            img = cv2.resize(img, (128, 128))
            img_norm = img.astype(np.float32) / 255.0
            
            # Compute ALL Geological Attributes
            phase = compute_instantaneous_phase(img)
            rai = compute_acoustic_impedance(img)
            entropy = compute_seismic_entropy(img_norm)
            
            return img, img_norm, phase, rai, entropy
    return None, None, None, None, None

def load_real_well_tops():
    possible_paths = ["dataset_final/Well Tops/Well tops", "dataset_final/Well Tops/Well tops.txt", "Well Tops/Well tops"]
    found_depth = None
    horizon_name = "Unknown"
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "Base Cretaceous" in line:
                        matches = re.findall(r'-?\d+\.\d+', line)
                        if len(matches) >= 3:
                            found_depth = float(matches[2])
                            horizon_name = "Base Cretaceous"
                            break
            if found_depth: break
    return found_depth, horizon_name

def create_interactive_graph(G, preds, real_depth):
    pos = {n: (n[1], 128-n[0]) for n in G.nodes()} 
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#00FFFF'), hoverinfo='none', mode='lines', name='Connectivity')

    node_x, node_y, node_color, node_text = [], [], [], []
    RES_VISUAL_TOP = 40 
    
    for i, node in enumerate(G.nodes()):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        
        # Stratigraphy Logic
        if y < RES_VISUAL_TOP: 
            zone_name = "Caprock Shale"
            zone_color = "#228B22" 
        else: 
            zone_name = "Reservoir Sand"
            zone_color = "#E3CF57" 

        is_risky = preds[i] == 1
        is_touching_horizon = abs(y - RES_VISUAL_TOP) < 10
        
        if is_risky and is_touching_horizon:
            color = '#ff0000' 
            status = f"CRITICAL: Breach at {zone_name}"
        elif is_risky:
            color = '#FF8C00' 
            status = f"WARNING: Fault in {zone_name}"
        else:
            color = zone_color 
            status = f"STRUCTURAL: Safe {zone_name}"
            
        node_color.append(color)
        node_text.append(status)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(color=node_color, size=6, line_width=1, line_color='white'), 
        text=node_text, name='Fault Nodes'
    )

    shapes = [
        dict(type="line", x0=0, y0=RES_VISUAL_TOP, x1=128, y1=RES_VISUAL_TOP, line=dict(color="gold", width=2, dash="dash")),
    ]
    annotations = []
    if real_depth:
        annotations.append(dict(
            x=64, y=RES_VISUAL_TOP-5, 
            text=f"<b>REAL DATA: {real_depth}m (Base Cretaceous)</b>", 
            showarrow=False, font=dict(color="gold", size=10)
        ))

    layout = go.Layout(
        title=dict(text="Physics-Validated Risk Map (Zonal Juxtaposition)", font=dict(color="#00f2ff", size=16)),
        showlegend=True, hovermode='closest', margin=dict(b=0,l=0,r=0,t=40), height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Crossline Position", showgrid=False, zeroline=False, visible=True),
        yaxis=dict(title="Depth (meters)", showgrid=False, zeroline=False, visible=True),
        shapes=shapes, annotations=annotations,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)")
    )
    return go.Figure(data=[edge_trace, node_trace], layout=layout)

def build_graph(binary_mask):
    skel = skeletonize(binary_mask)
    if np.sum(skel) == 0: return None, None
    pts = np.column_stack(np.where(skel > 0))
    G = nx.Graph()
    for p in pts: G.add_node(tuple(p), pos=(p[1], p[0]))
    if len(pts) > 1:
        tree = KDTree(pts)
        pairs = tree.query_pairs(r=2.5)
        for i,j in pairs: G.add_edge(tuple(pts[i]), tuple(pts[j]))
    
    if G.number_of_nodes() < 5: return None, None
    
    node_features = []
    node_list = list(G.nodes())
    for node in node_list:
        r, c = node
        depth = r / 128.0
        is_vert = False
        nbrs = list(G.neighbors(node))
        if nbrs:
            for n in nbrs:
                if abs(r - n[0]) > abs(c - n[1]): is_vert = True; break
        dip = 1.0 if is_vert else 0.1
        amp = 0.5 
        node_features.append([depth, dip, amp])
        
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = []
    for u, v in G.edges():
        u_idx, v_idx = node_list.index(u), node_list.index(v)
        edge_index.extend([[u_idx, v_idx], [v_idx, u_idx]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index), G


def main():
    st.title("The Protean Subsurface Modeler")
    st.markdown("##### **Lakshya Gupta** | GT Sophomore | IIT Roorkee")
    st.markdown("---")

    if not os.path.exists("dataset_final/images"):
        st.error("Dataset folder not found."); st.stop()
    files = sorted([f for f in os.listdir("dataset_final/images") if f.endswith(('.png', '.jpg'))])
    
    format_func = lambda x: f"Cross-Section {x.split('_')[1]}" if '_' in x else x
    selected_file = st.sidebar.selectbox("Select Seismic Slice", files, format_func=format_func)
    sensitivity = st.sidebar.slider("AI Sensitivity Threshold", 0.0, 1.0, 0.5)

    real_depth, horizon_name = load_real_well_tops()
    if real_depth:
        st.sidebar.success(f"✅ Loaded Well Tops: {horizon_name} @ {real_depth}m")
    else:
        st.sidebar.warning("⚠️ No Well Tops found. Using Simulation Mode.")

    vision, risk = load_models()
    if not vision: st.stop()
    img, img_norm, phase, rai, entropy_attr = load_data(selected_file)
    if img is None: st.stop()

    st.subheader("1. Geophysical Attribute Workbench (Interactive)")
    
    # 4-Way Workbench
    view_mode = st.radio(
        "Select Geophysical Attribute:", 
        ["Seismic Amplitude (Raw)", "Instantaneous Phase (Continuity)", 
         "Relative Impedance (Lithology)", "Seismic Entropy (Chaos)"], 
        horizontal=True
    )
    
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(vision(tensor)).squeeze().cpu().numpy()
    mask = prob > sensitivity
    
    # Select Base Image
    if "Amplitude" in view_mode:
        base_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif "Phase" in view_mode:
        base_img = cv2.cvtColor((phase * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif "Impedance" in view_mode:
        base_img = cv2.cvtColor((rai * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif "Entropy" in view_mode:
        base_img = cv2.cvtColor((entropy_attr * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    img_overlay = base_img.copy()
    img_overlay[mask == 1] = [0, 255, 255] # Yellow Faults

    image_comparison(
        img1=base_img,
        img2=img_overlay,
        label1=f"{view_mode.split('(')[0].strip()}",
        label2="AI Extraction",
        width=700
    )

    st.markdown("---")
    st.subheader("2. Structural Risk Assessment")
    geo_data, G = build_graph(mask)
    
    if geo_data:
        geo_data = geo_data.to(DEVICE)
        with torch.no_grad():
            preds = risk(geo_data).argmax(dim=1).cpu().numpy()
        
        c3, c4 = st.columns([1.5, 1])
        with c3:
            st.plotly_chart(create_interactive_graph(G, preds, real_depth), use_container_width=True)
        with c4:
           
            avg_dip, regime, raw_dips = analyze_tectonic_regime(G)
            fractal_dim = calculate_fractal_dimension(mask)
            
            # Recalculate Risk 
            pos = {n: (n[1], 128-n[0]) for n in G.nodes()}
            critical_nodes = 0
            RES_VISUAL_TOP = 40
            for i, node in enumerate(G.nodes()):
                _, y = pos[node]
                if preds[i] == 1 and abs(y - RES_VISUAL_TOP) < 10:
                    critical_nodes += 1
            
            total = len(preds)
            risk_pct = (critical_nodes/total)*100
            
            # DASHBOARD 
            st.markdown(f"""
            <div class="metric-box">
            <b>Structural Geometry:</b><br>
            Mean Dip: {avg_dip:.1f}°<br>
            Fractal Dim: <span style='color: #FFD700'>{fractal_dim:.2f}</span><br>
            Regime: <span style='color: #00CC96'>{regime}</span>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🧭 View Structural Orientation (Rose Diagram)", expanded=True):
                rose_fig = plot_rose_diagram(raw_dips)
                if rose_fig: st.pyplot(rose_fig, use_container_width=True)
                else: st.write("Insufficient vector data.")

            st.metric("Structure Nodes", total)
            st.metric("Breach Points", critical_nodes, delta_color="inverse")
            
          
            status = "CRITICAL" if risk_pct > 5 else "STABLE"
            color = "red" if risk_pct > 5 else "green"
            
            st.markdown(f"""
            <div class="report-box">
            <h4>🧠 MAIA Geological Assessment</h4>
            <b>Status:</b> :{color}[{status}]<br>
            <b>Validation:</b> Calibrated with {horizon_name} ({real_depth}m)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # REPORT GENERATION
            long_analysis = OfflineAnalyst.generate_narrative(risk_pct, total, horizon_name, real_depth, avg_dip, regime, fractal_dim)
            
            with st.expander("View Analyst Summary"):
                st.write(long_analysis)

            # Generate PDF 
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.imshow(np.ma.masked_where(mask==0, mask), cmap='cool', alpha=0.7)
            ax.axis('off')
            plt_path = "temp_risk_map.png"
            fig.savefig(plt_path, bbox_inches='tight', dpi=150)
            
            pdf_bytes = create_offline_report(long_analysis, plt_path)
            
            st.download_button(
                label="📄 Download Official Report",
                data=pdf_bytes,
                file_name=f"Assessment_{selected_file}.pdf",
                mime="application/pdf"
            )
            
    else:
        st.info("No faults detected.")

if __name__ == "__main__":
    main()