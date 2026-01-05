import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

class ActiveAttentionModule(nn.Module):
    def __init__(self, F_l_channels, F_h_channels, F_int_channels):
        super(ActiveAttentionModule, self).__init__()
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l_channels, F_int_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int_channels)
        )
        self.W_h = nn.Sequential(
            nn.Conv2d(F_h_channels, F_int_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int_channels)
        )
        self.W_s = nn.Sequential(
            nn.Conv2d(F_int_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, F_l, F_h):
        g_h = self.W_h(F_h)
        g_l = self.W_l(F_l)
        if g_h.shape[2:] != g_l.shape[2:]:
            g_h = F.interpolate(g_h, size=g_l.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g_l + g_h)
        return F_l * self.W_s(psi), self.W_s(psi)

class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SeismicAttentionUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SeismicAttentionUNet2D, self).__init__()
        self.enc1 = ConvBlock2D(in_channels, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock2D(32, 64); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock2D(64, 128); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock2D(128, 256)
        self.aam3 = ActiveAttentionModule(128, 256, 64)
        self.aam2 = ActiveAttentionModule(64, 128, 32)
        self.aam1 = ActiveAttentionModule(32, 64, 16)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock2D(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock2D(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock2D(64 + 32, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        b_up = self.up3(b)
        e3_f, a3 = self.aam3(e3, b_up); d3 = self.dec3(torch.cat([b_up, e3_f], dim=1))
        d3_up = self.up2(d3)
        e2_f, a2 = self.aam2(e2, d3_up); d2 = self.dec2(torch.cat([d3_up, e2_f], dim=1))
        d2_up = self.up1(d2)
        e1_f, a1 = self.aam1(e1, d2_up); d1 = self.dec1(torch.cat([d2_up, e1_f], dim=1))
        return self.final_conv(d1), [a1, a2, a3]



class FaultGAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(FaultGAT, self).__init__()
        # Heads=4 learns 4 different ways to look at neighbors
        self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 4, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply dropout during training to prevent overfitting on simulation data
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)


def build_topology_from_skeleton(skel):
    """Converts a skeletonized image into a NetworkX graph."""
    pts = np.column_stack(np.where(skel > 0))
    G = nx.Graph()
    for p in pts: 
        G.add_node(tuple(p), pos=(p[1], p[0]))
    
    if len(pts) > 1:
        tree = KDTree(pts)
        # Connect pixels within distance 1.5 (includes diagonals)
        pairs = tree.query_pairs(r=1.5)
        for i, j in pairs: 
            G.add_edge(tuple(pts[i]), tuple(pts[j]))
            
    return G, None

def image_to_graph_pipeline(binary_mask, is_simulation=False):
    """
    Transforms a binary fault mask into a PyTorch Geometric Data object.
    Extracts physics-informed features: Depth, Dip (Verticality), and Amplitude.
    """
    skeleton = skeletonize(binary_mask)
    if np.sum(skeleton) == 0: return None, None
    
    # 1. Build Graph Topology
    G_topo, _ = build_topology_from_skeleton(skeleton)
    
    # 2. Pruning (Optional / Fallback)
    try:
        # If you define prune_spurious_branches elsewhere, import it.
        # Otherwise, this falls back to the raw topology.
        # G_pruned, _ = prune_spurious_branches(G_topo.copy(), min_length=10)
        G_pruned = G_topo.copy()
    except:
        G_pruned = G_topo.copy()
    
    if G_pruned.number_of_nodes() < 2: return None, None

    # 3. Feature Extraction
    node_features = []
    center_r, center_c = binary_mask.shape[0] // 2, binary_mask.shape[1] // 2
    
    for node in G_pruned.nodes():
        r, c = node
        
        # Feature A: Depth (Normalized 0-1)
        # Deeper faults often carry higher leakage risk
        depth_norm = r / 128.0
        
        # Feature B: Dip / Verticality
        # Check neighbors to determine if structure is vertical
        is_vertical = False
        neighbors = list(G_pruned.neighbors(node))
        if len(neighbors) == 0:
            is_vertical = True 
        else:
            for n in neighbors:
                r_n, c_n = n
                # If change in Row > change in Col, it's vertical
                if abs(r - r_n) > abs(c - c_n): 
                    is_vertical = True
                    break
        
        dip_norm = 1.0 if is_vertical else 0.1 
        
        # Feature C: Amplitude (Distance from center proxy)
        dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        amp_norm = max(0.1, 1.0 - (dist / 100.0))
        
        node_features.append([depth_norm, dip_norm, amp_norm])
        
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 4. Build PyG Edge Index
    edge_index_list = []
    node_list = list(G_pruned.nodes())
    for u, v in G_pruned.edges():
        u_idx, v_idx = node_list.index(u), node_list.index(v)
        edge_index_list.extend([[u_idx, v_idx], [v_idx, u_idx]])
        
    if not edge_index_list: return None, None
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    # 5. Physics Rules (Labels for Training)
    # High Risk = Deep (> 0.2) AND Vertical (> 0.8)
    y_list = [1 if (f[0] > 0.2 and f[1] > 0.8) else 0 for f in node_features]
    y = torch.tensor(y_list, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y), G_pruned