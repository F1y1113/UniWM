import math
import re

# ===================================================================
# 1. Action Range Constants
# ===================================================================
DATASET_RANGES = {
    "tartan_drive": {
        "dxy": [-2.05, 2.05],
        "dyaw": [-0.17, 0.17]
    },
    "recon": {
        "dxy": [-2.46, 2.46],
        "dyaw": [-1.87, 1.87]
    },
    "scand": {
        "dxy": [-0.7879, 0.8518],
        "dyaw": [-0.48, 0.48]
    },
    "sacson": {
        "dxy": [-1.35, 1.35],
        "dyaw": [-2.82, 2.82]
    },
    "stanford": {
        "dxy": [-0.18, 0.18],
        "dyaw": [-0.63, 0.63]
    }
}
DEFAULT_RANGES = {"dxy": [-0.18, 0.18], "dyaw": [-0.4, 0.4]}


# ===================================================================
# 2. Action Calculation Utilities
# ===================================================================
def calculate_action_delta(current_pos_yaw, next_pos_yaw):
    """Calculates the [dx, dy, dyaw] action vector between two poses."""
    delta_x = next_pos_yaw[0] - current_pos_yaw[0]
    delta_y = next_pos_yaw[1] - current_pos_yaw[1]
    delta_yaw = next_pos_yaw[2] - current_pos_yaw[2]
    return [float(delta_x), float(delta_y), float(delta_yaw)]


# ===================================================================
# 3. Action Tokenization Toolkit (Encoder, Decoder, Generator)
# ===================================================================
def action_to_text(action, bin_width=0.01, epsilon=1e-5):
    """Encodes a numerical action vector [dx, dy, dyaw] into a token string."""
    if isinstance(action, str):
        return action

    def to_bin_token(val, prefix):
        token_prefix = f"<{prefix}_pos_bin" if val >= 0 else f"<{prefix}_neg_bin"
        idx = int(math.floor(abs(val) / bin_width))
        return f"{token_prefix}_{idx:02d}>"

    dx_token = to_bin_token(action[0], "dx")
    dy_token = to_bin_token(action[1], "dy")
    dyaw_token = to_bin_token(action[2], "dyaw")

    return f"Move by dx: {dx_token}, dy: {dy_token}, dyaw: {dyaw_token}"

def generate_bin_tokens(prefix, vmin, vmax, step):
    """
    Generates positive, negative, and a zero token.
    """
    tokens = []
    
    # Calculate and generate positive bins based on vmax
    if vmax >= 0:
        nbins_pos = int(math.floor(vmax / step))
        tokens += [f"<{prefix}_pos_bin_{i:02d}>" for i in range(0, nbins_pos + 1)]
        
    # Calculate and generate negative bins based on vmin
    if vmin < 0:
        nbins_neg = int(math.floor(abs(vmin) / step))
        tokens += [f"<{prefix}_neg_bin_{i:02d}>" for i in range(0, nbins_neg + 1)]
        
    return tokens

def extract_bin_values(token_str, prefix, step_val):
    pos_match = re.search(f"<{prefix}_pos_bin_(\d+)>", token_str)
    neg_match = re.search(f"<{prefix}_neg_bin_(\d+)>", token_str)
    
    if pos_match:
        return round(int(pos_match.group(1)) * step_val, 4)
    elif neg_match:
        return round(-int(neg_match.group(1)) * step_val, 4)
    else:
        return 0.0