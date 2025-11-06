import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import ConnectionPatch

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_min_entropy(filename, save_as="Pguess.png"):
    """
    Plot the min-entropy H_min from a data file containing (omega, non_ea, ea) values.

    Parameters
    ----------
    filename : str
        Path to the data file (e.g., 'data_avg_pg1.txt').
        Each line should contain three comma-separated values:
            omega, non_ea_value, ea_value
    save_as : str, optional
        Output filename for saving the figure (default is 'Pguess.png').

    The function reads the data, computes -log2(values),
    and plots both curves with an inset zoomed region.
    """

    # --- Load data from text file ---
    omega, non_ea, ea = [], [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # ⬅️ skip header/comment lines
                continue
            line = line.replace("(", "").replace(")", "")
            parts = line.split(",")
            omega.append(float(parts[0]))
            non_ea.append(float(parts[1]))
            ea.append(float(parts[2]))

    # Convert to numpy arrays
    omega = np.array(omega)
    non_ea = np.array(non_ea)
    ea = np.array(ea)

    # --- Main figure ---
    fig, ax = plt.subplots()

    ax.plot(omega, -np.log2(non_ea), label=r'$H_{\min}^{*, \mathrm{sep}}$ [22]', linewidth=2)
    ax.plot(omega, -np.log2(ea), label=r'Seesaw upper bound to $H_{\min}^*$', linewidth=2)

    ax.set_xlabel(r'$\omega$', fontsize=14)
    ax.set_ylabel(r'$H_{\min}$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # --- Inset plot (zoomed region) ---
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)

    axins.plot(omega, -np.log2(non_ea), linewidth=2)
    axins.plot(omega, -np.log2(ea), linewidth=2)

    # Define zoom region
    x1, x2 = 0.32, 0.47
    axins.set_xlim(x1, x2)

    # Compute y-range dynamically
    y_data = (
        [-np.log2(y) for x, y in zip(omega, ea) if x1 <= x <= x2] +
        [-np.log2(y) for x, y in zip(omega, non_ea) if x1 <= x <= x2]
    )
    if y_data:
        y_min, y_max = min(y_data), max(y_data)
        padding = 0.05 * (y_max - y_min)
        axins.set_ylim(y_min - padding, y_max + padding)

    # Style inset
    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(axis='both', which='major', labelsize=9)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(1.0)

    axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Connect inset with main plot
    xy_main_1 = (x1, y_min)
    xy_inset_1 = (x1, y_min)
    xy_main_2 = (x2, y_max)
    xy_inset_2 = (x2, y_max)

    con1 = ConnectionPatch(
        xyA=xy_inset_1, coordsA=axins.transData,
        xyB=xy_main_1, coordsB=ax.transData,
        color="gray", linestyle="--", lw=1.0, alpha=0.6, clip_on=False
    )
    con2 = ConnectionPatch(
        xyA=xy_inset_2, coordsA=axins.transData,
        xyB=xy_main_2, coordsB=ax.transData,
        color="gray", linestyle="--", lw=1.0, alpha=0.6, clip_on=False
    )
    ax.add_artist(con1)
    ax.add_artist(con2)

    # --- Save and show ---
    fig.savefig(save_as, dpi=1000, bbox_inches="tight")
    plt.show()
    print(f"✅ Figure saved as {save_as}")

def plot_channel_discrimination_advantage(filename="Channel_discr_adv_data.txt", save_as="Channel_discr_adv.png"):
    """
    Plot the channel discrimination advantage data from a text file.

    Parameters
    ----------
    filename : str
        Path to the .txt file containing the data.
        Expected format per line (without parentheses):
            omega, ea_value, non_ea_value
        Example:
            0.01, 1.05, 1.00
    save_as : str
        Filename to save the generated figure as (default: 'Channel_discr_adv.png').

    Notes
    -----
    - The green dashed line corresponds to the analytical bound in Eq. (19).
    - The first and second columns correspond respectively to:
        - entanglement-assisted advantage (ea)
        - non-entangled advantage (non_ea)
    """

    # === Load data from TXT ===
    omega, ea, non_ea = [], [], []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # skip empty or comment lines
                continue
            line = line.replace("(", "").replace(")", "")
            parts = line.split(",")
            if len(parts) < 3:
                continue
            omega.append(float(parts[0]))
            ea.append(float(parts[1]))
            non_ea.append(float(parts[2]))

    # Convert to numpy arrays
    omega = np.array(omega)
    ea = np.array(ea)
    non_ea = np.array(non_ea)

    # === Plot setup ===
    fig, ax = plt.subplots()

    ax.plot(omega, non_ea, label='Eq. (22)', linewidth=2)
    ax.plot(omega, ea, label='Eq. (23)', linewidth=2)

    # Theoretical bound line (Eq. 19)
    y_val = 0.5 + 1 / np.sqrt(2)
    ax.hlines(y=y_val, xmin=0.01, xmax=0.5, colors='green', linestyles='--', linewidth=2, label='Eq. (19)')

    # === Axis labels and style ===
    ax.set_xlabel(r'$\omega$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='best')

    # === Save and show ===
    plt.show()
    fig.savefig(save_as, dpi=1000, bbox_inches='tight')
    print(f"✅ Plot saved as '{save_as}'")