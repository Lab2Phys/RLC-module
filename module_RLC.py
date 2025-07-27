import numpy as np
from scipy import linalg
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def analyze_circuit(N, edges, source_branch_nodes, veff, i, ref_node):
    """
    تحلیل مدار با استفاده از روش تحلیل گره‌ای و محاسبه جریان‌ها و ولتاژها.
    """
    # 2. تحلیل گره‌ای
    V_source = veff * i

    source_z = None
    for n1, n2, z in edges:
        if (n1, n2) == source_branch_nodes or (n2, n1) == source_branch_nodes:
            source_z = z
            break
    if source_z is None:
        raise ValueError(f"Impedance for source branch {source_branch_nodes} not found.")

    I_norton = V_source / source_z
    num_nodes_to_solve = N - 1

    Y = np.zeros((num_nodes_to_solve, num_nodes_to_solve), dtype=complex)
    I_sources = np.zeros(num_nodes_to_solve, dtype=complex)
    node_map = [n for n in range(1, N + 1) if n != ref_node]

    for n1, n2, z_edge in edges:
        y_edge = 1 / z_edge
        if n1 != ref_node:
            Y[node_map.index(n1), node_map.index(n1)] += y_edge
        if n2 != ref_node:
            Y[node_map.index(n2), node_map.index(n2)] += y_edge
        if n1 != ref_node and n2 != ref_node:
            Y[node_map.index(n1), node_map.index(n2)] -= y_edge
            Y[node_map.index(n2), node_map.index(n1)] -= y_edge

    idx_n1, idx_n2 = source_branch_nodes
    I_sources[node_map.index(idx_n1)] -= I_norton
    I_sources[node_map.index(idx_n2)] += I_norton

    V_unknowns = linalg.solve(Y, I_sources)
    V_nodes = np.zeros(N, dtype=complex)
    for idx, node_num in enumerate(node_map):
        V_nodes[node_num - 1] = V_unknowns[idx]

    # 3. محاسبه جریان‌های شاخه‌ها
    T_currents = []
    for n1, n2, z_edge in edges:
        if (n1, n2) == source_branch_nodes:
            current = (V_nodes[n1-1] - V_nodes[n2-1] + V_source) / z_edge
        elif (n2, n1) == source_branch_nodes:
            current = (V_nodes[n1-1] - V_nodes[n2-1] - V_source) / z_edge
        else:
            current = (V_nodes[n1-1] - V_nodes[n2-1]) / z_edge
        T_currents.append([n1, n2, abs(current)])

    T_currents = np.array(T_currents, dtype=object)
    T_currents = T_currents[np.lexsort((T_currents[:, 1], T_currents[:, 0]))]

    # 4. محاسبه ولتاژ بین گره‌ها
    all_voltages = []
    for n1, n2 in itertools.combinations(range(1, N + 1), 2):
        v_diff = V_nodes[n1 - 1] - V_nodes[n2 - 1]
        all_voltages.append([n1, n2, abs(v_diff), np.angle(v_diff, deg=True)])

    T_voltages_all = np.array(all_voltages, dtype=object)
    T_voltages_all = T_voltages_all[np.lexsort((T_voltages_all[:, 1], T_voltages_all[:, 0]))]

    return T_currents, T_voltages_all

def save_results_to_pdf(pdf_filename, T_currents, T_voltages_all):
    """
    ذخیره جداول نتایج در یک فایل PDF.
    """
    with PdfPages(pdf_filename) as pdf:
        # جدول جریان‌ها
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.axis('off')
        col_labels_currents = ["Node i", "Node j", r"$|\mathbf{I(i,j)}|$ (mA)"]
        table_data_currents_for_pdf = [
            [int(row[0]), int(row[1]), f"{row[2]*1000:.4f}"]
            for row in T_currents
        ]
        table = ax.table(cellText=table_data_currents_for_pdf, colLabels=col_labels_currents,
                         loc='center', colWidths=[0.15, 0.15, 0.26])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)
        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(ha='center', va='center')
            if row == 0:
                cell.set_facecolor('#cce5ff')
                cell.set_text_props(weight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # جدول ولتاژها
        fig, ax = plt.subplots(figsize=(7, 10))
        ax.axis('off')
        col_labels_voltages = ["Node i", "Node j", "|V(i,j)| (V)", "Phase (deg)"]
        table_data_voltages_for_pdf = [[int(r[0]), int(r[1]), f"{r[2]:.4f}", f"{r[3]:.3f}"] for r in T_voltages_all]
        table = ax.table(cellText=table_data_voltages_for_pdf, colLabels=col_labels_voltages,
                         loc='center', colWidths=[0.15, 0.15, 0.22, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(ha='center', va='center')
            if row == 0:
                cell.set_facecolor('#cce5ff')
                cell.set_text_props(weight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()