import numpy as np
from scipy import linalg
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib

# تنظیمات فونت برای نمایش متن انگلیسی در matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

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

def interactive_node_selector(T_currents, T_voltages_all, edges, decimal_places):
    """
    باکس ویجیت کشویی برای انتخاب زوج نودها و نمایش ولتاژ و جریان
    """
    print("\n" + "="*60)
    print("Interactive Node Pair Selector")
    print("="*60)
    
    # تبدیل edges به مجموعه‌ای از tuple برای جستجوی سریع
    edge_set = set()
    for n1, n2, _ in edges:
        edge_set.add((min(n1, n2), max(n1, n2)))
    
    # ایجاد لیست تمام زوج‌های ممکن
    all_pairs = []
    for row in T_voltages_all:
        all_pairs.append((int(row[0]), int(row[1])))
    
    while True:
        print("\nزوج‌های نود موجود:")
        for idx, (i, j) in enumerate(all_pairs):
            print(f"{idx + 1}. ({i}, {j})")
        
        print(f"{len(all_pairs) + 1}. خروج")
        
        try:
            choice = int(input(f"\nیکی از گزینه‌های 1 تا {len(all_pairs) + 1} را انتخاب کنید: "))
            
            if choice == len(all_pairs) + 1:
                print("خروج از برنامه انتخاب نود.")
                break
            elif 1 <= choice <= len(all_pairs):
                selected_pair = all_pairs[choice - 1]
                i, j = selected_pair
                
                # پیدا کردن ولتاژ
                voltage_info = None
                for row in T_voltages_all:
                    if int(row[0]) == i and int(row[1]) == j:
                        voltage_info = row
                        break
                
                print(f"\n{'='*40}")
                print(f"نتایج برای زوج نود ({i}, {j}):")
                print(f"{'='*40}")
                
                if voltage_info is not None:
                    format_str = f"{{:.{decimal_places}f}}"
                    print(f"ولتاژ |V({i},{j})|: {format_str.format(voltage_info[2])} V")
                    print(f"فاز ولتاژ: {format_str.format(voltage_info[3])} درجه")
                
                # بررسی اینکه آیا این دو نود مستقیماً متصل هستند یا نه
                normalized_pair = (min(i, j), max(i, j))
                if normalized_pair in edge_set:
                    # پیدا کردن جریان
                    current_info = None
                    for row in T_currents:
                        if (int(row[0]) == i and int(row[1]) == j) or (int(row[0]) == j and int(row[1]) == i):
                            current_info = row
                            break
                    
                    if current_info is not None:
                        current_ma = current_info[2] * 1000
                        print(f"جریان |I({int(current_info[0])},{int(current_info[1])})|: {format_str.format(current_ma)} mA")
                    print("این دو نود مستقیماً متصل هستند.")
                else:
                    print("این دو نود مستقیماً متصل نیستند (جریان مستقیم وجود ندارد).")
                
            else:
                print("انتخاب نامعتبر! لطفاً دوباره تلاش کنید.")
                
        except ValueError:
            print("ورودی نامعتبر! لطفاً یک عدد صحیح وارد کنید.")
        except KeyboardInterrupt:
            print("\nبرنامه متوقف شد.")
            break

def save_results_to_pdf(pdf_filename, T_currents, T_voltages_all, decimal_places):
    """
    ذخیره جداول نتایج در یک فایل PDF با عنوان‌های مناسب.
    """
    format_str = f"{{:.{decimal_places}f}}"
    
    with PdfPages(pdf_filename) as pdf:
        # مشکل سوم: جدول جریان‌ها با عنوان
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.axis('off')
        
        # اضافه کردن عنوان
        fig.suptitle('Current Table of Circuit', fontsize=16, fontweight='bold', y=0.9)
        
        col_labels_currents = ["Node i", "Node j", r"$|\mathbf{I(i,j)}|$ (mA)"]
        table_data_currents_for_pdf = [
            [int(row[0]), int(row[1]), format_str.format(row[2]*1000)]
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
        
        # تنظیم موقعیت جدول تا جا برای عنوان باشد
        ax.set_position([0.1, 0.1, 0.8, 0.7])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # مشکل چهارم: جدول ولتاژها با عنوان
        fig, ax = plt.subplots(figsize=(8, 11))
        ax.axis('off')
        
        # اضافه کردن عنوان
        fig.suptitle('Voltage Table of Circuit', fontsize=16, fontweight='bold', y=0.95)
        
        col_labels_voltages = ["Node i", "Node j", "|V(i,j)| (V)", "Phase (deg)"]
        table_data_voltages_for_pdf = [
            [int(r[0]), int(r[1]), format_str.format(r[2]), format_str.format(r[3])] 
            for r in T_voltages_all
        ]
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
        
        # تنظیم موقعیت جدول تا جا برای عنوان باشد
        ax.set_position([0.1, 0.05, 0.8, 0.85])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"\nنتایج با موفقیت در فایل '{pdf_filename}' ذخیره شد.")