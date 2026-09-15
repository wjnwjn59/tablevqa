import random
import string
import json
from playwright.sync_api import sync_playwright

# Cố định seed
random.seed(42)

def generate_massive_unique_data(num_rows=80):
    """Tạo tập dữ liệu lớn, không lặp lại với nhiều yếu tố thị giác."""
    data = []
    regions = ['NA', 'EMEA', 'APAC', 'LATAM']
    sectors = ['Tech', 'Health', 'Finance', 'Retail', 'Energy']
    traffic_states = ['red', 'yellow', 'green']
    
    for i in range(num_rows):
        base_rev = random.uniform(50, 900)
        q1 = base_rev * random.uniform(0.8, 1.2)
        q2 = base_rev * random.uniform(0.7, 1.3)
        q3 = base_rev * random.uniform(0.9, 1.4)
        q4 = base_rev * random.uniform(0.5, 1.5)
        ytd = q1 + q2 + q3 + q4
        target = base_rev * 4 * random.uniform(0.9, 1.1)
        
        row = {
            'id': f"ITM-{10000+i}",
            'region': random.choice(regions),
            'sector': random.choice(sectors),
            'code': ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)),
            'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4,
            'ytd': ytd,
            'variance': (ytd / target - 1) * 100,
            'headcount': random.randint(150, 8000),
            'csat': random.uniform(3.0, 5.0),
            'defect': random.uniform(0.1, 4.5),
            'trend_data': [random.randint(10, 100) for _ in range(12)],
            'win_loss_data': [random.choice([-1, 1, 1, -1, 1]) for _ in range(8)],
            'rating': random.randint(1, 5),
            'target_progress': random.randint(40, 110),
            'traffic_light': random.choice(traffic_states)
        }
        data.append(row)
    return data

# --- CÁC HÀM TẠO SVG MICRO-CHARTS ---

def generate_sparkline_svg(data_points):
    avg = sum(data_points) / len(data_points)
    max_val = max(data_points) if max(data_points) > 0 else 1
    pts = []
    for i, val in enumerate(data_points):
        x = i * (120 / (len(data_points) - 1))
        y = 35 - (val / max_val * 30)
        pts.append(f"{x:.1f},{y:.1f}")
    
    poly = " ".join(pts)
    avg_y = 35 - (avg / max_val * 30)
    return f"""
    <svg width="120" height="40" viewBox="0 0 120 40" style="vertical-align: middle;">
        <line x1="0" y1="{avg_y}" x2="120" y2="{avg_y}" stroke="#94a3b8" stroke-dasharray="2,2" stroke-width="1.5"/>
        <polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="1.5"/>
        <circle cx="120" cy="{pts[-1].split(',')[1]}" r="2.5" fill="#dc2626"/>
    </svg>
    """

def generate_win_loss_svg(data_points):
    rects = ""
    for i, val in enumerate(data_points):
        x = i * 14
        if val > 0:
            rects += f'<rect x="{x}" y="5" width="10" height="15" fill="#16a34a"/>'
        else:
            rects += f'<rect x="{x}" y="20" width="10" height="15" fill="#dc2626"/>'
    return f"""
    <svg width="110" height="40" viewBox="0 0 110 40" style="vertical-align: middle;">
        <line x1="0" y1="20" x2="110" y2="20" stroke="#cbd5e1" stroke-width="1.5"/>
        {rects}
    </svg>
    """

def generate_rating_svg(rating):
    bars = ""
    for i in range(5):
        x = i * 12
        h = (i + 1) * 5
        y = 30 - h
        fill_color = "#eab308" if i < rating else "#e2e8f0"
        bars += f'<rect x="{x}" y="{y}" width="8" height="{h}" fill="{fill_color}"/>'
    return f"""
    <svg width="60" height="40" viewBox="0 0 60 40" style="vertical-align: middle;">
        {bars}
    </svg>
    """

def generate_bullet_svg(progress):
    bar_width = min(progress, 100)
    return f"""
    <svg width="100" height="40" viewBox="0 0 100 40" style="vertical-align: middle;">
        <rect x="0" y="15" width="100" height="10" fill="#e2e8f0"/>
        <rect x="0" y="15" width="{bar_width}" height="10" fill="#3b82f6"/>
        <line x1="80" y1="5" x2="80" y2="35" stroke="#0f172a" stroke-width="2"/>
    </svg>
    """

def generate_traffic_light_svg(state):
    c_red = "#ef4444" if state == "red" else "#fee2e2"
    c_yel = "#f59e0b" if state == "yellow" else "#fef3c7"
    c_grn = "#10b981" if state == "green" else "#d1fae5"
    return f"""
    <svg width="70" height="40" viewBox="0 0 70 40" style="vertical-align: middle;">
        <rect x="0" y="5" width="70" height="30" rx="15" fill="#f8fafc" stroke="#cbd5e1"/>
        <circle cx="15" cy="20" r="8" fill="{c_red}"/>
        <circle cx="35" cy="20" r="8" fill="{c_yel}"/>
        <circle cx="55" cy="20" r="8" fill="{c_grn}"/>
    </svg>
    """

def build_html_table(data, case_type="hierarchical"):
    css_base = """
        body { font-family: 'Segoe UI', Helvetica, sans-serif; background: #fff; margin: 0; padding: 0; color: #1e293b; }
        table { border-collapse: collapse; font-size: 13px; margin: 0; background: #fff; }
        th, td { border: 1px solid #cbd5e1; padding: 8px 12px; text-align: right; vertical-align: middle; }
        th { background-color: #f8fafc; text-align: center; font-weight: 600; color: #334155; }
        td.left { text-align: left; }
        .negative { color: #b91c1c; }
        .highlight { font-weight: 700; }
        .center-chart { text-align: center; padding: 2px; }
    """

    if case_type == "visual_noise":
        css_custom = css_base + """
        body {
            background: #e2e8f0; padding: 30px; 
            transform: perspective(2000px) rotateX(1.5deg) rotateZ(0.5deg);
            transform-origin: center center; filter: blur(1.1px) contrast(0.75) sepia(0.15) opacity(0.9);
        }
        table { width: 1000px; box-shadow: 10px 10px 30px rgba(0,0,0,0.1); border: 2px solid #94a3b8; }
        th, td { border-color: #94a3b8; color: #475569; }
        """
    elif case_type == "charts":
        css_custom = css_base + "table { min-width: 1800px; }"
    elif case_type == "borderless_misaligned":
        # Không viền, không nền header, padding siêu nhỏ ép sát nhau, font chữ monospace giống báo cáo raw
        css_custom = """
        body { font-family: 'Courier New', Courier, monospace; background: #fff; margin: 0; padding: 0; color: #000; }
        table { border-collapse: collapse; font-size: 11px; margin: 0; background: #fff; min-width: 1400px; border: none; }
        th, td { border: none !important; padding: 1px 4px; text-align: right; vertical-align: bottom; background: transparent !important;}
        th { font-weight: bold; border-bottom: 1px solid #000 !important; } /* Duy nhất đường gạch chân mỏng ở Header chính */
        td.left { text-align: left; }
        .negative { color: #000; } /* Bỏ màu đỏ, ép mô hình tự đọc dấu trừ */
        """
    else: 
        css_custom = css_base + "table { min-width: 1600px; border: 2px solid #334155; } th { border: 1px solid #94a3b8; }"

    html = f"<!DOCTYPE html><html><head><style>{css_custom}</style></head><body>"
    html += "<table><thead>"

    if case_type == "visual_noise":
        html += """
            <tr>
                <th>Report ID</th><th>Region</th><th>Sector</th>
                <th>Q1 ($M)</th><th>Q2 ($M)</th><th>Q3 ($M)</th><th>Q4 ($M)</th>
                <th>YTD Total</th><th>Variance %</th><th>Headcount</th>
            </tr>
        """
    elif case_type == "charts":
        html += """
            <tr>
                <th>ID</th><th>Region</th><th>Sector</th>
                <th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Var %</th>
                <th>Trend (12-Mo)</th><th>Win/Loss</th>
                <th>Signal Rating</th><th>Target Progress</th><th>Status</th>
            </tr>
        """
    elif case_type in ["hierarchical", "borderless_misaligned"]:
        # Cấu trúc chung nhưng hiển thị hoàn toàn khác do CSS
        html += """
            <tr>
                <th colspan="3" style="text-align:left;">Entity Profile</th>
                <th colspan="5">FY24 Revenue Distribution ($M)</th>
                <th colspan="3">Operational Cost & HR</th>
                <th colspan="2" style="text-align:right;">Quality Metrics</th>
            </tr>
            <tr>
                <th rowspan="2" class="left">Report ID</th><th rowspan="2">Region</th><th rowspan="2">Sector</th>
                <th colspan="4">Quarterly Actuals</th><th rowspan="2">YTD Total</th>
                <th rowspan="2">Variance %</th><th rowspan="2">Headcount</th><th rowspan="2">Budget Code</th>
                <th rowspan="2">CSAT (/5.0)</th><th rowspan="2">Defect %</th>
            </tr>
            <tr>
                <th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th>
            </tr>
        """

    html += "</thead><tbody>"

    # Hàm tạo độ lệch pixel ngẫu nhiên nhưng cố định theo ID (Deterministic misalign)
    def get_offset(id_str, col_idx):
        return (ord(id_str[-1]) * (col_idx + 1)) % 18

    for idx, row in enumerate(data):
        var_class = 'negative' if row['variance'] < 0 else ''
        
        if case_type == "visual_noise":
            html += f"""
            <tr>
                <td class="left highlight">{row['id']}</td><td class="left">{row['region']}</td><td class="left">{row['sector']}</td>
                <td>{row['q1']:,.1f}</td><td>{row['q2']:,.1f}</td><td>{row['q3']:,.1f}</td><td>{row['q4']:,.1f}</td>
                <td class="highlight">{row['ytd']:,.1f}</td><td class="{var_class}">{row['variance']:+.1f}%</td><td>{row['headcount']:,}</td>
            </tr>
            """
        elif case_type == "charts":
            html += f"""
            <tr>
                <td class="left highlight">{row['id']}</td><td class="left">{row['region']}</td><td class="left">{row['sector']}</td>
                <td>{row['q1']:,.0f}</td><td>{row['q2']:,.0f}</td><td>{row['q3']:,.0f}</td><td>{row['q4']:,.0f}</td>
                <td class="{var_class}">{row['variance']:+.1f}%</td>
                <td class="center-chart">{generate_sparkline_svg(row['trend_data'])}</td>
                <td class="center-chart">{generate_win_loss_svg(row['win_loss_data'])}</td>
                <td class="center-chart">{generate_rating_svg(row['rating'])}</td>
                <td class="center-chart">{generate_bullet_svg(row['target_progress'])}</td>
                <td class="center-chart">{generate_traffic_light_svg(row['traffic_light'])}</td>
            </tr>
            """
        elif case_type == "hierarchical":
            html += f"""
            <tr>
                <td class="left highlight">{row['id']}</td><td class="left">{row['region']}</td><td class="left">{row['sector']}</td>
                <td>{row['q1']:,.1f}</td><td>{row['q2']:,.1f}</td><td>{row['q3']:,.1f}</td><td>{row['q4']:,.1f}</td>
                <td class="highlight">{row['ytd']:,.1f}</td><td class="{var_class}">{row['variance']:+.2f}%</td>
                <td>{row['headcount']:,}</td><td class="left">{row['code']}</td>
                <td>{row['csat']:.2f}</td><td>{row['defect']:.2f}%</td>
            </tr>
            """
        elif case_type == "borderless_misaligned":
            # Áp dụng độ lệch padding-right hoặc padding-left để chữ bị zic-zac
            html += f"""
            <tr>
                <td class="left" style="padding-left: {get_offset(row['id'], 0)}px;">{row['id']}</td>
                <td class="left" style="padding-left: {get_offset(row['id'], 1)}px;">{row['region']}</td>
                <td class="left" style="padding-left: {get_offset(row['id'], 2)}px;">{row['sector']}</td>
                <td style="padding-right: {get_offset(row['id'], 3)}px;">{row['q1']:,.1f}</td>
                <td style="padding-right: {get_offset(row['id'], 4)}px;">{row['q2']:,.1f}</td>
                <td style="padding-right: {get_offset(row['id'], 5)}px;">{row['q3']:,.1f}</td>
                <td style="padding-right: {get_offset(row['id'], 6)}px;">{row['q4']:,.1f}</td>
                <td style="padding-right: {get_offset(row['id'], 7)}px; font-weight:bold;">{row['ytd']:,.1f}</td>
                <td class="{var_class}" style="padding-right: {get_offset(row['id'], 8)}px;">{row['variance']:+.2f}%</td>
                <td style="padding-right: {get_offset(row['id'], 9)}px;">{row['headcount']:,}</td>
                <td style="padding-right: {get_offset(row['id'], 10)}px;">{row['code']}</td>
                <td style="padding-right: {get_offset(row['id'], 11)}px;">{row['csat']:.2f}</td>
                <td style="padding-right: {get_offset(row['id'], 12)}px;">{row['defect']:.2f}%</td>
            </tr>
            """

    html += "</tbody></table></body></html>"
    return html

def capture_standardized_cases():
    full_data = generate_massive_unique_data(num_rows=80)
    
    cases = [
        {"name": "01_visual_noise_avg_size.png", "type": "visual_noise", "data": full_data[:20]},
        {"name": "02_charts_in_cell.png", "type": "charts", "data": full_data[:40]},
        {"name": "03_hierarchical_large.png", "type": "hierarchical", "data": full_data},
        # Case mới: 50 dòng, dữ liệu zic-zac, không có đường kẻ nào
        {"name": "04_borderless_misaligned.png", "type": "borderless_misaligned", "data": full_data[:50]} 
    ]

    qa_dataset = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 2000, "height": 4000}, device_scale_factor=1.5) 
        
        for case in cases:
            html_content = build_html_table(case["data"], case_type=case["type"])
            page.set_content(html_content, wait_until="networkidle")
            
            if case["type"] == "visual_noise":
                page.locator("body").screenshot(path=case["name"])
            else:
                page.locator("table").screenshot(path=case["name"])
                
            print(f"✅ Đã lưu ảnh: {case['name']}")
            
            # --- SINH CÂU HỎI VÀ ĐÁP ÁN (GROUND TRUTH) ---
            if case["type"] == "visual_noise":
                row_12 = next(r for r in case["data"] if r['id'] == "ITM-10012")
                row_08 = next(r for r in case["data"] if r['id'] == "ITM-10008")
                row_03 = next(r for r in case["data"] if r['id'] == "ITM-10003")
                row_18 = next(r for r in case["data"] if r['id'] == "ITM-10018")
                row_14 = next(r for r in case["data"] if r['id'] == "ITM-10014")
                
                qa_dataset.extend([
                    {"image_path": case["name"], "task_type": "Visual Noise", "question": "What is the Variance % for Report ID ITM-10012?", "answer": f"{row_12['variance']:+.1f}%"},
                    {"image_path": case["name"], "task_type": "Visual Noise", "question": "What is the Sector for Report ID ITM-10008?", "answer": row_08['sector']},
                    {"image_path": case["name"], "task_type": "Visual Noise", "question": "What is the Q1 ($M) value for Report ID ITM-10003?", "answer": f"{row_03['q1']:,.1f}"},
                    {"image_path": case["name"], "task_type": "Visual Noise", "question": "What is the Headcount for Report ID ITM-10018?", "answer": f"{row_18['headcount']:,}"},
                    {"image_path": case["name"], "task_type": "Visual Noise", "question": "What is the Region for Report ID ITM-10014?", "answer": row_14['region']}
                ])
                
            elif case["type"] == "charts":
                row_15 = next(r for r in case["data"] if r['id'] == "ITM-10015")
                row_35 = next(r for r in case["data"] if r['id'] == "ITM-10035")
                row_22 = next(r for r in case["data"] if r['id'] == "ITM-10022")
                row_10 = next(r for r in case["data"] if r['id'] == "ITM-10010")
                row_19 = next(r for r in case["data"] if r['id'] == "ITM-10019")
                row_25 = next(r for r in case["data"] if r['id'] == "ITM-10025")
                
                wins_15 = str(sum(1 for x in row_15['win_loss_data'] if x > 0))
                losses_22 = str(sum(1 for x in row_22['win_loss_data'] if x < 0))
                sparkline_status = "above" if row_35['trend_data'][-1] > (sum(row_35['trend_data']) / len(row_35['trend_data'])) else "below"
                progress_crossed = "yes" if row_19['target_progress'] >= 80 else "no"
                
                qa_dataset.extend([
                    {"image_path": case["name"], "task_type": "Charts in Cell", "question": "How many green bars are in the Win/Loss column for Report ID ITM-10015?", "answer": wins_15},
                    {"image_path": case["name"], "task_type": "Charts in Cell", "question": "In the Trend column for Report ID ITM-10035, is the final red dot 'above' or 'below' the dashed line?", "answer": sparkline_status},
                    {"image_path": case["name"], "task_type": "Charts in Cell", "question": "How many red bars are in the Win/Loss column for Report ID ITM-10022?", "answer": losses_22},
                    {"image_path": case["name"], "task_type": "Charts in Cell", "question": "How many active (yellow) bars are displayed in the Signal Rating for Report ID ITM-10010?", "answer": str(row_10['rating'])},
                    {"image_path": case["name"], "task_type": "Charts in Cell", "question": "Does the blue progress bar reach or exceed the vertical black target line for Report ID ITM-10019?", "answer": progress_crossed},
                    {"image_path": case["name"], "task_type": "Charts in Cell", "question": "What color is illuminated in the Status traffic light for Report ID ITM-10025?", "answer": row_25['traffic_light']}
                ])
                
            elif case["type"] == "hierarchical":
                row_65 = next(r for r in case["data"] if r['id'] == "ITM-10065")
                row_78 = next(r for r in case["data"] if r['id'] == "ITM-10078")
                row_50 = next(r for r in case["data"] if r['id'] == "ITM-10050")
                row_42 = next(r for r in case["data"] if r['id'] == "ITM-10042")
                row_71 = next(r for r in case["data"] if r['id'] == "ITM-10071")
                
                qa_dataset.extend([
                    {"image_path": case["name"], "task_type": "Hierarchical Header", "question": "What is the Defect % for Report ID ITM-10065?", "answer": f"{row_65['defect']:.2f}%"},
                    {"image_path": case["name"], "task_type": "Hierarchical Header", "question": "What is the Q3 value for Report ID ITM-10078?", "answer": f"{row_78['q3']:,.1f}"},
                    {"image_path": case["name"], "task_type": "Hierarchical Header", "question": "What is the Budget Code for Report ID ITM-10050?", "answer": row_50['code']},
                    {"image_path": case["name"], "task_type": "Hierarchical Header", "question": "What is the Headcount for Report ID ITM-10042?", "answer": f"{row_42['headcount']:,}"},
                    {"image_path": case["name"], "task_type": "Hierarchical Header", "question": "What is the CSAT score for Report ID ITM-10071?", "answer": f"{row_71['csat']:.2f}"}
                ])

            elif case["type"] == "borderless_misaligned":
                # Lấy dữ liệu của các dòng sâu bên dưới để buộc mô hình phải dóng hàng thật xa
                row_45 = next(r for r in case["data"] if r['id'] == "ITM-10045")
                row_32 = next(r for r in case["data"] if r['id'] == "ITM-10032")
                row_11 = next(r for r in case["data"] if r['id'] == "ITM-10011")
                row_48 = next(r for r in case["data"] if r['id'] == "ITM-10048")
                
                qa_dataset.extend([
                    # Test dóng hàng ngang cự ly xa (bên trái cùng -> tít bên phải cùng)
                    {"image_path": case["name"], "task_type": "Borderless Misaligned", "question": "What is the Defect % for Report ID ITM-10045?", "answer": f"{row_45['defect']:.2f}%"},
                    # Test dóng cột phân cấp không có border ngăn cách
                    {"image_path": case["name"], "task_type": "Borderless Misaligned", "question": "What is the Q2 value for Report ID ITM-10032?", "answer": f"{row_32['q2']:,.1f}"},
                    # Test đọc số nguyên có dấu phẩy
                    {"image_path": case["name"], "task_type": "Borderless Misaligned", "question": "What is the Headcount for Report ID ITM-10011?", "answer": f"{row_11['headcount']:,}"},
                    # Test tìm dữ liệu cột chữ ở giữa bảng số (dễ bị nhiễu do zic-zac)
                    {"image_path": case["name"], "task_type": "Borderless Misaligned", "question": "What is the Budget Code for Report ID ITM-10048?", "answer": row_48['code']}
                ])

        page.close()
        browser.close()

    with open("qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Đã lưu bộ Ground Truth vào file 'qa_dataset.json'.")

if __name__ == "__main__":
    capture_standardized_cases()