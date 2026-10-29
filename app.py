import os
import threading
import time
import io
import torch
import numpy as np
from flask import Flask, render_template, jsonify, request, send_file
from scapy.all import sniff, IP, TCP, UDP
from groq import Groq
from train_model import HybridCNNLSTM
from captum.attr import IntegratedGradients

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

# ==========================================
# AI Model & Groq Setup
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FEATURES = 77  # Matches trained feature shape

model = HybridCNNLSTM(num_features=NUM_FEATURES).to(DEVICE)
ig = IntegratedGradients(model)

try:
    model.load_state_dict(torch.load("cnn_lstm_sentinel.pth", map_location=DEVICE))
    model.eval()
    print("🛡️ CNN-LSTM Sentinel model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load model weights ({e}). Running with rule-based fallback.")

# Groq Client Initialization
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

packet_feature_buffer = []
SEQUENCE_LENGTH = 10

# Global state for live metrics and toggle control
sniffing_active = False  # Default to False so it doesn't sniff automatically on startup

traffic_data = {
    "total_packets": 0,
    "malicious_count": 0,
    "benign_count": 0,
    "alerts": [],
    "recent_packets": []
}

def explain_prediction(input_tensor, target_class=1):
    try:
        input_tensor.requires_grad_()
        attributions, delta = ig.attribute(input_tensor, target=target_class, return_convergence_delta=True)
        feature_importances = attributions.detach().cpu().numpy().mean(axis=(0, 1))
        return feature_importances.tolist()
    except Exception as e:
        print(f"XAI calculation error: {e}")
        return []

def analyze_packet(packet):
    global traffic_data, packet_feature_buffer
    
    # Only process packets if sniffing is toggled ON
    if not sniffing_active:
        return

    if packet.haslayer(IP):
        traffic_data["total_packets"] += 1
        
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = "TCP" if packet.haslayer(TCP) else ("UDP" if packet.haslayer(UDP) else "Other")
        length = len(packet)
        
        feat_vector = np.zeros(NUM_FEATURES, dtype=np.float32)
        feat_vector[0] = float(length)
        feat_vector[1] = float(packet[IP].ttl)
        feat_vector[2] = 1.0 if protocol == "TCP" else (2.0 if protocol == "UDP" else 0.0)
        
        if packet.haslayer(TCP):
            feat_vector[3] = float(packet[TCP].sport)
            feat_vector[4] = float(packet[TCP].dport)
            flags = str(packet[TCP].flags)
            feat_vector[5] = 1.0 if 'S' in flags else 0.0
            feat_vector[6] = 1.0 if 'A' in flags else 0.0
        elif packet.haslayer(UDP):
            feat_vector[3] = float(packet[UDP].sport)
            feat_vector[4] = float(packet[UDP].dport)

        packet_feature_buffer.append(feat_vector)
        if len(packet_feature_buffer) > SEQUENCE_LENGTH:
            packet_feature_buffer.pop(0)
            
        is_malicious = False
        reason = ""

        # Check Heuristic Rules First
        if packet.haslayer(TCP) and 'S' in str(packet[TCP].flags) and len(packet) < 60:
            is_malicious = True
            reason = "Heuristic SYN Scan pattern detected"

        # Check CNN-LSTM Model with confidence threshold
        if not is_malicious and len(packet_feature_buffer) == SEQUENCE_LENGTH:
            try:
                input_array = np.array(packet_feature_buffer, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                input_tensor = torch.tensor(input_array).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][pred_class].item()
                    
                    if pred_class == 1 and confidence >= 0.85:
                        is_malicious = True
                        reason = f"High-Confidence CNN-LSTM Threat ({confidence*100:.1f}% confidence)"
            except Exception as ml_err:
                print(f"ML Inference execution error: {ml_err}")

        if is_malicious:
            traffic_data["malicious_count"] += 1
            alert_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": protocol,
                "reason": reason
            }
            traffic_data["alerts"].insert(0, alert_entry)
            traffic_data["alerts"] = traffic_data["alerts"][:20]
        else:
            traffic_data["benign_count"] += 1

        packet_log = {
            "timestamp": time.strftime("%H:%M:%S"),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": protocol,
            "length": length,
            "status": "Malicious" if is_malicious else "Benign"
        }
        traffic_data["recent_packets"].insert(0, packet_log)
        traffic_data["recent_packets"] = traffic_data["recent_packets"][:15]

def start_sniffing():
    sniff(prn=analyze_packet, store=False)

# ==========================================
# Flask Routes & Endpoints
# ==========================================
@app.route("/")
def landing_page():
    return render_template("landingpage.html")

@app.route("/index")
def index():
    return render_template("index.html", active="dashboard")

@app.route("/chat")
def chat_page():
    return render_template("chat.html", active="chat")

@app.route("/news")
def news_page():
    news_api_key = os.environ.get("NEWS_API_KEY", "")
    url = f"https://newsapi.org/v2/everything?q=cybersecurity&language=en&sortBy=publishedAt&pageSize=12&apiKey={news_api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get("articles", []) if data.get("status") == "ok" else []
    except Exception:
        articles = []
    return render_template("news.html", active="news", articles=articles)

@app.route("/api/stats")
def get_stats():
    return jsonify({
        **traffic_data,
        "sniffing_active": sniffing_active
    })

@app.route("/api/toggle-sniffing", methods=["POST"])
def toggle_sniffing():
    global sniffing_active
    data = request.get_json(silent=True)
    if data and "active" in data:
        sniffing_active = bool(data["active"])
    else:
        sniffing_active = not sniffing_active
    return jsonify({"sniffing_active": sniffing_active})

@app.route("/api/ai-summary", methods=["POST"])
def generate_ai_summary():
    try:
        total = traffic_data["total_packets"]
        malicious = traffic_data["malicious_count"]
        benign = traffic_data["benign_count"]
        alerts = traffic_data["alerts"][:5]

        prompt = (
            f"Analyze these network intrusion detection metrics:\n"
            f"- Total Packets: {total}\n"
            f"- Benign Packets: {benign}\n"
            f"- Malicious Threats Detected: {malicious}\n"
            f"- Recent Threat Alerts: {str(alerts)}\n\n"
            f"Provide a concise executive security summary, potential risks, and recommendations for the network administrator."
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are XAI-Sentinel AI, an expert cybersecurity incident responder."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
        )
        return jsonify({"summary": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"summary": f"Error generating summary: {str(e)}"}), 500

@app.route("/api/download-pdf", methods=["GET"])
def download_pdf():
    try:
        total = traffic_data["total_packets"]
        malicious = traffic_data["malicious_count"]
        benign = traffic_data["benign_count"]
        alerts = traffic_data["alerts"][:5]

        prompt = (
            f"Provide a concise executive security summary and risk evaluation based on these live IDS metrics:\n"
            f"- Total Packets: {total}\n- Benign: {benign}\n- Malicious: {malicious}\n- Recent Alerts: {str(alerts)}"
        )
        
        try:
            ai_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are XAI-Sentinel AI, an expert cybersecurity incident responder."},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-120b",
            )
            ai_summary_text = ai_completion.choices[0].message.content
        except Exception:
            ai_summary_text = "AI summary unavailable at the time of export."

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#0ea5e9'), spaceAfter=10)
        heading_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#334155'), spaceBefore=12, spaceAfter=6)
        normal_style = styles['Normal']

        elements.append(Paragraph("🛡️ XAI-Sentinel Security Incident Report", title_style))
        elements.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Groq AI Security Assessment", heading_style))
        elements.append(Paragraph(ai_summary_text, normal_style))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Traffic Statistics Overview", heading_style))
        summary_data = [
            ["Metric", "Count"],
            ["Total Packets Inspected", str(total)],
            ["Benign Traffic", str(benign)],
            ["Malicious Threats Flagged", str(malicious)]
        ]
        t = Table(summary_data, colWidths=[250, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f1f5f9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1'))
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Active Threat Alerts Log", heading_style))
        if traffic_data["alerts"]:
            alert_data = [["Time", "Source IP", "Target IP", "Protocol", "Reason"]]
            for a in traffic_data["alerts"][:10]:
                alert_data.append([a["timestamp"], a["src_ip"], a["dst_ip"], a["protocol"], a["reason"]])
            
            at = Table(alert_data, colWidths=[65, 95, 95, 60, 135])
            at.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#991b1b')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTSIZE', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,0), 5),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1'))
            ]))
            elements.append(at)
        else:
            elements.append(Paragraph("No active threat alerts recorded during this session.", normal_style))

        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"XAI_Sentinel_Report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500

@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"response": "Please provide a message."}), 400

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are XAI-Sentinel AI, an expert cybersecurity assistant helping network administrators analyze traffic patterns and security threats."
                },
                {"role": "user", "content": user_message}
            ],
            model="openai/gpt-oss-120b",
        )
        return jsonify({"response": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"response": f"Error processing request: {str(e)}", "sniffing_active": sniffing_active}), 500

@app.route("/api/explain", methods=["POST"])
def get_explanation():
    if not packet_feature_buffer or len(packet_feature_buffer) < SEQUENCE_LENGTH:
        return jsonify({"error": "Insufficient buffer history for explanation."}), 400
    
    try:
        input_array = np.array(packet_feature_buffer[-SEQUENCE_LENGTH:], dtype=np.float32).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        input_tensor = torch.tensor(input_array).to(DEVICE)
        
        importances = explain_prediction(input_tensor, target_class=1)
        
        feature_labels = [
            "Packet Length", "Time-To-Live (TTL)", "Protocol Type", 
            "Source Port", "Destination Port", "TCP SYN Flag", "TCP ACK Flag"
        ]
        while len(feature_labels) < NUM_FEATURES:
            feature_labels.append(f"Feature Index #{len(feature_labels)}")

        combined = list(zip(feature_labels, importances))
        combined.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = combined[:10]

        labels = [item[0] for item in top_features]
        vals = [item[1] for item in top_features]

        return jsonify({
            "features": labels,
            "importances": vals
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/upload-logs", methods=["POST"])
def upload_logs():
    """Accepts uploaded packet log files and generates a Groq AI security overview."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    try:
        # Read uploaded log text content
        content = file.read().decode('utf-8', errors='ignore')
        
        # Truncate if the log file is excessively large to stay within context limits
        if len(content) > 15000:
            content = content[:15000] + "\n...[Log truncated due to length]"

        prompt = (
            f"Analyze these uploaded network packet logs or security data:\n\n"
            f"{content}\n\n"
            f"Provide a concise executive security summary, highlight any suspicious anomalies or attack patterns, "
            f"and give recommended mitigation steps for the network administrator."
        )

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are XAI-Sentinel AI, an expert cybersecurity incident responder."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
        )
        return jsonify({"summary": chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500
if __name__ == "__main__":
    # Start the continuous background sniffer daemon thread once
    sniff_thread = threading.Thread(target=start_sniffing, daemon=True)
    sniff_thread.start()
    app.run(debug=True, host="0.0.0.0", port=5001)