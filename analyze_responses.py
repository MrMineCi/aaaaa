import pandas as pd
from transformers import pipeline

# สร้างโมเดล Zero-shot Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# อ่านข้อมูลจาก CSV
df = pd.read_csv("responses.csv")

# ทักษะที่ต้องการตรวจจับ
labels = ["communication", "decision making", "leadership", "problem solving", "teamwork"]

# ฟังก์ชันเพื่อทำการวิเคราะห์คำตอบแต่ละบรรทัด
def analyze_response(response):
    result = classifier(response, candidate_labels=labels)
    
    # ตรวจสอบผลลัพธ์ที่ได้จาก classifier
    print(result)  # พิมพ์ผลลัพธ์เพื่อตรวจสอบโครงสร้าง
    
    # หากคืนค่ามาเป็น dictionary เราจะใช้ 'labels' และ 'scores'
    if isinstance(result, list) and len(result) > 0:
        return result[0]  # คืนค่าผลลัพธ์ตัวแรกจาก list ที่คืนมาจาก classifier
    return result

# วิเคราะห์คำตอบจากคอลัมน์ที่มีชื่อ
df['Communication_Analysis'] = df['โปรดอธิบายประสบการณ์ของคุณในการสื่อสารกับทีมในสถานการณ์วิกฤต'].apply(analyze_response)
df['DecisionMaking_Analysis'] = df['เล่าประสบการณ์การตัดสินใจที่คุณต้องทำในสถานการณ์ที่ยากลำบาก'].apply(analyze_response)
df['ProblemSolving_Analysis'] = df['เล่าประสบการณ์ที่คุณแก้ไขปัญหาในห้องนักบินหรือสนามบิน'].apply(analyze_response)
df['SkillToImprove_Analysis'] = df['ทักษะใดที่คุณคิดว่าคุณควรพัฒนาเพิ่มเติม?'].apply(analyze_response)

# การคำนวณเปอร์เซ็นต์ทักษะ
def calculate_skill_percentage(analysis):
    skills = {label: 0 for label in labels}
    
    # ใช้ 'labels' และ 'scores' จากผลลัพธ์
    for label, score in zip(analysis['labels'], analysis['scores']):
        skills[label] += score
    
    total_score = sum(skills.values())
    skills_percentage = {skill: (score / total_score) * 100 for skill, score in skills.items()}
    return skills_percentage

# ประมวลผลผลลัพธ์ทั้งหมด และแสดงผลเปอร์เซ็นต์ในรูปแบบที่อ่านง่าย
for index, row in df.iterrows():
    print(f"\nคำตอบของผู้ทดสอบ {row['ชื่อผู้ทดสอบ']}:")

    # แสดงผลลัพธ์ในรูปแบบที่เข้าใจง่าย
    for col in ['Communication_Analysis', 'DecisionMaking_Analysis', 'ProblemSolving_Analysis', 'SkillToImprove_Analysis']:
        analysis = row[col]
        percentage = calculate_skill_percentage(analysis)
        
        # จัดเรียงและแสดงผลทักษะตามเปอร์เซ็นต์ที่สูงสุด
        print(f"\n\tผลลัพธ์สำหรับ {col}:")
        sorted_skills = sorted(percentage.items(), key=lambda x: x[1], reverse=True)  # เรียงตามเปอร์เซ็นต์
        for skill, score in sorted_skills:
            print(f"\t\t{skill}: {score:.2f}%")

    print("-" * 50)
