import pandas as pd
from transformers import pipeline

# โหลดโมเดล Zero-shot Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv("responses.csv")

# ทักษะที่ต้องการวิเคราะห์
labels = ["communication", "decision making", "leadership", "problem solving", "teamwork"]

# วิเคราะห์แต่ละคำตอบ
def analyze_response(response):
    result = classifier(response, candidate_labels=labels)
    return result  # คืน dictionary ที่มี labels และ scores

# วิเคราะห์แต่ละคอลัมน์
df['Communication_Analysis'] = df['โปรดอธิบายประสบการณ์ของคุณในการสื่อสารกับทีมในสถานการณ์วิกฤต'].apply(analyze_response)
df['DecisionMaking_Analysis'] = df['เล่าประสบการณ์การตัดสินใจที่คุณต้องทำในสถานการณ์ที่ยากลำบาก'].apply(analyze_response)
df['ProblemSolving_Analysis'] = df['เล่าประสบการณ์ที่คุณแก้ไขปัญหาในห้องนักบินหรือสนามบิน'].apply(analyze_response)
df['SkillToImprove_Analysis'] = df['ทักษะใดที่คุณคิดว่าคุณควรพัฒนาเพิ่มเติม?'].apply(analyze_response)

# ฟังก์ชันคำนวณเปอร์เซ็นต์ทักษะจากผลลัพธ์
def calculate_skill_percentage(analysis):
    skills = {label: 0 for label in labels}
    for label, score in zip(analysis['labels'], analysis['scores']):
        skills[label] += score
    total_score = sum(skills.values())
    return {skill: (score / total_score) * 100 for skill, score in skills.items()}

# ฟังก์ชันหาค่าเฉลี่ยทักษะจาก 4 คำถาม
def average_skills_from_row(row):
    skill_totals = {label: 0 for label in labels}
    count = 0
    for col in ['Communication_Analysis', 'DecisionMaking_Analysis', 'ProblemSolving_Analysis', 'SkillToImprove_Analysis']:
        percentage = calculate_skill_percentage(row[col])
        for skill, score in percentage.items():
            skill_totals[skill] += score
        count += 1
    return {skill: score / count for skill, score in skill_totals.items()}

# ฟังก์ชันแนะนำทักษะที่ควรพัฒนา
def suggest_development(avg_skills):
    sorted_skills = sorted(avg_skills.items(), key=lambda x: x[1])
    weakest_skills = [skill for skill, score in sorted_skills[:2]]
    return f"จากการวิเคราะห์ คุณควรพัฒนาเรื่อง {', '.join(weakest_skills)} เพิ่มเติม เนื่องจากเป็นทักษะที่คุณมีคะแนนเฉลี่ยน้อยที่สุดเมื่อเทียบกับทักษะอื่น ๆ"

# วิเคราะห์ผลลัพธ์และแสดงข้อมูล
for index, row in df.iterrows():
    print(f"\nคำตอบของผู้ทดสอบ {row['ชื่อผู้ทดสอบ']}:")

    for col in ['Communication_Analysis', 'DecisionMaking_Analysis', 'ProblemSolving_Analysis', 'SkillToImprove_Analysis']:
        percentage = calculate_skill_percentage(row[col])
        print(f"\n\tผลลัพธ์สำหรับ {col}:")
        for skill, score in sorted(percentage.items(), key=lambda x: x[1], reverse=True):
            print(f"\t\t{skill}: {score:.2f}%")

    # คำนวณค่าเฉลี่ย
    avg_skills = average_skills_from_row(row)
    print(f"\n\tค่าเฉลี่ยทักษะจากทุกคำถาม:")
    for skill, avg in sorted(avg_skills.items(), key=lambda x: x[1], reverse=True):
        print(f"\t\t{skill}: {avg:.2f}%")

    # สร้างคำแนะนำ
    suggestion = suggest_development(avg_skills)
    print(f"\n\t📌 คำแนะนำจาก AI: {suggestion}")
    print("-" * 50)

    # เพิ่มลง DataFrame ถ้าต้องการ export
    df.at[index, 'Suggestion'] = suggestion
