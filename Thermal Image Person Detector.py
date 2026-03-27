import cv2
from ultralytics import YOLO
import os

# ===============================
# 1️⃣ إعداد المجلدات
# ===============================

INPUT_FOLDER = "my_image"      # ضع صورك هنا
OUTPUT_FOLDER = "results"            # سيتم حفظ النتائج هنا

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===============================
# 2️⃣ تحميل نموذج YOLO
# ===============================

model = YOLO("yolov8s.pt")   # أول مرة سيقوم بتحميله تلقائياً

# نكتشف الأشخاص فقط (class 0 في COCO)
PERSON_CLASS = 0
CONFIDENCE_THRESHOLD = 0.50

# ===============================
# 3️⃣ قراءة الصور من المجلد
# ===============================

image_files = [f for f in os.listdir(INPUT_FOLDER)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

if not image_files:
    print("❌ لا توجد صور داخل المجلد.")
    exit()

print("🔍 بدء كشف الأشخاص في الصور الحرارية...")
print("-" * 50)

# ===============================
# 4️⃣ معالجة كل صورة
# ===============================

for filename in image_files:

    image_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ تعذر قراءة الصورة: {filename}")
        continue

    # إذا كانت الصورة حرارية رمادية نحولها إلى 3 قنوات
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # تشغيل النموذج
    results = model.predict(
        source=image,
        classes=[PERSON_CLASS],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # رسم الصناديق
    output_image = results[0].plot()

    # حفظ الصورة الناتجة
    save_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(save_path, output_image)

    # حساب عدد الأشخاص
    boxes = results[0].boxes
    person_count = len(boxes)

    print(f"[{filename}] → عدد الأشخاص المكتشفين: {person_count}")

print("-" * 50)
print("✅ انتهى التنفيذ بنجاح!")
print(f"📁 النتائج محفوظة داخل مجلد: {OUTPUT_FOLDER}")