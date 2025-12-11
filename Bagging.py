import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 1. Tải và chuẩn bị dữ liệu
df = pd.read_csv('Student_performance_data _.csv')

X = df.drop(['GradeClass', 'StudentID', 'GPA'], axis=1)
y = df['GradeClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Huấn luyện mô hình Bagging
base_cls = DecisionTreeClassifier(random_state=42)
model = BaggingClassifier(estimator=base_cls, n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 3. Dự đoán
y_pred = model.predict(X_test)

# 4. Tính toán các chỉ số chi tiết

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# 5. In kết quả ra màn hình
print("--------------------------------------------------")
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BAGGING")
print("--------------------------------------------------")
print(f"1. Accuracy (Độ chính xác tổng thể):  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"2. Precision (Độ chuẩn - Weighted):   {precision:.4f}")
print(f"3. Recall (Độ nhạy - Weighted):       {recall:.4f}")
print(f"4. F1-Score (Trung bình điều hòa):    {f1:.4f}")
print("--------------------------------------------------\n")

print("BÁO CÁO CHI TIẾT TỪNG LỚP (Classification Report):")
print(classification_report(y_test, y_pred))

# 6. Vẽ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix\n(Hàng: Thực tế - Cột: Dự đoán)')
plt.ylabel('Thực tế (Actual GradeClass)')
plt.xlabel('Dự đoán (Predicted GradeClass)')

plt.show()
