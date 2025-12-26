import pandas as pd

# 1. Đọc dữ liệu
df = pd.read_csv('health_lifestyle_dataset.csv')

# 2. Drop cột không dùng
df = df.drop(columns=['id'])

# 3. One-Hot Encoding cho gender
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# 4. Kiểm tra nhanh
print(df.head().to_string())

# 5. Lưu dữ liệu SAU ENCODE
df.to_csv('health_lifestyle_encoded.csv', index=False)


