from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import rand
from pyspark.sql.functions import monotonically_increasing_id

#Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("DiseaseRiskLogisticRegression") \
    .getOrCreate()

#Đọc dữ liệu
df = spark.read.csv(
    "health_lifestyle_encoded.csv",
    header=True,
    inferSchema=True)

#Tạo Vector Feature
feature_cols = [c for c in df.columns if c != 'disease_risk']

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw")

#Chuẩn hoá z-score
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True)

#Cân bằng lớp ở biến mục tiêu
count_0 = df.filter(col("disease_risk") == 0).count()
count_1 = df.filter(col("disease_risk") == 1).count()
total = count_0 + count_1

weight_0 = total / (2 * count_0)
weight_1 = total / (2 * count_1)

df = df.withColumn(
    "class_weight",
    when(col("disease_risk") == 0, weight_0)
    .otherwise(weight_1))

#Mô hình Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="disease_risk",
    weightCol="class_weight",
    maxIter=100)

#Pipeline
pipeline = Pipeline(stages=[
    assembler,
    scaler,
    lr])

#Train-Test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

#Huấn luyện mô hình
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

##Đánh giá mô hình trên train/test
# Confusion Matrix
predictions.groupBy("disease_risk", "prediction").count().show()

# Evaluators
acc_eval = MulticlassClassificationEvaluator(
    labelCol="disease_risk",
    predictionCol="prediction",
    metricName="accuracy")

prec_eval = MulticlassClassificationEvaluator(
    labelCol="disease_risk",
    predictionCol="prediction",
    metricName="weightedPrecision")

rec_eval = MulticlassClassificationEvaluator(
    labelCol="disease_risk",
    predictionCol="prediction",
    metricName="weightedRecall")

f1_eval = MulticlassClassificationEvaluator(
    labelCol="disease_risk",
    predictionCol="prediction",
    metricName="f1")

roc_eval = BinaryClassificationEvaluator(
    labelCol="disease_risk",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC")

print("Accuracy :", acc_eval.evaluate(predictions))
print("Precision:", prec_eval.evaluate(predictions))
print("Recall   :", rec_eval.evaluate(predictions))
print("F1-score :", f1_eval.evaluate(predictions))
print("ROC-AUC  :", roc_eval.evaluate(predictions))


#Cross validation(10 fold)
k = 10
df_cv = df.withColumn("fold_id", monotonically_increasing_id() % k)
metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []}
best_model = None
best_roc = -1
best_fold = -1
print("\n========== KẾT QUẢ TỪNG FOLD ==========")
for i in range(k):
    print(f"\n--- Fold {i+1} ---")
    test_fold = df_cv.filter(col("fold_id") == i)
    train_fold = df_cv.filter(col("fold_id") != i)
    model = pipeline.fit(train_fold)
    preds = model.transform(test_fold)
    acc = acc_eval.evaluate(preds)
    prec = prec_eval.evaluate(preds)
    rec = rec_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)
    roc = roc_eval.evaluate(preds)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    metrics["accuracy"].append(acc)
    metrics["precision"].append(prec)
    metrics["recall"].append(rec)
    metrics["f1"].append(f1)
    metrics["roc_auc"].append(roc)
    #Tìm best_model
    if roc > best_roc:
        best_roc = roc
        best_model = model
        best_fold = i + 1

# ================== MEAN METRICS ==================
# print("\n========== TRUNG BÌNH 10-FOLD ==========")
# for m in metrics:
#     mean_val = sum(metrics[m]) / k
#     print(f"{m.upper():10}: {mean_val:.4f}")

#Lưu model
# ================= SAVE FINAL MODEL =================
final_model = pipeline.fit(df)  # fit trên toàn bộ dữ liệu

model_path = "saved_models/logistic_disease_risk_model"
final_model.write().overwrite().save(model_path)

print("Model saved successfully at:", model_path)