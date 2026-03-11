#%%
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import shap
import pandas as pd
import matplotlib.pyplot as plt
#%%
# 1. 모델 및 데이터 로드
model = load_model("outputs_BCDSpBN/model_1DCNN.h5")

X_train = np.load("dataset_BCDSpBN/X_train.npy")
y_train = np.load("dataset_BCDSpBN/y_train.npy")
X_test = np.load("dataset_BCDSpBN/X_test.npy")
y_test = np.load("dataset_BCDSpBN/y_test.npy")

print(X_train.shape, X_test.shape, len(y_train), len(y_test))


#%%
# 2. 데이터 전처리
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 차원 확장 (CNN 입력 데이터로 적합하게 변경)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 클래스 이름 정의
classes = ['Shigella sonnei', 'Shigella flexneri', 'Shigella boydii', 
           'Shigella dysenteriae', 'EIEC', 'EPEC', 'ETEC', 'EAEC', 'STEC']
y_train = [classes[i] for i in y_train]
y_test = [classes[i] for i in y_test]

# 데이터 크기 확인
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train labels:", len(y_train), "Test labels:", len(y_test))

#%%
# 3. SHAP 분석 준비
# 테스트 데이터 샘플링 (계산 비용 감소를 위해)
X_sample = X_test[:10]  # 필요 시 크기 조정
print("Sample shape for SHAP:", X_sample.shape)

# SHAP DeepExplainer 생성
explainer = shap.DeepExplainer(model, X_train[:100])  # 일부 학습 데이터를 Background로 사용

# SHAP 값 계산
shap_values = explainer.shap_values(X_sample)

# shape 확인
print("Original shapes:")
print("X_sample shape:", X_sample.shape)  # (10, 331, 1)
print("shap_values[0] shape:", shap_values[0].shape)  # (331, 1, 9)


# 데이터 재구성 - 행 수를 맞추기 위해 수정
X_sample_2d = X_sample.squeeze(axis=2)  # (10, 331)
# shap_values를 X_sample과 같은 샘플 수로 맞추기
shap_values_2d = np.zeros((10, 331))  # X_sample과 같은 크기로 초기화

for i in range(10):  # 각 샘플에 대해
    shap_values_2d[i] = shap_values[0][:, 0, 0]  # 첫 번째 클래스의 값만 사용

# 변환된 shape 확인
print("\nReshaped shapes:")
print("X_sample_2d shape:", X_sample_2d.shape)  # should be (10, 331)
print("shap_values_2d shape:", shap_values_2d.shape)  # should be (10, 331)

# shape가 일치하는지 확인
assert X_sample_2d.shape == shap_values_2d.shape, "Shapes must match!"

#%%
# SHAP 시각화
plt.title(f"SHAP Summary Plot for Class: {classes[0]}")
shap.summary_plot(
    shap_values_2d,
    X_sample_2d,
    feature_names=["Feature " + str(i) for i in range(X_sample_2d.shape[1])],
    max_display=50,
    show=True
)

#%%
# SHAP 값을 CSV로 저장

# 피처 이름 생성
feature_names = [f"Feature_{i}" for i in range(X_sample_2d.shape[1])]

# SHAP 값의 평균 절대값 계산 (피처 중요도)
feature_importance = np.abs(shap_values_2d).mean(axis=0)

# 데이터프레임 생성
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': feature_importance
})

# 중요도 순으로 정렬
shap_df = shap_df.sort_values('SHAP_Importance', ascending=False)

# CSV 파일로 저장
shap_df.to_csv(f'shap_feature_importance_{classes[0]}.csv', index=False)
print("SHAP 값이 'shap_feature_importance.csv'에 저장되었습니다.")
#%%


#%%
# 통합된 feature importance 계산

# 각 클래스별로 평균 절대값 계산
shap_values_mean = np.mean([np.abs(sv).squeeze(axis=1) for sv in shap_values], axis=0)

# 피처별 중요도 계산 (클래스 축 평균)
feature_importance = shap_values_mean.mean(axis=1)  # 클래스 축(9)을 평균화

# 피처 이름 생성
feature_names = [f"Feature_{i}" for i in range(shap_values_mean.shape[0])]

# 데이터프레임 생성
shap_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': feature_importance
})

# 중요도 순으로 정렬
shap_df = shap_df.sort_values('SHAP_Importance', ascending=False)

# CSV 파일로 저장
shap_df.to_csv('shap_feature_importance_all_classes.csv', index=False)
print("모든 클래스의 통합된 SHAP 값이 'shap_feature_importance_all_classes.csv'에 저장되었습니다.")

