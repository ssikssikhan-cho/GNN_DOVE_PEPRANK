import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 예측 결과 파일 경로 설정
predictions_path = '/home2/escho/GNN_DOVE_PEPRANK/inf_results/predictions_sorted.txt'  # 여기에 예측 결과 파일 경로를 입력

# y_true와 y_scores 리스트 생성
y_true = []  # 실제 레이블 (1 또는 0)
y_scores = []  # 예측 점수

# 파일 읽기 및 y_true, y_scores 채우기
with open(predictions_path, 'r') as file:
    lines = file.readlines()[1:]  # 첫 번째 라인은 헤더이므로 제외
    for line in lines:
        if line.strip():
            name, score = line.split('\t')
            y_scores.append(float(score))
            y_true.append(1 if 'crt' in name else 0)  # 'crt'가 포함되면 양성으로 간주

# ROC 곡선 및 AUC 계산
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# ROC 그래프 시각화
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('/home2/escho/GNN_DOVE_PEPRANK/roc_curve.png')
plt.show()
