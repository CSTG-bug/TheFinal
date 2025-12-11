import joblib
import pandas as pd

# 1. 加载已经训练好的模型
model_path = r"D:\MLDesignAl\TheFinal\XGBoost\ElementTreatmentEl-UTS\output-exceptEL\XGB_best_model.joblib"  # 把这里改成你的模型文件路径
model = joblib.load(model_path)

# 2. 读取需要预测的新数据
data_path = r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTS\DataProcess\ScaleX.csv"  # 或者 .csv
# 如果是 excel
# df = pd.read_excel(data_path)
# 如果是 csv 就用：
df = pd.read_csv(data_path)

# 3. 选取和训练时完全一致的特征列
feature_cols = [

    'Si','Fe','Cu','Mn','Mg','Cr','Zn','V','Ti','Zr','Li','Ni','Be','Sc','Ag','Bi','Pb','Al',
    'SS Temp','Ageing Temp','Ageing Time'

]
X_new = df[feature_cols]

# 4. 用模型进行预测
y_pred = model.predict(X_new)

# 5. 把预测结果加回原表，并保存
df['UTS-pred'] = y_pred  # 列名你可以改成你要的，比如 'UTS_pred', 'YS_pred'

output_path = r"D:\MLDesignAl\TheFinal\Data\ElementTreatment-UTS\DataProcess\RawX.csv"
df.to_csv(output_path, index=False)

print("预测完成！结果已保存到：", output_path)
