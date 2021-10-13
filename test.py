"""
1. 若某单品在某天有多种促销价格或销售价格，则取最低价格为当天价格，当天各种价格的总销量为当天促销销量。
此近似方法，假定促销每天的销量均是按同一种价格卖出，且是最低价格；则一个单品一天只有一条记录。
如果一天有多种促销价格，最好是将每种价格对应的销量分开，作为多条记录保留，则一个单品一天就可能有多条记录。

2. 当一个单品在历史上某段时间属于一种sort，在另一段时间属于另一种sort，则会出现一个code对应两个sort；
若一个code在某段时间没有sort，则向上取midsort作为sort。更好的做法是，如果某个code在某段时间没有sort，在以后出现了sort，则统一为那个sort；
如果一直没有sort，取midsort作为sort时将位数补足，对于将sort置为连续型变量时更好，如果置为分类型变量，则没有影响。

3. 是否可通过库存和销量来判断是否缺货，这样对于在销但销量为空的样本，可知哪些是没卖出去而销量为0，哪些是缺货而销量为0.

4. 对于农历节日特别是春节，可将busdate转化为日期，这样每个春节的day of year，month of year可对应上。

5. 特别重要：通常情况下，数据集的各个特征中的数据，各自与因变量构成的二维散点图聚类越明显，对模型越友好，训练和预测的结果会更好；
所以通常情况下，数据集越简单，复杂度越低，规律越明显，越容易符合前述条件，从而使结果更好。

6. 因为MAPE的分子是残差绝对值，SMAPE的分子是两倍残差绝对值，所以如果预测值比真实值小，则SMAPE的分母小于两倍MAPE的分母，这时SMAPE会大于MAPE。
对于销量预测，预测值的下限是0，上限没有，其实残差是非对称的，残差分布通常偏向于正半轴，即右偏的，所以通常SMAPE会小于MAPE；
如果相反，要么说明预测普遍不足，这是模型相关的问题；或者是有少量极大的真实值，而预测值对这部分极大的值偏小较多，导致在整体上SMAPE被拉高，这是数据源相关的问题。
当特征工程和模型调优做得不够，即模式学习不到位，可能会出现预测普遍偏小或偏大的情况；当特征工程和模型调优做得足够充分合理，还出现SMAPE普遍大于MAPE，
则更可能是数据源相关的问题。

7. about model training: whole dataset:6000, seen_data:5400, unseen_data:600, train set:3779, test set:1621
1. The 'tuned_model' variable is trained on the whole train set of 3779 samples, not only the CV samples of 3779*(1 - 1 / Kfold).
2. The 'final_model' variable is trained on the complete dataset of 5400 samples, including the test/hold-out set,
using the same hyperparemeters with 'tuned_model', but different parameters with 'tuned_model',
because of the different training dataset. So final_model is retrained with the same hyperparemeters and different dataset.
3. At the very end, it's better to retrain the model using the very whole dataset of 6000 samples,
and using the same hyperparemeters with 'tuned_model', not to re-tune the hyperparameters in order to prevent overfitting.
Because we couldn't ensure that after re-tuning hyperparameters, what results the completely new model will emerge, like not overfitting.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from data_tools import get_processed_data, get_data_from_s3, process_data
from preprocess import his_mean
from sklearn.model_selection import train_test_split
from pycaret.regression import *
from borax.calendars.lunardate import LunarDate
import datetime
import regression_evaluation_def as re
from sklearn import metrics

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 20)
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

# cpnid, organ = '0042', '1038'
# bucket_name = 'wuerp-nx'
# train_ori = get_data_from_s3(bucket_name, 'train_1038', cpnid, organ)
# train_ori.to_csv('/Users/ZC/Documents/company/promotion/0042-1038-train.csv', index=False)
# print('\n', '促销样本量：%.d' % (len(train_ori)-train_ori.isna().sum().max()), '\n')
# print('促销样本占总样本的百分比：%.3f' % ((1-train_ori.isna().sum().max()/len(train_ori))*100), '\n')

train_ori = pd.read_csv('/Users/ZC/Documents/company/promotion/0042-1038-train.csv')
# train_ori = pd.read_csv('/Users/ZC/Documents/company/promotion/0042/1021/train.csv')
print(train_ori.isnull().sum(), '\n')
# train_ori = train_ori[train_ori['flag'] == '促销']
train_ori = train_ori[(train_ori['price']>0) & (train_ori['costprice']>0) & (train_ori['pro_day']>0) & (train_ori['distance_day']>=0) & (train_ori['promotion_type']>0)]  # 将逻辑异常的样本剔除
train_ori = train_ori[~((train_ori['price_level']>3) | (train_ori['price_level']<=0) | pd.isna(train_ori['price_level']))]
# 将负库存、库存为0销售为0、库存为0但有销售、库存为空的四种情况剔除
train_ori = train_ori[~((train_ori['stock_begin']<=0) | pd.isna(train_ori['stock_begin']))]  # & (train_ori['amount_ch']!=0)

min_date = train_ori['busdate'].min()
train_df = process_data(train_ori, min_date)
print(train_df.isnull().sum(), '\n')
dfgb = his_mean(train_df)
train_data = pd.merge(train_df, dfgb, how='left', on=['class', 'bigsort', 'midsort', 'sort', 'code'])
# train_data = select_his_data(train_df, dfgb)
print(train_data.isnull().sum(), '\n')

train_data_01 = train_data[(train_data['festival_num'] == 0) & (train_data['workday'] == 1)]
train_data_00 = train_data[(train_data['festival_num'] == 0) & (train_data['workday'] == 0)]

train_data_split = []
# python iterable object 的索引默认从0开始，而range通常表达索引，所以默认从0开始，且含左不含右，可保持索引总个数不变
for i in range(1, len(train_data['festival_num'].value_counts())):
    for j in range(1, len(train_data['mark_1'].value_counts())):
        train_data_split.append(train_data[(train_data['festival_num'] == i) & (train_data['mark_1'] == j)])
# 因为移动假日的festival_num编号为2,3,8,9,10，所以train_data_split中各移动假日的节前节中节后的编号为[(x-1)*3-1+1,(x-1)*3-1+3]；
# 对这些df生成农历日期相关变量，并使用农历日期而不使用公历日期，使这些df的移动日期变量变为固定日期变量。
lunar_num_ori = list([2, 3, 8, 9, 10])
lunar_num = []
for i in lunar_num_ori:
    for j in range(1, 3+1):
        lunar_num.append((i-1)*3-1+j)
for i in lunar_num:
    train_data_split[i]['LunarDate'] = train_data_split[i]['busdate'].apply(lambda x: LunarDate.from_solar(x.date()))
    train_data_split[i]['LunarYear'] = train_data_split[i]['LunarDate'].apply(lambda x: x.year)
    train_data_split[i]['LunarMonth'] = train_data_split[i]['LunarDate'].apply(lambda x: x.month)
    train_data_split[i]['LunarDay'] = train_data_split[i]['LunarDate'].apply(lambda x: x.day)

data, data_len, data_train, data_unseen = [], [], [], []
data.extend((train_data_01, train_data_00))
for i in range(len(train_data_split)):
    data.append(train_data_split[i])
for i in range(len(data)):
    data_train.append(data[i][(data[i]['busdate'] < data[i].iloc[:int(len(data[i]) * 0.7)]['busdate'].values[-1])])
    data_unseen.append(data[i][~(data[i]['busdate'] < data[i].iloc[:int(len(data[i]) * 0.7)]['busdate'].values[-1])])
    data_len.append(len(data[i]))
    # print(len(data_train[i]), len(data_unseen[i]))
print(pd.Series(data_len).describe())

# iden_feat = ['organ', 'code', 'busdate', 'pro_count']
cate_feat = ['promotion_type', 'weekend']  # , 'class', 'bigsort', 'midsort', 'sort'
# special_feat = ['class', 'bigsort', 'midsort', 'sort']
# numb_feat = ['distance_day', 'pro_day', 'costprice', 'price', 'price_level', 'weekday',  'month_diff',
#              'year', 'month', 'day', 'mean', 'std', 'stock_begin']  # 'mark_1', 'mark_2', 'festival_num',
# numb_feat_lunar = ['distance_day', 'pro_day', 'costprice', 'price', 'price_level', 'weekday', 'month_diff',
#                    'LunarYear', 'LunarMonth', 'LunarDay', 'mean', 'std', 'stock_begin']  # 'mark_1', 'mark_2', 'festival_num'
numb_feat = ['distance_day', 'pro_day', 'costprice', 'price', 'price_level', 'weekday',  'month_diff',
             'year', 'month', 'day', 'stock_begin', 'code_mean', 'code_std', 'sort_mean', 'sort_std', 'mid_mean', 'mid_std', 'big_mean', 'big_std', 'class_mean', 'class_std']  # 'mark_1', 'mark_2', 'festival_num',
numb_feat_lunar = ['distance_day', 'pro_day', 'costprice', 'price', 'price_level', 'weekday', 'month_diff',
                   'LunarYear', 'LunarMonth', 'LunarDay', 'stock_begin', 'code_mean', 'code_std', 'sort_mean', 'sort_std', 'mid_mean', 'mid_std', 'big_mean', 'big_std', 'class_mean', 'class_std']  # 'mark_1', 'mark_2', 'festival_num'
feat = cate_feat + numb_feat
feat_lunar = cate_feat + numb_feat_lunar
label = ['y']

SMAPE, MAPE, RMSE, R2 = [], [], [], []
result = pd.DataFrame(columns=data_unseen[0].columns)

# for i in range(len(data_train)):
for i in [0,1]:
    if i < 2:  # 非节日
        # high_cardinality_features=special_feat, high_cardinality_method='clustering',
        # ,transform_target=True, transform_target_method='yeo-johnson', normalize=True, normalize_method='robust'
        # 设置session_id为1，使管道和算法中所有用到随机数的地方random_state=1，不然历史预测值将不可复现，会对后续优化造成干扰
        # Because Target transformation is applied separately from feature transformations, so it only applies
        # power transformation to dependent variable, not to independent variables, so it's easy to mismatch with
        # the characteristics of independent variables.
        # transformation only transforms independent variables, doesn't transform dependent variable,
        # it is different with the normal yeo-johnson process.
        # ignore_features=iden_feat,
        reg = setup(data=data_train[i][feat+label], target='y',
                    silent=True, categorical_features=cate_feat, numeric_features=numb_feat,
                    data_split_shuffle=False, fold_strategy='timeseries', fold=2,
                    remove_multicollinearity=True, multicollinearity_threshold=0.95, session_id=1,
                    transformation=True, normalize=True, normalize_method='robust',
                    feature_ratio=True, interaction_threshold=0.01,
                    remove_outliers=False, outliers_threshold=10/len(data_train[i]))
    elif i in np.array(lunar_num)+2:  # 将两个较大的非节假日的两个数据集放在前面
        if i in [5, 6, 7]:  # 春节
            reg = setup(data=data_train[i][feat_lunar+label+['workday', 'mark_2']], target='y', silent=True,
                        categorical_features=cate_feat+['workday'],
                        numeric_features=numb_feat_lunar+['mark_2'],
                        data_split_shuffle=False, fold_strategy='timeseries', fold=2,
                        remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                        transformation=True, feature_ratio=True,  interaction_threshold=0.005,
                        normalize=True, normalize_method='robust',
                        remove_outliers=False, outliers_threshold=10/len(data_train[i]))
        else:  # 其他农历节日
            reg = setup(data=data_train[i][feat_lunar+label+['workday', 'mark_2']], target='y', silent=True,
                        categorical_features=cate_feat+['workday'],
                        numeric_features=numb_feat_lunar+['mark_2'],
                        data_split_shuffle=False,
                        fold_strategy='timeseries', fold=2,
                        remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                        transformation=True, feature_ratio=False, normalize=True, normalize_method='robust',
                        remove_outliers=False, outliers_threshold=3/len(data_train[i]))
    else:  # 其他公历节日
        reg = setup(data=data_train[i][feat+label+['workday', 'mark_2']], target='y', silent=True,
                    categorical_features=cate_feat+['workday'],
                    numeric_features=numb_feat+['mark_2'],
                    data_split_shuffle=False,
                    fold_strategy='timeseries', fold=2,
                    remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                    transformation=True, feature_ratio=False, normalize=True, normalize_method='robust',
                    remove_outliers=False, outliers_threshold=3/len(data_train[i]))

    # include=list(models().index.values), exclude=list(['kr', 'mlp', 'tr']), Kernel Ridge and MLP is too slow
    # if cross_validation=False, then training on 70% train_set and evaluating on 30% test_set,
    # if cross_validation=True, then training and evaluating both on 70% train_set.
    best3 = compare_models(n_select=3, sort='RMSE', cross_validation=False)
    # best.append(best3)
    ensembled = []
    for j in range(len(best3[:])):  # 外层循环变量是i，内层的其他循环变量不能同名
        try:
            ensembled.append(ensemble_model(best3[j], method='Boosting', n_estimators=5, choose_better=True, optimize='RMSE'))
        except Exception as e:
            print('\n', f'模型{best3[j]}对提升流程做梯度下降失败，改用超参调优，报错信息：{e}', '\n')
            try:
                ensembled.append(tune_model(best3[j], n_iter=50, optimize='RMSE', choose_better=True, early_stopping=True))
            except Exception as e:
                print('\n', f'模型{best3[j]}超参调优失败，返回超参数为初始值的模型，报错信息：{e}', '\n')
                # , search_library='scikit-optimize', search_algorithm='bayesian'
                # 贝叶斯推理中，下一轮训练的超参数取决于在上一轮设置的超参数下，CV后模型的指标情况，所以无法多核并行训练，除非对每个核单独进行贝叶斯推理，
                # 最后比较每个核上的推理结果。但这样会降低每个核上做推理的次数，变为1/n，n为核心或超线程数，与贝叶斯推理的思想有所违背。
                ensembled.append(best3[j])
    voting = blend_models(ensembled, choose_better=True, optimize='RMSE')
    final_model = finalize_model(voting)  # 按照voting的模型结构和超参数，用data_train的所有数据，重新训练voting的参数
    # stacked = stack_models(ensembled[1:], meta_model=ensembled[0], choose_better=False, optimize='RMSE', restack=True)
    # ensembled = ensemble_model(best5[0], method='Boosting', n_estimators=5)
    # stacked = stack_models(best5[1:], meta_model=ensembled, choose_better=False, optimize='RMSE', restack=True)

#     lgbm = create_model('lightgbm', cross_validation=False)
    # catb = create_model('catboost', cross_validation=False)
    # gdb = create_model('gbr', cross_validation=False)

#     if i == [0, 1, 5, 6]:  # 对样本数较多的模型进行超参调优更有效
#         tm = tune_model(best5[0], n_iter=50, optimize='RMSE', return_tuner=True, choose_better=True, early_stopping=True)
#         pred = predict_model(tm[0])
#     else:
#         pred = predict_model(best5[0])
#     tm = tune_model(best5[0], n_iter=50, optimize='RMSE', return_tuner=True, choose_better=True, early_stopping=True)
    pred = predict_model(final_model, data=data_unseen[i])
    pred.loc[pred['Label'] < 0, 'Label'] = 0.1
    result = result.append(pred)
    SMAPE.append(re.smape(y_true=pred['y'], y_pred=pred['Label']))
    MAPE.append(re.mape(y_true=pred['y'], y_pred=pred['Label']))
    RMSE.append(re.rmse(y_true=pred['y'], y_pred=pred['Label']))
    R2.append(round(metrics.r2_score(y_true=pred['y'], y_pred=pred['Label']), 4))

print('\n', SMAPE, '\n', MAPE, '\n', RMSE, '\n', R2, '\n')
print(round(np.average(SMAPE),4), round(np.average(MAPE),4), round(np.average(RMSE),4), round(np.average(R2),4))
result.sort_values(by='busdate', inplace=True)
result['yhat'] = np.exp(result['Label'])-1
smape_final = re.smape(y_true=result['amount'], y_pred=result['yhat'])
mape_final = re.mape(y_true=result['amount'], y_pred=result['yhat'])
rmse_final = re.rmse(y_true=result['amount'], y_pred=result['yhat'])
r2_final = round(metrics.r2_score(y_true=result['amount'], y_pred=result['yhat']), 4)
print('\n', smape_final, '\n', mape_final, '\n', rmse_final, '\n', r2_final, '\n')
result.to_csv('/Users/ZC/Documents/company/promotion/result/0042-1038-1013-2.csv')


'''
transformation=True, transform_target=False, log: 
1.4119704545454546 0.42096818181818174 0.4297977272727273 0.5958727272727272
[1.6445, 1.593, 1.3459, 1.301, 0.9428, 1.198, 1.4028, 1.4514, 1.4807, 1.4468, 1.2676, 1.4399, 1.5326, 1.7019, 1.4544, 1.4642, 1.5832, 1.6662, 1.6944, 1.7598, 1.6113, 1.2532, 1.0797, 1.3912, 1.5048, 1.7263, 1.6945, 1.5952, 1.545, 1.1629, 1.1419, 1.2479, 1.2503, 1.1461, 1.3057, 1.4428, 1.4102, 1.5382, 1.4049, 0.9041, 1.4686, 1.4921, 1.2583, 1.1804] 
[0.5217, 0.4827, 0.3921, 0.4046, 0.2079, 0.3967, 0.5137, 0.6618, 0.3893, 0.3906, 0.2861, 0.4238, 0.4053, 0.4505, 0.4659, 0.4315, 0.5022, 0.4365, 0.53, 0.5149, 0.3247, 0.3658, 0.3404, 0.4779, 0.4518, 0.6177, 0.5575, 0.3615, 0.5634, 0.3589, 0.3625, 0.3792, 0.4219, 0.3833, 0.223, 0.4311, 0.355, 0.369, 0.4399, 0.2747, 0.5617, 0.4249, 0.3253, 0.3437] 
[0.4519, 0.4366, 0.4758, 0.4447, 0.2193, 0.5442, 0.5195, 0.5434, 0.3895, 0.4029, 0.4133, 0.5561, 0.4438, 0.4063, 0.5059, 0.4513, 0.4134, 0.3423, 0.3524, 0.2684, 0.3463, 0.3385, 0.2823, 0.4314, 0.4222, 0.4368, 0.3873, 0.419, 0.4308, 0.4604, 0.5043, 0.4669, 0.4885, 0.4795, 0.2958, 0.5079, 0.4956, 0.4116, 0.4854, 0.4432, 0.5675, 0.5163, 0.3533, 0.3593] 
[0.5448, 0.5868, 0.6787, 0.7472, 0.9277, 0.7204, 0.5871, 0.4823, 0.6703, 0.5233, 0.6038, 0.4177, 0.6345, 0.1944, 0.5328, 0.6239, 0.307, 0.5403, 0.4383, 0.679, 0.7451, 0.4248, 0.6236, 0.6504, 0.6142, 0.4481, 0.5129, 0.3557, 0.5645, 0.4987, 0.6914, 0.6081, 0.6975, 0.7245, 0.7376, 0.6069, 0.5607, 0.4822, 0.665, 0.8544, 0.5346, 0.6149, 0.8054, 0.7569] 

transformation=True, boosting, voting, 1038:
1.197540909090909 0.34184545454545456 0.4733272727272728 0.7624954545454545
[1.3712, 1.2927, 1.1333, 1.0091, 1.0846, 0.9692, 1.2232, 1.3124, 1.3802, 0.9399, 0.9783, 1.1499, 1.1192, 1.2965, 1.1686, 1.0456, 1.3941, 1.4216, 1.4467, 1.5452, 1.3201, 0.8502, 0.8792, 1.2487, 1.1573, 1.0666, 1.4443, 1.0218, 1.365, 1.496, 1.3228, 1.3849, 1.5069, 0.9858, 1.0151, 1.2421, 1.239, 1.3409, 1.0405, 0.9719, 1.0062, 1.2424, 1.167, 1.0956] 
[0.3752, 0.3574, 0.3504, 0.2704, 0.2836, 0.2816, 0.3972, 0.3998, 0.4425, 0.3362, 0.2868, 0.3209, 0.3115, 0.3749, 0.3299, 0.3202, 0.3884, 0.3701, 0.4225, 0.4137, 0.2762, 0.34, 0.2614, 0.3482, 0.309, 0.3504, 0.34, 0.3209, 0.3548, 0.3847, 0.3665, 0.4164, 0.3542, 0.3194, 0.2239, 0.3412, 0.3375, 0.416, 0.3605, 0.33, 0.3196, 0.3445, 0.2779, 0.3148] 
[0.4864, 0.5176, 0.5343, 0.5292, 0.513, 0.5149, 0.5393, 0.4994, 0.4364, 0.4439, 0.3776, 0.4752, 0.4567, 0.4627, 0.5891, 0.4937, 0.5923, 0.444, 0.4497, 0.4028, 0.4042, 0.3832, 0.3826, 0.4982, 0.5091, 0.5027, 0.4074, 0.4304, 0.427, 0.4341, 0.4942, 0.5085, 0.3861, 0.5243, 0.4109, 0.4688, 0.517, 0.3786, 0.5728, 0.4951, 0.5104, 0.5237, 0.4511, 0.4478] 
[0.7448, 0.7491, 0.7555, 0.8088, 0.7625, 0.8546, 0.7396, 0.7215, 0.7509, 0.7288, 0.814, 0.7577, 0.7813, 0.7038, 0.7102, 0.8189, 0.5418, 0.7616, 0.7041, 0.6884, 0.8586, 0.8022, 0.7911, 0.7795, 0.8274, 0.7858, 0.7931, 0.7437, 0.7545, 0.7705, 0.7896, 0.6683, 0.7756, 0.7875, 0.7815, 0.8164, 0.7831, 0.6795, 0.7349, 0.8012, 0.7365, 0.7347, 0.8252, 0.8315] 
'''
