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

8. 在ensemble_model中若choose_better=True，则会根据cross validation在optimize指标上的结果选择返回是做method=boosting前或后的模型；
由于第一，optimize上只能选择单一指标，该指标更好的模型不一定是综合精度更好的模型；第二，原始模型使用初始超参数，对于不同的数据集、特征变量、特征工程方法，
使用相同的初数超参数均会得到不同的预测结果，而初始超参数几乎可以认为是一定可以被优化的，由于在pycaret中使用超参调优暂时对预测结果没有多少改变，
则采用boosting的集成方法作用于单一模型，让其代替超参调优的作用，但n_estimators可选小些，防止对残差的极度逼近而过拟合。
所以，在ensemble_model中选择choose_better=False，n_estimators=3，不用tune_model。

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


# 读取数据并做一些预处理和初步的特征工程
train_ori = pd.read_csv('/Users/ZC/Documents/company/promotion/train.csv')
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

# 对因变量做变换
a = 2 ** (1 / 2)
if train_data['amount'].min() >= 0:
    train_data['y'] = train_data['amount']
    train_data['y_bc'] = train_data['amount'] + 1  # 当λ被极大似然估计优选为0时，box-cox是做对数变换，所以+1能使对数的因变量非负；由于amount≥0，则此时对y_bc和自变量做box-cox变换，与对amount和相同自变量做yeo-johnson变换等价。
    train_data['y_log1p'] = np.log1p(train_data['amount'])
    train_data['y_log2'] = np.log2(train_data['amount'] + 1)
    train_data['y_log'] = np.log1p(train_data['amount']) / np.log(a)
    train_data['y_norm'] = train_data['amount'] / np.mean(train_data['amount'])


# 整理数据集
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
    data_train.append(data[i][(data[i]['busdate'] < data[i].iloc[:int(len(data[i]) * 0.9)]['busdate'].values[-1])])
    data_unseen.append(data[i][~(data[i]['busdate'] < data[i].iloc[:int(len(data[i]) * 0.9)]['busdate'].values[-1])])
    data_len.append(len(data[i]))
    # print(len(data_train[i]), len(data_unseen[i]))
print(pd.Series(data_len).describe())

iden_feat = ['organ', 'code', 'busdate', 'class', 'bigsort', 'midsort', 'sort']
cate_feat = ['weekend', 'promotion_type']
numb_feat = ['weekday', 'year', 'month', 'day', 'distance_day', 'pro_day', 'costprice', 'price', 'price_level',
             'stock_begin', 'mean_code', 'std_code', 'mean_sort', 'std_sort', 'mean_mid', 'std_mid', 'mean_big', 'std_big', 'mean_class', 'std_class']  # 'mark_1', 'mark_2', 'festival_num', 'month_diff',
numb_feat_lunar = ['weekday', 'LunarYear', 'LunarMonth', 'LunarDay', 'distance_day', 'pro_day', 'costprice', 'price', 'price_level',
                   'stock_begin', 'mean_code', 'std_code', 'mean_sort', 'std_sort', 'mean_mid', 'std_mid', 'mean_big', 'std_big', 'mean_class', 'std_class']  # 'mark_1', 'mark_2', 'festival_num', 'month_diff',
feat = cate_feat + numb_feat
feat_lunar = cate_feat + numb_feat_lunar
flexible = ['festival_num', 'mark_1', 'mark_2', 'workday']
label = ['y_norm']

SMAPE, MAPE, RMSE, R2 = [], [], [], []
result = pd.DataFrame(columns=data_unseen[0].columns)
best_final, ensembled_final, voting_final = [], [], []
tuning, no_tune = [], []

# for i in range(len(data_train)):
for i in [5]:
    if i < 2:  # 非节日
        # 设置session_id为1，使管道和算法中所有用到随机数的地方random_state=1，不然历史预测值将不可复现，会对后续优化造成干扰
        # Because Target transformation is applied separately from feature transformations, so it only applies
        # power transformation to dependent variable, not to independent variables, so it's easy to mismatch with
        # the characteristics of independent variables.
        # transformation only transforms independent variables, doesn't transform dependent variable,
        # it is different with the normal yeo-johnson process.
        reg = setup(data=data_train[i][feat+label], target=label[0],
                    silent=True, categorical_features=cate_feat, numeric_features=numb_feat,
                    data_split_shuffle=False, fold_strategy='timeseries', fold=2,
                    remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                    transformation=True,  transformation_method='yeo-johnson', normalize=True, normalize_method='robust',
                    feature_ratio=True, interaction_threshold=0.01,
                    remove_outliers=False, outliers_threshold=10/len(data_train[i]),
                    transform_target=False, transform_target_method='yeo-johnson')
    elif i in np.array(lunar_num)+2:  # 将两个较大的非节假日的两个数据集放在前面
        if i in [5, 6, 7]:  # 春节
            reg = setup(data=data_train[i][feat_lunar+label+['workday', 'mark_2']], target=label[0], silent=True,
                        categorical_features=cate_feat+['workday'],
                        numeric_features=numb_feat_lunar+['mark_2'],
                        data_split_shuffle=False, fold_strategy='timeseries', fold=2,
                        remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                        transformation=True,  transformation_method='yeo-johnson', feature_ratio=True,  interaction_threshold=0.005,
                        normalize=True, normalize_method='robust',
                        remove_outliers=False, outliers_threshold=10/len(data_train[i]),
                        transform_target=False, transform_target_method='box-cox')
        else:  # 其他农历节日
            reg = setup(data=data_train[i][feat_lunar+label+['workday', 'mark_2']], target=label[0], silent=True,
                        categorical_features=cate_feat+['workday'],
                        numeric_features=numb_feat_lunar+['mark_2'],
                        data_split_shuffle=False,
                        fold_strategy='timeseries', fold=2,
                        remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                        transformation=True,  transformation_method='yeo-johnson', feature_ratio=False, normalize=True, normalize_method='robust',
                        remove_outliers=False, outliers_threshold=3/len(data_train[i]),
                        transform_target=False, transform_target_method='box-cox')
    else:  # 其他公历节日
        reg = setup(data=data_train[i][feat+label+['workday', 'mark_2']], target=label[0], silent=True,
                    categorical_features=cate_feat+['workday'],
                    numeric_features=numb_feat+['mark_2'],
                    data_split_shuffle=False,
                    fold_strategy='timeseries', fold=2,
                    remove_multicollinearity=True, multicollinearity_threshold=0.9, session_id=1,
                    transformation=True, transformation_method='yeo-johnson', feature_ratio=False, normalize=True, normalize_method='robust',
                    remove_outliers=False, outliers_threshold=3/len(data_train[i]),
                    transform_target=False, transform_target_method='box-cox')

    # include=list(models().index.values), exclude=list(['kr', 'mlp', 'tr']), Kernel Ridge and MLP is too slow
    # if cross_validation=False, then training on 70% train_set and evaluating on 30% test_set,
    # if cross_validation=True, then training and evaluating both on 70% train_set.
    # lgbm = create_model('lightgbm', cross_validation=False)
    best3 = compare_models(n_select=3, sort='RMSE', cross_validation=False)
    best_final.append(best3)
    ensembled = []
    for j in range(len(best3[:])):  # 外层循环变量是i，内层的其他循环变量不能同名
        try:
            ensembled.append(ensemble_model(best3[j], method='Boosting', n_estimators=3, choose_better=False, optimize='RMSE'))
        except Exception as e:
            print('\n', f'模型{best3[j]}对提升流程做梯度下降失败，改用超参调优，报错信息：{e}', '\n')
            tuning.append(best3[j])
            try:
                ensembled.append(tune_model(best3[j], n_iter=50, optimize='RMSE', choose_better=True, early_stopping=True))
            except Exception as e:
                print('\n', f'模型{best3[j]}超参调优失败，返回超参数为初始值的模型，报错信息：{e}', '\n')
                no_tune.append(best3[j])
                # , search_library='scikit-optimize', search_algorithm='bayesian'
                # 贝叶斯推理中，下一轮训练的超参数取决于在上一轮设置的超参数下，CV后模型的指标情况，所以无法多核并行训练，除非对每个核单独进行贝叶斯推理，
                # 最后比较每个核上的推理结果。但这样会降低每个核上做推理的次数，变为1/n，n为核心或超线程数，与贝叶斯推理的思想有所违背。
                ensembled.append(best3[j])
    ensembled_final.append(ensembled)
    voting = blend_models(ensembled, choose_better=False, optimize='RMSE')
    voting_final.append(voting)
    final_model = finalize_model(voting)  # 按照voting的模型结构和超参数，用data_train的所有数据，重新训练voting的参数
    # final_model = finalize_model(lgbm)
    # stacked = stack_models(ensembled[1:], meta_model=ensembled[0], choose_better=False, optimize='RMSE', restack=True)
    # ensembled = ensemble_model(best5[0], method='Boosting', n_estimators=5)
    # stacked = stack_models(best5[1:], meta_model=ensembled, choose_better=False, optimize='RMSE', restack=True)

    if i in np.array(lunar_num)+2:
        pred = predict_model(final_model, data=data_unseen[i][iden_feat+flexible+feat_lunar+label+['y']])
    else:
        pred = predict_model(final_model, data=data_unseen[i][iden_feat+flexible+feat+label+['y']])
    pred.loc[pred['Label'] < 0, 'Label'] = 0
    result = result.append(pred)

result.sort_values(by='busdate', inplace=True)
if label[0] == 'y_log1p':
    result['yhat'] = np.exp(result['Label'])-1
elif label[0] == 'y_log2':
    result['yhat'] = 2**(result['Label'])-1
elif label[0] == 'y_log':
    result['yhat'] = a**(result['Label'])-1
elif label[0] == 'y_norm':
    result['yhat'] = result['Label'] * np.mean(train_data['amount'])
elif label[0] == 'y_bc':
    result['yhat'] = result['Label'] - 1
else:
    result['yhat'] = result['Label']
smape_final = re.smape(y_true=result['y'], y_pred=result['yhat'])
mape_final = re.mape(y_true=result['y'], y_pred=result['yhat'])
rmse_final = re.rmse(y_true=result['y'], y_pred=result['yhat'])
r2_final = round(metrics.r2_score(y_true=result['y'], y_pred=result['yhat']), 4)
print(smape_final, mape_final, rmse_final, r2_final)
result.to_csv('/Users/ZC/Documents/company/promotion/result/90%-targetnorm-indepyj-boostingvoting.csv')
print('\n', tuning, '\n', no_tune)

# interpret_model(best3[0], plot='correlation')
