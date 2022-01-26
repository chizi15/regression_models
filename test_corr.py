import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pycaret.regression import *
from borax.calendars.lunardate import LunarDate
from joblib import Parallel, delayed
import sys
sys.path.append('/Users/ZC/PycharmProjects/PythonProjectProf/promotion/')
from data_tools import get_data_from_s3, process_data
from preprocess import his_mean
sys.path.insert(0, '/Users/ZC/PycharmProjects/PythonProjectProf/functions/')
import regression_evaluation_def as ref
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 20)


# 设置变量参数和模型超参数等
data_number = 1
label = ['y_ori']  # 'y_ori', 'y_log', 'y_comps'
feature = ['price_level', 'price', 'pro_day', 'stock_begin', 'distance_day', 'month_diff']
corr_level = abs(np.array([-0.65, -0.65, -0.55, 0.6, 0.5, 0.6]))
r = 0.9
weighted_type = 'amean_sigmoid'
weighted_len = 42
iden_feat = ['organ', 'code', 'busdate', 'class', 'bigsort', 'midsort', 'sort']  # 7
cate_feat = ['weekend', 'workday']  # 2, 'promotion_type'
numb_feat = ['weekday', 'year', 'month', 'day', 'distance_day', 'pro_day', 'costprice', 'price', 'price_level',
             'stock_begin', 'mean_code', 'mean_sort', 'mean_mid', 'mean_big',  'mean_class', 'month_diff',
             'mean_weighted_rolling', 'mean_weighted_group']  # 20, 'std_code', 'std_sort', 'std_mid', 'std_big', 'std_class',
numb_feat_lunar = ['weekday', 'LunarYear', 'LunarMonth', 'LunarDay', 'distance_day', 'pro_day', 'costprice', 'price', 'price_level',
                   'stock_begin', 'mean_code', 'std_code', 'mean_sort', 'std_sort', 'mean_mid', 'std_mid', 'mean_big', 'std_big', 'mean_class', 'std_class',
                   'mean_weighted_rolling', 'mean_weighted_group']  # 20, 'mark_1', 'mark_2', 'festival_num', 'month_diff',
feat = cate_feat + numb_feat  # 22
feat_lunar = cate_feat + numb_feat_lunar  # 22
flexible = ['festival_num', 'mark_1', 'mark_2']  # 3

# cpnid, organ = '0042', '1038'
# bucket_name = 'wuerp-nx'
# train_raw = get_data_from_s3(bucket_name, 'train', cpnid, organ)
# train_raw.to_csv('/Users/ZC/Documents/company/promotion/0042-1038-train.csv', index=False)
# print('\n', '促销样本量：%.d' % (len(train_raw)-train_raw.isna().sum().max()), '\n')
# print('促销样本占总样本的百分比：%.3f' % ((1-train_raw.isna().sum().max()/len(train_raw))*100), '\n')


# 读取数据并做一些预处理和初步的特征工程
train_raw = pd.read_csv('/Users/ZC/Documents/company/promotion/0042-1038-train.csv')
# train_raw = pd.read_csv('/Users/ZC/Documents/company/promotion/0042/1021/train.csv')
print('train_raw.isnull().sum():', '\n', train_raw.isnull().sum(), '\n')
# train_raw = train_raw[train_raw['flag'] == '促销']
train_raw = train_raw[(train_raw['price']>0) & (train_raw['costprice']>0) & (train_raw['pro_day']>0) & (train_raw['distance_day']>=0) & (train_raw['promotion_type']>0)]  # 将逻辑异常的样本剔除
train_raw = train_raw[~((train_raw['price_level']>3) | (train_raw['price_level']<=0) | pd.isna(train_raw['price_level']))]
# 将负库存、库存为0销售为0、库存为0但有销售、库存为空的四种情况剔除
train_raw = train_raw[~((train_raw['stock_begin']<=0) | pd.isna(train_raw['stock_begin']))]  # & (train_raw['amount_ch']!=0)
# train_raw.to_csv('/Users/ZC/Documents/company/promotion/result/amount.csv')
train_ori = train_raw[train_raw['busdate']>'2018-06-16'].copy()
train_ori = train_ori[train_ori['busdate']<'2021-09-10']
min_date = train_ori['busdate'].min()
train_df = process_data(train_ori, min_date)
print('\n', 'train_df.isnull().sum():', '\n', train_df.isnull().sum(), '\n')
dfgb = his_mean(train_df)
train_data = pd.merge(train_df, dfgb, how='left', on=['class', 'bigsort', 'midsort', 'sort', 'code'])
print('\n', 'train_data.isnull().sum():', '\n', train_data.isnull().sum(), '\n')
train_data.dropna(inplace=True)
# train_data.fillna(0, inplace=True)

# 对因变量做变换
a = np.sqrt(2)
if train_data['amount'].min() >= 0:
    train_data['y_ori'] = train_data['amount']
    train_data['y_bc'] = train_data['amount'] + 1  # 当λ被极大似然估计优选为0时，box-cox是做对数变换，所以+1能使对数的因变量非负；由于amount≥0，则此时对y_bc和自变量做box-cox变换，与对amount和相同自变量做yeo-johnson变换等价。
    train_data['y_log1p'] = np.log1p(train_data['amount'])
    train_data['y_log2'] = np.log2(train_data['amount'] + 1)
    train_data['y_log'] = np.log1p(train_data['amount']) / np.log(a)
    # 因为train_data['std_code']中存在0值，若不加a，则对应的train_data['y_zscr']会为空，会影响后续groupby；加上a，当train_data['std_code']为0时，对应的train_data['y_zscr']会是0，符合正态变换的逻辑，是对z-score的实用化处理
    train_data['y_zscr'] = (train_data['amount']-train_data['mean_code']) / (np.log(train_data['std_code']+a) / np.log(a))
    train_data['y_comps'] = train_data['amount'] / train_data['mean_class'].mean()

# 整理数据集
train_data_01 = train_data[(train_data['festival_num'] == 0) & (train_data['workday'] == 1)]  # 非节日工作日
train_data_00 = train_data[(train_data['festival_num'] == 0) & (train_data['workday'] == 0)]  # 非节日非工作日

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

data = []
data.extend((train_data_01, train_data_00))
for i in range(len(train_data_split)):  # for loop variable i is a global variable
    data.append(train_data_split[i])

train_data_011 = train_data_01[train_data_01['promotion_type']==1]  # 非节日工作日店促,50497
train_data_012 = train_data_01[train_data_01['promotion_type']==2]  # 非节日工作日档期,77578
train_data_001 = train_data_00[train_data_00['promotion_type']==1]  # 非节日非工作日店促,19335
train_data_002 = train_data_00[train_data_00['promotion_type']==2]  # 非节日非工作日档期,29722
data.extend((train_data_011, train_data_012, train_data_001, train_data_002))

##################################################################################################
data_use = data[data_number]

unit_codes = data_use['code'].drop_duplicates()
corr_code_select = pd.DataFrame()
for j in range(len(feature)):
    result = Parallel(n_jobs=4, backend="multiprocessing", verbose=1, batch_size=4*4) \
        (delayed(ref.regression_correlaiton_single)
         (data_use[data_use['code']==unit_codes.iloc[i]][feature[j]],
          data_use[data_use['code']==unit_codes.iloc[i]][label[0]], type='high')
         for i in range(len(unit_codes)))  # loop variable i in delayed is local variable
    corr_result = [result[i][0]['correlation'].values[0] for i in range(len(result))]
    d = {'code': list(unit_codes.values), 'corr': corr_result}
    corr_code = pd.DataFrame(d, index=unit_codes.index.values)
    print('\n', '本数据集总单品数：', len(unit_codes), '\n', '相关性非空的单品：', len(corr_code[corr_code['corr'].notna()]), '\n',
          '相关性非空单品占比(%)：', round(len(corr_code[corr_code['corr'].notna()])/len(unit_codes)*100, 2))
    len_single = len(corr_code_select)
    corr_code_select = corr_code_select.append(corr_code[abs(corr_code['corr']) > corr_level[j]])
    print('\n', f'特征{feature[j]}与应变量{label[0]}间相关性绝对值大于{corr_level[j]}的单品数有{len(corr_code_select)-len_single}个', '\n')
corr_code_select_union = corr_code_select['code'].drop_duplicates()
print('\n', '符合条件的单品数：', len(corr_code_select_union), '\n', '符合条件的单品数与本数据集总单品数之比：(%)', round(len(corr_code_select_union)/len(unit_codes)*100, 2))
samps_select = pd.DataFrame()
len_all = []
for i in range(len(corr_code_select_union)):
    # 这一步操作会打乱samps_select中busdate的升序排列，对后面截取训练集、预测集，以及训练和预测造成逻辑错误，对于时间序列问题，df需按升序排列
    samps_select = pd.concat([samps_select, data_use[data_use['code']==corr_code_select_union.values[i]]])
    len_all.append(len(data_use[data_use['code']==corr_code_select_union.values[i]]))
print('\n', f'平均每个单品含有{round(sum(len_all)/len(corr_code_select_union), 2)}个样本')
if len(samps_select) / (sum(len_all)/len(corr_code_select_union)) - len(corr_code_select_union) > 1e-3:
    raise Exception('筛选出的单品数与样本数间不匹配')
print('\n', '符合条件的样本数：', len(samps_select), '\n', '符合条件的样本数与本数据集总样本数之比：(%)', round(len(samps_select)/len(data_use)*100, 2))
# 将被打乱顺序的samps_select按busdate升序恢复成正常时间顺序，以保证后续逻辑正确
samps_select.sort_values(by=['busdate'], inplace=True)

# generating grouped rolling weighted average and grouped weighted average for individual maped data, compared to the whole data mean.
df_group, df_rolling, df_code, codes = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), list()
data_partn = samps_select[['busdate', 'amount', 'class', 'bigsort', 'midsort', 'sort', 'code']][:]
grouped = data_partn.groupby(['class', 'bigsort', 'midsort', 'sort', 'code'])
for code, group in grouped:
    codes.append(code[-1])
    df_group = pd.concat([df_group, group])
result_group = Parallel(n_jobs=4, backend="multiprocessing", verbose=1, batch_size=4*4) \
    (delayed(ref.dyn_seri_weighted)(group['amount'][-weighted_len:], type=weighted_type, critical_y=(1e-10, 0.1, 0.4, 1)) for code, group in grouped)
# result_group的值匹配codes中的值
df_code['code'], df_code['mean_weighted_group'] = codes, result_group

grouped_rolling = data_partn.groupby(['class', 'bigsort', 'midsort', 'sort', 'code']).rolling(window=weighted_len, min_periods=1, center=False, closed='both')
# for rolling in grouped_rolling:
#     df_rolling = df_rolling.append(rolling)  # unnecessary
result_rolling = Parallel(n_jobs=4, backend="multiprocessing", verbose=1, batch_size=4*4) \
    (delayed(ref.dyn_seri_weighted)(rolling['amount'], type=weighted_type, critical_y=(1e-10, 0.1, 0.4, 1)) for rolling in grouped_rolling)
# result_rolling的值匹配df_group的索引
df_group['mean_weighted_rolling'] = result_rolling
df_rolling_group = pd.merge(df_group, df_code, how='left', on=['code'])
df_rolling_group.drop(columns=['amount'], inplace=True)

samps_select_rolling_group = pd.merge(samps_select, df_rolling_group, how='left', on=['busdate', 'class', 'bigsort', 'midsort', 'sort', 'code'])

# pred_select = pd.DataFrame()
# for i in range(len(corr_code_select)):
#     pred_select = pd.concat([pred_select, data_unseen[0][data_unseen[0]['code']==corr_code_select['code'].values[i]]])


########################################
# coor = []
# for n in range(len(samps_select)):
#     coor.append(ref.regression_correlaiton_single(
#         samps_select[samps_select['code']==samps_select['code'].iloc[n]][feat],
#         samps_select[samps_select['code']==samps_select['code'].iloc[n]]['amount'], type='high')[0]['correlation'].values[0])
# print(np.nanmean(coor))


# data_use['amount'].memory_usage(deep=True)
# data_use.memory_usage(deep=True)/1024/1024  # unit: M
# data_use.info()
##############################################

# 'weekday', 'year', 'month', 'day'
# corr_data = data_use[['price_level', 'price', 'stock_begin', 'mean_code', 'mean_sort', 'mean_mid',
#                            'mean_big', 'mean_class', 'distance_day', 'pro_day', 'promotion_type']].loc[:]
# # corr_data[['price_level', 'pro_day']] = corr_data[['price_level', 'pro_day']]*-1
# samp_cardt = corr_data.loc[0]
# # batch_size越小，传batch的次数越多，对缓存占用越少，但可能造成对缓存空间的浪费，以及计算过程为传输过程总体的等待时间增加；
# # batch_size越大，传输次数越少，计算过程为传输过程总体的等待时间减少，但对缓存占用越大，当缓存放满时则溢出到内存中，造成取数计算的总时间增加
# result = Parallel(n_jobs=4, backend="multiprocessing", verbose=1, batch_size=4*4) \
#     (delayed(ref.regression_correlaiton_single)(samp_cardt, corr_data.loc[i]) for i in corr_data.index)
# corr_result = [result[i][0]['correlation'].values[0] for i in range(len(result))]
# corr_index = pd.Series(corr_result, index=corr_data.index.values)
# corr_filter = corr_index[corr_index > 0.7]
# data_train_filter = data_use.loc[corr_filter.index]
#
# train_code = data_train_filter['code'].drop_duplicates()
# data_pred_filter = pd.DataFrame()
# for i in range(len(train_code)):
#     data_pred_filter = pd.concat([data_pred_filter, data_unseen[0][data_unseen[0]['code']==train_code.values[i]]])


# corr_data_pred = data_unseen[0][['amount', 'price_level', 'price', 'stock_begin', 'mean_code', 'mean_sort', 'mean_mid',
#                            'mean_big', 'mean_class', 'distance_day', 'pro_day', 'promotion_type', 'weekday', 'year',
#                            'month', 'day']].loc[:]
# result_pred = Parallel(n_jobs=-1, backend="multiprocessing", verbose=1, batch_size=4*4) \
#     (delayed(ref.regression_correlaiton_single)(samp_cardt, corr_data_pred.loc[i]) for i in corr_data_pred.index)
# corr_result_pred = [result_pred[i][0]['correlation'].values[0] for i in range(len(result_pred))]
# corr_index_pred = pd.Series(corr_result_pred, index=corr_data_pred.index.values)
# corr_filter_pred = corr_index_pred[corr_index_pred > 0.8]
# data_pred_filter = data_unseen[0].loc[corr_filter_pred.index]
#################################################


# for i in range(len(data)):
#     data_train.append(data[i][(data[i]['busdate'] < data[i].iloc[:int(len(data[i]) * 0.9)]['busdate'].values[-1])])
#     data_unseen.append(data[i][~(data[i]['busdate'] < data[i].iloc[:int(len(data[i]) * 0.9)]['busdate'].values[-1])])
#     data_len.append(len(data[i]))
#     # print(len(data_train[i]), len(data_unseen[i]))
# print(pd.Series(data_len).describe())

data_len, data_train, data_unseen = [], [], []
# 以日期为界限进行筛选
data_train.append(samps_select_rolling_group[(pd.to_datetime(samps_select_rolling_group['busdate']) < samps_select_rolling_group.iloc[:int(len(samps_select_rolling_group) * r)]['busdate'].values[-1])])
data_unseen.append(samps_select_rolling_group[~(pd.to_datetime(samps_select_rolling_group['busdate']) < samps_select_rolling_group.iloc[:int(len(samps_select_rolling_group) * r)]['busdate'].values[-1])])
# # 以索引为界限进行筛选
# data_train.append(samps_select_rolling_group.iloc[:int(len(samps_select_rolling_group) * r)])
# data_unseen.append(samps_select_rolling_group[int(len(samps_select_rolling_group) * r):])
# data_len.append(len(samps_select_rolling_group))
# print(len(data_train[i]), len(data_unseen[i]))
# print(pd.Series(data_len).describe())
print('参与训练的单品数：', len(data_train[0]['code'].value_counts()))
print('进行预测的单品数：', len(data_unseen[0]['code'].value_counts()))
print('参与训练的样本数：', len(data_train[0]))
print('进行预测的样本数：', len(data_unseen[0]))
#############################################


SMAPE, MAPE, RMSE, R2 = [], [], [], []
result, result_final = pd.DataFrame(), pd.DataFrame()  # columns=data_unseen[0].columns
best_final, ensembled_final, models_final = [], [], []
tuning, no_tune = [], []

# for i in range(len(data_train)-4, len(data_train)):
for i in [0]:
    if i < 2 or i >= len(data_train)-4:  # 非节日
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
                    feature_ratio=False, interaction_threshold=0.01,
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
    # rf = create_model('rf', cross_validation=False)
    # 样本量较小，如小于1e4时，可用SVM，预测指标较好
    compare = list(models().index.values)
    compare.remove('kr')
    compare.remove('mlp')
    compare.remove('tr')
    best = compare_models(include=compare, n_select=3, sort='RMSE', cross_validation=False)
    ensembled_final_models = []
    for j in range(len(best[:])):  # 外层循环变量是i，内层的其他循环变量最好不要重名
        try:
            ensembled = ensemble_model(best[j], method='Boosting', n_estimators=5, choose_better=False, optimize='RMSE')
        except Exception as e:
            try:
                ensembled = tune_model(best[j], n_iter=50, optimize='RMSE', choose_better=True, early_stopping=True)
                print('\n', f'模型{best[j]}对提升流程做梯度下降失败，改用超参调优，报错信息：{e}', '\n')
                tuning.append(best[j])
            except Exception as e:
                # , search_library='scikit-optimize', search_algorithm='bayesian'
                # 贝叶斯推理中，下一轮训练的超参数取决于在上一轮设置的超参数下，CV后模型的指标情况，所以无法多核并行训练，除非对每个核单独进行贝叶斯推理，
                # 最后比较每个核上的推理结果。但这样会降低每个核上做推理的次数，变为1/n，n为核心或超线程数，与贝叶斯推理的思想有所违背。
                ensembled = best[j]
                print('\n', f'模型{best[j]}超参调优失败，返回超参数为初始值的模型，报错信息：{e}', '\n')
                no_tune.append(best[j])
        final_model_use = finalize_model(ensembled)
        if i in np.array(lunar_num)+2:  # 农历节假日
            # features in predict data which don't appear in train data will not be used in forecast
            pred = predict_model(final_model_use, data=data_unseen[i][iden_feat+flexible+feat_lunar+label+['amount']])  # 预测数据中的列需包含（即大于等于）训练数据中的列
        else:  # 非农历节假日
            pred = predict_model(final_model_use, data=data_unseen[i][iden_feat+flexible+feat+label+['amount']])
        result = result.append(pred)
        ensembled_final_models.append(final_model_use)
    models_final.append(ensembled_final_models)
    best_final.append(best)
result.sort_values(by='busdate', inplace=True)


# 对预测值进行后处理
pd.set_option('display.max_columns', None)
result_mean = result.groupby(list(result.columns)[:-1], as_index=False).agg(
            Label_mean = pd.NamedAgg(column='Label', aggfunc='mean'),
            Label_count = pd.NamedAgg(column='Label', aggfunc='count'))
if label[0] == 'y_log1p':
    result_mean['yhat'] = np.exp(result_mean['Label_mean'])-1
elif label[0] == 'y_log2':
    result_mean['yhat'] = 2**(result_mean['Label_mean'])-1
elif label[0] == 'y_log':
    result_mean['yhat'] = a**(result_mean['Label_mean'])-1
elif label[0] == 'y_zscr':
    result_mean['yhat'] = result_mean['Label_mean'] * (np.log(result_mean['std_code']+a) / np.log(a)) + result_mean['mean_code']
elif label[0] == 'y_bc':
    result_mean['yhat'] = result_mean['Label_mean'] - 1
elif label[0] == 'y_comps':
    result_mean['yhat'] = result_mean['Label_mean'] * result_mean['mean_class'].mean()
else:
    result_mean['yhat'] = result_mean['Label_mean']
result_mean.loc[result_mean['yhat'] < 0, 'yhat'] = 0


# 计算预测结果指标，输出结果
smape_final = ref.smape(y_true=result_mean['amount'], y_pred=result_mean['yhat'])
mape_final = ref.mape(y_true=result_mean['amount'], y_pred=result_mean['yhat'])
rmse_final = ref.rmse(y_true=result_mean['amount'], y_pred=result_mean['yhat'])
r2_final = round(metrics.r2_score(y_true=result_mean['amount'], y_pred=result_mean['yhat']), 4)
print(smape_final, mape_final, rmse_final, r2_final)
result_mean['DR'] = abs(result_mean['amount'] - result_mean['yhat']) / result_mean['amount']
result_mean['SDR'] = 2*abs(result_mean['amount'] - result_mean['yhat']) / (result_mean['amount'] + result_mean['yhat'])
# when result_mean['amount']==0 and abs(result_mean['amount'] - result_mean['yhat'])!=0, result_mean['DR'] == np.inf, then existing bias
result_mean.loc[result_mean['DR'] == np.inf, 'DR'] = abs(result_mean['amount'] - result_mean['yhat'])[result_mean['DR'] == np.inf]
# when result_mean['amount']==0 and abs(result_mean['amount'] - result_mean['yhat'])==0, result_mean['DR'] is NaN, then completely acurate
result_mean.loc[pd.isna(result_mean['DR']), 'DR'] = 0
# when (result_mean['amount'] + result_mean['yhat'])==0 and 2*abs(result_mean['amount'] - result_mean['yhat'])!=0, result_mean['SDR'] == np.inf, then existing bias
result_mean.loc[result_mean['SDR'] == np.inf, 'SDR'] = 2*abs(result_mean['amount'] - result_mean['yhat'])[result_mean['SDR'] == np.inf]
# when (result_mean['amount'] + result_mean['yhat'])==0 and 2*abs(result_mean['amount'] - result_mean['yhat'])==0, result_mean['SDR'] is NaN, then completely acurate
result_mean.loc[pd.isna(result_mean['SDR']), 'SDR'] = 0
result_mean['y'] = result_mean['amount']
result_mean.to_csv('/Users/ZC/Documents/company/promotion/result/0042-1038-0125-corr_pl-90%-y_log-indepyj-boostingavg-n_select=3.csv')

"""
1.1651 0.6075 9.77108141919013 0.796
0.9225 0.5263 11.641784773144042 0.7308
1.1919 0.4077 8.721994820209305 0.64
0.976 0.3861 3.7745797986909086 0.7047
"""
