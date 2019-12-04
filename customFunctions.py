import pandas as pd
import numpy as np
from sklearn import preprocessing


def preparazione_Datasets(sessions, age_gender_bkts):
	print('### START WORKING ON SESSION DATAFRAME ###')
	grouped_sessions = sessions[['user_id', 'action_type']].pivot_table(index='user_id', columns='action_type', 
                        aggfunc=len, fill_value=0)
	grouped_sessions['user_id'] = list(grouped_sessions.index)
	sessions_count = pd.DataFrame(sessions.user_id.value_counts(sort = False))
	sessions_count['action_count'] = sessions_count.user_id
	sessions_count['user_id'] = sessions_count.index
	sess = grouped_sessions.merge(sessions_count, on='user_id', how = 'left')
	
	secs_elapsed = sessions.groupby('user_id')['secs_elapsed']
	secs_elapsed = secs_elapsed.agg(
		{
			'secs_elapsed_sum': np.sum,
			'secs_elapsed_mean': np.mean,
			'secs_elapsed_min': np.min,
			'secs_elapsed_max': np.max,
			'secs_elapsed_median': np.median,
			'secs_elapsed_std': np.std,
			'secs_elapsed_var': np.var,
			'day_pauses': lambda x: (x > 86400).sum(),
			'long_pauses': lambda x: (x > 300000).sum(),
			'short_pauses': lambda x: (x < 3600).sum(),
			'session_length' : np.count_nonzero
		}
	)
	secs_elapsed.reset_index(inplace=True)

	sessions = secs_elapsed.merge(sess, on='user_id', how='left')

	print('### END SESSION DATAFRAME ###')
	print('### START WORKING ON AGE_GENDER_BKTS DATAFRAME ###')
	
	age_gender_bkts.loc[age_gender_bkts['age_bucket']=='100+', 'age_bucket'] = '100-200'
	age_gender_bkts.groupby('age_bucket').agg(['count']).stack()
	
	age_buck=list(age_gender_bkts['age_bucket'].unique())
	
	## find most visited country for each age_bucket - gender
	
	age_gen_country = age_gender_bkts.groupby(['age_bucket', 'gender', 'country_destination'])['population_in_thousands'].sum().reset_index()
	
	age_gen_country['age_gender'] = age_gen_country['age_bucket'] + '/' + age_gen_country['gender']
	
	age_gen_country=age_gen_country.pivot(index='age_gender', columns='country_destination', values='population_in_thousands').reset_index()
	
	
	age_gen_country['TOT']=age_gen_country[[x for x in age_gen_country.columns if x != 'age_gender']].sum(axis=1)
	
	for x in [x for x in age_gen_country.columns if x not in ['age_gender', 'TOT']]:
		age_gen_country['prob_tot_' + x] = age_gen_country[x]/age_gen_country['TOT']
	
	del age_gen_country['TOT']

	list_countries=["US",\
				"FR",\
				"IT",\
				"GB",\
				"ES",\
				"CA",\
				"DE",\
				"NL",\
				"AU",\
				"PT"]
	
	for x in [x for x in list_countries if x not in ['age_gender', 'TOT']]:
		age_gen_country['prob_' + x] = age_gen_country[x]/age_gen_country[x].sum()
	print('### END AGE_GENDER_BKTS DATAFRAME ###')

	return sessions, age_gen_country
	

	
def prepareTrainTest(train, test, sessions, age_gen_country, age_gender_bkts):

	train.index = train['id']
	test.index = test['id']

	print('### START ENCODING PROCESS ###')

	df = pd.concat((train, test), axis = 0, ignore_index = True)
	
	df['date_account_created'] = pd.to_datetime(df['date_account_created'])
	df['timestamp_first_active'] = pd.to_datetime((df.timestamp_first_active // 1000000), format='%Y%m%d')
	df['weekday_account_created'] = df.date_account_created.dt.weekday_name
	df['day_account_created'] = df.date_account_created.dt.day
	df['month_account_created'] = df.date_account_created.dt.month
	df['year_account_created'] = df.date_account_created.dt.year
	df['weekday_first_active'] = df.timestamp_first_active.dt.weekday_name
	df['day_first_active'] = df.timestamp_first_active.dt.day
	df['month_first_active'] = df.timestamp_first_active.dt.month
	df['year_first_active'] = df.timestamp_first_active.dt.year
	
	df['time_lag'] = (df['date_account_created'] - df['timestamp_first_active'])
	df['time_lag'] = df['time_lag'].astype(pd.Timedelta).apply(lambda l: l.days)
	df.drop( ['date_account_created', 'timestamp_first_active'], axis=1, inplace=True)
	
	### join all the tables
	# to join we lower case the gender columns
	# adjust age 

	## creating age_bucket column in df
	age_buck=list(age_gender_bkts['age_bucket'].unique())
	
	df['age_bucket']='nan'
	df['age'][df['age']> 120]=120
		
	for i in age_buck:
		age_max=int(i.split('-')[1])
		age_min=int(i.split('-')[0])
		df['age_bucket'][df['age'].between(age_min, age_max)] = i
	
	df['age_gender']=df['age_bucket'] + '/' + df['gender']
	df['age_gender']=df['age_gender'].str.lower()
	age_gen_country['age_gender']=age_gen_country['age_gender'].str.lower()
	
	df=df.merge(age_gen_country, on='age_gender', how='left')
	
	# join with grouped session
	
	sessions.columns = ['id'] + [x for x in sessions.columns if x.lower() != 'user_id']
	
	df=df.merge(sessions, on='id', how='left')
	
	### fill nan
	df_=df
	
	for i in [x for x in list(df['age_bucket'].unique()) if x.lower() != 'nan']:
		df_[df_['age_bucket']==i] = df_[df_['age_bucket']==i].fillna(df_[df_['age_bucket']==i].mean())
		
	### Encoding char variables
	from sklearn.preprocessing import LabelEncoder
	le =LabelEncoder()
	
	# column first_affiliate_tracked conteins nan
	df_['first_affiliate_tracked']=df_['first_affiliate_tracked'].fillna('unknown')
	# string cols
	str_cols = list(df_.dtypes[(df_.dtypes != 'float64') & (df_.dtypes != 'int64')].index)
	
	str_cols = [x for x in str_cols if (x.lower() != 'id') & ('date' not in x.lower()) & (x.lower() != 'country_destination')]
	
	for i in str_cols:
#		print('start ' + i)
		le.fit(df_[i])
		df_[i+'_encoded'] = le.transform(df_[i])
#		print('end ' + i)

	#### prepare train and test dataframes
	
	#drop missing
	#df = df_[(df_['age_gender'] != 'nan/-unknown-')]
	
	
	df=df.sort_values(['date_first_booking'], ascending=False)
	
	df=df.replace([np.inf, -np.inf], np.nan)
	df=df.fillna(0)
	
	num_cols = list(df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].index)
	num_cols = [x for x in num_cols if ('timestamp' not in x.lower()) & ('date' not in x.lower()) & (x.upper() not in [ 'AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NL', 'PT', 'US'])]

	df=df[['id'] + num_cols + ['country_destination']]
	
	print('### END ENCODING PROCESS ###')
	# print('### STD COLUMNS ###')	
	# x = df[[x for x in num_cols if x.lower()!= 'target']].values
	# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	# x_scaled = min_max_scaler.fit_transform(x)

	# x_scaled=pd.DataFrame(x_scaled)
	# x_scaled.columns=[x + '_scaled' for x in num_cols if x.lower()!= 'target']
	# df=pd.concat([df, x_scaled], axis=1)
	print('### SPLITTING TRAIN AND TEST ###')
	
	df.index=df['id']
	
	df_train = df.loc[train['id']]
	df_test = df.loc[test['id']]
	df_test=df_test.drop(columns=['country_destination'])

	
	dic_target={"NDF" :0,\
				"US" :1,\
				"other" :2,\
				"FR" :3,\
				"IT" :4,\
				"GB" :5,\
				"ES" :6,\
				"CA" :7,\
				"DE" :8,\
				"NL" :9,\
				"AU" :10,\
				"PT" :11}
	print('---- THE TARGET IS ENCODEDAS FOLLOW ----')
	print('--- the order follows the grups size ---')
	
	for i in dic_target.keys():
		print(i + ' : ' + str(dic_target[i]))
	
	
	df_train['TARGET']=0
	for i in dic_target.keys():
		df_train.loc[df_train['country_destination']==i, 'TARGET']=dic_target[i]
	
	return df_train, df_test