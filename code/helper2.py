def graph_sentiment_profit(publisher,positive_sentiment_threshold,min_date):
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 40

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    if publisher!='all':
        # Generate DataFrame for daily sentiment for each publisher
        asdf_df = df_news_sentiment[df_news_sentiment['date']>min_date]
        #df_news_sentiment = df_news_sentiment[df_news_sentiment['date']>min_date]
        sentiment_by_publisher = asdf_df.groupby(by=['date','publisher'],as_index=False).mean()

        sentiment_publisher_1 = sentiment_by_publisher[sentiment_by_publisher['publisher']==publisher]
        sentiment_publisher_1 = sentiment_publisher_1.merge(df_daily[['date','daily_sentiment_change','target_daily']], left_on='date', right_on='date')

        sentiment_publisher_1['vader_buy_sell'] = sentiment_publisher_1['compound'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['vader_profit'] = (sentiment_publisher_1['vader_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['txtblob_buy_sell'] = sentiment_publisher_1['txtblob'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['txtblob_profit'] = (sentiment_publisher_1['txtblob_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_buy_sell'] = sentiment_publisher_1['final_sentiment'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_profit'] = (sentiment_publisher_1['compound_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell'] = sentiment_publisher_1['daily_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit'] = (sentiment_publisher_1['compound_change_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1.dropna(inplace=True)

#         start_index = len(sentiment_publisher_1)-30
#         end_index = len(sentiment_publisher_1)

#         plt.figure(figsize=(20,8))
#         sns.lineplot(x=sentiment_publisher_1['date'][start_index:end_index],y=sentiment_publisher_1['compound'][start_index:end_index])
#         sns.lineplot(x=sentiment_publisher_1['date'][start_index:end_index],y=sentiment_publisher_1['txtblob'][start_index:end_index])
#         sns.lineplot(x=sentiment_publisher_1['date'][start_index:end_index],y=sentiment_publisher_1['final_sentiment'][start_index:end_index])

#         plt.xlabel('Date')
#         plt.ylabel('Sentiment')
#         plt.title('Daily Sentiment for Bitcoin ({}) - Pos Threshold = {}'.format(publisher,positive_sentiment_threshold))
#         plt.legend(['Vader','TextBlob','Vader+TextBlob'])
#     #     plt.xticks(sentiment_publisher_1['date'][start_index:end_index],rotation=60)
#         plt.grid()
#         plt.show()
    #     print('Vader \n   Mean : {}, Std : {}'.format(np.round(np.mean(sentiment_publisher_1['compound']),3),np.round(np.std(sentiment_publisher_1['compound']),3)))
    #     print('TextBlob \n   Mean : {}, Std : {}'.format(np.round(np.mean(sentiment_publisher_1['txtblob']),3),np.round(np.std(sentiment_publisher_1['txtblob']),3)))
    #     print('Vader&TextBlob \n   Mean : {}, Std : {}'.format(np.round(np.mean(sentiment_publisher_1['final_sentiment']),3),np.round(np.std(sentiment_publisher_1['final_sentiment']),3)))

        vader_profit = list(sentiment_publisher_1['vader_profit'])
        for i in range(1,len(vader_profit)):
            vader_profit[i] = vader_profit[i]+vader_profit[i-1]
        txtblob_profit = list(sentiment_publisher_1['txtblob_profit'])
        for i in range(1,len(txtblob_profit)):
            txtblob_profit[i] = txtblob_profit[i]+txtblob_profit[i-1]
        compound_profit = list(sentiment_publisher_1['compound_profit'])
        for i in range(1,len(compound_profit)):
            compound_profit[i] = compound_profit[i]+compound_profit[i-1]
        compound_change_profit = list(sentiment_publisher_1['compound_change_profit'])
        for i in range(1,len(compound_change_profit)):
            compound_change_profit[i] = compound_change_profit[i]+compound_change_profit[i-1]

        plt.figure(figsize=(20,8))
        sns.lineplot(x=sentiment_publisher_1['date'],y=vader_profit)
        sns.lineplot(x=sentiment_publisher_1['date'],y=txtblob_profit)
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_profit)
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_change_profit)

        plt.xlabel('Date')
        plt.ylabel('Profit ($)')
        plt.title('Cumulative Profit ({}) - Pos Threshold = {}'.format(publisher,positive_sentiment_threshold))
        plt.legend(['Vader','TextBlob','Vader+TextBlob','Vader+TextBlob (*Change in Sentiment)'])
        #plt.xticks(sentiment_publisher_1['date'],rotation='vertical')
        plt.grid()
        plt.show()
    else:
        # Generate DataFrame for daily sentiment for each publisher
        sentiment_by_publisher = df_news_sentiment.groupby(by=['date'],as_index=False).mean()

        sentiment_publisher_1 = sentiment_by_publisher[sentiment_by_publisher['date']>min_date]
        sentiment_publisher_1 = sentiment_publisher_1.merge(df_daily[['date','daily_sentiment_change','wkly_sentiment_change','2wk_sentiment_change','4wk_sentiment_change','target_daily']], left_on='date', right_on='date')

        sentiment_publisher_1['compound_buy_sell'] = sentiment_publisher_1['final_sentiment'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_profit'] = (sentiment_publisher_1['compound_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell'] = sentiment_publisher_1['daily_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit'] = (sentiment_publisher_1['compound_change_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_wkly'] = sentiment_publisher_1['wkly_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_wkly'] = (sentiment_publisher_1['compound_change_buy_sell_wkly']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_2wk'] = sentiment_publisher_1['2wk_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_2wk'] = (sentiment_publisher_1['compound_change_buy_sell_2wk']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_4wk'] = sentiment_publisher_1['4wk_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_monthly'] = (sentiment_publisher_1['compound_change_buy_sell_4wk']*100*sentiment_publisher_1['target_daily']).copy()

        
        sentiment_publisher_1.dropna(inplace=True)

#         start_index = len(sentiment_publisher_1)-30
#         end_index = len(sentiment_publisher_1)        
        
#         min_max_scaler = preprocessing.MinMaxScaler()

#         plt.figure(figsize=(15,5))
        
#         plt.figure(figsize=(20,8))
#         sns.lineplot(x=df_daily['date'][start_index:end_index],y=min_max_scaler.fit_transform(df_daily['close'][start_index:end_index].values.reshape(-1,1)).reshape(-1))
#         sns.lineplot(x=sentiment_publisher_1['date'][start_index:end_index],y=min_max_scaler.fit_transform(sentiment_publisher_1['final_sentiment'][start_index:end_index].values.reshape(-1,1)).reshape(-1))
# #        sns.lineplot(x=sentiment_publisher_1['date'],y=sentiment_publisher_1['final_sentiment'])


#         plt.xlabel('Date')
#         plt.ylabel('Sentiment')
#         plt.title('Daily Compound Sentiment for Bitcoin')
#         plt.legend(['Bitcoin Close Price','Vader+TextBlob'])
#         plt.grid()
#         plt.show()

        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        plt.figure(figsize=(20,8))
        sns.lineplot(x=df_daily['date'],y=min_max_scaler.fit_transform(df_daily['open'].values.reshape(-1,1)).reshape(-1))
        sns.lineplot(x=df_daily['date'],y=min_max_scaler.fit_transform(df_daily['volume'].values.reshape(-1,1)).reshape(-1))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        plt.axvline(x=date(2018, 9, 1),color='r',linestyle ='-.' )  #vertical line. Linestype : ['-', '--', '-.', ':', '',]

        plt.xlabel('Date')
        plt.ylabel('Scaled Range')
        plt.title('Bitcoin Open Price vs Volume - Pos Threshold = {}'.format(positive_sentiment_threshold))
        plt.grid()
        plt.legend(['open','volume'])
        plt.show()   

        compound_profit = list(sentiment_publisher_1['compound_profit'])
        compound_change_profit = list(sentiment_publisher_1['compound_change_profit'])
        compound_change_profit_wkly = list(sentiment_publisher_1['compound_change_profit_wkly'])
        compound_change_profit_2wk = list(sentiment_publisher_1['compound_change_profit_2wk'])
        compound_change_profit_montly = list(sentiment_publisher_1['compound_change_profit_monthly'])
        
        for i in range(1,len(compound_profit)):
            compound_profit[i] = compound_profit[i]+compound_profit[i-1]
        for i in range(1,len(compound_change_profit)):
            compound_change_profit[i] = compound_change_profit[i]+compound_change_profit[i-1]
        for i in range(1,len(compound_change_profit_wkly)):
            compound_change_profit_wkly[i] = compound_change_profit_wkly[i]+compound_change_profit_wkly[i-1]            
        for i in range(1,len(compound_change_profit_2wk)):
            compound_change_profit_2wk[i] = compound_change_profit_2wk[i]+compound_change_profit_2wk[i-1]  
        for i in range(1,len(compound_change_profit_montly)):
            compound_change_profit_montly[i] = compound_change_profit_montly[i]+compound_change_profit_montly[i-1]          

        plt.figure(figsize=(20,8))
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_profit)
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_change_profit)
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_change_profit_wkly)
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_change_profit_2wk)
        sns.lineplot(x=sentiment_publisher_1['date'],y=compound_change_profit_montly)
        plt.axhline(y=0,color='r',linestyle ='-.' )  #vertical line. Linestype : ['-', '--', '-.', ':', '',]


        
        plt.xlabel('Date')
        plt.ylabel("Trading Bot's Profit ($)")
        plt.title('Cumulative Profit - Pos Threshold = {}'.format(positive_sentiment_threshold))
        plt.legend(['Vader+TextBlob (*Sentiment - Daily)',
                    'Vader+TextBlob (*Change in Sentiment - Daily)',
                    'Vader+TextBlob (*Change in Sentiment - Weekly MA)',
                    'Vader+TextBlob (*Change in Sentiment - 2 Weeks MA)',
                    'Vader+TextBlob (*Change in Sentiment - Monthly MA)'])
        #plt.xticks(sentiment_publisher_1['date'],rotation='vertical')
        plt.grid()
        plt.show()
        
def best_threshold(min_threshold,max_threshold,step,min_date,max_date):
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 40
    
    # Parameters for graphs
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # Add all final profit later on to these empty lists
    thresholds,raw_senti,senti_change_daily,senti_change_weekly,senti_change_2wk,senti_change_monthly = [],[],[],[],[],[]
    threshold_list = np.arange(min_threshold,max_threshold,step)
    for i in trange(len(threshold_list)):
        positive_sentiment_threshold = threshold_list[i]

        # Generate DataFrame for daily sentiment for each publisher
        sentiment_by_publisher = df_news_sentiment.groupby(by=['date'],as_index=False).mean()

        # Mask to limit daterange for searching best threshhold
        mask = ((sentiment_by_publisher['date']>min_date)&(sentiment_by_publisher['date']<max_date))
        sentiment_publisher_1 = sentiment_by_publisher[mask]
        sentiment_publisher_1 = sentiment_publisher_1.merge(df_daily[['date','daily_sentiment_change','wkly_sentiment_change','2wk_sentiment_change','4wk_sentiment_change','target_daily']], left_on='date', right_on='date')

        sentiment_publisher_1['compound_buy_sell'] = sentiment_publisher_1['final_sentiment'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_profit'] = (sentiment_publisher_1['compound_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell'] = sentiment_publisher_1['daily_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit'] = (sentiment_publisher_1['compound_change_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_wkly'] = sentiment_publisher_1['wkly_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_wkly'] = (sentiment_publisher_1['compound_change_buy_sell_wkly']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_2wk'] = sentiment_publisher_1['2wk_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_2wk'] = (sentiment_publisher_1['compound_change_buy_sell_2wk']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_4wk'] = sentiment_publisher_1['4wk_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_monthly'] = (sentiment_publisher_1['compound_change_buy_sell_4wk']*100*sentiment_publisher_1['target_daily']).copy()


        sentiment_publisher_1.dropna(inplace=True)


        compound_profit = list(sentiment_publisher_1['compound_profit'])
        compound_change_profit = list(sentiment_publisher_1['compound_change_profit'])
        compound_change_profit_wkly = list(sentiment_publisher_1['compound_change_profit_wkly'])
        compound_change_profit_2wk = list(sentiment_publisher_1['compound_change_profit_2wk'])
        compound_change_profit_montly = list(sentiment_publisher_1['compound_change_profit_monthly'])

        for i in range(1,len(compound_profit)):
            compound_profit[i] = compound_profit[i]+compound_profit[i-1]
        for i in range(1,len(compound_change_profit)):
            compound_change_profit[i] = compound_change_profit[i]+compound_change_profit[i-1]
        for i in range(1,len(compound_change_profit_wkly)):
            compound_change_profit_wkly[i] = compound_change_profit_wkly[i]+compound_change_profit_wkly[i-1]            
        for i in range(1,len(compound_change_profit_2wk)):
            compound_change_profit_2wk[i] = compound_change_profit_2wk[i]+compound_change_profit_2wk[i-1]  
        for i in range(1,len(compound_change_profit_montly)):
            compound_change_profit_montly[i] = compound_change_profit_montly[i]+compound_change_profit_montly[i-1] 

        thresholds.append(positive_sentiment_threshold)
        raw_senti.append(compound_profit[-1])
        senti_change_daily.append(compound_change_profit[-1])
        senti_change_weekly.append(compound_change_profit_wkly[-1])
        senti_change_2wk.append(compound_change_profit_2wk[-1])
        senti_change_monthly.append(compound_change_profit_montly[-1])
    return thresholds,raw_senti,senti_change_daily,senti_change_weekly,senti_change_2wk,senti_change_monthly

   
def best_threshold2(min_threshold,max_threshold,step,min_date,max_date):
    
    # Add all final profit later on to these empty lists
    thresholds,senti_change_weekly,senti_change_2wk,senti_change_monthly = [],[],[],[]
    threshold_list = np.arange(min_threshold,max_threshold,step)
    for i in range(len(threshold_list)):
        positive_sentiment_threshold = threshold_list[i]

        # Generate DataFrame for daily sentiment for each publisher
        sentiment_by_publisher = df_news_sentiment.groupby(by=['date'],as_index=False).mean()

        # Mask to limit daterange for searching best threshhold
        #mask = ((sentiment_by_publisher['date']>min_date)&(sentiment_by_publisher['date']<max_date))
        mask = (sentiment_by_publisher['date']<min_date)
        sentiment_publisher_1 = sentiment_by_publisher.merge(df_daily[['date','wkly_sentiment_change','2wk_sentiment_change','4wk_sentiment_change','target_daily']], left_on='date', right_on='date')

        sentiment_publisher_1['compound_change_buy_sell_wkly'] = sentiment_publisher_1['wkly_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_wkly'] = (sentiment_publisher_1['compound_change_buy_sell_wkly']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_2wk'] = sentiment_publisher_1['2wk_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_2wk'] = (sentiment_publisher_1['compound_change_buy_sell_2wk']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_4wk'] = sentiment_publisher_1['4wk_sentiment_change'].apply(lambda x: 1 if x>positive_sentiment_threshold else 0).copy()
        sentiment_publisher_1['compound_change_profit_monthly'] = (sentiment_publisher_1['compound_change_buy_sell_4wk']*100*sentiment_publisher_1['target_daily']).copy()

        # Drop NAs
        sentiment_publisher_1.dropna(inplace=True)

        # Get Cumulative Profits
        compound_change_profit_wkly = list(sentiment_publisher_1['compound_change_profit_wkly'])
        compound_change_profit_2wk = list(sentiment_publisher_1['compound_change_profit_2wk'])
        compound_change_profit_montly = list(sentiment_publisher_1['compound_change_profit_monthly'])

        for i in range(1,len(compound_change_profit_wkly)):
            compound_change_profit_wkly[i] = compound_change_profit_wkly[i]+compound_change_profit_wkly[i-1]            
        for i in range(1,len(compound_change_profit_2wk)):
            compound_change_profit_2wk[i] = compound_change_profit_2wk[i]+compound_change_profit_2wk[i-1]  
        for i in range(1,len(compound_change_profit_montly)):
            compound_change_profit_montly[i] = compound_change_profit_montly[i]+compound_change_profit_montly[i-1] 

        thresholds.append(positive_sentiment_threshold)
        senti_change_weekly.append(compound_change_profit_wkly[-1])
        senti_change_2wk.append(compound_change_profit_2wk[-1])
        senti_change_monthly.append(compound_change_profit_montly[-1])
    return thresholds,senti_change_weekly,senti_change_2wk,senti_change_monthly


def threshold_vs_profit(thresholds,model_list):
        
    plt.figure(figsize=(20,8))
    for i in range(len(model_list)):
        sns.lineplot(x=thresholds,y=model_list[i])
        
    plt.xlabel('Thresholds')
    plt.ylabel("Trading Bot's Profit ($)")
    plt.title('Cumulative Profit Over Various Positive Sentiment Thresholds')
    plt.legend(['Vader+TextBlob (*Sentiment - Daily)',
                'Vader+TextBlob (*Change in Sentiment - Daily)',
                'Vader+TextBlob (*Change in Sentiment - Weekly MA)',
                'Vader+TextBlob (*Change in Sentiment - 2 Weeks MA)',
                'Vader+TextBlob (*Change in Sentiment - Monthly MA)'])
    #plt.xticks(sentiment_publisher_1['date'],rotation='vertical')
    plt.grid()
    plt.show()
    
    
    
 # Get profit based on model's predictions. Punish it if it predicted to buy but actually it was a loss. Else, no action = profit = 0.
def model_profit_graph(model_output,min_date,max_date):
    # Get Cumulative Profit for Generic Models       
    thresholds,senti_change_weekly,senti_change_2wk,senti_change_monthly = best_threshold2(0,0.3,0.002,min_date,max_date)
    best_threshold_weekly = thresholds[np.argmax(senti_change_weekly)]
    best_threshold_biweekly = thresholds[np.argmax(senti_change_2wk)]
    best_threshold_monthly = thresholds[np.argmax(senti_change_monthly)]
    

    asdf_df = df_news_sentiment[df_news_sentiment['date']>max_date]
    sentiment_by_publisher = asdf_df.groupby(by=['date'],as_index=False).mean()
    sentiment_publisher_1 = sentiment_by_publisher.merge(df_daily[['date','wkly_sentiment_change','2wk_sentiment_change','4wk_sentiment_change','target_daily']], left_on='date', right_on='date')
    sentiment_publisher_1['weekly_buy_sell'] = sentiment_publisher_1['wkly_sentiment_change'].apply(lambda x: 1 if x>best_threshold_weekly else 0).copy()
    sentiment_publisher_1['weekly_profit'] = (sentiment_publisher_1['weekly_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()
    sentiment_publisher_1['biweekly_buy_sell'] = sentiment_publisher_1['2wk_sentiment_change'].apply(lambda x: 1 if x>best_threshold_biweekly else 0).copy()
    sentiment_publisher_1['biweekly_profit'] = (sentiment_publisher_1['biweekly_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()
    sentiment_publisher_1['monthly_buy_sell'] = sentiment_publisher_1['4wk_sentiment_change'].apply(lambda x: 1 if x>best_threshold_monthly else 0).copy()
    sentiment_publisher_1['monthly_profit'] = (sentiment_publisher_1['monthly_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()
    sentiment_publisher_1.dropna(inplace=True)
        
    weekly_profit = list(sentiment_publisher_1['weekly_profit'])
    for i in range(1,len(weekly_profit)):
        weekly_profit[i] = weekly_profit[i]+weekly_profit[i-1]        
    biweekly_profit = list(sentiment_publisher_1['biweekly_profit'])
    for i in range(1,len(biweekly_profit)):
        biweekly_profit[i] = biweekly_profit[i]+biweekly_profit[i-1]        
    monthly_profit = list(sentiment_publisher_1['monthly_profit'])
    for i in range(1,len(monthly_profit)):
        monthly_profit[i] = monthly_profit[i]+monthly_profit[i-1]
        
    # Get Cumulative Profit for LightGBM
    df_test['pred'] = model_output   
    pred_profit = []
    for i in range(len(df_test['pred'])):
        if (df_test['pred'].iloc[i]>0):
            pred_profit.append(df_test['target_daily'].iloc[i]*100)
        else:
            pred_profit.append(0)            
            
    # Get ideal profit for the same day if you bought $100 worth of bitcoin in one day and sold it exactly day after
    df_test['target_daily_profit'] = df_test['target_daily'].apply(lambda x: 100*x if x>0 else 0)  
    # Gain cumulative profit
    target_profit = list(df_test['target_daily_profit'])
    for i in range(1,len(target_profit)):
        target_profit[i] = (target_profit[i]+target_profit[i-1])
        pred_profit[i] = (pred_profit[i]+pred_profit[i-1])
    print("Profit : ${}".format(pred_profit[-1]))
    
    # Plot cumulative profit
    plt.figure(figsize=(20,8))
    sns.lineplot(x=df_test['date'],y=target_profit)
    sns.lineplot(x=df_test['date'],y=pred_profit)
    sns.lineplot(x=sentiment_publisher_1['date'],y=weekly_profit)
    sns.lineplot(x=sentiment_publisher_1['date'],y=biweekly_profit)
    sns.lineplot(x=sentiment_publisher_1['date'],y=monthly_profit)
   
    plt.axhline(y=0,color='r',linestyle ='-.' )  #vertical line. Linestype : ['-', '--', '-.', ':', '',]

    plt.xlabel('Date')
    plt.ylabel("Profit ($)")
    plt.title('Cumulative Profit')
    plt.legend(['Actual','Trading Bot','Weekly Sentiment MA','Bi-weekly Sentiment MA','Monthly Sentiment MA'])
    plt.grid()
    plt.show()
    
def custom_train_test_split(df, train_cols,date_gap, min_date,max_date,max_max_date):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    # 2018.6 Market stabilized 
    min_date = min_date
    max_date = max_date
    max_max_date = max_max_date
    
    mask_train = ((df['date']>min_date)  & (df['date']<(max_date- timedelta(days=date_gap))))
    mask_test = ((df['date'] > max_date) & (df['date'] < max_max_date))

    df_train = df[mask_train]
    df_test  = df[mask_test]

    print("Train and Test size", len(df_train), len(df_test))
    # scale the feature MinMax, build array
    x = df_train.loc[:,train_cols].values
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(df_test.loc[:,train_cols]) 
    return x_train, x_test, scaler

# TIME_STEPS is how many units back in time you want your network to see
def build_timeseries(mat,TIME_STEPS, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

# Small batch_size increase train time and too big batch size reduce model's ability to generalize, but quicker
def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat
# Use of return_sequences : https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
def create_model(lr):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, #return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.4))
#     lstm_model.add(LSTM(60, dropout=0.0))
#     lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(16,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=lr)
    #optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model