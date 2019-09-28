def graph_sentiment_profit_gif(publisher,positive_sentiment_threshold,min_date,iteration):
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
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)


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
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=vader_profit[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=txtblob_profit[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_profit[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_change_profit[:10*iteration])
        plt.axhline(y=0,color='r',linestyle ='-.' )  #vertical line. Linestype : ['-', '--', '-.', ':', '',]

        plt.ylim(-350,30)
        plt.xlim(min(sentiment_publisher_1['date']), max(sentiment_publisher_1['date']))
        plt.xlabel('Date')
        plt.ylabel('Profit ($)')
        plt.title('Cumulative Profit ({}) - Pos Threshold = {}'.format(publisher,positive_sentiment_threshold))
        plt.legend(['Vader','TextBlob','Vader+TextBlob','Vader+TextBlob (*Change in Sentiment)'],loc='lower left')
        #plt.xticks(sentiment_publisher_1['date'],rotation='vertical')
        plt.grid()
        filename='gif/Cointelegraph/Gapminder_step'+str(iteration)+'.png'
        plt.savefig(filename)
        plt.close()
        
    else:
        # Generate DataFrame for daily sentiment for each publisher
        sentiment_by_publisher = df_news_sentiment.groupby(by=['date'],as_index=False).mean()

        sentiment_publisher_1 = sentiment_by_publisher[sentiment_by_publisher['date']>min_date]
        sentiment_publisher_1 = sentiment_publisher_1.merge(df_daily[['date','daily_sentiment_change','wkly_sentiment_change','2wk_sentiment_change','4wk_sentiment_change','target_daily']], left_on='date', right_on='date')

        sentiment_publisher_1['compound_buy_sell'] = sentiment_publisher_1['final_sentiment'].apply(lambda x: 1 if x>0.236 else 0).copy()
        sentiment_publisher_1['compound_profit'] = (sentiment_publisher_1['compound_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell'] = sentiment_publisher_1['daily_sentiment_change'].apply(lambda x: 1 if x>0.219 else 0).copy()
        sentiment_publisher_1['compound_change_profit'] = (sentiment_publisher_1['compound_change_buy_sell']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_wkly'] = sentiment_publisher_1['wkly_sentiment_change'].apply(lambda x: 1 if x>0.331 else 0).copy()
        sentiment_publisher_1['compound_change_profit_wkly'] = (sentiment_publisher_1['compound_change_buy_sell_wkly']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_2wk'] = sentiment_publisher_1['2wk_sentiment_change'].apply(lambda x: 1 if x>0.073 else 0).copy()
        sentiment_publisher_1['compound_change_profit_2wk'] = (sentiment_publisher_1['compound_change_buy_sell_2wk']*100*sentiment_publisher_1['target_daily']).copy()

        sentiment_publisher_1['compound_change_buy_sell_4wk'] = sentiment_publisher_1['4wk_sentiment_change'].apply(lambda x: 1 if x>0.027 else 0).copy()
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

        plt.figure(figsize=(20,8))
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_profit[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_change_profit[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_change_profit_wkly[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_change_profit_2wk[:10*iteration])
        sns.lineplot(x=sentiment_publisher_1['date'][:10*iteration],y=compound_change_profit_montly[:10*iteration])
        plt.axhline(y=0,color='r',linestyle ='-.' )  #vertical line. Linestype : ['-', '--', '-.', ':', '',]

        
        plt.ylim(-150,100)
        plt.xlim(min(sentiment_publisher_1['date']), max(sentiment_publisher_1['date']))
        plt.xlabel('Date')
        plt.ylabel("Trading Bot's Profit ($)")
        plt.title('Cumulative Profit'.format(positive_sentiment_threshold))
        plt.legend(['Vader+TextBlob Threshold = 0.236 (*Sentiment - Daily)',
                    'Vader+TextBlob Threshold = 0.219 (*Change in Sentiment - Daily)',
                    'Vader+TextBlob Threshold = 0.331 (*Change in Sentiment - Weekly MA)',
                    'Vader+TextBlob Threshold = 0.073 (*Change in Sentiment - 2 Weeks MA)',
                    'Vader+TextBlob Threshold = 0.027 (*Change in Sentiment - Monthly MA)'],loc='lower left')
        #plt.xticks(sentiment_publisher_1['date'],rotation='vertical')
        plt.grid()
        filename='gif/Cointelegraph/Gapminder_step'+str(iteration)+'.png'
        plt.savefig(filename)
        plt.close()
