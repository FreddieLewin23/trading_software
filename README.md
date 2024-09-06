# trading_software
 This trading software using an API from Quant Insight to find trades and when to exit trades.
 Quant Insight create models for assets. These models are 'fed' directly the macro-economic data, so in theory react quicker to fundamental changes in the market, than the asset itself.
 each day each model is given an R-squared value (measure of model confidence, or the percentage change in the asset price that can be explained from the model), and an FVG (fair value gap) figure.

 If the Rsq is above 65, the model is defined to be in a macro regime. My software iterates over all US stocks and if the model is in a macro regime and the FVG is less than -1 sigma (i.e. 
 the model value is at least one standard deviation above the real value) then it considers that trade possible. Since the fair value gap exists, this means the model has reacted quicker to the change in 
 the macro environment, and a lag has been created, allowing for trade opportunities.

 once my code finds all of the model with Rsq > 65 and FVG < -1, it iterates over those again. I assign each trade a backtest score based off of the average returns and hit rate of similar trades, found on
 my own volatility adjusted backtest framework, and a price trend score based of model value gradients. The model value gradients have a 3, 10 and 30 day look back and I fit a linear regression to find the 
 model value gradient. Based off of the backtest score and price trend score I change the amount that is put into each trade. For exmaple if i find a trade and the price trends are looking positive and it
 backtests very well, then my software will put a larger buy on that trade. If it performs really badly on the backtest, my software will not enter that trade. 
 right now the software is either in the trade or not (or there are no rolling trade ends) so there is still potential for better returns.

order sizes are vol adjusted (higher vol, lower order sizes), and max leverage of the account is also leverage adjusted (qi vol indicator increasing  leads to a lower max leverage). It dynamically delta hedges
using Russell 3000 (tracking around 98% of investable equity in the US) by buying puts (max risk being the premimum paid). It only hedges positions whose RSq have dropped below 65 after entry (since the entry
req for RSq is 65, but it may drop after that). It theory this means I am not exposed to idiosyncratic shifts of these US equities and only exposed to the model/spots currently responding well with macro.
