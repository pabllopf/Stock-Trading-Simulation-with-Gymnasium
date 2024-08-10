# Stock-Trading-Simulation-with-Gymnasium
Train a simple reinforcement learning agent in stock trading simulation.

# Description:
Financial markets are constantly changing and many companies now leverage AI in their attempts to beat the market. Have a try at it yourself as you train a simple reinforcement learning agent in Python using Gymnasium. You will then explore the decisions that the agent made and analyze its performance. Perfect for learners interested in finance and machine learning.

# Create and train a Proximal Policy Optimization (PPO) model using the provided environment. Train this model for a total_timesteps of 10000. Implement the trading logic to make buy, sell, or hold decisions based on the model's predictions.

Adjust the cash balance and shares_held based on the action taken (buy/sell) and allow the model to trade a certain percentage of its balance each step.
Monitor balance changes and store the history in balance_history for visualization.
If shares are still held at the end, sell them to update the final balance.
By the end of this project you should have two charts that display the performance of the model.

Chart 1 should show the stock price with buy/sell actions overlaid on the price.
Chart 2 should show the cash balance over time.
Questions:

What do you think about the performance of this model?
Why do you think it has performed the way it has?
How could the data we have selected impact the performance of the model?

![image](https://github.com/user-attachments/assets/6b5540ef-ade3-4a96-a59d-6cb7b3d16c52)


Your project is centered around developing a reinforcement learning (RL) simulation for stock trading in Python. This initiative is spearheaded by Quantum Trading, a fictional but ambitious trading firm looking to leverage cutting-edge machine learning techniques to gain a competitive edge in the financial markets. Quantum Trading is a small but highly specialised team of financial analysts, data scientists, and software engineers who are passionate about transforming the way trading decisions are made.

In the fast-paced world of financial markets, staying ahead of the curve is crucial. Traditional trading strategies, while effective, often rely on historical data and predefined rules that may not adapt quickly to changing market conditions. Reinforcement learning, a subfield of machine learning where an agent learns to make decisions by interacting with an environment, offers a promising alternative. It allows the trading algorithms to learn and adapt in real-time, improving their performance as they gain more experience.

By engaging with this project, you will gain valuable insights into the dynamic world of algorithmic trading and enhance your skill set in data science, finance, and machine learning. Remember, the journey of learning and experimentation is as important as the results. Good luck, and may your trading algorithms be ever profitable!

## The Data
The provided data `AAPL.csv` contains historical prices for AAPL (the ticker symbol for Apple Inc) and you will be using this in your model. It has been loaded for you already in the sample code below and contains two columns, described below.

| Column | Description |
|--------|-------------|
|`Date`    | The date corresponding to the closing price              |
|`Close`   | The closing price of the security on the given date      |

_**Disclaimer: This project is for educational purposes only. It is not financial advice, and should not be understood or construed as, financial advice.**_
