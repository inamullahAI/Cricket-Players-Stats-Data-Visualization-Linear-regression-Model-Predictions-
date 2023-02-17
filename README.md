# Cricket Players' Stats Data Visualization & prediction of player's Performance 

This is a project that visualizes the performance statistics of top cricket players using Python libraries. The data used in this project is obtained from Cricinfo Statsguru.

Data
The data set used in this project contains the statistics of the top 5 batsmen in Test cricket, as of September 2021. The data includes the following fields:

Player: name of the player
Span: span of the player's career
Mat: number of matches played
Inns: number of innings batted
NO: number of times not out
Runs: total number of runs scored
HS: highest score in a single innings
Ave: average number of runs per dismissal
BF: number of balls faced
SR: strike rate (runs per 100 balls)
100: number of centuries scored
50: number of half centuries scored
0: number of times the player was dismissed without scoring
4s: number of fours scored
6s: number of sixes scored
Libraries used
The following Python libraries were used in this project:

pandas: for data manipulation and analysis
matplotlib: for data visualization using line charts and scatter plots
seaborn: for data visualization using heatmaps and bar charts
plotly.express: for data visualization using heatmaps and bar charts

Visualizations
The following visualizations were created using the data set:

A line chart showing the total number of runs scored by each player over the course of their career.
A scatter plot showing the relationship between the average number of runs scored per dismissal and the strike rate.
A heatmap showing the frequency distribution of the number of centuries and half centuries scored by each player.
A bar chart showing the total number of fours and sixes scored by each player.
Running the code
To run the code, you will need to have Python 3 and the required libraries installed on your machine. You can install the required libraries using the following command:

Copy code
pip install matplotlib seaborn
pip install plotly.express
Once you have installed the libraries, you can run the code by executing the main.py file. The output visualizations will be saved to the output/ directory.

#Machine Learning Model

In addition to visualizing the data using Python, we can also build a machine learning model to predict a player's total runs based on their innings, number of times not out, balls faced, 4s, and 6s.

We use the scikit-learn library to train a linear regression model on the data. The code for the machine learning model can be found in the machine_learning.py file. The key steps are as follows:

Load the data using pandas: data = pd.read_csv('cricket_players_stats.csv')
Select the features and target variable: X = data[['Inns', 'NO', 'BF', '4s', '6s']] and y = data['Runs']
Split the data into training and testing sets: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Train a linear regression model: model = LinearRegression(); model.fit(X_train, y_train)
Make predictions on the test set: y_pred = model.predict(X_test)
Evaluate the model using mean squared error: mse = mean_squared_error(y_test, y_pred)
In this case, the mean squared error is used to evaluate the performance of the model. Other performance metrics could be used depending on the specific problem being solved.

It's important to note that this is just one example of a machine learning model that could be trained on the data. Depending on the problem you're trying to solve, there may be other models that are more appropriate, such as decision trees, random forests, or neural networks. Additionally, you may want to perform feature engineering or use more sophisticated techniques like cross-validation to ensure that your model is robust and accurate.




License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code as long as you give credit to the original author.

