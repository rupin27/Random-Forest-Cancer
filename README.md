# Random-Forest-Cancer
This project implements the Random Forest algorithm from scratch, without using existing machine learning code. The implementation includes the following components:

1. Decision Tree Algorithm:
   - The standard Decision Tree algorithm is implemented based on the Information Gain splitting criterion.
   - The implementation supports both categorical and numerical attributes.
   - For numerical nodes, the splitting threshold can be determined using either the average of attribute values or a more sophisticated technique discussed in lecture 04 (slide #66).

2. Bootstrap Dataset Creation:
   - A procedure is implemented to create bootstrap datasets by sampling with replacement from the current training set.
   - These datasets are used to train each tree in the random forest.

3. Random Attribute Selection:
   - A mechanism is implemented to randomly select a subset of m attributes from the complete set of attributes for each node split.
   - The value of m is typically set to be approximately equal to the total number of attributes in each instance of the dataset.

4. Majority Voting Mechanism:
   - A majority voting mechanism is implemented to combine the predictions made by each tree in the ensemble.
   - This mechanism is used to classify new instances.

5. Stratified Cross-Validation:
   - The stratified cross-validation technique is implemented to evaluate the generalization capability of the random forest.
   - The recommended value for k (number of folds) is 10.
   - During each step of the cross-validation process, multiple decision trees (ntree) are trained and combined to form a random forest.
   - The performance of the random forest is evaluated using the remaining fold, not used during training, and the average performance across all folds is calculated.

6. Impact of Number of Trees:
   - The main objective of this project is to study the impact of the number of trees (ntree hyper-parameter) on the performance of the random forest.
   - Different stopping criteria can be used for constructing each tree in the ensemble:
     - Minimal size for split criterion: Stop splitting nodes when the data partition contains fewer than n instances.
     - Minimal gain criterion: Stop splitting nodes when the information gain is not sufficiently high.
     - Maximal depth stopping criterion: Stop splitting when the tree's depth becomes larger than a pre-defined value.
   - Various possibilities should be explored to identify the best combination of stopping criteria and hyper-parameters that yield the best performance for the algorithm.

## Datasets

This project analyzes two datasets:

1. The Wine Dataset:
   - Goal: Predict the type of a wine based on its chemical contents.
   - Number of instances: 178
   - Attributes: 13 numerical attributes
   - Classes: 3

2. The 1984 United States Congressional Voting Dataset:
   - Goal: Predict the party (Democrat or Republican) of each U.S. House of Representatives Congressperson.
   - Number of instances: 435
   - Attributes: 16 categorical attributes
   - Classes: 2

## Instructions

To run the Random Forest algorithm and evaluate its performance, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/rupin27/Random-Forest_Cancer.git
   cd Random-Forest-Cancer
   ```
2. Set up the environment:
- Ensure you have Python 3 installed.
- Create and activate a virtual environment (optional, but recommended):
   ```
   python3 -m venv env
   source env/bin/activate
   ```
3. Run the Random Forest algorithm:
- Open the main script file, e.g., random_forest.py, in your preferred code editor.
- Modify the hyper-parameters and stopping criteria as desired.
- Specify the dataset to use (e.g., Wine Dataset or Congressional Voting Dataset).
- Run the script to execute the Random Forest algorithm:
```
evaluationCalc.py
```
4. Analyze the results:
- The script will output the performance metrics and evaluation results for the random forest.
- Study the impact of the number of trees (ntree hyper-parameter) on the algorithm's performance.
- Experiment with different stopping criteria and hyper-parameters to optimize the performance. <br>
Note: It is recommended to explore the code files, such as decisionTree.py and randomForest.py, to gain a deeper understanding of the implementation details.
