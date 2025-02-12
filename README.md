Fake News Detector
------------------------------------------------------------------------------------

Project Overview:

  This project aims to classify news articles as real or fake based on their content using machine learning. 
  The model is trained on a dataset of labeled news headlines, processed with natural language processing (NLP) techniques, and evaluated for accuracy.


Features:

- Preprocesses text data by removing punctuation, converting to lowercase, and eliminating numbers.

- Converts text into numerical representations using TF-IDF vectorization.

- Trains a Logistic Regression model for classification.

- Evaluates the model using accuracy and classification reports.

- Saves the trained model and vectorizer for future predictions.



Dataset:

  The dataset is stored in fake_news_dataset.csv and contains:

  text: The headline or snippet of a news article.

  label: 0 for real news and 1 for fake news.

Installation:
  Download the project and look at the text file "using_the_FND.txt" under the instruction folder.


Contributors:

- Uzair Azizuddin
- Firas Al Halaq
- Awab Elfadl


Future Goals: 
- User-Friendly Interface: Create an intuitive interface, such as a graphical user interface (GUI) or a web-based application, that allows users to easily interact with the model without needing to modify the code.

- Dataset Expansion: Add more diverse datasets to improve model accuracy and robustness. Continuously update the datasets with new data to ensure the model adapts to evolving trends in fake news.

- Enhanced Interactivity: Implement interactive features where users can input news headlines, see predictions, and understand the reasoning behind the model's decisions (e.g., using model explainability tools).

- Experiment with Additional Machine Learning Models: In the future, test and integrate different machine learning models (such as Random Forest, SVM, or deep learning approaches like LSTMs or transformers) to compare and potentially improve classification accuracy.




License:

This project is licensed under the MIT License.

