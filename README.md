# TSAI_Session_5_Assignment

This assignement is about building an LSTM network for sentiment analysis on  StanfordSentimentAnalysis Dataset. (http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) which contains movie reviews and sentiment lables which can be considered into 5 labels.
This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes.

Target is to achieve 60% + validation accuracy, for this we have done Data Augmentation using google translation, random delete, and random swap methods.

The dataset is loaded from the text files to dataframes, following files were used: dictionary.txt, datasetSplit.txt, sentiment_labels.txt and datasetSentences.txt
These files were uploaded to Colab and used pandas to read and create dataframes.

Then we have used sqllite to create tables for these datafrmaes in memory and inserted the data to respective tables.
Using cursor having join on these tables such that sentences are grouped with sentiment marking and datasplit columns, this was achieved using join on sentence and phrases, as sentences cosists of phrases.

Next we have used data augmentation technique called oogle translation, random delete, and random swap methods.
In google translation method we have certain limitation on the number of request sent for translation, if request exceeds that then it simply returns the output which is same as input.
The random deletion method takes a sentence splits it to words and deletes charaters randomly from the words, then we join back the words and reconstructe sentence using concatination.
The random swap method takes random sentences and swap them.
Using these methods we tried to rebalance the dataset and created final train dataset.

Using this train dataset we trained our LSTM network, having hyperparameters as :
     embedding_dim = 200
     num_hidden_nodes = 300
     num_output_nodes = 6
     num_layers = 5
     dropout = 0.7

Used Adam optimizer with learning rate as lr=2e-4
After 10 epochs of run following is the train and validation accuracy:

![alt text](https://github.com/sunny9sinha/TSAI_Session_5/blob/main/Images/trainvalidaccuracy.png)

As we can see that accuracy in validation is not even near to the target(60%+) , this may have to be because of very small validation dataset size
or network structure or any hyperparameter.

With the above training we tried following ten sentences to get the sentiment rating, these sentences were selected from the validation dataset:

![alt text](https://github.com/sunny9sinha/TSAI_Session_5/blob/main/Images/sentences_1.png)

![alt text](https://github.com/sunny9sinha/TSAI_Session_5/blob/main/Images/sentences_2.png)

We will try to improve this model in future.

Note: Taken help from following reference for sqlite code:
https://gist.github.com/soaxelbrooke/226480b00a86dad2e6f8bbb2b28c07a2
