########################
#                      #
#    data importing    #
#                      #
########################

#Importing train set
library(readr)
train = read_delim("path/train.txt", 
                    "\t", escape_double = FALSE, col_names = FALSE, 
                    trim_ws = TRUE)
colnames(train) = c("type","text")  #Rename column names

#Importing test set
test = read_delim("path/test.txt", 
                   "\t", escape_double = FALSE, col_names = FALSE, 
                   col_types = cols(X1 = col_integer()), 
                   trim_ws = TRUE)
colnames(test) = c("type","text") #Rename column names


########################
#                      #
#   data processing    #
#                      #
########################

library(NLP)
library(tm)
library(SnowballC)  #Extract words
library(slam)

train = train[order(train$type,decreasing = T),]  #Ordering data so that message 1-673 are spam 674-5000 are ham

#Creating Corpus 
CorpusPreProcessor = function(content){ 
  #Build a new corpus variable
  contentCorpus = Corpus(VectorSource(content))
  
  #Create plain text documents
  contentCorpus = tm_map(contentCorpus, PlainTextDocument)
  
  #Convert the text to lowercase
  contentCorpus = tm_map(contentCorpus, tolower)  
  
  #Remove all numbers
  contentCorpus = tm_map(contentCorpus, removeNumbers) 
  
  #Remove all English stopwords from the corpus
  contentCorpus = tm_map(contentCorpus, removeWords,stopwords("english"))
  
  #Remove all punctuation from the corpus
  contentCorpus = tm_map(contentCorpus, removePunctuation)
  
  #Strip extra whitespace from a text document
  contentCorpus = tm_map(contentCorpus, stripWhitespace)
  return(contentCorpus)
}

train.corpus = CorpusPreProcessor(train$text)
train.dtm = DocumentTermMatrix(train.corpus)  #Build a document term matrix from the train.corpus
test.corpus = CorpusPreProcessor(test$text)
test.dtm = DocumentTermMatrix(test.corpus)  #Build a document term matrix from the test.corpus


########################
#                      #
# Dimension Reduction  #
#                      #
########################

Dictionary = findFreqTerms(train.dtm,100) #Find out words whose frequency large than 100

train.selected = DocumentTermMatrix(train.corpus,list(dictionary=Dictionary)) #Process train.corpus with selected words
test.selected = DocumentTermMatrix(test.corpus,list(dictionary=Dictionary)) #Process test.corpus with selected words


########################
#                      #
#    Model Training    #
#                      #
########################

#A Convertor which turns 0 and 1 to factor "No" and "Yes"
convertor = function(x){
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels=c(0,1), labels=c("No","Yes"))
  return(x)
}

train.selected = apply(train.selected, MARGIN=2, convertor) #Make 0 & 1 to "No" & "Yes" in train.selected
test.selected = apply(test.selected, MARGIN = 2, convertor) #Make 0 & 1 to "No" & "Yes" in test.selected

#Training a model with NaiveBayes
library(e1071)
train.type = c(rep("spam",673),rep("ham",4327)) #Responses in train set
train.type = as.data.frame(train.type) 

model = naiveBayes(train.selected, train.type$train.type, laplace=1)  #Model 
Labels = predict(model, test.selected, type = "class")  #Predicted y label
Probabilities = predict(model, test.selected, type = "raw") #Probability of being spam


########################
#                      #
#   Result Combining   #
#                      #
########################

Submission = as.data.frame(cbind(seq(from = 1, to = 574, by = 1),Probabilities[,2],Labels))
colnames(Submission) = c("No.","Probability of Being Spam","Predicted Label")
Submission$`Predicted Label` = ifelse(Submission$`Predicted Label`==1,"Ham","Spam")
write.csv(Submission,file = "path",row.names = F)


