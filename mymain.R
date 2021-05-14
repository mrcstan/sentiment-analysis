library(text2vec)
library(glmnet)
library(pROC)

#####################################
# Load your vocabulary and training data
#####################################
myvocab <- scan(file = "myvocab.txt", what = character())
train <- read.table("train.tsv", stringsAsFactors = FALSE,
                     header = TRUE)


#####################################
#
# Train a binary classification model
#
#####################################
train$review <- gsub('<.*?>', ' ', train$review)
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)

vectorizer = vocab_vectorizer(create_vocabulary(myvocab, 
                                                ngram = c(1L, 4L)))
dtm_train = create_dtm(it_train, vectorizer)

# Logistic regression with ridge penalty
mylogit.fit = glmnet(x = dtm_train, 
                     y = train$sentiment, 
                     alpha = 0,
                     lambda = 0.07, 
                     family='binomial')


test <- read.table("test.tsv", stringsAsFactors = FALSE,
                    header = TRUE)
#####################################
# Compute prediction 
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predited probabilities
#####################################
test$review <- gsub('<.*?>', ' ', test$review)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)

pred = predict(mylogit.fit, dtm_test, type = "response")
output = data.frame(id = test$id, prob = as.vector(pred))

write.table(output, file = "my_predictions.txt", 
            row.names = FALSE, sep='\t')

