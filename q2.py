from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import json 

docs = []
docsList = []
tfidf = {}
docsString = ""
corpus = {}
idf = {}	
queryDict = {}
simScores = {}

def processDocuments():
	file = open("data.txt","r");
	docsString = file.read()
	docs = docsString.split("\n")
	# for doc in docs:
	# 	tokenizer = RegexpTokenizer(r'\w+')
	# 	wordList = tokenizer.tokenize(doc)
	# 	docList = []
	# 	for word in wordList:
	# 		lemmaWord = lemmatize(word)
	# 		lemmaWord = lemmaWord.lower()
	# 		docList.append(lemmaWord)
	# 	docsList.append(docList)	
	# print(docsList[0])	
	docsList = docs
	return docsList

def lemmatize(word):
	lemmatizer = WordNetLemmatizer()
	wordLemma = lemmatizer.lemmatize(word)
	return wordLemma

def getQueryvector(queryList):
	queryVector = model.infer_vector(queryList)
	return queryVector	

def getQueries():
	file = open("test.jsonl", "r")
	jsonq = file.read()
	# print(jsonq)
	jsonList = jsonq.split("\n")
	# queryDict = {}
	# print(jsonList[0])
	for i, query in enumerate(jsonList):
		queryDict[i] = json.loads(query)	

def getSimDocsCount(maxAnswers, correctAnswer):
	maxSim = 0
	count = 0
	maxAnswers.sort(key = lambda x: x[1])
	
	maxAnswers = maxAnswers[::-1]
	
	score = 0
	if(correctAnswer==maxAnswers[0][0]):
		score = 1
		for answer in maxAnswers[1:]:
			if(correctAnswer==answer[0]):
				score+=1
	# print(score)
	
	# print(maxAnswers)
	# print(maxAnswer)	
	maxSim = maxAnswers[0][1]
	return maxSim,score		


# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docsList)]
# model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
tagDocsList = []
tags = []
docsList = []
docsList = processDocuments()
getQueries()
# print(docsList)

for i, doc in enumerate(docsList):
		docN = word_tokenize(doc.lower())
		docN = TaggedDocument(words=docN, tags= str(i))
		tagDocsList.append(docN)
		tags.append(str(i))

# print(tagDocsList)

model = Doc2Vec(vector_size=100,alpha=0.5, min_alpha=0.0001,min_count=1,dm =1)
model.build_vocab(tagDocsList)

# tagged_data = TaggedDocument(words=tagDocsList, tags= for i, _d in enumerate(data)]
model.train(tagDocsList,total_examples=model.corpus_count, epochs=model.iter)

vector = model.infer_vector(["what", "is", "your", "name"])
# vector1 = model.infer_vector(["system", "system"])
# print(similarity(vector, vector1))
similar_doc = model.docvecs.most_similar([vector])

# print(similar_doc)

correct = 0
totalScore = 0

for key in queryDict:
	maxSimilarity = 0
	maxAnswers = []
	question = queryDict[key]

	answers = question['question']['choices']
	query =  question['question']['stem']
	simScores[key] = {}
	simScores[key]['id'] = queryDict[key]['id']
	simScores[key]['choices'] = []
	# print(query)
	maxAnswer = " "
	for answer in answers:
		queryList = []
		queryAnswer = query + " " + answer['text']
		tokenizer = RegexpTokenizer(r'\w+')
		queryWords = tokenizer.tokenize(queryAnswer)

		for word in queryWords:
			queryList.append(lemmatize(word).lower())

		queryVector = getQueryvector(queryList)
		simDocs = model.docvecs.most_similar([queryVector])
		maxDoc = simDocs[0][0]
		maxAnswers.append((answer['label'],simDocs[0][1]))
		dikt = {}
		dikt['label'] = answer['label']
		dikt['similarity'] = simDocs[0][1]
		dikt['doc'] = maxDoc

	maxSimilarity, score = getSimDocsCount(maxAnswers, queryDict[key]['answerKey'])
		
	simScores[key]['choices'].append(dikt)
	if(score!=0):
		totalScore+=float(1/score)
	# print(maxSimilarity)
	# if(maxAnswer==queryDict[key]['answerKey']):
	# 	correct+=1	

print(totalScore/500)
# print(simScores)
