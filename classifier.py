import xml.etree.ElementTree as ET
import os
import sklearn.preprocessing as PRE
import sklearn.metrics as M
import sklearn.pipeline as P
import sklearn.feature_extraction.text as T
import sklearn.tree as CF
import sklearn.cross_validation as CV 

# load data from DCAT-AP XML export into data and labels
def prepare():
	tree = ET.parse('datagovbe_edp.xml')
	root = tree.getroot()

	data = []
	labels = []
	unclassed = []

	for dataset in root.iter('{http://www.w3.org/ns/dcat#}Dataset'):
		about = dataset.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about']
		text = u""
		for title in dataset.iter('{http://purl.org/dc/terms/}title'):
			text = text + unicode(title.text) + "\n"
		for desc in dataset.iter('{http://purl.org/dc/terms/}description'):
			text = text + unicode(desc.text) + "\n"
		for keyword in dataset.iter('{http://www.w3.org/ns/dcat#}keyword'):
			text = text + unicode(keyword.text) + "\n"

		cat = False
		cats = []
		for themes in dataset.iter('{http://www.w3.org/ns/dcat#}theme'):
			theme = themes.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'].split("/")[-1]
			# ignore INSPIRE themes, we only use EU portal categories
			if len(theme) == 4:
				cats.append(theme)
				cat = True

		if cat == True:
			data.append(text)
			labels.append(cats)
		else:
			unclassed.append(text)

	return data, labels, unclassed


# train and test the data
def train_test(data, labels):
	X = data
	lb = PRE.MultiLabelBinarizer()
	Y = lb.fit_transform(labels)

	# split data into train and test
	train, test, train_target, test_target = CV.train_test_split(X, Y, train_size = 0.6)

	text_clf = P.Pipeline([('vect', T.CountVectorizer()),
					('tfidf', T.TfidfTransformer()),
					('clf', CF.DecisionTreeClassifier()),
	])
	text_clf = text_clf.fit(train, train_target)

	# flatten list and get unique labels
	target_names = sorted(list(set(x for l in labels for x in l)))
	predicted = text_clf.predict(test)

	# evaluation metrics	
	print(M.classification_report(test_target, predicted, target_names=target_names))
	return text_clf


data, labels, unclassed = prepare()
model = train_test(data, labels)
print(model.predict(unclassed))

