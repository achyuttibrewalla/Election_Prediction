import pandas as pd 
import numpy as np
import math
import time
import random
from copy import deepcopy
#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas


#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

def preprocess(panda):
        df1 = encodeData(panda, ['can_off', 'can_inc_cha_ope_sea'])
        dff = normalizeData(df1,["net_ope_exp","tot_loa", "net_con" ])
        f, l = getNumpy(dff)
        return (f,l)

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / labels.size



#===========================================================================================================
    
class KNN:
     
    def __init__(self):
		#KNN state here
		#Feel free to add methods
         self.k = 5
         self.d = {}

    def train(self,features, labels):
        
         print("\n")
         print ("KNN Training...")
         print ("...")
		
         #Training model
         #storing the dataset in dictionary for faster processing
         for i in range(len(labels)):
             if labels[i] in self.d:
                 self.d[labels[i]].append(features[i])
             else:
                 self.d[labels[i]]= []
                 self.d[labels[i]].append(features[i])

    def predict(self,features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
         print ("Predicting Labels for Testing Set")
         label = self.eucledianDistance(features)
        
         return label
     
    
    def eucledianDistance(self, features):
        testingSetLabels=[]
        for i in range(len(features)):
             eucledian=[]
             for labels in self.d:
                 for feat in self.d[labels]:
                     dist=0
                     for j in range(len(feat)):
                         dist += ((feat[j]-features[i][j])**2) 
                     dist = math.sqrt(dist)
                     eucledian.append((dist, labels))
                     
             testingSetLabels.append(self.outputLabel(eucledian))
             
        return testingSetLabels
    
    def outputLabel(self, edist):
        #function to 
        edist = sorted(edist, key=lambda x:x[0])[:self.k]
        countLabelZero = 0
        countLabelOne = 0
        for ele in edist:
            if ele[1] == 0:
                countLabelZero = countLabelZero + 1
            else:
                countLabelOne = countLabelOne + 1
             
        if countLabelZero > countLabelOne:
            return False
        else:
            return True
        
#===========================================================================================================
            
class Perceptron:
    def __init__(self):
		
        self.weight = []

    def train(self,features, labels):
        print("\n")
        print ("Training Perceptron...")
        print ("...")
        learningRate = 0.01
        labels = np.array(labels)
        labels[labels == 0] = -1
        lengthFeatures = len(features[0])     #number of attributes
        feat = np.array(features)              
        self.weight = np.random.uniform(0.0,1.0, lengthFeatures+1)
        feat = np.insert(feat,9,1, axis=1)             #appending bias
        t_end = time.time() + 60
        
        while time.time() < t_end:
        
            for i in range(len(feat)):
                output = self.dotProduct(feat[i],self.weight)
                if output != labels[i]:
                     self.weight += learningRate * labels[i] * feat[i]
                     
        print ("Training Completed for Perceptron!")

    def dotProduct(self, f, w):     
        product = np.dot(f, w)
        return self.activate(product)
    
    def activate(self,x):
        value = 1 / (1 + math.exp(-x))
        if value > 0.5:
            return 1
        else:
            return -1
        
#        if x>0:
#            return 1
#        else:
#            return -1
        
     
    def predict(self,features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
        print("Predicting Labels for Testing Data Set")
        testingSetLabels = []
        features = np.insert(features,9,1, axis=1)
        
        for i in range(len(features)):
            output = self.dotProduct(features[i], self.weight)
            if output == 1:
                testingSetLabels.append(True)
            else:
                testingSetLabels.append(False)
                
        return (testingSetLabels)
    
#===========================================================================================================
    
class MLP:
    def __init__(self):
		
        self.weight = []
        self.hiddenweight = []
        self.learningRate = 0.01

    def train(self,features, labels):
        
        print("\n")
        print ("Training Multi Layer Perceptron...")
        print ("...")
		
        hiddenNodes = 10
        labels = np.array(labels)
        labels[labels == 0] = -1
        inputLength = len(features[0])
        feat = np.array(features)              

        feat = np.insert(feat,9,1, axis=1)             #appending bias
        t_end = time.time() + 60                       #time function

        for i in range(hiddenNodes):
                self.weight.append(np.random.uniform(-1.0,1.0, inputLength+1))

        self.hiddenweight = np.random.uniform(-1.0, 1.0, hiddenNodes+1)
        
        Nodes = []
        deltaNodes = []
       
        while time.time() < t_end:
            
            for i in range(len(feat)):
                Nodes=[]
                for j in range(len(self.weight)):
                    output = np.dot(feat[i],self.weight[j])
                    deltaNodes.append(self.dactivate(output))
                    output = self.activate(output)
                    Nodes.append(output)
                Nodes.append(1)     #appending bias
                Nodes = np.array(Nodes)
                
                
                output = np.dot(Nodes,self.hiddenweight)
                delta_output = self.dactivate(output) * (output - labels[i])
                
                for k in range(len(Nodes)-1):
                    sum = 0
                    for w in (self.hiddenweight):
                        sum = sum + (w * delta_output)
                    deltaNodes[k] = deltaNodes[k] * sum
                    
                for l in range(len(self.hiddenweight)):
                    self.hiddenweight[l] -= (self.learningRate * (delta_output * Nodes[l]))
                    
                for m in range(hiddenNodes):
                    for n in range(inputLength):
                        self.weight[m][n] -= (self.learningRate * deltaNodes[m] * feat[i][n])
                        
        print ("Training Completed for Multi Layer Perceptron!")
        
    def activate(self,x):
        """
        Function to return the value after applying sigmoid function
        """
        return 1 / (1 + np.exp(-x))
    
    def dactivate(self, y):
        """
        Function to calculate the derivative of the activation function
        """
        return self.activate(y) * (1-self.activate(y))
    
    def oactivate(self,x):
        """
        Function for thresholding sigmoid function value
        """
        value = 1 / (1 + math.exp(-x))
        if value > 0.5:
            return 1
        else:
            return -1
     
    def predict(self,features):
        testingSetLabels = []
        features = np.insert(features,9,1, axis=1)

        for i in range(len(features)):
                Nodes=[]
                for j in range(len(self.weight)):
                    output = np.dot(features[i],self.weight[j])
                    output = self.activate(output)
                    Nodes.append(output)
                Nodes.append(1)
                Nodes = np.array(Nodes)
                
                output = np.dot(Nodes,self.hiddenweight)
                output = self.oactivate(output)
                
                if output == 1:
                    testingSetLabels.append(True)
                else:
                    testingSetLabels.append(False)
                
        return (testingSetLabels)
    
#===========================================================================================================
        
class Node(object):
    def __init__(self, name, edge =[], examples=[]): #constructor
        self.child = []
        self.name = name
        self.edge= edge
        self.examples = examples

    def addChild(self,obj):       #function to add child node to the parent node
        self.child.append(obj)
        
        

class ID3:
    def __init__(self):
        
         self.data_set = None
         self.root = None            #r is the root node of the tree
         
    def preprocess(self, panda):
        
        training, testing = trainingTestData(panda, 0.75)
        
        training = normalizeData(training,["net_ope_exp","tot_loa", "net_con" ])
        testing = normalizeData(testing,["net_ope_exp","tot_loa", "net_con" ])
        
        self.testing_set= testing
        
        f, l = getNumpy(training)
        
        return (f,l)
    
    def train(self, features, labels):
        
        print("\n")
        print ("Creating Decision Tree")
        print ("...")
        
        self.data_set = pd.DataFrame(features)
        length = len(self.data_set.columns) 
        ndf = pd.Series(labels)
        self.data_set = pd.concat([self.data_set, ndf.rename(length)], axis =1)

        column_names =  (self.data_set.columns.get_values()).tolist()
        column_names = column_names[:len(column_names)-1]

        
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        group_names = ['Low', 'Okay', 'Mid', 'Good', 'Great']
     
        self.data_set[0] = pd.cut(self.data_set[0], bins, include_lowest= True, labels=group_names)
        self.data_set[1] = pd.cut(self.data_set[1], bins, include_lowest= True, labels=group_names)
        self.data_set[2] = pd.cut(self.data_set[2], bins, include_lowest= True, labels=group_names)
        
        del self.data_set[5]
      
        features = self.data_set.values.tolist()

        labels = labels.tolist()

        self.root = self.decision_tree_learning((features, labels), column_names,(features, labels) )
        
        print ("Decision Tree Created! ")
        return self.root
        
        
    def decision_tree_learning(self, examples, attributes, parent_examples):

        feature, label = examples
        features = feature[:]
        labels = label[:]
#        attributes = attribute[:]

        if (len(features) == 0):
            pv = self.pluralityValue(parent_examples)
            return pv
        
        elif (type(self.getClassification(features, labels)) != int):
            return self.getClassification(features, labels)
        
        elif len(attributes) == 0:
            return self.pluralityValue(examples)
        
        else:
            A = []
            for at in attributes:
                A.append(self.infoGain(at, features, labels))

            a = (max (A))[1]
            attributes.remove(a)
            root1 = Node(a, list(pd.unique(self.data_set[a])), features)
            for v in root1.edge:
                exs = self.getExamples((features,labels), v, a)
                obj = self.decision_tree_learning(exs,attributes , (features, labels))
                root1.addChild(obj)
                
            return root1
        
    def getExamples(self, feat, v, a):
        ex , lab = feat
        l = [] #exampls to be returned
        l1 = []
        for i in range(len(ex)):
            ele = ex[i][a]
            if ele == v:
                l.append(ex[i])
                l1.append(lab[i])
        return (l, l1)
        
    def getClassification(self,features, labels):
        countZero = 0
        countOne = 1
        l = len(features)
        label = labels[:l]
        for ele in label:
            if ele == 0:
                countZero += 1
            else:
                countOne += 1
                
        if countZero == len(label):
            return Node("False")
        
        elif countOne == len(label):
            return Node("True")
        
        else:
            return -1
        
    def pluralityValue(self,examples):
        count0 = 0
        count1 = 0
        f , l = examples
        for ele in l:
            if ele == 0:
                count0 += 1
            else:
                count1 += 1
                
        if count0 > count1:
            return Node("False")
        else:
            return Node("True")
        
    
        
    def infoGain(self,attr, data, target_attr):
        """
        Calculates the information gain (reduction in entropy) that would
        result by splitting the data on the chosen attribute (attr).
        """
        remainder = 0
        p = 0
        ent = 0
        for ele in target_attr:
            if ele == 1:
                p +=1
                
        q = p / (len(target_attr)) 
        if 0 < q < 1:
            ent = -((q * math.log2(q)) + ((1-q) * math.log2(1-q))) 
            
        unique = list(pd.unique(self.data_set[attr])) 
        l = self.data_set[attr]
        for ele in unique:
            pk =0
            nk=0
            j=0
            for i in range (0, len(data)):          #len (l) changed to len(data)
                j = j+1
                ele1 = l[i]
                if ele1 == ele:
                    out = target_attr[i]
                    if out == 1:
                        pk += 1
                    else:
                        nk += 1
            if (pk+nk) != 0:
                q1 = pk / (pk +nk)
                if 0 < q1 < 1:
                    e = -((q1 * math.log2(q1)) + ((1-q1) * math.log2(1-q1)))
                    remainder += (pk + nk)/(len(target_attr)) * e
            
        return (ent - remainder, attr)
    
    def predict(self, features):
        
        print ("Predicting Labels for Testing Set")
        
        predict_dataset = pd.DataFrame(features)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        group_names = ['Low', 'Okay', 'Mid', 'Good', 'Great']
        
        predict_dataset[0] = pd.cut(predict_dataset[0], bins, include_lowest= True, labels=group_names)
        predict_dataset[1] = pd.cut(predict_dataset[1], bins, include_lowest= True, labels=group_names)
        predict_dataset[2] = pd.cut(predict_dataset[2], bins, include_lowest= True, labels=group_names)
        
        features = predict_dataset.values.tolist()
        
        testingSetLabels =[]
        
        r1 = deepcopy(self.root)
        
        for i in range(len(features)):
            traverse_output = self.traverse(r1, features[i])
            if traverse_output == "True":    
                testingSetLabels.append(1)
            else:
                testingSetLabels.append(0)
            
        return testingSetLabels
        
        
        
    def traverse(self, node, feature):
        if (node.name == "True") or (node.name == "False"):
            return node.name
        else:
            n = node.name
            p = feature[n]
            if p in node.edge:
                pos = node.edge.index(p)
                ob = node.child[pos]
                return self.traverse(ob, feature)
            else:
                return random.choice(["True", "False"])
      
        
dataset_from_csv = pd.read_csv("data.csv")  
print (dataset_from_csv)   

train_dataset, test_dataset = trainingTestData(dataset_from_csv, 0.75)

train_features, train_labels = preprocess(train_dataset)
print (train_labels)
test_features, test_labels = preprocess(test_dataset)
#
kNN = KNN()
kNN.train(train_features, train_labels)
predictions = kNN.predict(test_features)
accuracy = evaluate(predictions, test_labels)  
print ("KNN Accuracy =", accuracy)

perceptron = Perceptron()
perceptron.train(train_features, train_labels)
predictions = perceptron.predict(test_features)
accuracy = evaluate(predictions, test_labels)  
print ("Perceptron Accuracy =", accuracy)

mlp = MLP()
mlp.train(train_features, train_labels)
predictions = mlp.predict(test_features)
accuracy = evaluate(predictions, test_labels)  
print ("Multi Layer Perceptron Accuracy =", accuracy)



# For Decision Tree the preprocess() function is inside the class ID3 itself, 
# since I needed the object on which preprocess is called.

id3 = ID3()
train_dataset_features, train_dataset_labels = id3.preprocess(dataset_from_csv)
ou = id3.train(train_dataset_features, train_dataset_labels)

testing_dataset_features, testing_dataset_labels = getNumpy(id3.testing_set)
classification = id3.predict(testing_dataset_features)       
 
accuracy = evaluate(classification, testing_dataset_labels)   
print ("Decision Tree Accuracy =", accuracy)




























    
        

