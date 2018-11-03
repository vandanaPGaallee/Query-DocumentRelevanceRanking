
# coding: utf-8

# In[144]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt


# In[155]:


#initializing all parameters
maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False


# In[146]:


def GetTargetVector(filePath): #extracting all the target data from the csv file and return target vector
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

def GenerateRawData(filePath, IsSynthetic):  #read the input 65000 data with 46 features from the csv file 
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False : #IsSynthetic is to eliminate all same value features because it does not affect the model and also to eliminate 0 value for covariance
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)
    dataMatrix = np.transpose(dataMatrix)  #   
    #print ("Data Matrix Generated..")
    return dataMatrix

#store the 80% of the training target data which is 55699
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): 
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

#store the 80% of the training  data which is 55699
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent)) # this is computing the column lenth 0 to 55699
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

#store the 10% of the validation data and testing data which is 6962 which is 41 * 6962
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix


#store the 10% of the validation target and testing target which is 6962 which is 41 * 6962
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

#Generates covariance for 41 features accross the training data
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data))) #41 * 41
    DataT       = np.transpose(Data) #65000 * 41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))  #55699   
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])   #storing 55699 for each feature in vct 
        varVect.append(np.var(vct))  #computing variance for 55699 values for each feature and storing in varVect
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j] #storing the variance for 41 features across the diagnol in 41 * 41 matrix
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

#computes the scalar value for phi design matrix
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow) # computes x - u where x is 41 features and muRow is the 41 mean values for each of feature
    T = np.dot(BigSigInv,np.transpose(R))  # computes dot product of 41 Inverse covariance and R 
    L = np.dot(R,T)# returns a scalar value
    return L

# computes the values of gaussian radial basis scalar value for each of the input
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv)) #computes the exponential of scalar value
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data) # Transposing 41 * 65000  to 65000 * 41 
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))  # 80% 55699     
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) # 55699 * 10 where 10 is number of clusters 
    BigSigInv = np.linalg.inv(BigSigma) # computes inverse of covariance matrix
    for  C in range(0,len(MuMatrix)): # 0 t0 10
        for R in range(0,int(TrainingLen)): # 0 to 55699
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv) # computes the phi(x) for each cluster
    #print ("PHI Generated..")
    return PHI

#computing weight matrix from phi vector
def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0])) # Regularization added to the weight matrix to balance the wight values to create a smooth curve
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda # identity matrix * lamda
    PHI_T       = np.transpose(PHI) # transpose of phi
    PHI_SQR     = np.dot(PHI_T,PHI) # phi * phi_T
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR) # phi * phi_T + lamda
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI) # inverse of PHI_SQR_LI
    INTER       = np.dot(PHI_SQR_INV, PHI_T) 
    W           = np.dot(INTER, T) # closed form solution with least squared regularization
    ##print ("Training Weights Generated..")
    return W

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI)) # compute the linear regression function y(x,w)
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)): #computing the root mean squared error
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2) # summation of squares of error
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]): # classifying the regression output to three ranks 0,1,2 by rounding the y value to nearest even number
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) #computes the ratio of correct prediction to total input
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT)))) #accuracy  and root mean squared error


# ## Fetch and Prepare Dataset

# In[147]:


RawTarget = GetTargetVector('Querylevelnorm_t.csv') #Extract the da
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)


# ## Prepare Training Data

# In[148]:


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[149]:


ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[150]:


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]

# In[151]:


ErmsArr = []
AccuracyArr = []

#this step is a optimazation technique to reduce the dimentionality
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData)) #Here we define the cluster size as 10 and random state to take random centroids initially.
Mu = kmeans.cluster_centers_ #It takes 55699 * 41 values and reduces it to 10 clusters and returns 10 * 41 values where the each of the 41 features represent the average value in each cluster

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[152]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set 

# In[153]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[154]:


print ('UBITname      = vandanap')
print ('Person Number (= 50289877')
print ('--------------)--------------------------------------')
print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = %s \nLambda = %s" %(M, C_Lambda))
print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("Accuracy Validation = " + str(float(ValidationAccuracy.split(',')[0])))
print ("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0])))
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


# ## Gradient Descent solution for Linear Regression

# In[120]:


print ('----------------------------------------------------')
print ('--------------Please Wait for 2 mins!----------------')
print ('----------------------------------------------------')


# In[121]:


W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.1
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,100): 
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i]) # Computing Delta E_D which is the rate of change of error with respect to w 
    La_Delta_E_W  = np.dot(La,W_Now) # Error regularization
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  # adding regularization to gradient error
    Delta_W       = -np.dot(learningRate,Delta_E) # multipying learning rate to computed error
    W_T_Next      = W_Now + Delta_W # subtracting error from output
    W_Now         = W_T_Next # updating the weight
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[ ]:


print ('----------Gradient Descent Solution--------------------')
print('learning rate %s' % learningRate)
print('learning rate %s' % la)
print ("Accuracy Training   = " + str(float(Erms_TR.split(',')[0])))
print ("Accuracy Validation = " + str(float(Erms_Val.split(',')[0])))
print ("Accuracy Testing    = " + str(float(Erms_Test.split(',')[0])))
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

