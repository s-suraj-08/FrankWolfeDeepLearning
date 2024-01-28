from train_test.train import training
from train_test.test import testing
import torch

def run1_1(glob,l_vals,device,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'sgd',
            LOSS = 'ce',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 0,
            EARLYSTOP = False,
            AUGMENT = False,
            DROPOUT = False,
            DROP_PERCENT = 0.4,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run1_2(glob,l_vals,device,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'sgd',
            LOSS = 'ce',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-3,
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 0,
            EARLYSTOP = False,
            AUGMENT = False,
            DROPOUT = False,
            DROP_PERCENT = 0.4,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc


def run2_1(glob,l_vals,device,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = True,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 0,
            EARLYSTOP = False,
            AUGMENT = False,
            DROPOUT = False,
            DROP_PERCENT = 0.4,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run2_2(glob,l_vals,device,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = True,
            EPOCHS = 50,
            LEARNING_RATE = 1e-3,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 0,
            EARLYSTOP = False,
            AUGMENT = False,
            DROPOUT = False,
            DROP_PERCENT = 0.4,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run3(glob,l_vals,device,reg,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'sgd',
            LOSS = 'ce',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = reg,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run4(glob,l_vals,device,reg,self):

    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = True,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = reg,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc


def run5(glob,l_vals,device,reg,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'sgd',
            LOSS = 'ce',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L1',
            REGPARAM = reg,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run6(glob,l_vals,device,reg,self):

    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = True,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L1',
            REGPARAM = reg,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run7(glob,l_vals,device,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "pretrained",
            OPT = 'sgd',
            LOSS = 'ce',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 5e-4,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run8(glob,l_vals,device,self):

    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "pretrained",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = True,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 5e-4,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc


def run9(glob,l_vals,device,self):
    glob.set(DATASET = "mnist",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "transfer",
            OPT = 'sgd',
            LOSS = 'ce',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 5e-4,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run10(glob,l_vals,device,self):

    glob.set(DATASET = "mnist",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "transfer",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = True,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,               ##REASON FOR CHOOSING THIS
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 5e-4,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc

def run11(glob,l_vals,device,self):
    glob.set(DATASET = "cifar10",
            ROOT = self.root,
            BATCH_SIZE = 64,
            MODEL_TYPE = "cnn",
            OPT = 'dfw',
            LOSS = 'svm',
            SMOOTHING = False,
            EPOCHS = 50,
            LEARNING_RATE = 1e-1,
            DECAY_FACTOR = 0.85,
            TYPE='L2',
            REGPARAM = 5e-4,
            EARLYSTOP = False,
            AUGMENT = True,
            DROPOUT = True,
            DROP_PERCENT = 0.2,
            MOMENTUM  = 0,
            PARAMS_TO_UPDATE = [])

    #l_vals = loadVals()
    #load dataset
    self.loader_train, self.loader_val, self.loader_test = l_vals.g_data(glob)

    #get the model
    self.model,self.nparams = l_vals.g_model(glob,device)

    #get the required loss function
    self.loss = l_vals.g_loss(glob,device)

    #get optimizer
    self.optim = l_vals.g_optim(glob,self.model)

    #training
    train_acc, train_loss, val_acc, val_loss = training(glob,
                                                        self.loader_train,
                                                        self.loader_val,
                                                        self.model,
                                                        self.loss,
                                                        self.optim,
                                                        glob.SMOOTHING,
                                                        device)
    #
    torch.save(self.model.state_dict(), './saved_model/best_dfw.pkl')
    #testing
    test_acc = testing(glob,self.model,self.loader_test,device)
    return train_acc, train_loss, val_acc, val_loss, test_acc