from . import run

class whichRun():
    def __init__(self,setGlobals,loadVals,device,root) -> None:
        self.setGlobals = setGlobals
        self.loadVals = loadVals
        self.device = device
        self.root = root
        self.loader_train = None
        self.loader_val = None
        self.loader_test = None
        self.model = None
        self.nparams = None
        self.loss = None
        self.optim = None
    
    def run(self,num,reg=0):
        if(num=='1_1'):
            ##run 1
            #####################
            # dataset = cifar10
            # model = cnn
            # opt = sgd
            # no regularisation/early stopping/dropout/augment
            #lr = 1e-1
            #####################
            #glob = setGlobals()
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run1_1(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc

        elif(num=='1_2'):
            #lr = 1e-3
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run1_2(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc
        
        elif(num=='2_1'):
            ##run 2
            #####################
            # dataset = cifar10
            # model = cnn
            # opt = dfw
            # no regularisation/early stopping/dropout/augment
            #####################
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run2_1(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc
        
        elif(num=='2_2'):
            #same as 2_1 but
            #lr = 1e-3
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run2_2(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc

        elif(num=='3'):
            ##run 3
            #####################
            # dataset = cifar10
            # model = cnn
            # opt = sgd
            # with regularisation:
            # early stopping = False
            # dropout = True, 0.2
            # augment = True
            #DIFFERENT REGULARISATION PARAMS
            #####################
            reg = [5e-1, 5e-2, 5e-3, 5e-4]
            temp = [run.run3(self.setGlobals,self.loadVals,self.device,reg[i],self) for i in range(len(reg))]
            return temp

        elif(num=='4'):
            ##run 4
            #####################
            # same as run3
            # opt = dfw
            #DIFFERENT REGULARISATION PARAMS
            #####################
            reg = [5e-1, 5e-2, 5e-3, 5e-4]
            temp = [run.run4(self.setGlobals,self.loadVals,self.device,reg[i],self) for i in range(len(reg))]
            return temp
        
        elif(num=='5'):
            ##run 5
            #####################
            # run3 with L1 regularisation
            #DIFFERENT REGULARISATION PARAMS
            #####################
            reg = [5e-3, 5e-4]
            temp = [run.run5(self.setGlobals,self.loadVals,self.device,reg[i],self) for i in range(len(reg))]
            return temp

        elif(num=='6'):
            ##run 6
            #####################
            # run4 with L1 regularisation
            # opt = dfw
            #DIFFERENT REGULARISATION PARAMS
            #####################
            reg = [5e-3, 5e-4]
            temp = [run.run6(self.setGlobals,self.loadVals,self.device,reg[i],self) for i in range(len(reg))]
            return temp
        
        elif(num=='7'):
            ##run 7
            #####################
            # dataset = cifar10
            # model = pretained vgg                     #main diff
            # with regularisation:
            # early stopping = False
            # dropout = True, 0.2
            # augment = True
            # L2 regparam = 5e-4
            # learning rate 1e-1
            #####################
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run7(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc
        
        elif(num=='8'):
            #same as run7 for opt = 'dfw'
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run8(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc
        
        elif(num=='9'):
            ##run 9
            #####################
            # dataset = mnist
            # model = cnn trained on cifar, finetuning on mnist                #main diff
            # with regularisation:
            # early stopping = False
            # dropout = True, 0.2
            # augment = True
            # L2 regparam = 5e-4
            # learning rate 1e-1
            #####################
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run9(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc
        
        elif(num=='10'):
            #same as run9 for opt = 'dfw'
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run10(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc
        
        elif(num=='11'):
            train_acc, train_loss, val_acc, val_loss, test_acc = run.run11(self.setGlobals,
                                                                            self.loadVals,
                                                                            self.device,
                                                                            self)
            return train_acc, train_loss, val_acc, val_loss, test_acc