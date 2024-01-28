
class EarlyStopping():
    def __init__(self, patience = 7, delta = 1e-3):
        self.delta = delta
        self.patience = patience
        self.accuracy = None
        self.count = 0
        self.stop = False
        self.accuracy_increased = True

    def __call__(self, accuracy):
        print(accuracy, self.accuracy)
        if self.accuracy is None:
            self.accuracy = accuracy
        elif accuracy - self.accuracy > self.delta:
            self.accuracy = accuracy
            self.count = 0
            self.accuracy_increased = True
        else:
            self.count += 1
            self.accuracy_increased = False

        if self.count == self.patience:
            self.stop = True