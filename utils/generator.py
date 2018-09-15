import h5py
import numpy as np

from keras.utils.np_utils import to_categorical

class Generator:
    def __init__(self, path, BS, epochs, preprocessors=None, classes=11):
        self.BS = BS
        self.preprocessors = preprocessors
        self.classes = classes
        self.epochs = epochs
        
        self.db = h5py.File(path)
        self.n_img = self.db["labels"].shape[0]
        
    def generate(self):
        for _ in range(self.epochs):
            for i in np.arange(0, self.n_img, self.BS):
                images = self.db["images"][i:i+self.BS]
                labels = self.db["labels"][i:i+self.BS]
                
                # comment this following line if using generator for 
                # the bounding box detection model
                labels = self.convert_labels(labels)
                
                if self.preprocessors is not None:
                    proc_img = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        
                        proc_img.append(image)
                        
                    images = np.array(proc_img)
                    
                    
                yield images, labels
            
            
    def close(self):
        self.db.close()
        
    def convert_labels(self, labels):
        """
        https://github.com/sajal2692/data-science-portfolio/blob/master/digit_recognition-mnist-sequence.ipynb
        """
        digit1 = np.ndarray(shape=(len(labels), self.classes))
        digit2 = np.ndarray(shape=(len(labels), self.classes))
        digit3 = np.ndarray(shape=(len(labels), self.classes))
        digit4 = np.ndarray(shape=(len(labels), self.classes))
        digit5 = np.ndarray(shape=(len(labels), self.classes))
        seq_len = np.ndarray(shape=(len(labels), 6))
        
        for idx, label in enumerate(labels):
            digit1[idx,:] = to_categorical(label[0], self.classes)
            digit2[idx,:] = to_categorical(label[1], self.classes)
            digit3[idx,:] = to_categorical(label[2], self.classes)
            digit4[idx,:] = to_categorical(label[3], self.classes)
            digit5[idx,:] = to_categorical(label[4], self.classes)
            seq_len[idx,:] = to_categorical(label[5], 6)
            
        return [digit1, digit2, digit3, digit4, digit5, seq_len]