from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from tensorflow.keras import Model


class Conv2Block(Model):
    def __init__(self, filters, kernal_size, pooling_size, drop_value):
        super(Conv2Block, self).__init__(name='')
        self.conv = Conv2D(filters, kernal_size, activation='relu')
        self.maxpool = MaxPool2D(pooling_size)
        self.bn = BatchNormalization()
        self.dropout = Dropout(drop_value)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.maxpool(x)
        x = self.bn(x)
        return self.dropout(x)
        
class Classifier(Model):
    def __init__(self, num_classes, units=128):
        super(Classifier, self).__init__(name='')
        self.flatten = Flatten()
        self.dense1= Dense(units, activation='relu')
        self.bn = BatchNormalization()
        self.drop = Dropout(0.2)
        self.dense2= Dense(128, activation='relu')
        self.out = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.dense2(x)
        return self.out(x)

class MyModel(Model):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        self.conv2block_1 = Conv2Block(filters=32, kernal_size=(3, 3), pooling_size=(2, 2), drop_value=0.2)
        self.conv2block_2 = Conv2Block(filters=64, kernal_size=(3, 3), pooling_size=(2, 2), drop_value=0.2)
        self.conv2block_3 =Conv2Block(filters=128, kernal_size=(3, 3), pooling_size=(2, 2), drop_value=0.2)
        self.conv2block_4 =Conv2Block(filters=256, kernal_size=(3, 3), pooling_size=(2, 2), drop_value=0.2)
        self.conv2block_5 =Conv2Block(filters=512, kernal_size=(3, 3), pooling_size=(2, 2), drop_value=0.2)
        self.classifier = Classifier(num_classes=num_classes, units=512)
    
    def call(self, inputs):
        x = self.conv2block_1(inputs)
        x = self.conv2block_2(x)
        x = self.conv2block_3(x)
        x = self.conv2block_4(x)
        x = self.conv2block_5(x)
        return self.classifier(x)

    
