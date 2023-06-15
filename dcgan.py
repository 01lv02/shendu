import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, concatenate

# 加载数据集并做预处理
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# 构建模型
inputs = Input(shape=(100,))
embedding_layer = Embedding(100000, 128)(inputs)

# 模型 1：使用一维卷积神经网络（CNN）提取文本特征
conv1 = Conv1D(64, 3, activation='relu')(embedding_layer)
conv2 = Conv1D(64, 5, activation='relu')(embedding_layer)
pool1 = GlobalMaxPooling1D()(conv1)
pool2 = GlobalMaxPooling1D()(conv2)

# 模型 2：使用 LSTM 层提取文本特征
lstm_layer = LSTM(128)(embedding_layer)

# 模型 3：使用多层感知机（MLP）提取文本特征
mlp_layer = Dense(64, activation='relu')(embedding_layer)
mlp_layer = Dense(32, activation='relu')(mlp_layer)

# 将三个模型的输出合并，并输入到一个全连接层中进行分类
merged_layer = concatenate([pool1, pool2, lstm_layer, mlp_layer])
output_layer = Dense(1, activation='sigmoid')(merged_layer)

model = Model(inputs=inputs, outputs=output_layer)

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# 评估模型
score, acc = model.evaluate(x_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
