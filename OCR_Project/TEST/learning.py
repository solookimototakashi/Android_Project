

#1 ライブラリのインポート等
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import glob
import numpy as np
from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
#from keras.utils import plot_model #これはKerasのバージョンなどにより使えないのでコメントアウト
from keras.utils import np_utils #keras.utils.to_categoricalでエラーが出るので追加
# from keras.optimizers import Adam # ここでエラーとなるので以下のコードに変更
from keras.optimizers import Adam # 「tensorflow.」を追加
import matplotlib.pyplot as plt
import time

#2 各種設定 

train_data_path = os.path.join(os.path.dirname(__file__),'Japanese_text_dataset') # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

image_size = 64 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。

color_setting = 1  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。

for curDir, dirs, files in os.walk(train_data_path):
    if curDir == train_data_path:
        folders = [curDir + r"\\" + i for i in dirs]

# folder = [] # ここを変更。データセット画像のフォルダ名（クラス名）を半角英数で入力

class_number = len(folders)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')


#3 データセットの読み込みとデータ形式の設定・正規化・分割 

X_image = []  
Y_label = []
index = 0
for foldersItem in folders: 
    for curDir, dirs, files in os.walk(foldersItem):
        if curDir == foldersItem:
            for i, file in enumerate(files):
                if ".png" in file:
                    image_name = curDir + r"\\" + file  
                    if color_setting == 1:
                        img = load_img(image_name, color_mode = 'grayscale' ,target_size=(image_size, image_size))  
                    elif color_setting == 3:
                        img = load_img(image_name, color_mode = 'rgb' ,target_size=(image_size, image_size))
                    array = img_to_array(img)
                    X_image.append(array)
                    Y_label.append(index)
    index += 1

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype('float32') / 255
#Y_label = keras.utils.to_categorical(Y_label, class_number) #Kerasのバージョンなどにより使えないのでコメントアウト
Y_label = np_utils.to_categorical(Y_label, class_number) #上記のコードのかわり

train_images, valid_images, train_labels, valid_labels = train_test_split(X_image, Y_label, test_size=0.10)
x_train = train_images
y_train = train_labels
x_test = valid_images
y_test = valid_labels


#4 機械学習（人工知能）モデルの作成 – 畳み込みニューラルネットワーク（CNN）・学習の実行等

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
          input_shape=(image_size, image_size, color_setting), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))               
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                
model.add(Dropout(0.5))                                   
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))                                 
model.add(Dense(class_number, activation='softmax'))

model.summary()
#plot_model(model, to_file='model.png') #ここはKerasのバージョンなどにより使えないのでコメントアウト

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

start_time = time.time()

# ここを変更。必要に応じて「batch_size=」「epochs=」の数字を変更してみてください。
history = model.fit(x_train,y_train, batch_size=1024, epochs=50, verbose=1, validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# open('cnn_model.json','w').write(model.to_json())
# model.save_weights('cnn_weights.h5')
model.save('cnn_model_weight.h5') #モデル構造と重みを1つにまとめることもできます

score = model.evaluate(x_test, y_test, verbose=0)
print('Loss:', score[0], '（損失関数値 - 0に近いほど正解に近い）') 
print('Accuracy:', score[1] * 100, '%', '（精度 - 100% に近いほど正解に近い）') 
print('Computation time（計算時間）:{0:.3f} sec（秒）'.format(time.time() - start_time))