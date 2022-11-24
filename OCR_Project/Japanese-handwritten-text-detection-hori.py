#1 各種インポート
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam # ここでエラーとなるので以下のコードに変更
# from tensorflow.keras.optimizers import Adam # 「tensorflow.」を追加
import matplotlib.pyplot as plt
import time


#2 各種設定  
train_data_path = os.path.join(os.path.dirname(__file__),'Japanese_text_dataset/') # ここを変更。
# Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力
image_width = 14   # ここを変更。
# 必要に応じて変更してください。「14」を指定した場合、縦の高さ14ピクセルの画像に変換します
image_height = 14  # ここを変更。
# 必要に応じて変更してください。「14」を指定した場合、横の幅14ピクセルの画像に変換します
# 画像のサイズを変更すると精度が大きく変動するようでした
color_setting = 1  # ここを変更。
# データセット画像のカラー指定：「1」はモノクロ・グレースケール。「3」はカラーとして画像を処理


#3 データセットの読み込みとデータ形式の設定・正規化・分割 

# パス内の全てのファイル・フォルダ名を取得
# folder = glob.glob(train_data_path,recursive=True)
for current_dir, sub_dirs, files_list in os.walk(train_data_path):
  if current_dir == train_data_path:
    fol_list = [os.path.join(current_dir,sub_dirs_item) for sub_dirs_item in sub_dirs]
# 並び替え
folder =  sorted(fol_list)  
print(folder) 

class_number = len(folder)
print('今回のデータで分類するクラス数は「', str(class_number), '」です。')


X_image = []  
Y_label = [] 
for index, name in enumerate(folder):
  read_data = name
  files = glob.glob(read_data + '/*.png') # ここを変更。png形式のファイルを利用する場合のサンプルです。
  print('--- 読み込んだデータセットは',  read_data, 'です。')

  for i, file in enumerate(files):  
    if color_setting == 1:
      img = load_img(file, color_mode = 'grayscale' ,target_size=(image_width, image_height))  
    elif color_setting == 3:
      img = load_img(file, color_mode = 'rgb' ,target_size=(image_width, image_height))
    array = img_to_array(img)
    X_image.append(array)
    Y_label.append(index)

X_image = np.array(X_image)
Y_label = np.array(Y_label)

X_image = X_image.astype('float32') / 255
Y_label = np_utils.to_categorical(Y_label, class_number) 

train_images, valid_images, train_labels, valid_labels = train_test_split(X_image, Y_label, test_size=0.10)
x_train = train_images
y_train = train_labels
x_test = valid_images
y_test = valid_labels


#4 機械学習（人工知能）モデルの作成 – 畳み込みニューラルネットワーク（CNN）・学習の実行等

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',input_shape=(image_width, image_height, color_setting), activation='relu'))
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

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])


start_time = time.time()


# ここを変更。
# 必要に応じて
# 「batch_size=」（バッチサイズ：重みとバイアスの更新を行う間隔の数）「epochs=」（学習回数）の数字を変更してみてください
history = model.fit(x_train,y_train, batch_size=32, epochs=30, verbose=1, validation_data=(x_test, y_test))



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model acc')
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

score = model.evaluate(x_test, y_test, verbose=0)
print('Loss:', score[0], '（損失関数値 - 0に近いほど正解に近い）') 
print('Accuracy:', score[1] * 100, '%', '（精度 - 100% に近いほど正解に近い）') 
print('Computation time（計算時間）:{0:.3f} sec（秒）'.format(time.time() - start_time))


# 学習済みモデル（モデル構造と学習済みの重み）の保存
# 名前は自分がわかりやすい名前にしてください
model.save(f'{image_width}_{color_setting}_model.h5') 

# モノクロ・グレー形式の学習済みモデルの例：color_setting = 1 にした場合  
#model.save('keras_cnn_japanese_handwritten_gray14*14_model.h5')

# カラー形式の学習済みモデルの例：color_setting = 3 にした場合  
#model.save('keras_cnn_japanese_handwritten_color14*14_model.h5') 