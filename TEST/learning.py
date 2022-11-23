from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout



from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape
from keras.utils import np_utils
import numpy as np

from keras.models import load_model

def learn_MNIST():
    # MNISTデータのダウンロード
    # MNISTは60,000件の訓練用データと10,000件のテストデータを持つ
    # (a, b), (c, d) 2セットのタプル
    # a,c:shape(num_samples,28,28)の白黒画像uint8配列
    # b,d:shape(num_samples,)のカテゴリラベル(1～9)のuint8配列を戻り値として返す
    (X_train, y_train),(X_test, y_test) = mnist.load_data()

    # 0～1の値じゃないと扱えないので、255で割って0～1の範囲の値に変換
    X_train = np.array(X_train)/255
    X_test = np.array(X_test)/255

    # カテゴリラベルをバイナリのダミー変数に変換する
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # モデルの構築
    model = Sequential()

    # MNISTはチャンネル情報を持っていないので畳み込み層に入れるため追加する
    model.add(Reshape((28,28,1),input_shape=(28,28)))

    # ネットワーク生成
    model.add(Conv2D(32,(3,3))) # 畳み込み層1
    model.add(Activation("relu"))
    model.add(Conv2D(32,(3,3))) # 畳み込み層2
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2))) # プーリング層
    model.add(Dropout(0.5))

    model.add(Conv2D(16,(3,3))) # 畳み込み層3
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2))) # プーリング層2
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(784)) # 全結合層
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(10)) # 出力層
    model.add(Activation("softmax"))
    # 学習
    # モデルのコンパイル
    # 損失関数：交差エントロピー、最適化関数：sgd、評価関数：正解率(acc)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    # 学習。バッチサイズ:200、ログ出力：プログレスバー、反復数：50、検証データの割合：0.1
    hist = model.fit(X_train, y_train, batch_size=200,
                    verbose=1, epochs=50, validation_split=0.1)

    # 学習結果の評価。今回は正答率(acc)
    score = model.evaluate(X_test, y_test, verbose=1)
    print("test accuracy：", score[1])

    # モデルを保存,削除
    model.save("kerastest.h5")
    # del model

    return model

def load_MNIST():
    # # MNISTデータをロード
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train = np.array(X_train)/255
    # X_test = np.array(X_test)/255

    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    #学習済モデルをロード
    model = load_model("kerastest.h5")

    # #学習実行
    # hist = model.fit(X_train, y_train, batch_size=200, verbose=1, 
    #                 epochs=10, validation_split=0.1)

    # #評価
    # score = model.evaluate(X_test, y_test, verbose=1)
    # print("正解率(acc)：", score[1])

    # #モデルを保存
    # model.save("kerastest.h5")

    return model