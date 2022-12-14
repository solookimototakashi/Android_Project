from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape
from keras.utils import np_utils
import numpy as np
import cv2

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
del model

# カメラから画像を取得する処理
# 動画表示
# 使用するカメラの指定, 0:内カメラ, 1:外カメラ
# ※PCの機種やUSBカメラの場合等で番号が違う可能性あり
cap = cv2.VideoCapture(1)

# 無限ループ
while(True):
    ret, frame = cap.read()

    # 画像のサイズを取得。グレースケールの場合,shape[:2]
    h, w, _ = frame.shape[:3]

    # 画像の中心点を計算
    w_center = w//2
    h_center = h//2

    # 画像の真ん中に142×142サイズの四角を描く
    cv2.rectangle(frame, (w_center-71, h_center-71), 
                 (w_center+71, h_center+71),(255, 0, 0))
    cv2.imshow("frame",frame) # カメラ画像を表示

    # キーが押下されるのを待つ。1秒置き。64ビットマシンの場合,& 0xFFが必要
    k =  cv2.waitKey(1) & 0xFF

    # ウィンドウのアスペクト比を取得。閉じられると-1.0になる
    prop_val = cv2.getWindowProperty("frame", cv2.WND_PROP_ASPECT_RATIO)

    if k == ord("q") or (prop_val < 0): # qが押下されるorウィンドウが閉じられたら終了
        break
    elif k == ord("s"): # sが押下されたらキャプチャして終了
        # 画像を中心から140×140サイズでトリミング
        # xとyが逆なので注意
        im = frame[h_center-70:h_center+70, w_center-70:w_center+70]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # グレースケールに変換

        # 大津の方法で2値化。retはいらないので取得しない
        _, th = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
        th = cv2.bitwise_not(th) # 白黒反転
        th = cv2.GaussianBlur(th,(9,9), 0) # ガウスブラーをかけて補間
        cv2.imwrite("capture.jpg", th) # 処理が終わった画像を保存
        break

cap.release() # カメラを解放。つかんだままだとハングの原因になる
cv2.destroyAllWindows() # ウィンドウを消す

# 学習完了したモデルで画像判別を試してみる
Xt = []
img = cv2.imread("capture.jpg", 0)
img = cv2.resize(img,(28, 28), cv2.INTER_CUBIC) # 訓練データと同じサイズに整形

Xt.append(img)
Xt = np.array(Xt)/255

model = load_model("kerastest.h5") # 学習済みモデルをロード

# 判定
# predict_classは予測結果のクラスを返す。MNISTは正解ラベル＝クラスなのでそのまま利用
result = model.predict_classes(Xt)
print(result[0])