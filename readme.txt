python config.py
で学習のハイパパラメータなどを設定したファイルを生成

python train.py -p Pkl/~.pkl
で設定したハイパパラメータに応じて学習を行う（学習したパラメータをWeightsに保存）

python plot.py -p Weights/~.pt -n 問題サイズ -b 乱数のseed
で学習したパラメータによって例題を解き、図示

python sudden_request.py -p Weights/~.pt -n 問題サイズ -b 乱数のseed
で往診を発生させた問題を解き、図示

Decoder.pyで報酬の重み定数を設定できる(72行目)

