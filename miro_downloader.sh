# download zip file
wget https://data.airc.aist.go.jp/kanezaki.asako/data/MIRO.zip

unzip MIRO.zip

mv ./MIRO/ ./dataset

rm -rf MIRO.zip

python3 redefine_dataset.py