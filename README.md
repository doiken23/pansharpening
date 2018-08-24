# Requirements
- joblib
- labelme
- tifffile

# tools
## merge_landsat.py
LandsatのB2~4までのバンドを重ね合わせてRGB画像を作成する。
## json_to_mask.py
labelmeで作成したjson形式の海マスクをラスタ画像に変換する。
## crop_img.py
RGB画像とL8バンド画像を同時に切り出す。マスクの領域は除外する。  
引数
- data_dir データが格納されているディレクトリ
- out 出力先
- size(option) 切り出し画像サイズ
