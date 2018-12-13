    files = glob.glob(dirname + "/*.jpg")
    for imgfile in files:
        img = load_img(imgfile, target_size=(hw["height"], hw["width"]))    # 画像ファイルの読み込み
        array = img_to_array(img) / 255                                     # 画像ファイルのnumpy化
        arrlist.append(array)                 # numpy型データをリストに追加
        for i in range(var_amount-1):
            arr2 = array
            arr2 = random_rotation(arr2, rg=360)
            arrlist.append(arr2)              # numpy型データをリストに追加
        num += 1