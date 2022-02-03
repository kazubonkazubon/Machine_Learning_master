import sys
import module_1 as func
import cv2
import os
import numpy as np
import glob
import cv2
import numpy as np


def equalizeHistRGB(src):
   RGB = cv2.split(src)
   Blue   = RGB[0]
   Green = RGB[1]
   Red    = RGB[2]
   for i in range(3):
       cv2.equalizeHist(RGB[i])
   img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
   return img_hist

# ガウシアンノイズ
def addGaussianNoise(src):
   row,col,ch= src.shape
   mean = 0
   var = 0.1
   sigma = 15
   gauss = np.random.normal(mean,sigma,(row,col,ch))
   gauss = gauss.reshape(row,col,ch)
   noisy = src + gauss
   return noisy

# salt&pepperノイズ
def addSaltPepperNoise(src):
   row,col,ch = src.shape
   s_vs_p = 0.5
   amount = 0.004
   out = src.copy()
   # Salt mode
   num_salt = np.ceil(amount * src.size * s_vs_p)
   coords = [np.random.randint(0, i-1 , int(num_salt))
                for i in src.shape]
   out[coords[:-1]] = (255,255,255)
   # Pepper mode
   num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
   coords = [np.random.randint(0, i-1 , int(num_pepper))
            for i in src.shape]
   out[coords[:-1]] = (0,0,0)
   return out

def white_cut(z1,org_img):
  #画像の読み込み
  img = cv2.imread(org_img)
  #グレースケール変換
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  #閾値の設定
  threshold_value = 255

  #配列の作成（output用）
  threshold_img = gray.copy()

  #実装(numpy)
  threshold_img[gray < threshold_value] = 0
  threshold_img[gray >= threshold_value] = 255
  #Output
  cv2.imwrite("/content/img.jpg",threshold_img)

  #画像の読み込み
  frame = cv2.imread('/content/img.jpg')

  #白黒反転
  frame_bitwise = cv2.bitwise_not(frame)
  #画像の保存
  cv2.imwrite('/content/img_1.jpg',frame_bitwise)
  for a in z1:
    # マスク対象画像読み込み
    img = cv2.imread(a,cv2.IMREAD_COLOR)

    # マスク画像読み込み
    imgMask = cv2.imread("/content/img_1.jpg",cv2.IMREAD_GRAYSCALE)

    # マスク画像合成
    img[imgMask==0] = [255, 255, 255]  # マスク画像の明度 0 の画素を灰色（R:128 G:128 B:128）で塗りつぶす

    # マスク結果画像を保存
    cv2.imwrite(a, img)

def get_labels():
  labels=[]
  if not os.path.exists("/content/model/labels.txt"):
    labels=glob.glob("/content/based_image_data/*")
    for i in range(len(labels)):
      labels[i]=labels[i].split("/")[-1]

  else :
    f = open("/content/model/labels.txt", 'r')
    datalist = f.readlines()
    for z in datalist:
      labels.append(z[:-1])
  return labels



def train_dulation(i_count,train_white_cut):
  # ルックアップテーブルの生成
  min_table = 50
  max_table = 205
  diff_table = max_table - min_table
  gamma1 = 0.75
  gamma2 = 1.5
  LUT_HC = np.arange(256, dtype = 'uint8' )
  LUT_LC = np.arange(256, dtype = 'uint8' )
  LUT_G1 = np.arange(256, dtype = 'uint8' )
  LUT_G2 = np.arange(256, dtype = 'uint8' )
  LUTs = []
  # 平滑化用
  average_square = (10,10)

  # ハイコントラストLUT作成
  for i in range(0, min_table):
    LUT_HC[i] = 0

  for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table

  for i in range(max_table, 255):
    LUT_HC[i] = 255

  # その他LUT作成
  for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

  LUTs.append(LUT_HC)
  LUTs.append(LUT_LC)
  LUTs.append(LUT_G1)
  LUTs.append(LUT_G2)

  sys.path.append("/content/drive/MyDrive/functions/")
  if i_count==0:
    print("trainデータ水増し")
  image_file_names = []

  labels=get_labels()
  if os.path.exists("/content/train_data"):
    for label in labels:
      image_dirs = func.cmd('ls '+'./train_data/' + label).decode('utf-8')
      image_files = image_dirs.splitlines()
      for image_file in image_files:
        image_file_names.append('./train_data/' + label + '/' + image_file)


  for image_file in image_file_names:
    # 画像の読み込み
    img_src = cv2.imread(image_file, 1)
    trans_img = []
    trans_img.append(img_src)
    # LUT変換
    """
    for i, LUT in enumerate(LUTs):
      trans_img.append(cv2.LUT(img_src, LUT))
    # 平滑化
    trans_img.append(cv2.blur(img_src, average_square))

    # ヒストグラム均一化
    trans_img.append(equalizeHistRGB(img_src))
    """
    # ノイズ付加
    #trans_img.append(addGaussianNoise(img_src))
    #trans_img.append(addSaltPepperNoise(img_src))

    # 反転
    flip_img = []
    for img in trans_img:
      flip_img.append(cv2.flip(img, 1))
    trans_img.extend(flip_img)

    dir_name =  os.path.splitext(os.path.dirname(image_file))[0]
    base_name =  os.path.splitext(os.path.basename(image_file))[0]

    img_src.astype(np.float64)
    train_white_path_list=[]
    for i, img in enumerate(trans_img):
      if i > 0:
        cv2.imwrite(dir_name + '/trans_' + base_name + '_' + str(i-1) + '.jpg' ,img)
        train_white_path_list.append(dir_name + '/trans_' + base_name + '_' + str(i-1) + '.jpg')

    if train_white_cut==True:
      white_cut(train_white_path_list,image_file)

def arc_face_train_dulation():
  # ルックアップテーブルの生成
  min_table = 50
  max_table = 205
  diff_table = max_table - min_table
  gamma1 = 0.75
  gamma2 = 1.5
  LUT_HC = np.arange(256, dtype = 'uint8' )
  LUT_LC = np.arange(256, dtype = 'uint8' )
  LUT_G1 = np.arange(256, dtype = 'uint8' )
  LUT_G2 = np.arange(256, dtype = 'uint8' )
  LUTs = []
  # 平滑化用
  average_square = (10,10)

  # ハイコントラストLUT作成
  for i in range(0, min_table):
    LUT_HC[i] = 0

  for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table

  for i in range(max_table, 255):
    LUT_HC[i] = 255

  # その他LUT作成
  for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

  LUTs.append(LUT_HC)
  LUTs.append(LUT_LC)
  LUTs.append(LUT_G1)
  LUTs.append(LUT_G2)

  sys.path.append("/content/drive/MyDrive/functions/")
  image_file_names = []

  labels=get_labels()
  if os.path.exists("/content/train_data"):
    for label in labels:
      image_dirs = func.cmd('ls '+'./train_data/' + label).decode('utf-8')
      image_files = image_dirs.splitlines()
      for image_file in image_files:
        image_file_names.append('./train_data/' + label + '/' + image_file)


  for image_file in image_file_names:
    # 画像の読み込み
    img_src = cv2.imread(image_file, 1)
    trans_img = []
    trans_img.append(img_src)
    # LUT変換
    """
    for i, LUT in enumerate(LUTs):
      trans_img.append(cv2.LUT(img_src, LUT))
    # 平滑化
    trans_img.append(cv2.blur(img_src, average_square))

    # ヒストグラム均一化
    trans_img.append(equalizeHistRGB(img_src))

    # ノイズ付加
    trans_img.append(addGaussianNoise(img_src))
    trans_img.append(addSaltPepperNoise(img_src))
    """
    # 反転
    flip_img = []
    for img in trans_img:
      flip_img.append(cv2.flip(img, 1))
    trans_img.extend(flip_img)

    dir_name =  os.path.splitext(os.path.dirname(image_file))[0]
    base_name =  os.path.splitext(os.path.basename(image_file))[0]

    img_src.astype(np.float64)
    train_white_path_list=[]
    for i, img in enumerate(trans_img):
      if i > 0:
        cv2.imwrite(dir_name + '/trans_' + base_name + '_' + str(i-1) + '.jpg' ,img)
        train_white_path_list.append(dir_name + '/trans_' + base_name + '_' + str(i-1) + '.jpg')


