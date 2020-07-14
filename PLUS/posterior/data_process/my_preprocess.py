
from LIBS.ImgPreprocess import my_preprocess_dir

dir_original = '/media/ubuntu/data2/未分后极部_2020_6_16/original/ROP训练l图集汇总_20200616_后极部标签修正'
dir_preprocess = '/media/ubuntu/data2/未分后极部_2020_6_16/preprocess384/ROP训练l图集汇总_20200616_后极部标签修正'

my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess, image_size=384,
                                    convert_jpg=False, add_black_pixel_ratio=0.07)



print('OK')

