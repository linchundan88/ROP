
from LIBS.ImgPreprocess import my_preprocess_dir

dir_original = '/media/ubuntu/data1/ROP_dataset/Hemorrhage/original/'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset/Hemorrhage/preprocess384/'

my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess, image_size=384,
                                    convert_jpg=False, add_black_pixel_ratio=0.07)

# dir_preprocess = '/media/ubuntu/data1/ROP项目/ROP训练图集汇总_20200102_修正20191208出血 +对照/preprocess384_1/'
# my_preprocess_dir.do_process_dir(dir_original, dir_preprocess, image_size=384,
#             is_rop=True, convert_jpg=False, add_black_pixel_ratio=0.07)

print('OK')

