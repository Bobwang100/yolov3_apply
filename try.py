import os
dir1 = [img for img in os.listdir('./test_tile_result/or-qual/dp-qual')]
dir2 = [img for img in os.listdir('./test_tile_result/or-qual/dp-unqual')]
print('qual:', len(dir1), '\n', 'unqual:', len(dir2), '\n', 'all:',
      (len(dir1) + len(dir2)), '\n' ,'ratio:', len(dir1)/(len(dir1) + len(dir2)))